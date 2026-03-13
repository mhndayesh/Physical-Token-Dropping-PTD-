from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import Qwen2ForCausalLM

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTD Phase 2 warmup with business penalties.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--block-size", type=int, default=6)
    p.add_argument("--segment-size", type=int, default=16)
    p.add_argument("--keep-rate", type=float, default=0.5)
    p.add_argument("--router-type", default="mq", choices=["mq", "transformer"])
    p.add_argument("--router-rank", type=int, default=16)
    p.add_argument("--router-queries", type=int, default=8)
    p.add_argument("--router-dim", type=int, default=128)
    p.add_argument("--router-heads", type=int, default=2)
    p.add_argument("--router-layers", type=int, default=1)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--sparsity-reg", type=float, default=1.0)
    p.add_argument("--critical-weight", type=float, default=2.0)
    p.add_argument("--recent-weight", type=float, default=1.0)
    p.add_argument("--diversity-reg", type=float, default=0.05)
    return p.parse_args()


def load_batch(data: Dict[str, torch.Tensor], batch: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
    idx = torch.randint(0, data["input_ids"].shape[0], (batch,))
    x = data["input_ids"][idx].to(device)
    attn = data["attention_mask"][idx].to(device)
    crit = data.get("critical_mask", torch.zeros_like(attn))[idx].to(device)
    recent = data.get("recent_mask", torch.zeros_like(attn))[idx].to(device)
    return x[:, :-1], attn[:, :-1], crit[:, :-1], recent[:, :-1]


def kl_distill(student_logits, teacher_logits, attn_mask, temperature: float):
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.softmax(teacher_logits.float() / temperature, dim=-1)
    kl = F.kl_div(s, t, reduction="none").sum(dim=-1)
    denom = attn_mask.float().sum().clamp_min(1.0)
    return (kl * attn_mask.float()).sum() / denom * (temperature ** 2)


def orth_loss(routers) -> torch.Tensor:
    losses = []
    for r in routers:
        if hasattr(r, "queries"):
            q = F.normalize(getattr(r, "queries").float(), dim=-1)
            sim = q @ q.transpose(0, 1)
            eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
            losses.append(((sim * (1.0 - eye)) ** 2).mean())
    if not losses:
        return torch.zeros(())
    return sum(losses) / len(losses)


def miss_penalty(selection_mask: torch.Tensor, required_mask: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    required = required_mask.bool() & token_mask.bool()
    miss = required & ~selection_mask.bool()
    return miss.float().sum() / required.float().sum().clamp_min(1.0)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    teacher = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device=device, dtype=dtype)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    cfg = PTDConfig(
        block_size=args.block_size,
        segment_size=args.segment_size,
        keep_rate=args.keep_rate,
        router_type=args.router_type,
        router_rank=args.router_rank,
        router_queries=args.router_queries,
        router_dim=args.router_dim,
        router_heads=args.router_heads,
        router_layers=args.router_layers,
        drop_tokens=False,
        ste_gating=False,
        recent_window_tokens=0,
    )
    student = PTDQwen2ForCausalLM.from_pretrained(args.model, ptd_config=cfg, torch_dtype=dtype).to(device=device, dtype=dtype)
    student.freeze_backbone()
    student.train()

    data = torch.load(args.data, weights_only=True)
    opt = AdamW(student.router_parameters(), lr=args.lr)
    os.makedirs("checkpoints", exist_ok=True)
    t0 = time.time()
    hist = []
    for step in range(1, args.steps + 1):
        inp, attn, crit, recent = load_batch(data, args.batch, device)
        with torch.no_grad():
            t_logits = teacher(input_ids=inp, attention_mask=attn).logits
        s_out, aux = student.forward_with_aux(input_ids=inp, attention_mask=attn)
        loss_kl = kl_distill(s_out.logits, t_logits, attn, args.temperature)
        loss_reg = ((aux["gate_means"] - args.keep_rate) ** 2).mean() if aux["gate_means"].numel() > 0 else torch.zeros((), device=device)
        loss_crit = miss_penalty(aux["selection_mask"], crit, aux["token_mask"])
        loss_recent = miss_penalty(aux["selection_mask"], recent, aux["token_mask"])
        loss_div = orth_loss(student.routers).to(device=device)
        loss = loss_kl + args.sparsity_reg * loss_reg + args.critical_weight * loss_crit + args.recent_weight * loss_recent + args.diversity_reg * loss_div
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        hist.append((loss.item(), loss_kl.item(), loss_crit.item(), loss_recent.item()))
        if step % args.log_every == 0:
            sl = hist[-args.log_every:]
            av = [sum(x[i] for x in sl) / len(sl) for i in range(4)]
            print(f"step {step}/{args.steps} | loss {av[0]:.4f} | kl {av[1]:.4f} | crit {av[2]:.4f} | recent {av[3]:.4f} | {time.time()-t0:.1f}s")
        if step % args.save_every == 0 or step == args.steps:
            path = f"checkpoints/ptd_prod_phase2_step{step:06d}.pt"
            torch.save({"step": step, "ptd_config": student.ptd_config_dict(), "router_state": student.routers.state_dict()}, path)
            print(f"saved: {path}")


if __name__ == "__main__":
    main()
