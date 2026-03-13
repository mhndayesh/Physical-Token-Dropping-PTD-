from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import Qwen2ForCausalLM

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTD Phase 3 with business-safe penalties.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data", required=True)
    p.add_argument("--router-ckpt", default=None)
    p.add_argument("--schedule", default="0.99,0.9,0.8,0.7")
    p.add_argument("--steps-per-stage", type=int, default=1500)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--block-size", type=int, default=6)
    p.add_argument("--segment-size", type=int, default=16)
    p.add_argument("--router-type", default="mq", choices=["mq", "transformer"])
    p.add_argument("--router-rank", type=int, default=16)
    p.add_argument("--router-queries", type=int, default=8)
    p.add_argument("--router-dim", type=int, default=128)
    p.add_argument("--router-heads", type=int, default=2)
    p.add_argument("--router-layers", type=int, default=1)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--mask-loss", action="store_true", default=True)
    p.add_argument("--coverage-window", type=int, default=4)
    p.add_argument("--coverage-weight", type=float, default=0.1)
    p.add_argument("--critical-weight", type=float, default=3.0)
    p.add_argument("--recent-weight", type=float, default=1.5)
    return p.parse_args()


def parse_schedule(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def load_batch(data: Dict[str, torch.Tensor], batch: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
    idx = torch.randint(0, data["input_ids"].shape[0], (batch,))
    x = data["input_ids"][idx].to(device)
    attn = data["attention_mask"][idx].to(device)
    crit = data.get("critical_mask", torch.zeros_like(attn))[idx].to(device)
    recent = data.get("recent_mask", torch.zeros_like(attn))[idx].to(device)
    return x[:, :-1], attn[:, :-1], crit[:, :-1], recent[:, :-1]


def kl_distill(student_logits, teacher_logits, token_mask, temperature: float, selection_mask=None, mask_loss=False):
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.softmax(teacher_logits.float() / temperature, dim=-1)
    kl = F.kl_div(s, t, reduction="none").sum(dim=-1)
    eff = token_mask.bool()
    if mask_loss and selection_mask is not None:
        eff = eff & selection_mask.bool()
    return (kl * eff.float()).sum() / eff.float().sum().clamp_min(1.0) * (temperature ** 2)


def coverage_penalty_soft(segment_scores: torch.Tensor, segment_valid: torch.Tensor | None, window: int) -> torch.Tensor:
    if window <= 0 or segment_scores.numel() == 0:
        return torch.zeros((), device=segment_scores.device)
    probs = torch.sigmoid(segment_scores.float())
    if segment_valid is not None and segment_valid.numel() > 0:
        probs = probs * segment_valid.to(probs.dtype)
    n_blocks, bsz, n_seg = probs.shape
    pad = (window - (n_seg % window)) % window
    if pad:
        probs = torch.cat([probs, torch.zeros(n_blocks, bsz, pad, device=probs.device)], dim=-1)
        n_seg = probs.size(-1)
    probs = probs.view(n_blocks, bsz, n_seg // window, window)
    prob_none = (1.0 - probs).clamp(1e-6, 1.0).prod(dim=-1)
    return prob_none.mean()


def miss_penalty(selection_mask: torch.Tensor, required_mask: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    required = required_mask.bool() & token_mask.bool()
    miss = required & ~selection_mask.bool()
    return miss.float().sum() / required.float().sum().clamp_min(1.0)


def main() -> None:
    args = parse_args()
    schedule = parse_schedule(args.schedule)
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
        keep_rate=schedule[0],
        router_type=args.router_type,
        router_rank=args.router_rank,
        router_queries=args.router_queries,
        router_dim=args.router_dim,
        router_heads=args.router_heads,
        router_layers=args.router_layers,
        drop_tokens=True,
        ste_gating=True,
        recent_window_tokens=0,
    )
    student = PTDQwen2ForCausalLM.from_pretrained(args.model, ptd_config=cfg, torch_dtype=dtype).to(device=device, dtype=dtype)
    student.unfreeze_all()
    student.train()

    if args.router_ckpt:
        ckpt = torch.load(args.router_ckpt, map_location="cpu", weights_only=True)
        if "router_state" in ckpt:
            student.routers.load_state_dict(ckpt["router_state"], strict=True)

    data = torch.load(args.data, weights_only=True)
    opt = AdamW(student.parameters(), lr=args.lr)
    os.makedirs("checkpoints", exist_ok=True)
    t0 = time.time()
    gstep = 0

    for stage_idx, keep_rate in enumerate(schedule, start=1):
        student.set_keep_rate(keep_rate)
        print(f"stage {stage_idx}/{len(schedule)} keep={keep_rate:.2f}")
        hist = []
        for step in range(1, args.steps_per_stage + 1):
            gstep += 1
            inp, attn, crit, recent = load_batch(data, args.batch, device)
            with torch.no_grad():
                t_logits = teacher(input_ids=inp, attention_mask=attn).logits
            s_out, aux = student.forward_with_aux(input_ids=inp, attention_mask=attn)
            loss_full = kl_distill(s_out.logits, t_logits, aux["token_mask"], args.temperature, aux["selection_mask"], False)
            loss_sel = kl_distill(s_out.logits, t_logits, aux["token_mask"], args.temperature, aux["selection_mask"], True)
            loss_cov = coverage_penalty_soft(aux["segment_scores"], aux.get("segment_valid"), args.coverage_window)
            loss_crit = miss_penalty(aux["selection_mask"], crit, aux["token_mask"])
            loss_recent = miss_penalty(aux["selection_mask"], recent, aux["token_mask"])
            loss = (loss_sel if args.mask_loss else loss_full) + args.coverage_weight * loss_cov + args.critical_weight * loss_crit + args.recent_weight * loss_recent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            hist.append((loss.item(), loss_sel.item(), loss_crit.item(), loss_recent.item(), loss_cov.item()))
            if step % args.log_every == 0:
                sl = hist[-args.log_every:]
                av = [sum(x[i] for x in sl) / len(sl) for i in range(5)]
                print(f"stage {stage_idx} step {step}/{args.steps_per_stage} | loss {av[0]:.4f} | sel {av[1]:.4f} | crit {av[2]:.4f} | recent {av[3]:.4f} | cov {av[4]:.4f} | {time.time()-t0:.1f}s")
            if gstep % args.save_every == 0:
                path = f"checkpoints/ptd_prod_phase3_step{gstep:06d}.pt"
                torch.save({"global_step": gstep, "keep_rate": keep_rate, "ptd_config": student.ptd_config_dict(), "model_state": student.state_dict()}, path)
                print(f"saved: {path}")
        stage_path = f"checkpoints/ptd_prod_phase3_stage{stage_idx}_keep{int(keep_rate*100)}.pt"
        torch.save({"global_step": gstep, "keep_rate": keep_rate, "ptd_config": student.ptd_config_dict(), "model_state": student.state_dict()}, stage_path)
        print(f"stage complete: {stage_path}")


if __name__ == "__main__":
    main()
