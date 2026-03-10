from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import Qwen2ForCausalLM

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PTD Phase 2: router warm-up with soft routing.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data", default="data/tinystories_packed_qwen.pt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--block-size", type=int, default=6)
    p.add_argument("--segment-size", type=int, default=16)
    p.add_argument("--keep-rate", type=float, default=0.3)
    p.add_argument("--router-type", default="mq", choices=["mq", "transformer"])
    p.add_argument("--router-rank", type=int, default=16)
    p.add_argument("--router-queries", type=int, default=8)
    p.add_argument("--router-dim", type=int, default=128)
    p.add_argument("--router-heads", type=int, default=2)
    p.add_argument("--router-layers", type=int, default=1)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ste-gating", action="store_true", help="use identity-STE gate instead of soft gate")
    p.add_argument(
        "--sparsity-reg",
        type=float,
        default=1.0,
        help="weight for gate-usage regularizer toward keep-rate (0 disables)",
    )
    p.add_argument(
        "--target-gate",
        type=float,
        default=None,
        help="target mean gate value for regularization (default: keep-rate)",
    )
    p.add_argument(
        "--diversity-reg",
        type=float,
        default=0.0,
        help="weight for router query diversity loss (0 disables)",
    )
    p.add_argument(
        "--block-distill-weight",
        type=float,
        default=0.0,
        help="weight for block-level hidden-state distillation (0 disables)",
    )
    p.add_argument("--save-full-model", action="store_true")
    return p.parse_args()


def get_batch(data: torch.Tensor, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, data.shape[0], (batch,))
    x = data[idx].to(device)
    inp = x[:, :-1]
    attn = torch.ones_like(inp, dtype=torch.bool)
    return inp, attn


def kl_distill(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attn_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    s_log_prob = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_prob = F.softmax(teacher_logits.float() / temperature, dim=-1)
    kl = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=-1)
    kl = (kl * attn_mask.float()).sum() / attn_mask.float().sum().clamp_min(1.0)
    return kl * (temperature ** 2)


def diversity_loss(queries: torch.Tensor) -> torch.Tensor:
    # Encourage query vectors to be orthogonal (low cosine similarity).
    if queries.numel() == 0:
        return torch.zeros((), device=queries.device, dtype=queries.dtype)
    q = F.normalize(queries.float(), dim=-1)
    sim = q @ q.transpose(0, 1)
    eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
    off_diag = sim * (1.0 - eye)
    return (off_diag ** 2).mean()


def router_diversity_loss(routers: Iterable[nn.Module]) -> torch.Tensor:
    losses = []
    router_list = list(routers)
    for router in router_list:
        if hasattr(router, "queries"):
            q = getattr(router, "queries")
            if isinstance(q, torch.Tensor):
                losses.append(diversity_loss(q))
    if not losses:
        try:
            dev = next(router_list[0].parameters()).device
        except Exception:
            dev = torch.device("cpu")
        return torch.zeros((), device=dev)
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    print(f"Device: {device}, dtype={dtype}")
    teacher = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device=device, dtype=dtype)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ptd_cfg = PTDConfig(
        block_size=args.block_size,
        segment_size=args.segment_size,
        keep_rate=args.keep_rate,
        router_type=args.router_type,
        router_rank=args.router_rank,
        router_queries=args.router_queries,
        router_dim=args.router_dim,
        router_heads=args.router_heads,
        router_layers=args.router_layers,
        drop_tokens=False,  # phase 2: no physical token deletion
        ste_gating=args.ste_gating,
    )
    student = PTDQwen2ForCausalLM.from_pretrained(args.model, ptd_config=ptd_cfg, torch_dtype=dtype).to(
        device=device, dtype=dtype
    )
    student.freeze_backbone()
    student.train()

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Missing data file: {args.data}")
    data = torch.load(args.data, weights_only=True)
    print(f"Loaded data: {tuple(data.shape)}")

    optimizer = AdamW(student.router_parameters(), lr=args.lr)
    os.makedirs("checkpoints", exist_ok=True)

    losses = []
    losses_kl = []
    losses_reg = []
    losses_div = []
    losses_block = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        inp, attn = get_batch(data, args.batch, device)

        with torch.no_grad():
            t_out = teacher(input_ids=inp, attention_mask=attn, output_hidden_states=args.block_distill_weight > 0)
            t_logits = t_out.logits

        s_out, aux = student.forward_with_aux(
            input_ids=inp, attention_mask=attn, return_block_hidden=args.block_distill_weight > 0
        )
        loss_kl = kl_distill(s_out.logits, t_logits, attn, args.temperature)
        target_gate = args.keep_rate if args.target_gate is None else args.target_gate
        if aux["gate_means"].numel() > 0 and args.sparsity_reg > 0:
            loss_reg = ((aux["gate_means"] - target_gate) ** 2).mean()
        else:
            loss_reg = torch.zeros((), device=device, dtype=loss_kl.dtype)
        if args.diversity_reg > 0:
            loss_div = router_diversity_loss(student.routers)
        else:
            loss_div = torch.zeros((), device=device, dtype=loss_kl.dtype)
        if args.block_distill_weight > 0 and "block_hidden" in aux:
            t_hid = t_out.hidden_states
            block_hidden = aux["block_hidden"]
            losses_blk = []
            for i, s_h in enumerate(block_hidden):
                layer_idx = (i + 1) * args.block_size
                if layer_idx >= len(t_hid):
                    break
                t_h = t_hid[layer_idx]
                mask = attn.unsqueeze(-1).float()
                mse = ((s_h.float() - t_h.float()) ** 2 * mask).sum()
                denom = mask.sum().clamp_min(1.0) * s_h.size(-1)
                mse = mse / denom
                losses_blk.append(mse)
            loss_block = sum(losses_blk) / max(len(losses_blk), 1)
        else:
            loss_block = torch.zeros((), device=device, dtype=loss_kl.dtype)

        loss = (
            loss_kl
            + args.sparsity_reg * loss_reg
            + args.diversity_reg * loss_div
            + args.block_distill_weight * loss_block
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        losses_kl.append(loss_kl.item())
        losses_reg.append(loss_reg.item())
        losses_div.append(loss_div.item())
        losses_block.append(loss_block.item())
        if step % args.log_every == 0:
            avg = sum(losses[-args.log_every:]) / args.log_every
            avg_kl = sum(losses_kl[-args.log_every:]) / args.log_every
            avg_reg = sum(losses_reg[-args.log_every:]) / args.log_every
            avg_div = sum(losses_div[-args.log_every:]) / args.log_every
            avg_blk = sum(losses_block[-args.log_every:]) / args.log_every
            print(
                f"step {step:>6d}/{args.steps} | loss {avg:.4f} | kl {avg_kl:.4f} | "
                f"reg {avg_reg:.4f} | div {avg_div:.4f} | blk {avg_blk:.4f} | {time.time() - t0:.1f}s"
            )

        if step % args.save_every == 0 or step == args.steps:
            ckpt = {
                "step": step,
                "ptd_config": student.ptd_config_dict(),
                "router_state": student.routers.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": losses[-1],
                "loss_kl": losses_kl[-1],
                "loss_reg": losses_reg[-1],
                "loss_div": losses_div[-1],
                "loss_block": losses_block[-1],
            }
            if args.save_full_model:
                ckpt["model_state"] = student.state_dict()
            path = f"checkpoints/ptd_v2_phase2_step{step:06d}.pt"
            torch.save(ckpt, path)
            print(f"saved: {path}")

    print("Phase 2 complete.")


if __name__ == "__main__":
    main()
