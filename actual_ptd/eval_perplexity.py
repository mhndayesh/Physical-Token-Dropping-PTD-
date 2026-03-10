from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense vs PTD perplexity check.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data", default="data/tinystories_packed_qwen.pt")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--keep-rate", type=float, default=0.3)
    p.add_argument("--block-size", type=int, default=6)
    p.add_argument("--segment-size", type=int, default=16)
    p.add_argument("--n-seq", type=int, default=100)
    p.add_argument(
        "--mask-loss",
        dest="mask_loss",
        action="store_true",
        default=True,
        help="compute PTD loss on selected tokens only (recommended)",
    )
    p.add_argument(
        "--full-loss",
        dest="mask_loss",
        action="store_false",
        help="compute PTD loss on all tokens (includes dropped-token penalty)",
    )
    return p.parse_args()


def ppl_dense(model: Qwen2ForCausalLM, data: torch.Tensor, n_seq: int, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_toks = 0
    with torch.no_grad():
        for i in range(min(n_seq, data.shape[0])):
            x = data[i : i + 1].to(device)
            inp, tgt = x[:, :-1], x[:, 1:]
            attn = torch.ones_like(inp, dtype=torch.bool)
            logits = model(input_ids=inp, attention_mask=attn).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_toks += tgt.numel()
    return math.exp(total_loss / max(total_toks, 1))


def ppl_ptd(
    model: PTDQwen2ForCausalLM,
    data: torch.Tensor,
    n_seq: int,
    device: torch.device,
    mask_loss: bool,
) -> tuple[float, float | None, float | None]:
    model.eval()
    total_loss = 0.0
    total_toks = 0
    keep_fracs = []
    entropies = []
    with torch.no_grad():
        for i in range(min(n_seq, data.shape[0])):
            x = data[i : i + 1].to(device)
            inp, tgt = x[:, :-1], x[:, 1:]
            attn = torch.ones_like(inp, dtype=torch.bool)
            out, aux = model.forward_with_aux(input_ids=inp, attention_mask=attn)
            if aux.get("segment_selection") is not None and aux["segment_selection"].numel() > 0:
                keep_fracs.append(aux["segment_selection"].float().mean().item())
            if aux.get("router_entropy") is not None and aux["router_entropy"].numel() > 0:
                entropies.append(aux["router_entropy"].float().mean().item())
            logits = out.logits
            if mask_loss:
                mask = aux["selection_mask"] & aux["token_mask"]
                if mask.any():
                    # apply mask by filtering tokens
                    logits_sel = logits[mask]
                    tgt_sel = tgt[mask]
                    loss = F.cross_entropy(logits_sel, tgt_sel, reduction="sum")
                    total_loss += loss.item()
                    total_toks += int(mask.sum().item())
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
                total_loss += loss.item()
                total_toks += tgt.numel()
    ppl = math.exp(total_loss / max(total_toks, 1))
    keep_mean = sum(keep_fracs) / len(keep_fracs) if keep_fracs else None
    ent_mean = sum(entropies) / len(entropies) if entropies else None
    return ppl, keep_mean, ent_mean


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    data = torch.load(args.data, weights_only=True)
    dense = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device=device, dtype=dtype)
    ptd_cfg = PTDConfig(
        block_size=args.block_size,
        segment_size=args.segment_size,
        keep_rate=args.keep_rate,
        drop_tokens=True,
    )
    sparse = PTDQwen2ForCausalLM.from_pretrained(args.model, ptd_config=ptd_cfg, torch_dtype=dtype).to(
        device=device, dtype=dtype
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state" in ckpt:
            sparse.load_state_dict(ckpt["model_state"], strict=True)
        elif "router_state" in ckpt:
            sparse.routers.load_state_dict(ckpt["router_state"], strict=True)

    dense_ppl = ppl_dense(dense, data, args.n_seq, device)
    sparse_ppl, keep_mean, ent_mean = ppl_ptd(sparse, data, args.n_seq, device, args.mask_loss)
    delta = (sparse_ppl - dense_ppl) / dense_ppl * 100

    print(f"Dense PPL : {dense_ppl:.3f}")
    print(f"Sparse PPL: {sparse_ppl:.3f}")
    print(f"Delta     : {delta:+.2f}%")
    if keep_mean is not None:
        print(f"Avg keep fraction: {keep_mean:.3f}")
    if ent_mean is not None:
        print(f"Avg router entropy: {ent_mean:.3f}")


if __name__ == "__main__":
    main()
