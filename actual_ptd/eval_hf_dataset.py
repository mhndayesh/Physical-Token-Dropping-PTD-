from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen2ForCausalLM

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate dense vs PTD on a Hugging Face dataset.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--keep-rate", type=float, default=0.7)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--subset", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="test")
    p.add_argument("--text-field", default="text")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--n-seq", type=int, default=100)
    p.add_argument("--mask-loss", action="store_true", default=True)
    p.add_argument("--full-loss", dest="mask_loss", action="store_false")
    return p.parse_args()


def pack_sequences(tokenizer, texts, seq_len: int, n_seq: int) -> torch.Tensor:
    ids = []
    seqs = []
    for text in texts:
        if not text:
            continue
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        while len(ids) >= seq_len + 1 and len(seqs) < n_seq:
            seqs.append(ids[: seq_len + 1])
            ids = ids[seq_len + 1 :]
        if len(seqs) >= n_seq:
            break
    if not seqs:
        raise ValueError("No sequences packed. Check dataset and seq_len.")
    return torch.tensor(seqs, dtype=torch.long)


def ppl_dense(model: Qwen2ForCausalLM, data: torch.Tensor, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_toks = 0
    with torch.no_grad():
        for i in range(data.shape[0]):
            x = data[i : i + 1].to(device)
            inp, tgt = x[:, :-1], x[:, 1:]
            attn = torch.ones_like(inp, dtype=torch.bool)
            logits = model(input_ids=inp, attention_mask=attn).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_toks += tgt.numel()
    return math.exp(total_loss / max(total_toks, 1))


def ppl_ptd(model: PTDQwen2ForCausalLM, data: torch.Tensor, device: torch.device, mask_loss: bool) -> float:
    model.eval()
    total_loss = 0.0
    total_toks = 0
    with torch.no_grad():
        for i in range(data.shape[0]):
            x = data[i : i + 1].to(device)
            inp, tgt = x[:, :-1], x[:, 1:]
            attn = torch.ones_like(inp, dtype=torch.bool)
            out, aux = model.forward_with_aux(input_ids=inp, attention_mask=attn)
            logits = out.logits
            if mask_loss:
                mask = aux["selection_mask"] & aux["token_mask"]
                if mask.any():
                    logits_sel = logits[mask]
                    tgt_sel = tgt[mask]
                    loss = F.cross_entropy(logits_sel, tgt_sel, reduction="sum")
                    total_loss += loss.item()
                    total_toks += int(mask.sum().item())
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
                total_loss += loss.item()
                total_toks += tgt.numel()
    return math.exp(total_loss / max(total_toks, 1))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    ds = load_dataset(args.dataset, args.subset, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    texts = ds[args.text_field][: args.n_seq * 10]
    packed = pack_sequences(tokenizer, texts, args.seq_len, args.n_seq)

    dense = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device=device, dtype=dtype)
    ptd_cfg = PTDConfig(keep_rate=args.keep_rate, drop_tokens=True)
    sparse = PTDQwen2ForCausalLM.from_pretrained(args.model, ptd_config=ptd_cfg, torch_dtype=dtype).to(
        device=device, dtype=dtype
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state" in ckpt:
            sparse.load_state_dict(ckpt["model_state"], strict=True)
        elif "router_state" in ckpt:
            sparse.routers.load_state_dict(ckpt["router_state"], strict=True)

    dense_ppl = ppl_dense(dense, packed, device)
    sparse_ppl = ppl_ptd(sparse, packed, device, args.mask_loss)
    delta = (sparse_ppl - dense_ppl) / dense_ppl * 100

    print(f"Dense PPL : {dense_ppl:.3f}")
    print(f"Sparse PPL: {sparse_ppl:.3f}")
    print(f"Delta     : {delta:+.2f}%")


if __name__ == "__main__":
    main()
