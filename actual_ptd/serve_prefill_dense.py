from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM


HARD_PATTERNS = [
    r"\ballergy\b",
    r"\bpeanut\b",
    r"\bvegan\b",
    r"\bgluten[- ]?free\b",
    r"\bno\s+onions\b",
    r"\bextra\s+\w+\b",
    r"\bwithout\s+\w+\b",
    r"\btable\s+\d+\b",
    r"\border\s*#?\d+\b",
    r"\bpaid\b",
    r"\bcash\b",
    r"\bcard\b",
    r"\brefund\b",
    r"\bcancel\b",
    r"\bdeliver(?:y)?\b",
    r"\baddress\b",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Business-safe PTD serving: prune prompt, dense decode.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--keep-rate", type=float, default=0.7)
    p.add_argument("--recent-window", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    return p.parse_args()


def _normalize_offsets(offsets) -> list[tuple[int, int]]:
    if isinstance(offsets, torch.Tensor):
        # return_tensors="pt" gives [batch, seq, 2]
        if offsets.ndim == 3:
            offsets = offsets[0]
        return [(int(s), int(e)) for s, e in offsets.tolist()]
    if isinstance(offsets, list):
        # non-tensor tokenizers may return [[(s,e), ...]]
        if offsets and isinstance(offsets[0], list):
            offsets = offsets[0]
        return [(int(x[0]), int(x[1])) for x in offsets]
    raise TypeError(f"Unsupported offset_mapping type: {type(offsets)!r}")


def build_mandatory_mask(prompt: str, enc) -> torch.Tensor:
    offsets = _normalize_offsets(enc["offset_mapping"])
    mask = torch.zeros(len(offsets), dtype=torch.bool)
    spans = []
    for pat in HARD_PATTERNS:
        for m in re.finditer(pat, prompt, flags=re.IGNORECASE):
            spans.append((m.start(), m.end()))
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        for a, b in spans:
            if max(s, a) < min(e, b):
                mask[i] = True
                break
    return mask


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    mandatory = build_mandatory_mask(args.prompt, enc).unsqueeze(0).to(device)

    cfg = PTDConfig(keep_rate=args.keep_rate, recent_window_tokens=args.recent_window)
    model = PTDQwen2ForCausalLM.from_pretrained(args.model, ptd_config=cfg, torch_dtype=dtype).to(device=device, dtype=dtype)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state", ckpt.get("router_state"))
    if state is not None and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    elif "router_state" in ckpt:
        model.routers.load_state_dict(ckpt["router_state"], strict=False)
    model.eval()

    generated = model.generate_prefill_dense(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mandatory_keep_mask=mandatory,
        force_keep_last_n=args.recent_window,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(generated[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
