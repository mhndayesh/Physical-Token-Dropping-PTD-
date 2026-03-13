from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from transformers import AutoTokenizer


DEFAULT_CRITICAL_PATTERNS = [
    r"\bno\s+onions\b",
    r"\bextra\s+\w+\b",
    r"\bwithout\s+\w+\b",
    r"\ballergy\b",
    r"\bpeanut\b",
    r"\bvegan\b",
    r"\bgluten[- ]?free\b",
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
    p = argparse.ArgumentParser(description="Build business PTD dataset from JSONL.")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--seq-len", type=int, default=1024, help="training input tokens; saved sequences are seq_len+1")
    p.add_argument("--recent-window", type=int, default=128)
    p.add_argument("--max-examples", type=int, default=0)
    return p.parse_args()


def _render_messages(messages: Iterable[dict]) -> str:
    chunks = []
    for m in messages:
        role = str(m.get("role", "user")).strip().upper()
        content = str(m.get("content", "")).strip()
        chunks.append(f"{role}: {content}")
    return "\n".join(chunks).strip()


def _find_spans(prompt: str, row: dict) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for s in row.get("critical_spans", []) or []:
        s = str(s)
        start = 0
        while s and (idx := prompt.lower().find(s.lower(), start)) >= 0:
            spans.append((idx, idx + len(s)))
            start = idx + len(s)
    for pat in DEFAULT_CRITICAL_PATTERNS:
        for m in re.finditer(pat, prompt, flags=re.IGNORECASE):
            spans.append((m.start(), m.end()))
    merged: List[Tuple[int, int]] = []
    for s, e in sorted(spans):
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def _token_mask_from_char_spans(offsets: List[Tuple[int, int]], spans: List[Tuple[int, int]]) -> List[int]:
    mask = [0] * len(offsets)
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        for a, b in spans:
            if max(s, a) < min(e, b):
                mask[i] = 1
                break
    return mask


def encode_row(tokenizer, row: dict, seq_len: int, recent_window: int):
    if "messages" in row:
        prompt = _render_messages(row["messages"])
    else:
        prompt = str(row.get("prompt", "")).strip()
    response = str(row.get("response", row.get("target", ""))).strip()
    if not prompt or not response:
        return None

    prompt_enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_enc["input_ids"]
    offsets = prompt_enc["offset_mapping"]

    spans = _find_spans(prompt, row)
    crit_prompt = _token_mask_from_char_spans(offsets, spans)
    recent_prompt = [0] * len(prompt_ids)
    if recent_window > 0 and prompt_ids:
        start = max(0, len(prompt_ids) - recent_window)
        for i in range(start, len(prompt_ids)):
            recent_prompt[i] = 1

    eos = []
    if tokenizer.eos_token_id is not None:
        eos = [tokenizer.eos_token_id]

    full_ids = prompt_ids + resp_ids + eos
    total = seq_len + 1
    full_ids = full_ids[:total]
    attn = [1] * len(full_ids)
    crit = (crit_prompt + [0] * (len(resp_ids) + len(eos)))[:total]
    recent = (recent_prompt + [0] * (len(resp_ids) + len(eos)))[:total]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    while len(full_ids) < total:
        full_ids.append(pad_id)
        attn.append(0)
        crit.append(0)
        recent.append(0)

    return {
        "input_ids": full_ids,
        "attention_mask": attn,
        "critical_mask": crit,
        "recent_mask": recent,
    }


def main() -> None:
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if args.max_examples and len(rows) >= args.max_examples:
                break

    packed = {"input_ids": [], "attention_mask": [], "critical_mask": [], "recent_mask": []}
    kept = 0
    for row in rows:
        ex = encode_row(tok, row, args.seq_len, args.recent_window)
        if ex is None:
            continue
        for k in packed:
            packed[k].append(ex[k])
        kept += 1

    if kept == 0:
        raise ValueError("No valid rows found.")

    out = {k: torch.tensor(v, dtype=torch.long if k == "input_ids" else torch.bool) for k, v in packed.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print({k: tuple(v.shape) for k, v in out.items()})
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
