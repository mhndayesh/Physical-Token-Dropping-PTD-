from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

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

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple head-to-head benchmark: dense Qwen vs PTD checkpoint.")
    p.add_argument("--input-jsonl", default="data/restaurant_eval.jsonl")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--keep-rate", type=float, default=0.7)
    p.add_argument("--recent-window", type=int, default=64)
    p.add_argument("--max-examples", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--router-confidence-threshold", type=float, default=0.55)
    p.add_argument("--max-protected-ratio", type=float, default=0.85)
    p.add_argument("--force-ptd", action="store_true", help="Disable safety fallback to dense for PTD benchmarking.")
    p.add_argument("--out-json", default="logs/dense_vs_ptd_results.json")
    return p.parse_args()


def _quantile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(len(s) * q)))
    return float(s[idx])


def _normalize_offsets(offsets) -> List[Tuple[int, int]]:
    if isinstance(offsets, torch.Tensor):
        if offsets.ndim == 3:
            offsets = offsets[0]
        return [(int(s), int(e)) for s, e in offsets.tolist()]
    if isinstance(offsets, list):
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


def load_rows(path: str, max_examples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_examples:
                break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def get_prompt(row: Dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if prompt:
        return str(prompt)
    msgs = row.get("messages", [])
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)


def token_f1(pred: str, ref: str) -> float:
    p = TOKEN_RE.findall(pred.lower())
    r = TOKEN_RE.findall(ref.lower())
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    pc = Counter(p)
    rc = Counter(r)
    overlap = sum((pc & rc).values())
    precision = overlap / max(1, len(p))
    recall = overlap / max(1, len(r))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def critical_hits(pred: str, crit_list: List[str]) -> Tuple[int, int]:
    if not crit_list:
        return 0, 0
    p = pred.lower()
    hits = sum(1 for x in crit_list if str(x).lower() in p)
    return hits, len(crit_list)


def _gpu_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated() / (1024 ** 2))


def _gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def run_dense(
    rows: List[Dict[str, Any]],
    tok,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    dense = Qwen2ForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device=device, dtype=dtype)
    dense.eval()
    load_sec = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    resident_mb = _gpu_mem_mb()

    latencies: List[float] = []
    f1s: List[float] = []
    crit_hit = 0
    crit_tot = 0
    gen_tokens = 0

    with torch.inference_mode():
        for row in rows:
            prompt = get_prompt(row)
            ref = str(row.get("response", row.get("target", ""))).strip()
            crit = list(row.get("critical_spans", []) or row.get("expected_substrings", []) or [])

            enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            attn = torch.ones_like(input_ids, dtype=torch.bool, device=device)

            _gpu_sync()
            s = time.perf_counter()
            out = dense.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(1e-5, args.temperature),
                top_p=args.top_p,
                pad_token_id=tok.eos_token_id,
            )
            _gpu_sync()
            latencies.append(time.perf_counter() - s)

            prompt_len = int(input_ids.shape[1])
            comp_ids = out[0, prompt_len:]
            gen_tokens += int(comp_ids.numel())
            comp_text = tok.decode(comp_ids, skip_special_tokens=True)

            if ref:
                f1s.append(token_f1(comp_text, ref))
            h, t = critical_hits(comp_text, crit)
            crit_hit += h
            crit_tot += t

    total_sec = float(sum(latencies))
    summary = {
        "name": "dense_qwen",
        "examples": len(rows),
        "load_sec": round(load_sec, 4),
        "total_infer_sec": round(total_sec, 4),
        "latency_mean_sec": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "latency_p50_sec": round(_quantile(latencies, 0.50), 4),
        "latency_p95_sec": round(_quantile(latencies, 0.95), 4),
        "generated_tokens_total": gen_tokens,
        "tokens_per_sec": round(gen_tokens / max(total_sec, 1e-6), 4),
        "critical_recall": round((crit_hit / max(1, crit_tot)), 4),
        "critical_hits": int(crit_hit),
        "critical_total": int(crit_tot),
        "response_token_f1_mean": round(statistics.mean(f1s), 4) if f1s else 0.0,
        "vram_resident_mb": round(resident_mb, 2),
        "vram_peak_mb": round(_gpu_peak_mb(), 2),
    }

    del dense
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def run_ptd(
    rows: List[Dict[str, Any]],
    tok,
    model_name: str,
    checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = PTDConfig(
        keep_rate=args.keep_rate,
        recent_window_tokens=args.recent_window,
        router_confidence_threshold=(-1.0 if args.force_ptd else args.router_confidence_threshold),
        max_protected_ratio=(1.0 if args.force_ptd else args.max_protected_ratio),
    )
    ptd = PTDQwen2ForCausalLM.from_pretrained(model_name, ptd_config=cfg, torch_dtype=dtype).to(device=device, dtype=dtype)
    ckpt = torch.load(checkpoint, map_location="cpu")
    if "model_state" in ckpt:
        ptd.load_state_dict(ckpt["model_state"], strict=False)
    elif "router_state" in ckpt:
        ptd.routers.load_state_dict(ckpt["router_state"], strict=False)
    ptd.eval()
    load_sec = time.perf_counter() - t0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    resident_mb = _gpu_mem_mb()

    latencies: List[float] = []
    f1s: List[float] = []
    crit_hit = 0
    crit_tot = 0
    gen_tokens = 0
    fallback_count = 0

    with torch.inference_mode():
        for row in rows:
            prompt = get_prompt(row)
            ref = str(row.get("response", row.get("target", ""))).strip()
            crit = list(row.get("critical_spans", []) or row.get("expected_substrings", []) or [])

            enc = tok(prompt, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
            input_ids = enc["input_ids"].to(device)
            attn = torch.ones_like(input_ids, dtype=torch.bool, device=device)
            mandatory = build_mandatory_mask(prompt, enc).unsqueeze(0).to(device)

            _gpu_sync()
            s = time.perf_counter()
            _, aux = ptd.forward_with_aux(
                input_ids=input_ids,
                attention_mask=attn,
                mandatory_keep_mask=mandatory,
                force_keep_last_n=args.recent_window,
            )

            if ptd.should_fallback(aux):
                fallback_count += 1
                out = ptd.base_model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(1e-5, args.temperature),
                    top_p=args.top_p,
                    pad_token_id=tok.eos_token_id,
                )
                prompt_len = int(input_ids.shape[1])
            else:
                sel = aux["selection_mask"] | mandatory
                if args.recent_window > 0:
                    end = int(attn[0].sum().item())
                    start = max(0, end - args.recent_window)
                    sel[:, start:end] = True
                compact = input_ids[0][sel[0]]
                compact_attn = torch.ones_like(compact, dtype=attn.dtype, device=device)
                out = ptd.base_model.generate(
                    input_ids=compact.unsqueeze(0),
                    attention_mask=compact_attn.unsqueeze(0),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(1e-5, args.temperature),
                    top_p=args.top_p,
                    pad_token_id=tok.eos_token_id,
                )
                prompt_len = int(compact.shape[0])

            _gpu_sync()
            latencies.append(time.perf_counter() - s)

            comp_ids = out[0, prompt_len:]
            gen_tokens += int(comp_ids.numel())
            comp_text = tok.decode(comp_ids, skip_special_tokens=True)

            if ref:
                f1s.append(token_f1(comp_text, ref))
            h, t = critical_hits(comp_text, crit)
            crit_hit += h
            crit_tot += t

    total_sec = float(sum(latencies))
    summary = {
        "name": "ptd_prefill_dense_decode",
        "examples": len(rows),
        "load_sec": round(load_sec, 4),
        "total_infer_sec": round(total_sec, 4),
        "latency_mean_sec": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "latency_p50_sec": round(_quantile(latencies, 0.50), 4),
        "latency_p95_sec": round(_quantile(latencies, 0.95), 4),
        "generated_tokens_total": gen_tokens,
        "tokens_per_sec": round(gen_tokens / max(total_sec, 1e-6), 4),
        "critical_recall": round((crit_hit / max(1, crit_tot)), 4),
        "critical_hits": int(crit_hit),
        "critical_total": int(crit_tot),
        "response_token_f1_mean": round(statistics.mean(f1s), 4) if f1s else 0.0,
        "fallback_rate": round(fallback_count / max(1, len(rows)), 4),
        "vram_resident_mb": round(resident_mb, 2),
        "vram_peak_mb": round(_gpu_peak_mb(), 2),
    }

    del ptd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_jsonl, args.max_examples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)

    dense = run_dense(rows, tok, args.model, device, dtype, args)
    ptd = run_ptd(rows, tok, args.model, args.checkpoint, device, dtype, args)

    speedup = dense["total_infer_sec"] / max(ptd["total_infer_sec"], 1e-6)
    vram_delta = dense["vram_peak_mb"] - ptd["vram_peak_mb"]
    vram_reduction_pct = (vram_delta / max(dense["vram_peak_mb"], 1e-6)) * 100.0

    result = {
        "config": {
            "input_jsonl": args.input_jsonl,
            "examples": len(rows),
            "model": args.model,
            "checkpoint": args.checkpoint,
            "max_new_tokens": args.max_new_tokens,
            "keep_rate": args.keep_rate,
            "recent_window": args.recent_window,
            "router_confidence_threshold": (-1.0 if args.force_ptd else args.router_confidence_threshold),
            "max_protected_ratio": (1.0 if args.force_ptd else args.max_protected_ratio),
            "force_ptd": bool(args.force_ptd),
            "device": str(device),
            "dtype": str(dtype),
        },
        "dense": dense,
        "ptd": ptd,
        "comparison": {
            "speedup_dense_over_ptd": round(speedup, 4),
            "vram_peak_delta_mb_dense_minus_ptd": round(vram_delta, 2),
            "vram_peak_reduction_pct_ptd_vs_dense": round(vram_reduction_pct, 2),
            "critical_recall_delta_ptd_minus_dense": round(ptd["critical_recall"] - dense["critical_recall"], 4),
            "response_f1_delta_ptd_minus_dense": round(ptd["response_token_f1_mean"] - dense["response_token_f1_mean"], 4),
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
