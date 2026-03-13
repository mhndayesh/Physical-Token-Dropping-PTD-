from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Dense Qwen vs PTD on long contexts (e.g., 4k/8k tokens).")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source-jsonl", default="data/general_train.jsonl")
    p.add_argument("--lengths", default="4096,8192")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--keep-rate", type=float, default=0.7)
    p.add_argument("--recent-window", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--force-ptd", action="store_true", help="Disable PTD fallback for pure PTD path benchmarking.")
    p.add_argument("--warmup", action="store_true", help="Run one warmup call before each timed run.")
    p.add_argument("--out-json", default="logs/long_context_dense_vs_ptd.json")
    return p.parse_args()


def _gpu_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_peak_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def _load_source_text(path: str, min_rows: int = 100) -> List[str]:
    rows: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", row.get("target", ""))).strip()
            text = f"{prompt}\n{response}".strip()
            if text:
                rows.append(text)
            if len(rows) >= min_rows:
                # enough for 8k construction
                break
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    return rows


def _build_prompt_ids(tokenizer, texts: List[str], target_len: int) -> torch.LongTensor:
    ids: List[int] = []
    i = 0
    while len(ids) < target_len:
        text = texts[i % len(texts)]
        chunk = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not chunk:
            i += 1
            continue
        ids.extend(chunk)
        if tokenizer.eos_token_id is not None:
            ids.append(int(tokenizer.eos_token_id))
        i += 1
    ids = ids[:target_len]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def _run_dense_once(
    model,
    input_ids: torch.LongTensor,
    attn: torch.Tensor,
    tokenizer,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _gpu_sync()
    t0 = time.perf_counter()
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=max(1e-5, args.temperature),
        top_p=args.top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    _gpu_sync()
    dt = time.perf_counter() - t0
    gen_tokens = int(out.shape[1] - input_ids.shape[1])
    return {
        "latency_sec": float(dt),
        "generated_tokens": gen_tokens,
        "tokens_per_sec": float(gen_tokens / max(dt, 1e-6)),
        "vram_peak_mb": _gpu_peak_mb(),
    }


def _run_ptd_once(
    model: PTDQwen2ForCausalLM,
    input_ids: torch.LongTensor,
    attn: torch.Tensor,
    tokenizer,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _gpu_sync()
    t0 = time.perf_counter()

    with torch.inference_mode():
        _, aux = model.forward_with_aux(
            input_ids=input_ids,
            attention_mask=attn,
            mandatory_keep_mask=None,
            force_keep_last_n=args.recent_window,
        )
        fallback = bool(model.should_fallback(aux))
        if args.force_ptd:
            fallback = False

        if fallback:
            out = model.base_model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(1e-5, args.temperature),
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            prompt_len = int(input_ids.shape[1])
        else:
            sel = aux["selection_mask"]
            if args.recent_window > 0:
                end = int(attn[0].sum().item())
                start = max(0, end - args.recent_window)
                sel[:, start:end] = True
            compact = input_ids[0][sel[0]]
            compact_attn = torch.ones_like(compact, dtype=attn.dtype, device=attn.device)
            out = model.base_model.generate(
                input_ids=compact.unsqueeze(0),
                attention_mask=compact_attn.unsqueeze(0),
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(1e-5, args.temperature),
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            prompt_len = int(compact.shape[0])

    _gpu_sync()
    dt = time.perf_counter() - t0
    gen_tokens = int(out.shape[1] - prompt_len)
    return {
        "latency_sec": float(dt),
        "generated_tokens": gen_tokens,
        "tokens_per_sec": float(gen_tokens / max(dt, 1e-6)),
        "vram_peak_mb": _gpu_peak_mb(),
        "fallback_used": fallback,
    }


def main() -> None:
    args = parse_args()
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    if not lengths:
        raise ValueError("No valid lengths provided.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)

    texts = _load_source_text(args.source_jsonl)
    prompts = {L: _build_prompt_ids(tok, texts, L).to(device) for L in lengths}

    dense = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device=device, dtype=dtype)
    dense.eval()

    cfg = PTDConfig(
        keep_rate=args.keep_rate,
        recent_window_tokens=args.recent_window,
        router_confidence_threshold=(-1.0 if args.force_ptd else 0.55),
        max_protected_ratio=(1.0 if args.force_ptd else 0.85),
    )
    ptd = PTDQwen2ForCausalLM.from_pretrained(args.model, ptd_config=cfg, torch_dtype=dtype).to(device=device, dtype=dtype)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state" in ckpt:
        ptd.load_state_dict(ckpt["model_state"], strict=False)
    elif "router_state" in ckpt:
        ptd.routers.load_state_dict(ckpt["router_state"], strict=False)
    ptd.eval()

    results: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "lengths": lengths,
            "max_new_tokens": args.max_new_tokens,
            "keep_rate": args.keep_rate,
            "recent_window": args.recent_window,
            "force_ptd": bool(args.force_ptd),
            "device": str(device),
            "dtype": str(dtype),
        },
        "runs": [],
    }

    for L in lengths:
        ids = prompts[L]
        attn = torch.ones_like(ids, dtype=torch.bool, device=device)

        if args.warmup:
            _ = _run_dense_once(dense, ids, attn, tok, args)
            _ = _run_ptd_once(ptd, ids, attn, tok, args)

        dense_res = _run_dense_once(dense, ids, attn, tok, args)
        ptd_res = _run_ptd_once(ptd, ids, attn, tok, args)

        entry = {
            "context_tokens": L,
            "dense": dense_res,
            "ptd": ptd_res,
            "comparison": {
                "latency_speedup_dense_over_ptd": round(dense_res["latency_sec"] / max(ptd_res["latency_sec"], 1e-6), 4),
                "vram_peak_delta_mb_dense_minus_ptd": round(dense_res["vram_peak_mb"] - ptd_res["vram_peak_mb"], 2),
                "tokens_per_sec_delta_ptd_minus_dense": round(ptd_res["tokens_per_sec"] - dense_res["tokens_per_sec"], 4),
            },
        }
        results["runs"].append(entry)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
