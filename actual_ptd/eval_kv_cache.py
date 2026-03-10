from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM, PTDSparseCache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate PTD sparse KV-cache correctness/perf.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--keep-rate", type=float, default=0.7)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--ideal-answer-file", required=True)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    p.add_argument("--tokenizer", default=None, help="Optional tokenizer path/name (defaults to --model).")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--report-json", default="kv_cache_report.json")
    return p.parse_args()


def pick_device_dtype(args: argparse.Namespace) -> Tuple[torch.device, torch.dtype]:
    if args.device == "auto":
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = args.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

    if device.type == "cpu" and dtype == torch.bfloat16:
        dtype = torch.float32
    return device, dtype


def build_prompt_answer_tensors(
    tokenizer: Any,
    prompt_path: str,
    answer_path: str,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    with open(answer_path, "r", encoding="utf-8") as f:
        answer_text = f.read().strip()

    answer_tokens = tokenizer.encode(f" {answer_text}\n", add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    if seq_len <= len(answer_tokens) + 1:
        raise ValueError("seq_len too small for answer tokens.")
    max_prompt = seq_len - len(answer_tokens)
    if len(prompt_tokens) > max_prompt:
        prompt_tokens = prompt_tokens[-max_prompt:]

    prompt_ids = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0)
    answer_ids = torch.tensor(answer_tokens, dtype=torch.long).unsqueeze(0)
    return prompt_ids, answer_ids


def compute_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="mean")
    ppl = math.exp(loss.item())
    preds = logits.argmax(dim=-1)
    correct = (preds == labels)
    acc = correct.float().mean().item()
    exact = bool(correct.all().item())
    return {
        "ppl": float(ppl),
        "acc": float(acc),
        "exact": exact,
    }


def eval_no_cache(
    model: PTDQwen2ForCausalLM,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    answer_len = answer_ids.size(1)
    if answer_len > 1:
        full_input = torch.cat([prompt_ids, answer_ids[:, :-1]], dim=1)
    else:
        full_input = prompt_ids

    with torch.no_grad():
        if full_input.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, _ = model.forward_with_aux(
            input_ids=full_input,
            attention_mask=torch.ones_like(full_input, dtype=torch.bool),
            logits_to_keep=answer_len,
        )
        if full_input.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    logits = out.logits
    met = compute_metrics_from_logits(logits, answer_ids)
    met["forward_sec"] = float(elapsed)
    met["mode"] = "sparse_no_cache"
    return met, logits


def cache_stats(cache: PTDSparseCache) -> Dict[str, Any]:
    active_layers = 0
    total_bytes = 0
    max_valid_tokens = 0
    for entry in cache.entries:
        if entry is None:
            continue
        active_layers += 1
        total_bytes += entry.key.numel() * entry.key.element_size()
        total_bytes += entry.value.numel() * entry.value.element_size()
        valid_tokens = int(entry.mask.sum(dim=1).max().item())
        max_valid_tokens = max(max_valid_tokens, valid_tokens)
    return {
        "active_layers": int(active_layers),
        "cache_mb": float(total_bytes / (1024 ** 2)),
        "max_valid_tokens_per_layer": int(max_valid_tokens),
    }


def eval_with_sparse_cache(
    model: PTDQwen2ForCausalLM,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    cache = model.init_ptd_cache()
    answer_len = answer_ids.size(1)
    logits_steps: List[torch.Tensor] = []
    nll_sum = 0.0
    n_correct = 0
    n_total = 0
    exact = True

    with torch.no_grad():
        if prompt_ids.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids, dtype=torch.bool),
            past_key_values=cache,
            use_cache=True,
            ptd_use_sparse_cache=True,
            logits_to_keep=1,
        )
        if prompt_ids.device.type == "cuda":
            torch.cuda.synchronize()
        prefill_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        first_logits = out.logits[:, -1, :]
        logits_steps.append(first_logits)
        first_target = answer_ids[:, 0]
        nll_sum += F.cross_entropy(first_logits, first_target, reduction="sum").item()
        first_pred = first_logits.argmax(dim=-1)
        first_correct = int((first_pred == first_target).sum().item())
        n_correct += first_correct
        n_total += first_target.numel()
        if first_correct != first_target.numel():
            exact = False

        for i in range(answer_len - 1):
            tok = answer_ids[:, i : i + 1]
            out = model(
                input_ids=tok,
                attention_mask=torch.ones_like(tok, dtype=torch.bool),
                past_key_values=cache,
                use_cache=True,
                ptd_use_sparse_cache=True,
                logits_to_keep=1,
            )
            step_logits = out.logits[:, -1, :]
            logits_steps.append(step_logits)
            target = answer_ids[:, i + 1]
            nll_sum += F.cross_entropy(step_logits, target, reduction="sum").item()
            pred = step_logits.argmax(dim=-1)
            corr = int((pred == target).sum().item())
            n_correct += corr
            n_total += target.numel()
            if corr != target.numel():
                exact = False
        if prompt_ids.device.type == "cuda":
            torch.cuda.synchronize()
        decode_sec = time.perf_counter() - t1

    logits = torch.cat([x.unsqueeze(1) for x in logits_steps], dim=1)
    ppl = math.exp(nll_sum / max(n_total, 1))
    met: Dict[str, Any] = {
        "mode": "sparse_kv_cache",
        "ppl": float(ppl),
        "acc": float(n_correct / max(n_total, 1)),
        "exact": bool(exact),
        "prefill_sec": float(prefill_sec),
        "decode_sec": float(decode_sec),
        "total_sec": float(prefill_sec + decode_sec),
        "decode_tokens_per_sec": float(answer_len / max(decode_sec, 1e-6)),
    }
    met.update(cache_stats(cache))
    return met, logits


def main() -> None:
    args = parse_args()
    device, dtype = pick_device_dtype(args)
    tokenizer_src = args.tokenizer or args.model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_src,
            use_fast=True,
            local_files_only=args.local_files_only,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_src,
            use_fast=False,
            local_files_only=args.local_files_only,
        )
    prompt_ids, answer_ids = build_prompt_answer_tensors(
        tokenizer=tokenizer,
        prompt_path=args.prompt_file,
        answer_path=args.ideal_answer_file,
        seq_len=args.seq_len,
    )
    prompt_ids = prompt_ids.to(device)
    answer_ids = answer_ids.to(device)

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Prompt tokens: {prompt_ids.size(1)}")
    print(f"Answer tokens: {answer_ids.size(1)}")

    ptd_cfg = PTDConfig(keep_rate=args.keep_rate, drop_tokens=True)
    model = PTDQwen2ForCausalLM.from_pretrained(
        args.model,
        ptd_config=ptd_cfg,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device=device, dtype=dtype)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
    elif "router_state" in ckpt:
        model.routers.load_state_dict(ckpt["router_state"], strict=True)
    else:
        raise ValueError("Unsupported checkpoint format: expected model_state or router_state.")
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    no_cache_met, no_cache_logits = eval_no_cache(model, prompt_ids, answer_ids)
    kv_cache_met, kv_cache_logits = eval_with_sparse_cache(model, prompt_ids, answer_ids)

    diff = (no_cache_logits.float() - kv_cache_logits.float()).abs()
    compare = {
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "ppl_delta_pct": float((kv_cache_met["ppl"] - no_cache_met["ppl"]) / max(no_cache_met["ppl"], 1e-12) * 100.0),
        "acc_delta_pct_points": float((kv_cache_met["acc"] - no_cache_met["acc"]) * 100.0),
    }

    report: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "keep_rate": args.keep_rate,
            "seq_len": args.seq_len,
            "device": str(device),
            "dtype": str(dtype),
        },
        "sparse_no_cache": no_cache_met,
        "sparse_kv_cache": kv_cache_met,
        "comparison": compare,
    }
    if device.type == "cuda":
        report["cuda_peak_alloc_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        report["cuda_peak_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 ** 2))

    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"No-cache PPL {no_cache_met['ppl']:.4f}, cache PPL {kv_cache_met['ppl']:.4f}, "
        f"logit max diff {compare['max_abs_logit_diff']:.6f}"
    )
    print(f"Wrote report: {args.report_json}")


if __name__ == "__main__":
    main()
