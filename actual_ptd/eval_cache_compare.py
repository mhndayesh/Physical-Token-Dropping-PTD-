from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare dense vs PTD(keep=K) using KV-cache decode.")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--keep-rate", type=float, default=0.7)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--ideal-answer-file", required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--report-json", default="cache_compare_report.json")
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


def load_tokenizer(src: str, local_files_only: bool):
    try:
        return AutoTokenizer.from_pretrained(src, use_fast=True, local_files_only=local_files_only)
    except Exception:
        return AutoTokenizer.from_pretrained(src, use_fast=False, local_files_only=local_files_only)


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


def tensor_tree_bytes(obj: Any, visited: Optional[set[int]] = None) -> int:
    if visited is None:
        visited = set()
    oid = id(obj)
    if oid in visited:
        return 0
    visited.add(oid)
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(tensor_tree_bytes(v, visited) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_tree_bytes(v, visited) for v in obj)
    if hasattr(obj, "__dict__"):
        return tensor_tree_bytes(vars(obj), visited)
    return 0


def metrics_from_decode(
    nll_sum: float,
    n_correct: int,
    n_total: int,
    exact: bool,
) -> Dict[str, Any]:
    ppl = math.exp(nll_sum / max(n_total, 1))
    acc = n_correct / max(n_total, 1)
    return {
        "ppl": float(ppl),
        "acc": float(acc),
        "exact": bool(exact),
    }


def eval_dense_cache(
    model: Qwen2ForCausalLM,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(
            input_ids=prompt_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_sec = time.perf_counter() - t0

        cache = out.past_key_values
        logits_steps: List[torch.Tensor] = [out.logits[:, -1, :]]
        nll_sum = F.cross_entropy(logits_steps[0], answer_ids[:, 0], reduction="sum").item()
        first_correct = int((logits_steps[0].argmax(dim=-1) == answer_ids[:, 0]).sum().item())
        n_correct = first_correct
        n_total = int(answer_ids[:, 0].numel())
        exact = first_correct == n_total

        t1 = time.perf_counter()
        for i in range(answer_ids.size(1) - 1):
            tok = answer_ids[:, i : i + 1]
            out = model(
                input_ids=tok,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
            step_logits = out.logits[:, -1, :]
            logits_steps.append(step_logits)
            target = answer_ids[:, i + 1]
            nll_sum += F.cross_entropy(step_logits, target, reduction="sum").item()
            corr = int((step_logits.argmax(dim=-1) == target).sum().item())
            n_correct += corr
            n_total += int(target.numel())
            if corr != int(target.numel()):
                exact = False
        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_sec = time.perf_counter() - t1

    m = metrics_from_decode(nll_sum, n_correct, n_total, exact)
    m.update(
        {
            "prefill_sec": float(prefill_sec),
            "decode_sec": float(decode_sec),
            "total_sec": float(prefill_sec + decode_sec),
            "decode_tokens_per_sec": float(answer_ids.size(1) / max(decode_sec, 1e-6)),
            "cache_mb_est": float(tensor_tree_bytes(cache) / (1024**2)),
        }
    )
    if device.type == "cuda":
        m["cuda_peak_alloc_mb"] = float(torch.cuda.max_memory_allocated() / (1024**2))
        m["cuda_peak_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024**2))
    return m


def eval_ptd_cache(
    model: PTDQwen2ForCausalLM,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    cache = model.init_ptd_cache()
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
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
        if device.type == "cuda":
            torch.cuda.synchronize()
        prefill_sec = time.perf_counter() - t0

        logits_steps: List[torch.Tensor] = [out.logits[:, -1, :]]
        nll_sum = F.cross_entropy(logits_steps[0], answer_ids[:, 0], reduction="sum").item()
        first_correct = int((logits_steps[0].argmax(dim=-1) == answer_ids[:, 0]).sum().item())
        n_correct = first_correct
        n_total = int(answer_ids[:, 0].numel())
        exact = first_correct == n_total

        t1 = time.perf_counter()
        for i in range(answer_ids.size(1) - 1):
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
            corr = int((step_logits.argmax(dim=-1) == target).sum().item())
            n_correct += corr
            n_total += int(target.numel())
            if corr != int(target.numel()):
                exact = False
        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_sec = time.perf_counter() - t1

    m = metrics_from_decode(nll_sum, n_correct, n_total, exact)
    m.update(
        {
            "prefill_sec": float(prefill_sec),
            "decode_sec": float(decode_sec),
            "total_sec": float(prefill_sec + decode_sec),
            "decode_tokens_per_sec": float(answer_ids.size(1) / max(decode_sec, 1e-6)),
            "cache_mb_est": float(tensor_tree_bytes(cache) / (1024**2)),
        }
    )
    if device.type == "cuda":
        m["cuda_peak_alloc_mb"] = float(torch.cuda.max_memory_allocated() / (1024**2))
        m["cuda_peak_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024**2))
    return m


def compare(dense: Dict[str, Any], ptd: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ppl_delta_pct_vs_dense": float((ptd["ppl"] - dense["ppl"]) / max(dense["ppl"], 1e-12) * 100.0),
        "acc_delta_pct_points_vs_dense": float((ptd["acc"] - dense["acc"]) * 100.0),
        "decode_speed_ratio_ptd_over_dense": float(ptd["decode_tokens_per_sec"] / max(dense["decode_tokens_per_sec"], 1e-12)),
        "total_time_ratio_ptd_over_dense": float(ptd["total_sec"] / max(dense["total_sec"], 1e-12)),
        "cache_mb_ratio_ptd_over_dense": float(ptd["cache_mb_est"] / max(dense["cache_mb_est"], 1e-12)),
    }


def main() -> None:
    args = parse_args()
    device, dtype = pick_device_dtype(args)
    tok_src = args.tokenizer or args.model
    tokenizer = load_tokenizer(tok_src, local_files_only=args.local_files_only)
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

    print("Loading dense model...")
    t_load = time.perf_counter()
    dense = Qwen2ForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device=device, dtype=dtype)
    dense.eval()
    dense_load_sec = time.perf_counter() - t_load
    print(f"Dense loaded in {dense_load_sec:.2f}s")

    dense_metrics = eval_dense_cache(dense, prompt_ids, answer_ids, device)
    del dense
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("Loading PTD model...")
    t_load = time.perf_counter()
    ptd_cfg = PTDConfig(keep_rate=args.keep_rate, drop_tokens=True)
    ptd = PTDQwen2ForCausalLM.from_pretrained(
        args.model,
        ptd_config=ptd_cfg,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device=device, dtype=dtype)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model_state" in ckpt:
        ptd.load_state_dict(ckpt["model_state"], strict=True)
    elif "router_state" in ckpt:
        ptd.routers.load_state_dict(ckpt["router_state"], strict=True)
    else:
        raise ValueError("Unsupported checkpoint format.")
    ptd.eval()
    ptd_load_sec = time.perf_counter() - t_load
    print(f"PTD loaded in {ptd_load_sec:.2f}s")

    ptd_metrics = eval_ptd_cache(ptd, prompt_ids, answer_ids, device)
    del ptd
    if device.type == "cuda":
        torch.cuda.empty_cache()

    report = {
        "config": {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "keep_rate": args.keep_rate,
            "seq_len": args.seq_len,
            "device": str(device),
            "dtype": str(dtype),
            "local_files_only": bool(args.local_files_only),
        },
        "dense_cache": {"load_sec": float(dense_load_sec), **dense_metrics},
        "ptd_cache": {"load_sec": float(ptd_load_sec), **ptd_metrics},
        "comparison_vs_dense": compare(dense_metrics, ptd_metrics),
    }
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report: {args.report_json}")


if __name__ == "__main__":
    main()
