from __future__ import annotations

import argparse
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from transformers import Qwen2ForCausalLM

from actual_ptd import PTDConfig, PTDQwen2ForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile dense vs PTD forward pass.")
    p.add_argument("--mode", choices=["dense", "ptd"], default="ptd")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--keep-rate", type=float, default=0.7)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--data", default="data/tinystories_packed_qwen.pt")
    p.add_argument("--trace-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    if os.path.exists(args.data):
        data = torch.load(args.data, weights_only=True)
        idx = torch.randint(0, data.shape[0], (args.batch,))
        x = data[idx][:, : args.seq_len + 1]
        input_ids = x[:, :-1].to(device)
        attn = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        vocab = 32000
        input_ids = torch.randint(0, vocab, (args.batch, args.seq_len), device=device)
        attn = torch.ones_like(input_ids, dtype=torch.bool)

    if args.mode == "dense":
        model = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device=device, dtype=dtype)
    else:
        cfg = PTDConfig(keep_rate=args.keep_rate, drop_tokens=True)
        model = PTDQwen2ForCausalLM.from_pretrained(args.model, ptd_config=cfg, torch_dtype=dtype).to(
            device=device, dtype=dtype
        )
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"], strict=True)
            elif "router_state" in ckpt:
                model.routers.load_state_dict(ckpt["router_state"], strict=True)
    model.eval()

    prof_schedule = schedule(wait=1, warmup=1, active=max(1, args.steps - 2), repeat=1)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=None if args.trace_dir is None else torch.profiler.tensorboard_trace_handler(args.trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.steps):
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attn)
            if device.type == "cuda":
                torch.cuda.synchronize()
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=20))


if __name__ == "__main__":
    main()
