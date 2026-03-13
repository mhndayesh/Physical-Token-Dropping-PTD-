# PTD Production Bridge Update

Date: 2026-03-14

This document records the practical production updates added on top of PTD V2 for real deployment workflows.

## What Was Added

Core runtime updates:
- `actual_ptd/model.py`
  - mandatory keep masks
  - recent-window protection
  - router confidence + protected-ratio fallback checks
  - `generate_prefill_dense()` for sparse prefill + dense decode serving

New data/training/serving scripts:
- `actual_ptd/prepare_business_dataset.py`
- `actual_ptd/train_phase2_business.py`
- `actual_ptd/train_phase3_business.py`
- `actual_ptd/train_full_production.py`
- `actual_ptd/serve_prefill_dense.py`
- `actual_ptd/eval_business_replay.py`
- `actual_ptd/prepare_general_hf_dataset.py`
- `actual_ptd/data_quality_report.py`
- `actual_ptd/compare_dense_vs_ptd.py`
- `actual_ptd/benchmark_long_context.py`

## Bug Fixes Included

- `serve_prefill_dense.py`: fixed tokenizer `offset_mapping` handling that caused:
  - `ValueError: too many values to unpack (expected 2)`
- `eval_business_replay.py`: switched JSONL load to `utf-8-sig` to avoid:
  - `JSONDecodeError: Unexpected UTF-8 BOM`
- `train_full_production.py`: improved orchestration behavior and path fallback handling when `_prod` data names are missing.

## Why This Path

The repository already documents that sparse-cache decode is approximate (not bit-exact).  
For safer operations, this bridge keeps PTD for prompt pruning and then decodes with dense Qwen (`prefill sparse, decode dense`).

## Benchmark Snapshot (Keep70 Checkpoint)

Checkpoint used:
- `checkpoints/ptd_prod_phase3_stage4_keep70.pt` (keep-rate 0.7)

Dense vs PTD replay run (10 eval samples, 96 new tokens):
- Dense
  - mean latency: `1.9674s`
  - throughput: `48.795 tok/s`
  - peak VRAM: `986.52 MB`
  - critical recall: `0.425`
- PTD (default fallback behavior)
  - mean latency: `1.9729s`
  - throughput: `48.6602 tok/s`
  - peak VRAM: `996.66 MB`
  - critical recall: `0.500`
  - fallback rate: `1.0`
- PTD (forced PTD path, fallback disabled)
  - mean latency: `1.9537s`
  - throughput: `49.1377 tok/s`
  - peak VRAM: `996.67 MB`
  - critical recall: `0.500`
  - fallback rate: `0.0`

Long-context benchmark (32 new tokens):
- 4k context
  - Dense latency: `1.6097s`, peak VRAM: `4150.69 MB`
  - PTD latency: `1.1198s`, peak VRAM: `4234.99 MB`
  - PTD speedup: about `1.44x` faster, but slightly higher peak VRAM (`+84 MB`)
- 8k context
  - Dense latency: `22.9846s`, peak VRAM: `10527.85 MB`
  - PTD latency: `25.7360s`, peak VRAM: `8580.28 MB`
  - PTD used about `1947.57 MB` less peak VRAM, but was slower in this run

Reference logs used:
- `logs/dense_vs_ptd_results_10.json`
- `logs/dense_vs_ptd_results_10_force_ptd.json`
- `logs/long_context_dense_vs_ptd_4k_8k.json`

## Notes

- 16k benchmark was started but interrupted; no final `16k` JSON report was produced.
- This bridge is an engineering deployment compromise, not a mathematically exact PTD-to-dense hidden-state handoff.
