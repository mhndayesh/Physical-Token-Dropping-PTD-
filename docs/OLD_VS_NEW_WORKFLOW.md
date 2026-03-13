# PTD Workflow Report: Old Way vs New Way

Date: 2026-03-14

This report explains the difference between the repository's older PTD V2 workflow and the new production bridge workflow that was added.

## Executive Summary

- Old way optimized for PTD research and sparse-cache benchmarking.
- New way optimized for practical deployment safety and easier operations.
- Main change: move serving to **sparse prefill + dense decode** with explicit fallback checks.

## 1) What "Old Way" Means

The older path in `actual_ptd` is centered on:
- `train_phase2.py`
- `train_phase3.py`
- `eval_cache_compare.py`
- `eval_kv_cache.py`

Characteristics:
- Trains router and sparse behavior with generic PTD curriculum.
- Strong focus on sparse-cache and long-context benchmark experiments.
- Decode path can use sparse cache mode, which is documented as approximate (not bit-exact to full PTD forward).

## 2) What "New Way" Means

The new production bridge path adds:
- `prepare_business_dataset.py`
- `train_phase2_business.py`
- `train_phase3_business.py`
- `train_full_production.py`
- `serve_prefill_dense.py`
- `eval_business_replay.py`
- `prepare_general_hf_dataset.py`
- `data_quality_report.py`
- `compare_dense_vs_ptd.py`
- `benchmark_long_context.py`

Characteristics:
- Adds mandatory keep masks for critical text and recent window protection.
- Adds router-confidence and protected-ratio fallback logic.
- Serving uses `generate_prefill_dense()`:
  - PTD prunes prompt/prefill tokens.
  - Base dense model performs decode.
- Built to reduce deployment risk and keep behavior predictable.

## 3) Side-by-Side Comparison

| Area | Old Way (PTD V2 baseline) | New Way (Production Bridge) |
| --- | --- | --- |
| Primary goal | PTD algorithm training/eval | Safe deployable PTD integration |
| Training scripts | `train_phase2.py`, `train_phase3.py` | `train_phase2_business.py`, `train_phase3_business.py`, `train_full_production.py` |
| Data pipeline | Existing packed datasets / eval scripts | JSONL-first pipeline + HF data builder + quality report |
| Critical token protection | Limited/implicit | Explicit mandatory mask + recent window enforcement |
| Fallback safety | Not central in old flow | Router-confidence + protected-ratio fallback checks |
| Decode path | Can use sparse-cache decode | Dense decode after sparse prefill pruning |
| Operational risk | Better for research experiments | Better for production reliability |
| Exactness vs full PTD | Sparse-cache decode is approximate | Also a compromise (prefill prune, dense decode), but operationally safer |

## 4) Why We Switched

- The repo already states sparse-cache decode is approximate.
- For production, incorrect omission of important tokens is costly.
- The new flow keeps PTD's prefill efficiency ideas while making decode conservative and robust.

## 5) Concrete Behavior Changes in Runtime

In `actual_ptd/model.py`, the new path introduces:
- `mandatory_keep_mask` handling in forward-with-aux path.
- `force_keep_last_n` / `recent_window_tokens` enforcement.
- `should_fallback(aux)` with:
  - `router_confidence_threshold`
  - `max_protected_ratio`
- `generate_prefill_dense()` helper for deployment serving.

## 6) Measured Outcomes (Current Logs)

Checkpoint used:
- `checkpoints/ptd_prod_phase3_stage4_keep70.pt` (Keep70 = 70% target keep rate)

From saved logs:
- `logs/dense_vs_ptd_results_10.json`
- `logs/dense_vs_ptd_results_10_force_ptd.json`
- `logs/long_context_dense_vs_ptd_4k_8k.json`

Observed:
- Small-sample quality improved on critical recall in PTD bridge runs vs dense baseline.
- 4k/8k latency and VRAM tradeoffs vary by context length; PTD is not always faster, but can reduce VRAM significantly at longer context.
- No completed 16k result is included (run was intentionally not completed to avoid system freeze).

## 7) Recommendation

Use the **new production bridge workflow** for real deployment:
- train: `train_full_production.py`
- serve: `serve_prefill_dense.py`
- evaluate: `compare_dense_vs_ptd.py` + `eval_business_replay.py`

Use the **old workflow** for PTD research experiments and legacy sparse-cache comparisons.
