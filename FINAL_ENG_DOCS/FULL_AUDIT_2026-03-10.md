# Full Audit Report (PTD V2)

Date: 2026-03-10
Scope: actual_ptd/ runtime + training + evaluation + new long-context tests.

## Summary
The codebase is functional and the new long-context tests are a strong addition. Most issues are about correctness of auxiliary losses and evaluation methodology rather than crashes. The two largest gaps remain sparse KV-cache (inference) and profiling/optimization work.

## High Severity Findings

1) Coverage penalty has no gradient effect
- Location: actual_ptd/train_phase3.py (coverage_penalty + loss_cov)
- Root cause: segment_selection is derived from hard top-k selection using detached scores. The boolean mask makes coverage_penalty constant w.r.t. router weights, so the loss does not influence training.
- Impact: enabling --coverage-weight does not actually change router behavior.
- Fix: compute coverage on soft scores (sigmoid/softmax) or use straight-through relaxed masks.

2) Block distillation loss scaling (fixed)
- Location: actual_ptd/train_phase2.py
- Root cause: MSE was summed over hidden dimension without normalization, causing loss to explode (~1e4) and dominate KL.
- Fix applied (not committed): normalize by hidden size. Keep block-distill weights small (0.1 or less).

## Medium Severity Findings

3) Per-block keep rates lack validation
- Location: actual_ptd/train_phase3.py + actual_ptd/model.py
- Issue: --per-block-keep length mismatch triggers runtime error in set_keep_rates, but no early validation or clear message.
- Recommendation: validate length against n_blocks after model config load and raise a clear error.

4) Transformer router diversity loss is a no-op
- Location: actual_ptd/train_phase2.py
- Issue: diversity loss only applies to routers with `queries` (MultiQueryRouter). TransformerRouter has no queries so loss=0.0.
- Recommendation: either disable diversity for transformer router or add an alternative regularizer (e.g., attention-head diversity).

5) Long-context batch test performance is limited by repeated JSON load
- Location: actual_ptd/run_long_test_batch.py
- Issue: each sample reloads chat.json from disk. This is fine for small runs but slows large batches.
- Recommendation: cache chat text in memory or precompute prompts.

## Low Severity Observations

6) HF dataset eval packs raw text without EOS/BOS tokens
- Location: actual_ptd/eval_hf_dataset.py
- Impact: acceptable for PPL comparisons but may differ from official LM eval protocols.

7) Router entropy and keep fraction are printed but not persisted
- Location: actual_ptd/eval_perplexity.py
- Impact: no CSV/JSON log for deeper analysis.
- Recommendation: add optional --report-json flag if needed.

## Gaps (Not Yet Implemented)

- Sparse KV-cache for generation. This is required for true inference speedups.
- Fused gather/scatter kernels. Necessary if profiling shows this is a bottleneck.
- Profiling runs across multiple keep rates and sequence lengths.
- Real dataset evaluation beyond TinyStories (WikiText-2 or The Pile slice).

## Recommended Next Steps (In Order)

1) Fix coverage penalty to be differentiable.
2) Add validation for --per-block-keep length.
3) Add transformer-router diversity alternative or disable diversity when router_type=transformer.
4) Run WikiText-2 eval with eval_hf_dataset.py.
5) Run profile_eval.py for dense/PTD to identify bottlenecks.
6) Plan sparse KV-cache implementation (separate engineering effort).

## Notes
- Trainable params in Phase 2 should remain small (router-only). If trainable count is tiny, that is expected.
- Large KL gaps between dense/PTD at long context are expected at 0.5B scale.
