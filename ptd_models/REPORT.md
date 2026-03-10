# PTD POC Report (Qwen2.5-0.5B)

Date: 2026-03-10

## Summary

This POC trains a PTD router on top of Qwen2.5-0.5B and evaluates sparsity at 50% keep.
The model is stable and shows the expected accuracy tradeoff: fewer tokens means higher perplexity.

## Checkpoint

- Keep 50%: https://huggingface.co/mhndayesh/PDT/resolve/main/ptd_v2_phase3_stage4_keep50.pt

## Evaluation (TinyStories packed)

- Dense PPL: 7.813
- Sparse PPL: 10.646
- Delta: +36.25%
- Ratio: 1.36x

## Interpretation (Simple)

Dense is best because it uses all tokens. At 50% keep, the model is faster but less accurate.
The gap is moderate and consistent with expected PTD behavior.

## Reproduce

```bash
python -m actual_ptd.eval_perplexity \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --checkpoint ptd_v2_phase3_stage4_keep50.pt \
  --keep-rate 0.5
```

## Caveats

- This checkpoint requires the custom PTD runtime (`actual_ptd/`).
- It is not a standard HF model.
