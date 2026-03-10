# PTD Accuracy Scoreboard (Sparse vs Dense)

Date: 2026-03-10
Model: Qwen/Qwen2.5-0.5B
Dataset: data/tinystories_packed_qwen.pt
Eval script: actual_ptd/eval_perplexity.py (selected-token loss)

## Results

- Dense PPL (baseline): 7.813

## Baseline (no coverage penalty)

### Keep 70% (stage3_keep70)
- Checkpoint: checkpoints/ptd_v2_phase3_stage3_keep70.pt
- Sparse PPL: 9.358
- Delta vs dense: +19.77%
- Ratio (Sparse / Dense): 1.20x

### Keep 50% (stage4_keep50)
- Checkpoint: checkpoints/ptd_v2_phase3_stage4_keep50.pt
- Sparse PPL: 10.646
- Delta vs dense: +36.25%
- Ratio (Sparse / Dense): 1.36x

### Keep 30% (stage5_keep30)
- Checkpoint: checkpoints/ptd_v2_phase3_stage5_keep30.pt
- Sparse PPL: 12.698
- Delta vs dense: +62.52%
- Ratio (Sparse / Dense): 1.63x

## Coverage penalty (coverage-window=4, coverage-weight=0.1)

### Keep 70% (stage3_keep70)
- Checkpoint: checkpoints/ptd_v2_phase3_stage3_keep70.pt
- Sparse PPL: 9.450
- Delta vs dense: +20.95%
- Ratio (Sparse / Dense): 1.21x

### Keep 50% (stage4_keep50)
- Checkpoint: checkpoints/ptd_v2_phase3_stage4_keep50.pt
- Sparse PPL: 10.631
- Delta vs dense: +36.06%
- Ratio (Sparse / Dense): 1.36x

### Keep 30% (stage5_keep30)
- Checkpoint: checkpoints/ptd_v2_phase3_stage5_keep30.pt
- Sparse PPL: 12.466
- Delta vs dense: +59.55%
- Ratio (Sparse / Dense): 1.60x

## Simple Explanation

Lower PPL is better. Dense is still best because it uses all tokens.
As we drop more tokens, accuracy gets worse:
- 70% keep: small drop (about 20-21% worse)
- 50% keep: medium drop (about 36% worse)
- 30% keep: larger drop (about 60-63% worse)

So your model is behaving correctly: more speed (fewer tokens) means lower accuracy.

## Notes

- These numbers are computed on selected tokens only (same objective as training).
- Full-token PPL is not the training target and will look extremely large.

## Dense vs PTD Cache Inference (Keep 70%, 0.5B model)

Method
- Script: `actual_ptd/eval_cache_compare.py`
- Prompt and answer: `long_context_test/prompt.txt`, `long_context_test/ideal_answer.txt`
- Cache mode for both models (dense KV-cache vs PTD sparse KV-cache)

### 4K context
- Dense: PPL 12.686, accuracy 42.86%, total 1.262s, peak VRAM 3179.56 MB
- PTD 70%: PPL 12.904, accuracy 42.86%, total 0.702s, peak VRAM 1141.95 MB
- PTD vs dense:
  - PPL: +1.72%
  - Accuracy: 0.00 points
  - Total latency: 44.38% lower
  - Peak VRAM: 64.09% lower

### 8K context
- Dense: PPL 13.184, accuracy 47.62%, total 2.949s, peak VRAM 9535.77 MB
- PTD 70%: PPL 13.468, accuracy 42.86%, total 0.822s, peak VRAM 1377.26 MB
- PTD vs dense:
  - PPL: +2.16%
  - Accuracy: -4.76 points
  - Total latency: 72.11% lower
  - Peak VRAM: 85.56% lower
