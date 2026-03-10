# PTD Accuracy Scoreboard

Date: 2026-03-10
Model: Qwen2.5-0.5B
Dataset: data/tinystories_packed_qwen.pt
Eval: actual_ptd/eval_perplexity.py (selected-token loss)

## Baseline

- Dense PPL: 7.813

## PTD Checkpoint (Keep 50%)

- HF Repo: https://huggingface.co/mhndayesh/PDT
- File: https://huggingface.co/mhndayesh/PDT/resolve/main/ptd_v2_phase3_stage4_keep50.pt
- Sparse PPL: 10.646
- Delta vs dense: +36.25%
- Ratio (Sparse / Dense): 1.36x
