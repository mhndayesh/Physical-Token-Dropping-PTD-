# Evaluation and Results

Evaluation method
- Dense model PPL is computed with cross-entropy over all tokens.
- PTD model PPL is computed on selected tokens only (default in eval_perplexity.py).
- This matches the Phase 3 training objective and avoids penalizing dropped tokens.

Why full-token PPL explodes
When keep-rate is low, dropped tokens are not computed by PTD. If you evaluate over all tokens, the loss includes positions the model never processed, so the PPL looks extremely large. That is expected and not a training failure.

Results summary (TinyStories packed)
Model: Qwen/Qwen2.5-0.5B
Eval script: actual_ptd/eval_perplexity.py (selected-token loss)
Dense baseline PPL: 7.813

Baseline results (no coverage penalty)
| Keep-rate | Checkpoint | Sparse PPL | Delta vs Dense | Ratio |
| 70% | checkpoints/ptd_v2_phase3_stage3_keep70.pt | 9.358 | +19.77% | 1.20x |
| 50% | checkpoints/ptd_v2_phase3_stage4_keep50.pt | 10.646 | +36.25% | 1.36x |
| 30% | checkpoints/ptd_v2_phase3_stage5_keep30.pt | 12.698 | +62.52% | 1.63x |

Coverage penalty results (coverage-window=4, coverage-weight=0.1)
| Keep-rate | Checkpoint | Sparse PPL | Delta vs Dense | Ratio |
| 70% | checkpoints/ptd_v2_phase3_stage3_keep70.pt | 9.450 | +20.95% | 1.21x |
| 50% | checkpoints/ptd_v2_phase3_stage4_keep50.pt | 10.631 | +36.06% | 1.36x |
| 30% | checkpoints/ptd_v2_phase3_stage5_keep30.pt | 12.466 | +59.55% | 1.60x |

Interpretation
Lower PPL is better. As keep-rate decreases, accuracy drops because the model uses fewer tokens. The current results show a reasonable tradeoff curve for a small POC.

Relevant code
- actual_ptd/eval_perplexity.py
- PTD_SCOREBOARD.md (root)

Cache-mode comparison (dense vs PTD keep=70)
Model: Qwen2.5-0.5B
Script: actual_ptd/eval_cache_compare.py

4K context
| Metric | Dense | PTD 70% | PTD vs Dense |
| --- | ---: | ---: | ---: |
| PPL | 12.686 | 12.904 | +1.72% |
| Accuracy | 42.86% | 42.86% | 0.00 points |
| Total time | 1.262s | 0.702s | 44.38% less |
| Peak VRAM | 3179.56 MB | 1141.95 MB | 64.09% less |

8K context
| Metric | Dense | PTD 70% | PTD vs Dense |
| --- | ---: | ---: | ---: |
| PPL | 13.184 | 13.468 | +2.16% |
| Accuracy | 47.62% | 42.86% | -4.76 points |
| Total time | 2.949s | 0.822s | 72.11% less |
| Peak VRAM | 9535.77 MB | 1377.26 MB | 85.56% less |

Detailed report
- FINAL_ENG_DOCS/CACHE_COMPARE_REPORT_2026-03-10.md
