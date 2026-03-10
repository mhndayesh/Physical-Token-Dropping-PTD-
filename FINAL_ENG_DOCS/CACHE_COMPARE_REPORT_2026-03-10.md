# Dense vs PTD(Keep=70) Cache Comparison

Date
- 2026-03-10

Model
- Qwen2.5-0.5B (0.5B parameters)

Goal
- Compare dense KV-cache inference against PTD KV-cache inference at keep-rate 70%.
- Use the same prompt, answer, and hardware.

Script
- `actual_ptd/eval_cache_compare.py`

Run commands
```powershell
python -m actual_ptd.eval_cache_compare --model "C:\Users\mhnda\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\060db6499f32faf8b98477b0a26969ef7d8b9987" --tokenizer "C:\Users\mhnda\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\060db6499f32faf8b98477b0a26969ef7d8b9987" --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --prompt-file long_context_test/prompt.txt --ideal-answer-file long_context_test/ideal_answer.txt --seq-len 4096 --local-files-only --report-json cache_compare_4k_keep70.json
```

```powershell
python -m actual_ptd.eval_cache_compare --model "C:\Users\mhnda\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\060db6499f32faf8b98477b0a26969ef7d8b9987" --tokenizer "C:\Users\mhnda\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\060db6499f32faf8b98477b0a26969ef7d8b9987" --checkpoint checkpoints/ptd_v2_phase3_stage3_keep70.pt --keep-rate 0.7 --prompt-file long_context_test/prompt.txt --ideal-answer-file long_context_test/ideal_answer.txt --seq-len 8192 --local-files-only --report-json cache_compare_8k_keep70.json
```

Raw outputs
- `reports/cache_compare_4k_keep70.json`
- `reports/cache_compare_8k_keep70.json`

## 4K Results (Cache vs Cache)

| Metric | Dense | PTD Keep=70 | PTD vs Dense |
| --- | ---: | ---: | ---: |
| PPL | 12.686 | 12.904 | +1.72% |
| Accuracy | 42.86% | 42.86% | 0.00 points |
| Prefill time | 0.779s | 0.148s | 5.25x faster |
| Decode speed | 43.56 tok/s | 37.96 tok/s | 12.86% slower |
| Total time | 1.262s | 0.702s | 44.38% less time |
| Cache size | 47.99 MB | 34.20 MB | 28.73% less |
| Peak VRAM | 3179.56 MB | 1141.95 MB | 64.09% less |

## 8K Results (Cache vs Cache)

| Metric | Dense | PTD Keep=70 | PTD vs Dense |
| --- | ---: | ---: | ---: |
| PPL | 13.184 | 13.468 | +2.16% |
| Accuracy | 47.62% | 42.86% | -4.76 points |
| Prefill time | 2.520s | 0.291s | 8.65x faster |
| Decode speed | 48.90 tok/s | 39.53 tok/s | 19.16% slower |
| Total time | 2.949s | 0.822s | 72.11% less time |
| Cache size | 95.99 MB | 68.35 MB | 28.79% less |
| Peak VRAM | 9535.77 MB | 1377.26 MB | 85.56% less |

## Simple Conclusion

- What PTD wins:
  - Very large prefill speedup for long context.
  - Much lower VRAM use (especially at 8K).
  - Smaller KV cache footprint.
- What PTD loses:
  - Slight quality drop (small PPL increase).
  - Lower decode tokens/sec than dense.

For this 0.5B model, keep=70 gives strong practical tradeoff: close quality with much better long-context cost.
