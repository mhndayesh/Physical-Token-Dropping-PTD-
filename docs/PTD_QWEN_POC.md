# PTD Qwen2.5-0.5B POC

This document summarizes the PTD Qwen2.5-0.5B proof-of-concept using the custom runtime in `actual_ptd/`.

## Assets

- Runtime code: `actual_ptd/`
- Checkpoint link + results: `ptd_models/README.md`
- Scoreboard: `ptd_models/SCOREBOARD.md`
- Report: `ptd_models/REPORT.md`

## HF Checkpoint

- Repo: https://huggingface.co/mhndayesh/PDT
- File: https://huggingface.co/mhndayesh/PDT/resolve/main/ptd_v2_phase3_stage4_keep50.pt

## Quick Eval

```bash
python -m actual_ptd.eval_perplexity \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --checkpoint ptd_v2_phase3_stage4_keep50.pt \
  --keep-rate 0.5
```
