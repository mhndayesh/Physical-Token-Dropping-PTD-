# PTD Qwen2.5-0.5B Models

This folder documents the PTD checkpoints hosted on Hugging Face.

## Hugging Face Repo

- https://huggingface.co/mhndayesh/PDT

## Available Checkpoint

- Keep 50%: `ptd_v2_phase3_stage4_keep50.pt`
  - Direct download: https://huggingface.co/mhndayesh/PDT/resolve/main/ptd_v2_phase3_stage4_keep50.pt

## How To Use

These checkpoints require the PTD runtime code in `actual_ptd/`.

```bash
python -m actual_ptd.eval_perplexity \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --checkpoint ptd_v2_phase3_stage4_keep50.pt \
  --keep-rate 0.5
```

## Notes

These are not standard Hugging Face models. They require the custom PTD forward pass.
