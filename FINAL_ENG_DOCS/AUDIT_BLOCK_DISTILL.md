# Audit: Phase 2 Block Distillation Loss Explosion

Date: 2026-03-10

## Symptom
Phase 2 with `--block-distill-weight 1.0` produced very large losses:
- loss ~12k
- blk (block loss) ~12k
- kl ~1.1
- reg/div small

This makes training unstable and hides useful signals.

## Root Cause
The block distillation loss was an unnormalized MSE over all hidden dimensions.
For large hidden sizes (e.g., 1024), the raw MSE sum is huge.
That overwhelms the KL loss unless the weight is extremely small.

## Fix Applied
The block MSE is now normalized by the hidden size:

mse = sum((s - t)^2 * mask) / (mask_sum * hidden_dim)

This brings the block loss to the same scale as KL.

## Recommended Usage
- Start with `--block-distill-weight 0.1` or smaller.
- Check logs for `blk` to ensure it is in the same scale as `kl`.

## Status
Fix applied in `actual_ptd/train_phase2.py` (not committed).
