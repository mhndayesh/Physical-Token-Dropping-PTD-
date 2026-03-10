# PTD V2 (Fresh Start)

This directory is a fresh implementation built from the repo concepts, with cleaner contracts for real model work:

- Keeps Hugging Face output contracts in `forward`.
- Uses explicit `forward_with_aux` for PTD-only metadata used by training.
- Honors `attention_mask` in PTD path.
- Preserves generation semantics by delegating cache-based calls to dense forward.
- Uses a clean Phase 2/Phase 3 split aligned with the blueprint.

## Files

- `model.py`: `PTDQwen2ForCausalLM` + `PTDConfig`.
- `train_phase2.py`: Router warm-up (soft routing, no physical drop).
- `train_phase3.py`: Curriculum sparsity with full-model training.
- `eval_perplexity.py`: Dense vs PTD perplexity check.

## Commands

Phase 2:

```bash
python -m actual_ptd.train_phase2 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --steps 3000 --batch 4 --lr 1e-4
```

Note: Phase 2 defaults to true soft gating (`ste_gating=False`) so router receives non-trivial distillation signal.
Phase 2 also applies gate-usage regularization toward `keep-rate` (`--sparsity-reg`, default `1.0`) to avoid pass-all collapse.

Phase 3:

```bash
python -m actual_ptd.train_phase3 --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --router-ckpt checkpoints/ptd_v2_phase2_step003000.pt --batch 2 --lr 1e-5
```

Eval:

```bash
python -m actual_ptd.eval_perplexity --model Qwen/Qwen2.5-0.5B --data data/tinystories_packed_qwen.pt --checkpoint checkpoints/ptd_v2_phase3_stage5_keep30.pt --keep-rate 0.3
```
