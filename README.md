# Smarter, Not Bigger: Physical Token Dropping (PTD)

Proof-of-concept repository for a transformer variant that physically keeps only top-scored tokens during block execution.

## Repository Layout

```text
.
├── src/
│   ├── transformer_0_5b.py
│   └── sparse_transformer.py
├── benchmarks/
├── eval/
├── docs/
├── prepare_data.py
└── README.md
```

## Quick Start

1) Install dependencies

```bash
pip install torch pandas datasets transformers
```

2) Prepare TinyStories tensor used by eval scripts

```bash
python prepare_data.py --samples 2000 --seq-len 513 --output tinystories_tokenized.pt
```

3) Run eval/benchmark scripts from repo root

```bash
python eval/verify_accuracy.py
python eval/verify_tinystories.py
python eval/true_baseline_accuracy.py
python benchmarks/scientific_validation.py
python benchmarks/oom_boundary_test.py
python benchmarks/true_baseline_full.py
```

## PTD Qwen2.5-0.5B POC

This repo now includes the PTD runtime code and links to a trained checkpoint.

- Runtime: `actual_ptd/`
- Checkpoint + results: `ptd_models/README.md`

Quick eval:

```bash
python -m actual_ptd.eval_perplexity \
  --model Qwen/Qwen2.5-0.5B \
  --data data/tinystories_packed_qwen.pt \
  --checkpoint ptd_v2_phase3_stage4_keep50.pt \
  --keep-rate 0.5
```

## Notes

- Import paths in scripts are now root-safe (`src/` is auto-added).
- `benchmarks/true_baseline_full.py` no longer uses a machine-specific hardcoded path.
- The docs describe the concept and POC results, not a production training stack.

## Core Docs

- [MASTER_POC](docs/MASTER_POC.md)
- [ARCHITECTURE](docs/ARCHITECTURE.md)
- [MATHEMATICAL_PROOFS](docs/MATHEMATICAL_PROOFS.md)
- [WALKTHROUGH](docs/WALKTHROUGH.md)
- [TRAINING_RECIPE](docs/TRAINING_RECIPE.md)
- [SCALABILITY](docs/SCALABILITY.md)
- [PTD_QWEN_POC](docs/PTD_QWEN_POC.md)
