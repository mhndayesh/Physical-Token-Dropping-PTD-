# PTD Production Training Commands (One-Line PowerShell)

`train_full_production` is an orchestrator. It intentionally runs these scripts in sequence:
1. dataset tensor preparation
2. phase2 router warm-up
3. phase3 sparsity training

## 0) Setup

```powershell
cd C:\udgrade_project_y\Physical-Token-Dropping-PTD; New-Item -ItemType Directory -Force data,checkpoints,logs | Out-Null
```

## 1) Build production JSONL (only if you do not already have it)

```powershell
python -m actual_ptd.prepare_general_hf_dataset --dataset HuggingFaceFW/fineweb-edu --config sample-10BT --train-out data/general_train_prod.jsonl --eval-out data/general_eval_prod.jsonl --train-examples 120000 --eval-examples 5000 --eval-ratio 0.04 --min-chars 220 --max-chars 2400 --min-words 40
```

## 2) Quality check

```powershell
python -m actual_ptd.data_quality_report --input-jsonl data/general_train_prod.jsonl --model Qwen/Qwen2.5-0.5B --seq-len 512
```

## 3) Full production loop (prepare + phase2 + phase3)

```powershell
python -m actual_ptd.train_full_production --model Qwen/Qwen2.5-0.5B --train-jsonl data/general_train_prod.jsonl --eval-jsonl data/general_eval_prod.jsonl --train-pt data/general_train_prod.pt --eval-pt data/general_eval_prod.pt --seq-len 512 --recent-window 64 --phase2-steps 3000 --phase2-batch 1 --phase2-keep-rate 0.5 --phase2-critical-weight 0.0 --phase2-recent-weight 0.2 --phase3-schedule 0.99,0.9,0.8,0.7 --phase3-steps-per-stage 1500 --phase3-batch 1 --phase3-critical-weight 0.0 --phase3-recent-weight 0.2 --phase3-coverage-weight 0.1 --save-every 250 --log-every 50
```

## 4) Fast fix if you already have `general_train.jsonl/.pt`

```powershell
python -m actual_ptd.train_full_production --skip-prepare --model Qwen/Qwen2.5-0.5B --train-pt data/general_train.pt --eval-pt data/general_eval.pt --phase2-steps 3000 --phase2-batch 1 --phase2-keep-rate 0.5 --phase2-critical-weight 0.0 --phase2-recent-weight 0.2 --phase3-schedule 0.99,0.9,0.8,0.7 --phase3-steps-per-stage 1500 --phase3-batch 1 --phase3-critical-weight 0.0 --phase3-recent-weight 0.2 --phase3-coverage-weight 0.1 --save-every 250 --log-every 50
```

## 5) Phase3 only (if phase2 checkpoint already exists)

```powershell
python -m actual_ptd.train_phase3_business --model Qwen/Qwen2.5-0.5B --data data/general_train.pt --router-ckpt checkpoints/ptd_prod_phase2_step003000.pt --schedule 0.99,0.9,0.8,0.7 --steps-per-stage 1500 --batch 1 --critical-weight 0.0 --recent-weight 0.2 --coverage-weight 0.1 --save-every 250 --log-every 50
```

## 6) Serve test (sparse prefill + dense decode)

```powershell
python -m actual_ptd.serve_prefill_dense --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_prod_phase3_stage4_keep70.pt --prompt "Explain PTD in simple terms and why sparse prefill + dense decode is used." --keep-rate 0.7 --recent-window 64 --max-new-tokens 160
```

## 7) Replay eval

```powershell
python -m actual_ptd.eval_business_replay --input-jsonl data/restaurant_eval.jsonl --model Qwen/Qwen2.5-0.5B --checkpoint checkpoints/ptd_prod_phase3_stage4_keep70.pt --keep-rate 0.7 --recent-window 64 --max-examples 40
```

## Common Errors

`FileNotFoundError: data/general_train_prod.jsonl`
- Meaning: `_prod` files do not exist yet.
- Fix: run step 1, or use existing `data/general_train.jsonl` paths.

`JSONDecodeError: Unexpected UTF-8 BOM`
- Fixed in `eval_business_replay.py` using `utf-8-sig`.
- If this appears elsewhere, re-save JSONL as UTF-8 (without BOM) or read with `utf-8-sig`.

`ValueError: too many values to unpack (expected 2)` in `serve_prefill_dense.py`
- Fixed by robust handling of tokenizer `offset_mapping` shape.
