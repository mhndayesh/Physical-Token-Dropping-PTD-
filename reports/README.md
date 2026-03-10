# Reports Folder

This folder stores local JSON outputs from evaluations and benchmarks.
These files are ignored by git by default (see `.gitignore`).

Common outputs
- `long_test_8k.json`
- `long_test_batch_4k_gpu.json`
- `kv_cache_report_4k.json`
- `kv_cache_report_8k.json`
- `cache_compare_4k_keep70.json`
- `cache_compare_8k_keep70.json`

Main benchmark scripts
- `actual_ptd/run_long_test.py`
- `actual_ptd/run_long_test_batch.py`
- `actual_ptd/eval_kv_cache.py`
- `actual_ptd/eval_cache_compare.py`
