from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quality report for PTD JSONL training data.")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--max-samples", type=int, default=50000)
    p.add_argument("--seq-len", type=int, default=512)
    return p.parse_args()


def _pct_over(values: list[int], limit: int) -> float:
    if not values:
        return 0.0
    return round(100.0 * sum(1 for v in values if v > limit) / len(values), 3)


def _quantile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    idx = min(len(sorted_vals) - 1, max(0, int(len(sorted_vals) * q)))
    return sorted_vals[idx]


def main() -> None:
    args = parse_args()
    path = Path(args.input_jsonl)
    tok = AutoTokenizer.from_pretrained(args.model)

    rows = 0
    invalid = 0
    missing_prompt = 0
    missing_response = 0
    hashed = set()
    dupes = 0
    token_lens: list[int] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if rows >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
            except Exception:
                invalid += 1
                continue
            prompt = str(row.get("prompt", "")).strip()
            response = str(row.get("response", "")).strip()
            if not prompt:
                missing_prompt += 1
            if not response:
                missing_response += 1
            digest = hashlib.sha1((prompt + "\n" + response).encode("utf-8")).hexdigest()
            if digest in hashed:
                dupes += 1
            else:
                hashed.add(digest)
            if prompt and response:
                text = prompt + " " + response
                token_lens.append(len(tok(text, add_special_tokens=False)["input_ids"]))

    token_lens.sort()
    report = {
        "path": str(path),
        "rows_checked": rows,
        "invalid_json_rows": invalid,
        "missing_prompt_rows": missing_prompt,
        "missing_response_rows": missing_response,
        "duplicate_rows": dupes,
        "token_len": {
            "mean": round(mean(token_lens), 2) if token_lens else 0.0,
            "p50": _quantile(token_lens, 0.50),
            "p90": _quantile(token_lens, 0.90),
            "p95": _quantile(token_lens, 0.95),
            "p99": _quantile(token_lens, 0.99),
            "max": token_lens[-1] if token_lens else 0,
            "over_seq_len_pct": _pct_over(token_lens, args.seq_len),
            "over_1024_pct": _pct_over(token_lens, 1024),
        },
    }

    verdict: list[str] = []
    if rows < 20000:
        verdict.append("dataset_small_for_production")
    if report["duplicate_rows"] > max(1, int(rows * 0.01)):
        verdict.append("high_duplicate_rate")
    if report["token_len"]["over_seq_len_pct"] > 5.0:
        verdict.append("high_truncation_risk_at_seq_len")
    if not verdict:
        verdict.append("looks_ok_for_training")
    report["verdict"] = verdict

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
