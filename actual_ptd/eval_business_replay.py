from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay business prompts and score critical substring recall.")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--keep-rate", type=float, default=0.7)
    p.add_argument("--recent-window", type=int, default=128)
    p.add_argument("--max-examples", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with open(args.input_jsonl, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= args.max_examples:
                    break
    total = 0
    hits = 0
    for row in rows:
        prompt = row.get("prompt") or "\n".join(f"{m['role'].upper()}: {m['content']}" for m in row.get("messages", []))
        cmd = [
            "python", "-m", "actual_ptd.serve_prefill_dense",
            "--model", args.model,
            "--checkpoint", args.checkpoint,
            "--prompt", prompt,
            "--keep-rate", str(args.keep_rate),
            "--recent-window", str(args.recent_window),
            "--max-new-tokens", "96",
        ]
        out = subprocess.check_output(cmd, text=True)
        crit = row.get("critical_spans", []) or row.get("expected_substrings", [])
        local_total = len(crit)
        local_hits = sum(1 for x in crit if str(x).lower() in out.lower())
        total += local_total
        hits += local_hits
        print({"prompt": prompt[:80], "score": f"{local_hits}/{local_total}"})
    score = hits / max(1, total)
    print({"critical_recall": score, "hits": hits, "total": total})


if __name__ == "__main__":
    main()
