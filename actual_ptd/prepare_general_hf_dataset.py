from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build general-domain PTD JSONL from Hugging Face streaming datasets.")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--train-out", default="data/general_train.jsonl")
    p.add_argument("--eval-out", default="data/general_eval.jsonl")
    p.add_argument("--train-examples", type=int, default=120000)
    p.add_argument("--eval-examples", type=int, default=5000)
    p.add_argument("--eval-ratio", type=float, default=0.04)
    p.add_argument("--min-chars", type=int, default=220)
    p.add_argument("--max-chars", type=int, default=2400)
    p.add_argument("--min-words", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _row_from_text(text: str) -> dict | None:
    words = text.split()
    if len(words) < 2:
        return None
    cut = max(20, int(len(words) * 0.55))
    if cut >= len(words):
        return None
    return {
        "prompt": "TEXT: " + " ".join(words[:cut]),
        "response": " ".join(words[cut:]),
        "critical_spans": [],
    }


def _choose_eval(text: str, eval_ratio: float, seed: int) -> bool:
    digest = hashlib.sha1((str(seed) + text).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return bucket < eval_ratio


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    train_path = Path(args.train_out)
    eval_path = Path(args.eval_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    stream_kwargs = {"split": args.split, "streaming": True}
    ds = load_dataset(args.dataset, args.config, **stream_kwargs) if args.config else load_dataset(args.dataset, **stream_kwargs)

    n_train = 0
    n_eval = 0
    n_seen = 0
    n_filtered = 0

    with train_path.open("w", encoding="utf-8") as f_train, eval_path.open("w", encoding="utf-8") as f_eval:
        for ex in ds:
            if n_train >= args.train_examples and n_eval >= args.eval_examples:
                break
            n_seen += 1
            text = _clean_text(ex.get("text", ""))
            if len(text) < args.min_chars or len(text) > args.max_chars:
                n_filtered += 1
                continue
            if len(text.split()) < args.min_words:
                n_filtered += 1
                continue
            row = _row_from_text(text)
            if row is None:
                n_filtered += 1
                continue
            line = json.dumps(row, ensure_ascii=False)
            target_eval = _choose_eval(text, args.eval_ratio, args.seed)
            if target_eval and n_eval < args.eval_examples:
                f_eval.write(line + "\n")
                n_eval += 1
            elif n_train < args.train_examples:
                f_train.write(line + "\n")
                n_train += 1
            elif n_eval < args.eval_examples:
                f_eval.write(line + "\n")
                n_eval += 1

    print(
        {
            "dataset": args.dataset,
            "config": args.config,
            "seen": n_seen,
            "filtered": n_filtered,
            "train_rows": n_train,
            "eval_rows": n_eval,
            "train_out": str(train_path),
            "eval_out": str(eval_path),
        }
    )


if __name__ == "__main__":
    main()
