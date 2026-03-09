import argparse
import os

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare TinyStories token tensor for eval scripts.")
    p.add_argument("--dataset", default="roneneldan/TinyStories", help="HF dataset name")
    p.add_argument("--split", default="train")
    p.add_argument("--model", default="gpt2", help="Tokenizer model")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--seq-len", type=int, default=513, help="Saved rows are length seq_len")
    p.add_argument("--output", default="tinystories_tokenized.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "Missing dependency. Install with: pip install datasets transformers"
        ) from e

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset(args.dataset, split=args.split)
    rows = []

    for ex in ds:
        ids = tok.encode(ex.get("text", ""), add_special_tokens=False)
        if not ids:
            continue
        ids = ids[: args.seq_len]
        if len(ids) < args.seq_len:
            pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
            ids = ids + [pad_id] * (args.seq_len - len(ids))
        rows.append(ids)
        if len(rows) >= args.samples:
            break

    if not rows:
        raise RuntimeError("No samples were prepared.")

    data = torch.tensor(rows, dtype=torch.long)
    out = os.path.abspath(args.output)
    torch.save(data, out)
    print(f"saved: {out} shape={tuple(data.shape)}")


if __name__ == "__main__":
    main()
