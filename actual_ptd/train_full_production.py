from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full PTD production training loop (prepare -> phase2 -> phase3).")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--train-jsonl", default="data/general_train.jsonl")
    p.add_argument("--eval-jsonl", default="data/general_eval.jsonl")
    p.add_argument("--train-pt", default="data/general_train.pt")
    p.add_argument("--eval-pt", default="data/general_eval.pt")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--recent-window", type=int, default=64)
    p.add_argument("--skip-prepare", action="store_true")

    p.add_argument("--phase2-steps", type=int, default=3000)
    p.add_argument("--phase2-batch", type=int, default=1)
    p.add_argument("--phase2-keep-rate", type=float, default=0.5)
    p.add_argument("--phase2-critical-weight", type=float, default=0.0)
    p.add_argument("--phase2-recent-weight", type=float, default=0.2)

    p.add_argument("--phase3-schedule", default="0.99,0.9,0.8,0.7")
    p.add_argument("--phase3-steps-per-stage", type=int, default=1500)
    p.add_argument("--phase3-batch", type=int, default=1)
    p.add_argument("--phase3-critical-weight", type=float, default=0.0)
    p.add_argument("--phase3-recent-weight", type=float, default=0.2)
    p.add_argument("--phase3-coverage-weight", type=float, default=0.1)

    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--router-ckpt", default="")
    return p.parse_args()


def _run(cmd: list[str], cwd: str) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _fallback_non_prod(path: Path) -> Path:
    if path.exists():
        return path
    alt_name = path.name.replace("_prod", "")
    alt = path.with_name(alt_name)
    return alt if alt.exists() else path


def _raise_missing_jsonl(path: Path) -> None:
    raise FileNotFoundError(
        f"Missing dataset file: {path}\n"
        "Run dataset build first, e.g.:\n"
        "python -m actual_ptd.prepare_general_hf_dataset --dataset HuggingFaceFW/fineweb-edu --config sample-10BT "
        "--train-out data/general_train_prod.jsonl --eval-out data/general_eval_prod.jsonl --train-examples 120000 "
        "--eval-examples 5000 --eval-ratio 0.04 --min-chars 220 --max-chars 2400 --min-words 40"
    )


def main() -> None:
    args = parse_args()
    root = Path(".").resolve()
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)
    print("[info] train_full_production runs multiple steps sequentially: prepare -> phase2 -> phase3.")

    train_jsonl = _fallback_non_prod(Path(args.train_jsonl))
    eval_jsonl = _fallback_non_prod(Path(args.eval_jsonl))
    train_pt = Path(args.train_pt)
    eval_pt = Path(args.eval_pt)

    if not args.skip_prepare:
        if not train_jsonl.exists():
            _raise_missing_jsonl(train_jsonl)
        if train_jsonl != Path(args.train_jsonl):
            print(f"[info] requested {args.train_jsonl} not found; using {train_jsonl} instead.")
        if eval_jsonl != Path(args.eval_jsonl) and eval_jsonl.exists():
            print(f"[info] requested {args.eval_jsonl} not found; using {eval_jsonl} instead.")

        print("[stage 1/3] preparing tensor datasets")
        _run(
            [
                sys.executable,
                "-m",
                "actual_ptd.prepare_business_dataset",
                "--input-jsonl",
                str(train_jsonl),
                "--out",
                str(train_pt),
                "--model",
                args.model,
                "--seq-len",
                str(args.seq_len),
                "--recent-window",
                str(args.recent_window),
            ],
            str(root),
        )
        if eval_jsonl.exists():
            _run(
                [
                    sys.executable,
                    "-m",
                    "actual_ptd.prepare_business_dataset",
                    "--input-jsonl",
                    str(eval_jsonl),
                    "--out",
                    str(eval_pt),
                    "--model",
                    args.model,
                    "--seq-len",
                    str(args.seq_len),
                    "--recent-window",
                    str(args.recent_window),
                ],
                str(root),
            )
        else:
            print(f"[warn] eval JSONL not found: {eval_jsonl}. Continuing without eval tensor export.")
    elif not train_pt.exists():
        raise FileNotFoundError(
            f"--skip-prepare was set, but train tensor dataset is missing: {train_pt}\n"
            "Either remove --skip-prepare or build it first with prepare_business_dataset."
        )

    print("[stage 2/3] phase2 router warmup")
    _run(
        [
            sys.executable,
            "-m",
            "actual_ptd.train_phase2_business",
            "--model",
            args.model,
            "--data",
            str(train_pt),
            "--steps",
            str(args.phase2_steps),
            "--batch",
            str(args.phase2_batch),
            "--keep-rate",
            str(args.phase2_keep_rate),
            "--critical-weight",
            str(args.phase2_critical_weight),
            "--recent-weight",
            str(args.phase2_recent_weight),
            "--save-every",
            str(args.save_every),
            "--log-every",
            str(args.log_every),
        ],
        str(root),
    )

    router_ckpt = args.router_ckpt.strip()
    if not router_ckpt:
        router_ckpt = f"checkpoints/ptd_prod_phase2_step{args.phase2_steps:06d}.pt"
    if not Path(router_ckpt).exists():
        raise FileNotFoundError(f"Router checkpoint not found: {router_ckpt}")

    print("[stage 3/3] phase3 sparsity training")
    _run(
        [
            sys.executable,
            "-m",
            "actual_ptd.train_phase3_business",
            "--model",
            args.model,
            "--data",
            str(train_pt),
            "--router-ckpt",
            router_ckpt,
            "--schedule",
            args.phase3_schedule,
            "--steps-per-stage",
            str(args.phase3_steps_per_stage),
            "--batch",
            str(args.phase3_batch),
            "--critical-weight",
            str(args.phase3_critical_weight),
            "--recent-weight",
            str(args.phase3_recent_weight),
            "--coverage-weight",
            str(args.phase3_coverage_weight),
            "--save-every",
            str(args.save_every),
            "--log-every",
            str(args.log_every),
        ],
        str(root),
    )

    print("done")


if __name__ == "__main__":
    main()
