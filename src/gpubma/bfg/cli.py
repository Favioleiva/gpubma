"""Command-line interface for standalone execution of BFG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gpubma.bfg.config import BFGConfig
from gpubma.bfg.engine import fit_bfg


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `gpubma-bfg`."""
    parser = argparse.ArgumentParser(
        prog="gpubma-bfg",
        description="BFG (Budgeted Fast GPU) Bayesian Model Averaging CLI",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to input data (Parquet, CSV, or Feather format).",
    )
    parser.add_argument(
        "--outcome",
        type=str,
        required=True,
        help="Column name of the dependent outcome variable.",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        nargs="+",
        help="Candidate predictor column names (or path to text file with one name per line).",
    )
    parser.add_argument(
        "--controls",
        type=str,
        nargs="*",
        default=[],
        help="Always-included control variable names.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100_000,
        help="Total model evaluation budget (default: 100,000).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device: 'cuda', 'cuda:0', 'cpu' (default: 'cuda').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260715,
        help="Deterministic random seed (default: 20260715).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory to save/load progressive checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in checkpoint-dir if available.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to save summary text output.",
    )

    args = parser.parse_args(argv)

    # 1. Load data
    data_path = args.data
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        return 1

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix == ".feather":
        df = pd.read_feather(data_path)
    else:
        try:
            df = pd.read_parquet(data_path)
        except Exception:
            df = pd.read_csv(data_path)

    # 2. Parse candidates
    if not args.candidates:
        # Default to all other columns
        cols = [c for c in df.columns if c != args.outcome and c not in args.controls]
    elif len(args.candidates) == 1 and Path(args.candidates[0]).exists():
        with open(args.candidates[0], "r", encoding="utf-8") as f:
            cols = [line.strip() for line in f if line.strip()]
    else:
        cols = args.candidates

    y = df[args.outcome]
    X = df[cols]
    always_in = df[args.controls] if args.controls else None

    print(f"Loaded {len(df):,} observations with {len(cols)} candidate predictors.")
    print(f"Executing BFG on device '{args.device}' with budget {args.budget:,}...")

    # 3. Execute BFG
    result = fit_bfg(
        y=y,
        X=X,
        candidate_names=cols,
        always_in=always_in,
        budget_models=args.budget,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        outcome_name=args.outcome,
    )

    summary_text = result.summary()
    print("\n" + summary_text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")
        print(f"Summary saved to: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
