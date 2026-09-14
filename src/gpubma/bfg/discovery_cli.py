"""CSV-to-discovery CLI. No oracle input or global posterior output."""
import argparse
from pathlib import Path
import pandas as pd
from .config import BFGConfig
from .engine import fit_bfg

def main(argv=None):
    parser = argparse.ArgumentParser(description='Budgeted high-evidence model discovery')
    parser.add_argument('--data', required=True, type=Path, help='Numeric CSV')
    parser.add_argument('--outcome', required=True)
    parser.add_argument('--candidates', required=True, help='Comma-separated candidate columns')
    parser.add_argument('--controls', default='', help='Comma-separated always-included columns')
    parser.add_argument('--budget-models', type=int, default=100000)
    parser.add_argument('--beam-width', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260715)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--out', type=Path, required=True, help='New discovery JSON; refuses overwrite')
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('--out exists; choose a new path')
    frame = pd.read_csv(args.data)
    names = [v.strip() for v in args.candidates.split(',')]
    controls = [v.strip() for v in args.controls.split(',') if v.strip()]
    if args.outcome in names or set(names) & set(controls) or args.outcome in controls:
        parser.error('Outcome, candidates and controls must be distinct')
    result = fit_bfg(frame[args.outcome], frame[names],
        always_in=frame[controls] if controls else None,
        config=BFGConfig(budget_models=args.budget_models, beam_width=args.beam_width,
                         seed=args.seed, device=args.device, checkpoint_dir=args.checkpoint_dir,
                         resume=args.resume), outcome_name=args.outcome)
    result.save_json(args.out)
    print(result.summary())
    return 0
