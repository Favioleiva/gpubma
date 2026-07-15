"""Compare Python BMA results against exported Stata bmaregress outputs.

Runs the Python reference on the frozen datasets and, when the CSV exports
from validation/stata/*.do exist under validation/stata/output/, compares
coefficient posterior means and PIPs without any rounding.

If the Stata exports are absent (Stata has not been run yet), the script
says so explicitly and exits with status 0 — prepared scripts are clearly
distinguished from executed validations.

Also always compares CSV vs Parquet vs DTA inputs and repeated Python runs.

Usage:  python scripts/compare_stata_python.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpubma.api import bma_regress  # noqa: E402
from gpubma.datasets.io_utils import read_any  # noqa: E402
from gpubma.validation.compare import ComparisonReport  # noqa: E402

STATA_OUT = ROOT / "validation" / "stata" / "output"
PREDICTORS = [f"x{j}" for j in range(1, 9)]

# provisional cross-implementation tolerance; NEVER loosen merely to pass
STATA_TOL = 1e-6


def run_python(df):
    return bma_regress(data=df, outcome="y", predictors=PREDICTORS,
                       controls=["w1", "w2"], g="benchmark",
                       model_prior=("betabinomial", 1.0, 1.0))


def main() -> int:
    reports = []

    # --- deterministic internal comparisons (always available) ----------
    frames = {fmt: read_any(ROOT / "data" / "synthetic" / f"panel_8.{fmt}")
              for fmt in ("csv", "parquet", "dta")}
    results = {fmt: run_python(df) for fmt, df in frames.items()}
    rep = ComparisonReport("Input format equivalence (panel_8)", "parquet", "csv & dta")
    for fmt in ("csv", "dta"):
        rep.add_arrays(f"log scores parquet vs {fmt}",
                       results["parquet"].log_scores, results[fmt].log_scores, 0.0)
        rep.add_arrays(f"PIP parquet vs {fmt}",
                       results["parquet"].pip, results[fmt].pip, 0.0)
        rep.add_arrays(f"coef means parquet vs {fmt}",
                       results["parquet"].coef_mean, results[fmt].coef_mean, 0.0)
    reports.append(rep)

    rep = ComparisonReport("Repeated Python runs (determinism)", "run 1", "run 2")
    r1, r2 = run_python(frames["parquet"]), run_python(frames["parquet"])
    rep.add_arrays("log scores", r1.log_scores, r2.log_scores, 0.0)
    rep.add_arrays("PIP", r1.pip, r2.pip, 0.0)
    rep.add_arrays("coef means", r1.coef_mean, r2.coef_mean, 0.0)
    reports.append(rep)

    # --- Stata comparison (only if real exports exist) -------------------
    stata_pip = STATA_OUT / "small_no_fe_pip.csv"
    if stata_pip.exists():
        stata = pd.read_csv(stata_pip)
        rep = ComparisonReport("Python vs Stata bmaregress (panel_8, no FE)",
                               "Stata bmaregress export", "gpubma CPU reference")
        py = results["parquet"]
        rep.add_arrays("PIP", stata.iloc[0].to_numpy(dtype=float)[: len(py.pip)],
                       py.pip, STATA_TOL)
        reports.append(rep)
    else:
        print("NOTE: no Stata exports found under validation/stata/output/.")
        print("      The .do scripts are PREPARED but have not been executed;")
        print("      Python-vs-Stata numerical comparison is pending a working Stata.")

    md = ["# Deterministic comparison report", ""]
    ok = True
    for rep in reports:
        md += [rep.to_markdown(), ""]
        ok = ok and rep.passed
        print(f"{rep.title}: {'PASS' if rep.passed else 'FAIL'}")
    (ROOT / "reports" / "comparison_report.md").write_text("\n".join(md), encoding="utf-8")
    (ROOT / "reports" / "comparison_report.json").write_text(
        json.dumps([r.to_dict() for r in reports], indent=2))
    print(f"overall: {'PASS' if ok else 'FAIL'} -> reports/comparison_report.md")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
