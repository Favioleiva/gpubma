"""Compare explicit-dummy vs within-residualization fixed effects.

For each FE configuration (individual, time, two-way) on the frozen
panel_8 dataset, compares:
  1. OLS slopes of the full model (all 8 predictors + controls) via
     explicit dummies vs within transform (Frisch-Waugh-Lovell);
  2. full BMA outputs (log scores, PMPs, PIPs, coefficients) from
     bma_regress under both fe_method settings with identical g and df.

Writes reports/fixed_effects_comparison.{md,json}.

Usage:  python scripts/compare_fixed_effects.py
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
from gpubma.fixed_effects.design import build_always_block  # noqa: E402
from gpubma.validation.compare import ComparisonReport  # noqa: E402

PREDICTORS = [f"x{j}" for j in range(1, 9)]
CONTROLS = ["w1", "w2"]
CONFIGS = [["individual"], ["time"], ["individual", "time"]]


def ols_slopes(block) -> np.ndarray:
    """OLS slopes of y on [A, X] for the optional predictors, via FWL."""
    A, y, X = block["A"], block["y_work"], block["X_work"]
    if A.shape[1]:
        Q, _ = np.linalg.qr(A)
        y = y - Q @ (Q.T @ y)
        X = X - Q @ (Q.T @ X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def main() -> int:
    df = pd.read_parquet(ROOT / "data" / "synthetic" / "panel_8.parquet")
    y = df["y"].to_numpy(np.float64)
    X = df[PREDICTORS].to_numpy(np.float64)

    reports = []
    for fe in CONFIGS:
        label = "+".join(fe)
        rep = ComparisonReport(
            title=f"Fixed effects: explicit dummies vs within residualization ({label})",
            reference_label="explicit dummies (reference approach)",
            candidate_label="within transform (candidate production approach)",
        )

        b_dum = build_always_block(df, CONTROLS, fe, "dummies",
                                   entity_col="individual_id", time_col="period", y=y, X=X)
        b_win = build_always_block(df, CONTROLS, fe, "within",
                                   entity_col="individual_id", time_col="period", y=y, X=X)

        # effective base rank must agree (dummies rank == absorbed + controls)
        rank_dum = b_dum["base_rank"]
        rank_win = b_win["base_rank"] + b_win["absorbed_rank"]
        rep.add("effective always-block rank", rank_dum, rank_win, 0.0)

        # OLS slope equality (Frisch-Waugh-Lovell)
        rep.add_arrays("OLS slopes (8 predictors)", ols_slopes(b_dum), ols_slopes(b_win), 1e-9)

        # full BMA comparison with identical priors and aligned df
        kw = dict(data=df, outcome="y", predictors=PREDICTORS, controls=CONTROLS,
                  fixed_effects=fe, entity_col="individual_id", time_col="period",
                  g="benchmark", model_prior=("betabinomial", 1.0, 1.0))
        r_dum = bma_regress(fe_method="dummies", **kw)
        r_win = bma_regress(fe_method="within", **kw)
        rep.add("BMA effective df", r_dum.df_resid, r_win.df_resid, 0.0)
        rep.add_arrays("BMA log scores (256 models)", r_dum.log_scores, r_win.log_scores, 1e-8)
        rep.add_arrays("BMA posterior model probs", r_dum.pmp, r_win.pmp, 1e-12)
        rep.add_arrays("BMA PIPs", r_dum.pip, r_win.pip, 1e-12)
        rep.add_arrays("BMA coefficient means", r_dum.coef_mean, r_win.coef_mean, 1e-10)
        rep.add_arrays("BMA coefficient sds", r_dum.coef_sd, r_win.coef_sd, 1e-10)
        rep.add("BMA mean model size", r_dum.mean_model_size, r_win.mean_model_size, 1e-12)
        reports.append(rep)
        print(f"{label}: {'PASS' if rep.passed else 'FAIL'}")

    out_md = ["# Fixed-effects comparison: explicit dummies vs absorption", "",
              "Same fixed g and effective degrees of freedom were used on both sides; "
              "under that alignment the Bayesian model scores coincide by construction "
              "(Frisch-Waugh-Lovell). See docs/FIXED_EFFECTS_DESIGN.md for why this "
              "alignment is a statistical CHOICE that Stata may not share; BMA "
              "equivalence with Stata remains unverified.", ""]
    for rep in reports:
        out_md += [rep.to_markdown(), ""]
    (ROOT / "reports" / "fixed_effects_comparison.md").write_text("\n".join(out_md), encoding="utf-8")
    (ROOT / "reports" / "fixed_effects_comparison.json").write_text(
        json.dumps([r.to_dict() for r in reports], indent=2))
    ok = all(r.passed for r in reports)
    print(f"overall: {'PASS' if ok else 'FAIL'} -> reports/fixed_effects_comparison.md")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
