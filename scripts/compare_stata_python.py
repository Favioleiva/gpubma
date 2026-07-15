"""Compare Python BMA results against ACTUAL Stata bmaregress exports.

The Stata oracle was executed in batch mode (StataNow/SE 19.5, bmaregress
1.0.2) via the scripts in validation/stata/, which export e(b_bma), e(pip),
vecdiag(e(V_bma)), and key e() scalars to CSV at full double precision.

For each of six designs this script runs the Python reference with the same
fixed g and priors and compares, without any rounding:
  - PIP per optional predictor;
  - BMA posterior mean per optional predictor (+ w1, w2 for no-FE designs);
  - BMA posterior sd per optional predictor (sqrt of V_bma diagonal);
  - posterior mean model size (Stata counts always-included variables in
    model size: Stata msize_mean - p_always == Python mean_model_size);
  - number of models evaluated.

Also compares CSV vs Parquet vs DTA inputs and repeated Python runs.

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
PANEL_PREDICTORS = [f"x{j}" for j in range(1, 9)]

# provisional cross-implementation tolerances; NEVER loosen merely to pass
TOL_PIP = 1e-9
TOL_COEF = 1e-9
TOL_SD = 1e-9
TOL_MSIZE = 1e-9

CONFIGS = [
    dict(stem="small_no_fe", data="panel_8", outcome="y", predictors=PANEL_PREDICTORS,
         controls=["w1", "w2"], fixed_effects=None, g=1000.0,
         compare_always=["w1", "w2"]),
    dict(stem="small_individual_fe", data="panel_8", outcome="y", predictors=PANEL_PREDICTORS,
         controls=["w1", "w2"], fixed_effects=["individual"], g=1000.0),
    dict(stem="small_time_fe", data="panel_8", outcome="y", predictors=PANEL_PREDICTORS,
         controls=["w1", "w2"], fixed_effects=["time"], g=1000.0),
    dict(stem="small_two_way_fe", data="panel_8", outcome="y", predictors=PANEL_PREDICTORS,
         controls=["w1", "w2"], fixed_effects=["individual", "time"], g=1000.0),
    dict(stem="grunfeld_no_fe", data="grunfeld", outcome="invest",
         predictors=["mvalue", "kstock"], controls=[], fixed_effects=None, g=200.0),
    dict(stem="grunfeld_company_fe", data="grunfeld", outcome="invest",
         predictors=["mvalue", "kstock"], controls=[], fixed_effects=["individual"],
         g=200.0),
]


def load_stata(stem: str) -> dict | None:
    files = {kind: STATA_OUT / f"{stem}_{kind}.csv"
             for kind in ("b_bma", "pip", "v_diag", "scalars")}
    names_file = STATA_OUT / f"{stem}_colnames.txt"
    if not all(f.exists() for f in files.values()) or not names_file.exists():
        return None
    names = names_file.read_text().split()
    b = pd.read_csv(files["b_bma"], float_precision="round_trip").iloc[0].to_numpy(float)
    pip = pd.read_csv(files["pip"], float_precision="round_trip").iloc[0].to_numpy(float)
    v = pd.read_csv(files["v_diag"], float_precision="round_trip").iloc[0].to_numpy(float)
    scalars = pd.read_csv(files["scalars"], float_precision="round_trip").iloc[0].to_dict()
    assert len(names) == len(b) == len(pip) == len(v)
    idx = {n: i for i, n in enumerate(names)}
    return {"names": names, "idx": idx, "b": b, "pip": pip, "sd": np.sqrt(v),
            "scalars": scalars}


def run_python(cfg) -> object:
    if cfg["data"] == "panel_8":
        df = pd.read_parquet(ROOT / "data" / "synthetic" / "panel_8.parquet")
        entity, time = "individual_id", "period"
    else:
        df = pd.read_parquet(ROOT / "data" / "public" / "grunfeld.parquet")
        entity, time = "company", "year"
    return bma_regress(
        data=df, outcome=cfg["outcome"], predictors=cfg["predictors"],
        controls=cfg["controls"], fixed_effects=cfg["fixed_effects"],
        fe_method="dummies", entity_col=entity, time_col=time,
        g=cfg["g"], model_prior=("betabinomial", 1.0, 1.0),
    )


def compare_config(cfg) -> ComparisonReport | None:
    stata = load_stata(cfg["stem"])
    if stata is None:
        print(f"{cfg['stem']}: Stata exports NOT found — skipped (prepared only)")
        return None
    py = run_python(cfg)
    rep = ComparisonReport(
        title=f"Python vs Stata bmaregress — {cfg['stem']}",
        reference_label="Stata bmaregress (StataNow/SE 19.5, batch export)",
        candidate_label="gpubma CPU reference (float64 enumeration)",
    )
    rep.add("models evaluated", stata["scalars"]["k_models"], py.n_models_evaluated, 0.0)
    rep.add("mean model size (Stata minus p_always)",
            stata["scalars"]["msize_mean"] - stata["scalars"]["p_always"],
            py.mean_model_size, TOL_MSIZE)
    for j, name in enumerate(cfg["predictors"]):
        i = stata["idx"][name]
        rep.add(f"PIP[{name}]", stata["pip"][i], py.pip[j], TOL_PIP)
        rep.add(f"coef mean[{name}]", stata["b"][i], py.coef_mean[j], TOL_COEF)
        rep.add(f"coef sd[{name}]", stata["sd"][i], py.coef_sd[j], TOL_SD)
    return rep


def internal_comparisons() -> list:
    reports = []
    frames = {fmt: read_any(ROOT / "data" / "synthetic" / f"panel_8.{fmt}")
              for fmt in ("csv", "parquet", "dta")}

    def run(df):
        return bma_regress(data=df, outcome="y", predictors=PANEL_PREDICTORS,
                           controls=["w1", "w2"], g=1000.0,
                           model_prior=("betabinomial", 1.0, 1.0))

    results = {fmt: run(df) for fmt, df in frames.items()}
    rep = ComparisonReport("Input format equivalence (panel_8)", "parquet", "csv & dta")
    for fmt in ("csv", "dta"):
        rep.add_arrays(f"log scores parquet vs {fmt}",
                       results["parquet"].log_scores, results[fmt].log_scores, 0.0)
        rep.add_arrays(f"PIP parquet vs {fmt}", results["parquet"].pip,
                       results[fmt].pip, 0.0)
        rep.add_arrays(f"coef means parquet vs {fmt}", results["parquet"].coef_mean,
                       results[fmt].coef_mean, 0.0)
    reports.append(rep)

    rep = ComparisonReport("Repeated Python runs (determinism)", "run 1", "run 2")
    r1, r2 = run(frames["parquet"]), run(frames["parquet"])
    rep.add_arrays("log scores", r1.log_scores, r2.log_scores, 0.0)
    rep.add_arrays("PIP", r1.pip, r2.pip, 0.0)
    rep.add_arrays("coef means", r1.coef_mean, r2.coef_mean, 0.0)
    reports.append(rep)
    return reports


def main() -> int:
    reports = internal_comparisons()
    n_stata = 0
    for cfg in CONFIGS:
        rep = compare_config(cfg)
        if rep is not None:
            reports.append(rep)
            n_stata += 1

    md = ["# Deterministic comparison report", "",
          f"Stata designs compared against REAL executed bmaregress output: {n_stata}/6.",
          "Values are never rounded before comparison; rounding is display-only.", ""]
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
