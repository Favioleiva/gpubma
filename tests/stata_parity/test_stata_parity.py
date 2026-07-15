"""Numerical parity against REAL executed Stata bmaregress exports.

The oracle was run in batch mode on StataNow/SE 19.5 (bmaregress 1.0.2,
2026-07-15); the scripts in validation/stata/ exported e(b_bma), e(pip),
vecdiag(e(V_bma)) and key scalars at full double precision to
validation/stata/output/. These tests compare the Python reference under
always_prior="shrink" (the verified Stata convention) against those exports.

If an export set is missing the corresponding test SKIPS explicitly.
Tolerances must never be loosened merely to pass (CLAUDE.md rule 9).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gpubma.api import bma_regress

ROOT = Path(__file__).resolve().parents[2]
STATA_OUT = ROOT / "validation" / "stata" / "output"
PANEL_PREDICTORS = [f"x{j}" for j in range(1, 9)]

TOL = 1e-9  # observed worst diff is ~2e-12; 1e-9 leaves honest headroom

CONFIGS = {
    "small_no_fe": dict(data="panel_8", outcome="y", predictors=PANEL_PREDICTORS,
                        controls=["w1", "w2"], fixed_effects=None, g=1000.0),
    "small_individual_fe": dict(data="panel_8", outcome="y", predictors=PANEL_PREDICTORS,
                                controls=["w1", "w2"], fixed_effects=["individual"], g=1000.0),
    "small_time_fe": dict(data="panel_8", outcome="y", predictors=PANEL_PREDICTORS,
                          controls=["w1", "w2"], fixed_effects=["time"], g=1000.0),
    "small_two_way_fe": dict(data="panel_8", outcome="y", predictors=PANEL_PREDICTORS,
                             controls=["w1", "w2"], fixed_effects=["individual", "time"],
                             g=1000.0),
    "grunfeld_no_fe": dict(data="grunfeld", outcome="invest",
                           predictors=["mvalue", "kstock"], controls=[],
                           fixed_effects=None, g=200.0),
    "grunfeld_company_fe": dict(data="grunfeld", outcome="invest",
                                predictors=["mvalue", "kstock"], controls=[],
                                fixed_effects=["individual"], g=200.0),
}


def _load_stata(stem):
    files = {k: STATA_OUT / f"{stem}_{k}.csv" for k in ("b_bma", "pip", "v_diag", "scalars")}
    names_file = STATA_OUT / f"{stem}_colnames.txt"
    if not all(f.exists() for f in files.values()) or not names_file.exists():
        pytest.skip(f"Stata exports for {stem} not present; oracle not executed "
                    "for this design on this machine")
    names = names_file.read_text().split()
    read = lambda f: pd.read_csv(f, float_precision="round_trip").iloc[0]
    return {
        "idx": {n: i for i, n in enumerate(names)},
        "b": read(files["b_bma"]).to_numpy(float),
        "pip": read(files["pip"]).to_numpy(float),
        "sd": np.sqrt(read(files["v_diag"]).to_numpy(float)),
        "scalars": read(files["scalars"]).to_dict(),
    }


def _run_python(cfg):
    if cfg["data"] == "panel_8":
        df = pd.read_parquet(ROOT / "data" / "synthetic" / "panel_8.parquet")
        entity, time = "individual_id", "period"
    else:
        df = pd.read_parquet(ROOT / "data" / "public" / "grunfeld.parquet")
        entity, time = "company", "year"
    return bma_regress(data=df, outcome=cfg["outcome"], predictors=cfg["predictors"],
                       controls=cfg["controls"], fixed_effects=cfg["fixed_effects"],
                       fe_method="dummies", entity_col=entity, time_col=time,
                       always_prior="shrink", g=cfg["g"],
                       model_prior=("betabinomial", 1.0, 1.0))


@pytest.mark.parametrize("stem", list(CONFIGS))
def test_parity_with_executed_stata_oracle(stem):
    cfg = CONFIGS[stem]
    stata = _load_stata(stem)
    py = _run_python(cfg)

    assert stata["scalars"]["k_models"] == py.n_models_evaluated
    assert stata["scalars"]["g_used"] == py.g_spec.g
    # Stata counts always-included variables in model size
    assert (stata["scalars"]["msize_mean"] - stata["scalars"]["p_always"]
            == pytest.approx(py.mean_model_size, abs=TOL))
    for j, name in enumerate(cfg["predictors"]):
        i = stata["idx"][name]
        assert stata["pip"][i] == pytest.approx(py.pip[j], abs=TOL), f"PIP {name}"
        assert stata["b"][i] == pytest.approx(py.coef_mean[j], abs=TOL), f"mean {name}"
        assert stata["sd"][i] == pytest.approx(py.coef_sd[j], abs=TOL), f"sd {name}"
