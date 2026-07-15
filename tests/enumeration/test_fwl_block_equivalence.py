"""Numerical equivalence of the FWL/block-matrix formulation.

Chain of evidence (docs/FWL_BLOCK_FORMULATION.md):

    explicit joint design  ==  FWL fast path (CPU oracle)  ==  Stata

The explicit scorer (gpubma/validation/joint_reference.py) rebuilds the
full joint design [X_gamma, W] for every model and evaluates the shrink
convention directly; the fast path never touches W after a one-time
residualization. Section 4 of the doc — flat residualization is NOT
sufficient — is asserted here as an executable fact.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gpubma.api import bma_regress
from gpubma.fixed_effects.design import dummy_design
from gpubma.priors.model_priors import log_model_prior_function
from gpubma.validation.joint_reference import joint_explicit_reference

ROOT = Path(__file__).resolve().parents[2]
STATA_OUT = ROOT / "validation" / "stata" / "output"

# Same-formula, different-linear-algebra-route comparisons: exact-arithmetic
# equality, so only accumulation order differs. Observed diffs are ~1e-12
# on log scores and below 1e-13 on probabilities.
ATOL_LOGSCORE = 1e-8
ATOL_PROB = 1e-10
ATOL_COEF = 1e-10
ATOL_STATA = 1e-9  # cross-implementation (matches the parity suite)


def _panel(name):
    return pd.read_parquet(ROOT / "data" / "synthetic" / f"{name}.parquet")


def _run_pair(df, predictors, controls, fixed_effects):
    """Run the FWL fast path and the explicit joint reference on one design."""
    fwl = bma_regress(
        data=df, outcome="y", predictors=predictors, controls=controls,
        fixed_effects=fixed_effects, fe_method="dummies",
        entity_col="individual_id", time_col="period",
        always_prior="shrink", g=1000.0, model_prior=("betabinomial", 1.0, 1.0),
    )
    X = df[predictors].to_numpy(np.float64)
    y = df["y"].to_numpy(np.float64)
    W_parts = [df[controls].to_numpy(np.float64)] if controls else []
    w_names = list(controls)
    if fixed_effects:
        D, dinfo = dummy_design(df, fixed_effects, "individual_id", "period")
        W_parts.append(D)
        w_names += dinfo["dummy_names"]
    W = np.hstack(W_parts) if W_parts else np.empty((len(df), 0))
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), len(predictors))
    joint = joint_explicit_reference(X, W, y, g=1000.0, log_model_prior=log_prior)
    return fwl, joint, w_names


def _assert_equivalent(fwl, joint):
    np.testing.assert_allclose(joint["log_scores"], fwl.log_scores,
                               rtol=0.0, atol=ATOL_LOGSCORE)
    np.testing.assert_allclose(joint["pmp"], fwl.pmp, rtol=0.0, atol=ATOL_PROB)
    np.testing.assert_allclose(joint["pip"], fwl.pip, rtol=0.0, atol=ATOL_PROB)
    np.testing.assert_allclose(joint["coef_mean"], fwl.coef_mean,
                               rtol=0.0, atol=ATOL_COEF)
    np.testing.assert_allclose(joint["coef_sd"], fwl.coef_sd,
                               rtol=0.0, atol=ATOL_COEF)
    np.testing.assert_allclose(joint["size_distribution"], fwl.size_distribution,
                               rtol=0.0, atol=ATOL_PROB)
    assert joint["mean_model_size"] == pytest.approx(fwl.mean_model_size,
                                                     abs=ATOL_PROB)


def test_fwl_equals_explicit_joint_no_fe_panel12():
    """q = 2 controls, 4,096 models: FWL fast path == explicit joint design."""
    df = _panel("panel_12")
    fwl, joint, _ = _run_pair(df, [f"x{j}" for j in range(1, 13)], ["w1", "w2"], None)
    _assert_equivalent(fwl, joint)


def test_fwl_equals_explicit_joint_two_way_fe_panel8():
    """q = 110 always slopes (w1 w2 + 99 + 9 dummies), 256 models.

    This is the case that matters for the production enumerator: the joint
    design is 1000 x (k + 110) per model, the FWL path solves k x k only.
    """
    df = _panel("panel_8")
    fwl, joint, _ = _run_pair(df, [f"x{j}" for j in range(1, 9)], ["w1", "w2"],
                              ["individual", "time"])
    _assert_equivalent(fwl, joint)


def test_explicit_joint_matches_stata_per_model_panel12():
    """Explicit joint scorer vs executed Stata saving() export, all 4,096
    models: normalized log PMPs (unnormalized constants differ)."""
    path = STATA_OUT / "medium_no_fe_models.dta"
    if not path.exists():
        pytest.skip("per-model Stata export not present")
    df = _panel("panel_12")
    p = 12
    _, joint, _ = _run_pair(df, [f"x{j}" for j in range(1, 13)], ["w1", "w2"], None)

    st = pd.read_stata(path)
    states = st[[f"state_eq1_p{j}" for j in range(1, p + 1)]].to_numpy(np.int64)
    masks = (states << np.arange(p, dtype=np.int64)).sum(axis=1)
    order = np.argsort(masks)
    st_logpost = st["_logposterior"].to_numpy(np.float64)[order]

    st_norm = st_logpost - np.logaddexp.reduce(np.sort(st_logpost))
    jt_norm = joint["log_scores"] - np.logaddexp.reduce(np.sort(joint["log_scores"]))
    np.testing.assert_allclose(st_norm, jt_norm, rtol=0.0, atol=ATOL_STATA)


def test_explicit_joint_matches_stata_moments_two_way_fe():
    """Explicit joint scorer vs executed Stata e() exports (two-way FE):
    PIPs and posterior means/sds for optional AND always columns by name."""
    stem = "small_two_way_fe"
    files = {k: STATA_OUT / f"{stem}_{k}.csv" for k in ("b_bma", "pip", "v_diag")}
    names_file = STATA_OUT / f"{stem}_colnames.txt"
    if not all(f.exists() for f in files.values()) or not names_file.exists():
        pytest.skip("Stata exports for small_two_way_fe not present")
    names = names_file.read_text().split()
    read = lambda f: pd.read_csv(f, float_precision="round_trip").iloc[0].to_numpy(float)
    st_b, st_pip, st_sd = read(files["b_bma"]), read(files["pip"]), np.sqrt(
        read(files["v_diag"]))
    idx = {n: i for i, n in enumerate(names)}

    df = _panel("panel_8")
    predictors = [f"x{j}" for j in range(1, 9)]
    _, joint, w_names = _run_pair(df, predictors, ["w1", "w2"],
                                  ["individual", "time"])

    for j, name in enumerate(predictors):
        i = idx[name]
        assert st_pip[i] == pytest.approx(joint["pip"][j], abs=ATOL_STATA), name
        assert st_b[i] == pytest.approx(joint["coef_mean"][j], abs=ATOL_STATA), name
        assert st_sd[i] == pytest.approx(joint["coef_sd"][j], abs=ATOL_STATA), name

    # always block by name: w1, w2 and every non-base dummy.
    # gpubma name "individual[individual_id=42]" == Stata "42.individual_id";
    # "time[period=7]" == "7.period". Base levels (1b.) are excluded columns.
    def stata_name(local):
        if local in ("w1", "w2"):
            return local
        factor, rest = local.split("[", 1)
        col, level = rest.rstrip("]").split("=")
        return f"{level}.{col}"

    checked = 0
    for j, local in enumerate(w_names):
        i = idx[stata_name(local)]
        assert st_b[i] == pytest.approx(joint["always_mean"][j], abs=ATOL_STATA), local
        assert st_sd[i] == pytest.approx(joint["always_sd"][j], abs=ATOL_STATA), local
        checked += 1
    assert checked == 110  # w1 w2 + 99 individual + 9 period dummies


def test_flat_residualization_is_not_sufficient():
    """Section 4 of the doc as an executable fact: treating the always block
    as flat produces materially different posteriors than joint shrinkage."""
    df = _panel("panel_8")
    kwargs = dict(
        data=df, outcome="y", predictors=[f"x{j}" for j in range(1, 9)],
        controls=["w1", "w2"], fixed_effects=["individual"], fe_method="dummies",
        entity_col="individual_id", time_col="period",
        g=1000.0, model_prior=("betabinomial", 1.0, 1.0),
    )
    shrink = bma_regress(always_prior="shrink", **kwargs)
    flat = bma_regress(always_prior="flat", **kwargs)
    assert np.max(np.abs(shrink.pip - flat.pip)) > 1e-2
    # and the difference is not a constant shift of the log scores
    d = shrink.log_scores - flat.log_scores
    assert np.max(np.abs(d - d.mean())) > 1e-2
