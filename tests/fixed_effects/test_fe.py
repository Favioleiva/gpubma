import numpy as np
import pytest

from gpubma.api import bma_regress
from gpubma.fixed_effects.design import build_always_block, dummy_design, within_transform

PREDICTORS = [f"x{j}" for j in range(1, 9)]
KW = dict(entity_col="individual_id", time_col="period")


def _ols_slopes(block):
    A, y, X = block["A"], block["y_work"], block["X_work"]
    if A.shape[1]:
        Q, _ = np.linalg.qr(A)
        y = y - Q @ (Q.T @ y)
        X = X - Q @ (Q.T @ X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def test_explicit_dummy_rank_no_trap(panel8):
    D, info = dummy_design(panel8, ["individual", "time"], **KW)
    n_ind, n_per = 100, 10
    assert D.shape[1] == (n_ind - 1) + (n_per - 1)
    with_intercept = np.hstack([np.ones((len(panel8), 1)), D])
    assert np.linalg.matrix_rank(with_intercept) == with_intercept.shape[1]
    assert info["base_categories"] == {"individual": "1", "time": "1"}


def test_shrink_convention_rejects_within(panel8):
    """always_prior='shrink' (Stata) is incompatible with absorbed FE."""
    with pytest.raises(ValueError, match="shrink"):
        bma_regress(data=panel8, outcome="y", predictors=PREDICTORS,
                    fixed_effects=["individual"], fe_method="within",
                    always_prior="shrink", **KW)


@pytest.mark.parametrize("fe", [["individual"], ["time"]])
def test_one_way_residualization_matches_dummies(panel8, fe):
    y = panel8["y"].to_numpy(float)
    X = panel8[PREDICTORS].to_numpy(float)
    dum = build_always_block(panel8, ["w1", "w2"], fe, "dummies", y=y, X=X, **KW)
    win = build_always_block(panel8, ["w1", "w2"], fe, "within", y=y, X=X, **KW)
    assert dum["base_rank"] == win["base_rank"] + win["absorbed_rank"]
    np.testing.assert_allclose(_ols_slopes(dum), _ols_slopes(win), atol=1e-10)


def test_two_way_residualization_matches_dummies(panel8):
    y = panel8["y"].to_numpy(float)
    X = panel8[PREDICTORS].to_numpy(float)
    dum = build_always_block(panel8, [], ["individual", "time"], "dummies", y=y, X=X, **KW)
    win = build_always_block(panel8, [], ["individual", "time"], "within", y=y, X=X, **KW)
    assert dum["base_rank"] == win["base_rank"] + win["absorbed_rank"]
    np.testing.assert_allclose(_ols_slopes(dum), _ols_slopes(win), atol=1e-10)


def test_two_way_within_rejects_unbalanced(panel8):
    unbalanced = panel8.iloc[:-3]
    with pytest.raises(ValueError, match="balanced"):
        within_transform(unbalanced["y"].to_numpy(float), unbalanced,
                         ["individual", "time"], **KW)


@pytest.mark.parametrize("fe", [["individual"], ["time"], ["individual", "time"]])
def test_bma_scores_equal_dummies_vs_within_when_df_aligned(panel8, fe):
    """Under the flat (conditional) always-block convention with identical
    fixed g and aligned effective df the BMA scores must coincide (FWL).
    Under the Stata 'shrink' convention this equivalence does NOT hold —
    see docs/FIXED_EFFECTS_DESIGN.md."""
    kw = dict(data=panel8, outcome="y", predictors=PREDICTORS, controls=["w1", "w2"],
              fixed_effects=fe, always_prior="flat", **KW)
    r_dum = bma_regress(fe_method="dummies", **kw)
    r_win = bma_regress(fe_method="within", **kw)
    assert r_dum.df_resid == r_win.df_resid
    np.testing.assert_allclose(r_dum.log_scores, r_win.log_scores, atol=1e-8)
    np.testing.assert_allclose(r_dum.pip, r_win.pip, atol=1e-12)
    np.testing.assert_allclose(r_dum.coef_mean, r_win.coef_mean, atol=1e-10)


def test_fixed_effects_do_not_change_model_count(panel8):
    r = bma_regress(data=panel8, outcome="y", predictors=PREDICTORS,
                    controls=["w1", "w2"], fixed_effects=["individual", "time"], **KW)
    assert r.n_models_evaluated == 2 ** len(PREDICTORS)
