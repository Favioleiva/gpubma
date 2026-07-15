"""Deterministic synthetic balanced-panel generator.

Data-generating process (documented in docs/DATA_GENERATING_PROCESS.md):

    y_it = alpha_i + lambda_t + sum_j beta_j * x_j,it + sum_c delta_c * w_c,it + eps_it

    alpha_i  ~ N(0, sigma_alpha^2)   individual fixed effects
    lambda_t ~ N(0, sigma_lambda^2)  time fixed effects
    x_j,it   ~ N(0, 1) i.i.d.        optional predictors (identity covariance)
    w_c,it   ~ N(0, 1) i.i.d.        always-included controls
    eps_it   ~ N(0, sigma_eps^2)     idiosyncratic error

True coefficients: beta = (1.5, -1.0, 0.75, 0.5, 0.25, 0, 0, ...) — only the
first five optional predictors enter the DGP; all higher-numbered predictors
are true noise. delta = (0.8, -0.6) for the two controls.

Determinism and nesting: every random column is drawn from its own
``numpy.random.default_rng([seed, stream_tag])`` stream that depends only on
the seed and the panel dimensions — NOT on n_predictors. Therefore the first
8 predictor columns of the 30-predictor dataset are bit-identical to the
8-predictor dataset's columns (same seed, same panel dimensions), and y is
identical too because beta_j = 0 for j > 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SEED = 20260715

TRUE_BETA_NONZERO = (1.5, -1.0, 0.75, 0.5, 0.25)  # beta_1..beta_5; beta_j = 0 for j > 5
CONTROL_DELTA = (0.8, -0.6)

_STREAM_ALPHA = 1
_STREAM_LAMBDA = 2
_STREAM_EPS = 3
_STREAM_X_BASE = 1000  # predictor j uses stream _STREAM_X_BASE + j
_STREAM_W_BASE = 2000  # control c uses stream _STREAM_W_BASE + c


def true_beta(n_predictors: int) -> np.ndarray:
    beta = np.zeros(n_predictors, dtype=np.float64)
    k = min(n_predictors, len(TRUE_BETA_NONZERO))
    beta[:k] = TRUE_BETA_NONZERO[:k]
    return beta


def generate_panel(
    n_individuals: int,
    n_periods: int,
    n_predictors: int,
    seed: int = DEFAULT_SEED,
    *,
    n_controls: int = 2,
    sigma_alpha: float = 1.0,
    sigma_lambda: float = 0.5,
    sigma_eps: float = 1.0,
):
    """Generate a balanced panel. Returns ``(df, meta)``.

    Rows are ordered by (individual_id, period), individual_id = 1..N,
    period = 1..T. Columns: individual_id, period, y, x1..xp, w1..wC.
    """
    n_obs = n_individuals * n_periods

    def rng(tag: int) -> np.random.Generator:
        return np.random.default_rng([seed, n_individuals, n_periods, tag])

    alpha = sigma_alpha * rng(_STREAM_ALPHA).standard_normal(n_individuals)
    lam = sigma_lambda * rng(_STREAM_LAMBDA).standard_normal(n_periods)
    eps = sigma_eps * rng(_STREAM_EPS).standard_normal(n_obs)

    X = np.empty((n_obs, n_predictors), dtype=np.float64)
    for j in range(1, n_predictors + 1):
        X[:, j - 1] = rng(_STREAM_X_BASE + j).standard_normal(n_obs)
    W = np.empty((n_obs, n_controls), dtype=np.float64)
    for c in range(1, n_controls + 1):
        W[:, c - 1] = rng(_STREAM_W_BASE + c).standard_normal(n_obs)

    ind = np.repeat(np.arange(1, n_individuals + 1), n_periods)
    per = np.tile(np.arange(1, n_periods + 1), n_individuals)

    beta = true_beta(n_predictors)
    delta = np.array(CONTROL_DELTA[:n_controls], dtype=np.float64)
    y = alpha[ind - 1] + lam[per - 1] + X @ beta + W @ delta + eps

    df = pd.DataFrame({"individual_id": ind.astype(np.int32), "period": per.astype(np.int32), "y": y})
    for j in range(1, n_predictors + 1):
        df[f"x{j}"] = X[:, j - 1]
    for c in range(1, n_controls + 1):
        df[f"w{c}"] = W[:, c - 1]

    meta = {
        "generator": "gpubma.datasets.synthetic.generate_panel",
        "seed": seed,
        "data_generating_equation": (
            "y_it = alpha_i + lambda_t + sum_j beta_j x_j_it "
            "+ sum_c delta_c w_c_it + eps_it"
        ),
        "n_individuals": n_individuals,
        "n_periods": n_periods,
        "n_observations": n_obs,
        "n_optional_predictors": n_predictors,
        "n_always_included_controls": n_controls,
        "expected_number_of_models": 2**n_predictors,
        "true_beta": beta.tolist(),
        "control_delta": delta.tolist(),
        "individual_effects": alpha.tolist(),
        "time_effects": lam.tolist(),
        "predictor_covariance_structure": "identity (i.i.d. standard normal)",
        "sigma_alpha": sigma_alpha,
        "sigma_lambda": sigma_lambda,
        "sigma_eps": sigma_eps,
        "error_variance": sigma_eps**2,
        "row_order": "sorted by (individual_id, period)",
        "rng": "numpy.random.default_rng([seed, n_individuals, n_periods, stream_tag]), "
               "one independent stream per column",
    }
    return df, meta
