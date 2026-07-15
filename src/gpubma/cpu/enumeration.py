"""Exact CPU reference enumeration for Gaussian linear-regression BMA.

Correctness over speed. All arithmetic is float64. Every one of the 2^p
candidate models is reconstructed directly from the sufficient statistics
(X'X, X'y, y'y) of the residualized data and scored in closed form.

Marginal likelihood (see docs/STATISTICAL_SPECIFICATION.md for derivation):
after residualizing y and X on the always-included block of rank r
(plus ``absorbed_rank`` for within transforms), with effective degrees of
freedom  df = n - r_total  and a fixed Zellner g-prior,

    log m(gamma)  =  ((df - k)/2) * log(1 + g)
                   - (df/2) * log(1 + g * (1 - R2_gamma))       (+ const)

where k = |gamma| and R2_gamma = (b' Z^{-1} b) / (y'y) on the residualized
data. The additive constant is common to all models and cancels in the
normalized posterior model probabilities.

Posterior moments per model (conditional on gamma):
    E[beta | gamma]   = s * beta_hat,                s = g / (1 + g)
    Var[beta | gamma] = E[sigma^2 | gamma] * s * Z^{-1}
    E[sigma^2|gamma]  = (y'y - s * b'Z^{-1}b) / (df - 2)
BMA moments combine these across models by the law of total expectation /
variance, with excluded coefficients exactly zero.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp


def _mask_indices(mask: int, p: int) -> np.ndarray:
    return np.array([j for j in range(p) if (mask >> j) & 1], dtype=np.intp)


def enumerate_models(
    X: np.ndarray,
    y: np.ndarray,
    *,
    df_resid: int,
    g: float,
    log_model_prior,
    compute_coefficients: bool = True,
    top_k: int = 10,
    keep_masks: bool | None = None,
):
    """Score all 2^p models. X, y must already be residualized on the
    always-included block. Returns a plain dict of arrays and scalars."""
    t0 = time.perf_counter()
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n, p = X.shape
    n_models = 1 << p
    if df_resid <= 2:
        raise ValueError(f"effective degrees of freedom too small: {df_resid}")

    # --- sufficient statistics -------------------------------------------
    Zxx = X.T @ X
    Zxy = X.T @ y
    tss = float(y @ y)
    if tss <= 0:
        raise ValueError("residualized outcome has zero variation")
    t_suff = time.perf_counter() - t0

    log1pg = np.log1p(g)
    shrink = g / (1.0 + g)

    # --- pass 1: log scores for every model ------------------------------
    t1 = time.perf_counter()
    masks = np.arange(n_models, dtype=np.int64)
    sizes = np.bitwise_count(masks).astype(np.int64)
    log_prior_by_size = np.array([log_model_prior(k) for k in range(p + 1)])

    log_ml = np.empty(n_models, dtype=np.float64)
    ess_arr = np.empty(n_models, dtype=np.float64)
    log_ml[0] = 0.0
    ess_arr[0] = 0.0
    for mask in range(1, n_models):
        idx = _mask_indices(mask, p)
        k = idx.size
        c, low = cho_factor(Zxx[np.ix_(idx, idx)], lower=True, check_finite=False)
        w = cho_solve((c, low), Zxy[idx], check_finite=False)
        ess = float(Zxy[idx] @ w)
        one_minus_r2 = max(1.0 - ess / tss, np.finfo(np.float64).tiny)
        log_ml[mask] = 0.5 * (df_resid - k) * log1pg - 0.5 * df_resid * np.log1p(g * one_minus_r2)
        ess_arr[mask] = ess
    log_scores = log_ml + log_prior_by_size[sizes]
    t_enum = time.perf_counter() - t1

    # --- normalization and reductions ------------------------------------
    t2 = time.perf_counter()
    log_norm = float(logsumexp(log_scores))
    pmp = np.exp(log_scores - log_norm)

    pip = np.empty(p, dtype=np.float64)
    for j in range(p):
        pip[j] = pmp[((masks >> j) & 1).astype(bool)].sum()
    # Summation of up to 2^(p-1) terms can overshoot 1 by a few ulp. Verify
    # the overshoot is numerical noise, then clamp to the mathematical range.
    overshoot = float(np.max(pip) - 1.0)
    if overshoot > 1e-12:
        raise FloatingPointError(f"PIP exceeds 1 beyond numerical noise: 1 + {overshoot:.3e}")
    pip = np.clip(pip, 0.0, 1.0)

    size_dist = np.bincount(sizes, weights=pmp, minlength=p + 1)
    mean_size = float(sizes @ pmp)

    order = np.argsort(pmp)[::-1][: min(top_k, n_models)]
    top_models = [
        {
            "mask": int(m),
            "included": [int(j) for j in range(p) if (m >> j) & 1],
            "size": int(sizes[m]),
            "pmp": float(pmp[m]),
            "log_score": float(log_scores[m]),
        }
        for m in order
    ]
    t_reduce = time.perf_counter() - t2

    # --- pass 2 (optional): BMA coefficient moments -----------------------
    t3 = time.perf_counter()
    coef_mean = coef_sd = None
    if compute_coefficients:
        mean_acc = np.zeros(p, dtype=np.float64)
        m2_acc = np.zeros(p, dtype=np.float64)
        for mask in range(1, n_models):
            wgt = pmp[mask]
            idx = _mask_indices(mask, p)
            Z = Zxx[np.ix_(idx, idx)]
            b = Zxy[idx]
            c, low = cho_factor(Z, lower=True, check_finite=False)
            beta_hat = cho_solve((c, low), b, check_finite=False)
            zinv_diag = np.diag(cho_solve((c, low), np.eye(idx.size), check_finite=False))
            cond_mean = shrink * beta_hat
            e_sigma2 = (tss - shrink * ess_arr[mask]) / (df_resid - 2)
            cond_var = e_sigma2 * shrink * zinv_diag
            mean_acc[idx] += wgt * cond_mean
            m2_acc[idx] += wgt * (cond_var + cond_mean**2)
        coef_mean = mean_acc
        coef_sd = np.sqrt(np.maximum(m2_acc - mean_acc**2, 0.0))
    t_coef = time.perf_counter() - t3

    total = time.perf_counter() - t0
    return {
        "n_models_expected": n_models,
        "n_models_evaluated": n_models,
        "masks": masks if (keep_masks or (keep_masks is None and p <= 16)) else None,
        "sizes": sizes,
        "log_scores": log_scores,
        "log_normalizer": log_norm,
        "pmp": pmp,
        "pip": pip,
        "mean_model_size": mean_size,
        "size_distribution": size_dist,
        "coef_mean": coef_mean,
        "coef_sd": coef_sd,
        "top_models": top_models,
        "tss_residualized": tss,
        "runtime": {
            "backend": "cpu",
            "precision": "float64",
            "sufficient_statistics_s": t_suff,
            "enumeration_s": t_enum,
            "reduction_s": t_reduce,
            "coefficients_s": t_coef,
            "total_s": total,
            "models_per_second_enumeration": n_models / t_enum if t_enum > 0 else float("nan"),
        },
    }
