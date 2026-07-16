"""Explicit joint-design reference scorer (validation only, never optimized).

This module scores every candidate model by building the FULL joint design
[X_gamma, W] per model — controls and fixed-effect dummies included as
explicit columns — and evaluating the Stata "shrink" convention directly:
flat intercept, joint Zellner g-prior over optional and always slopes,
df = n - 1. It is deliberately naive (O(2^p * (k+q)^3)): its only purpose
is to prove numerically that the FWL/block-matrix fast path used by
``gpubma.cpu.enumeration`` (residualized sufficient statistics + ESS_W)
computes exactly the same quantities. See docs/FWL_BLOCK_FORMULATION.md
for the derivation and tests/enumeration/test_fwl_block_equivalence.py
for the equivalence suite.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp


def joint_explicit_reference(
    X: np.ndarray,
    W: np.ndarray,
    y: np.ndarray,
    *,
    g: float,
    log_model_prior,
    top_k: int = 10,
):
    """Score all 2^p models with the explicit joint design (shrink convention).

    Parameters
    ----------
    X : (n, p) optional predictors (raw, NOT residualized).
    W : (n, q) always-included slope columns (controls + FE dummies, raw;
        NO intercept column — the intercept is flat and handled by centering).
    y : (n,) outcome (raw).
    g : fixed Zellner g.
    log_model_prior : callable k -> log prior mass for a model of size k.

    Returns a dict with the same core arrays as the fast path:
    log_scores (mask order), pmp, pip, coef_mean/coef_sd for the optional
    predictors, always_mean/always_sd for the always slopes, mean model
    size, size distribution, and top models.
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    q = W.shape[1]
    n_models = 1 << p
    if n - 1 <= 2:
        raise ValueError("need n - 1 > 2 for the shrink convention")

    # Flat intercept == everything enters centered (projection on the 1 vector).
    yc = y - y.mean()
    Xc = X - X.mean(axis=0, keepdims=True)
    Wc = W - W.mean(axis=0, keepdims=True) if q else W
    tss_c = float(yc @ yc)

    df = n - 1
    log1pg = float(np.log1p(g))
    shrink = g / (1.0 + g)
    tiny = np.finfo(np.float64).tiny

    log_scores = np.empty(n_models, dtype=np.float64)
    sizes = np.empty(n_models, dtype=np.int64)
    cond_mean = np.zeros((n_models, p), dtype=np.float64)
    cond_var = np.zeros((n_models, p), dtype=np.float64)
    cond_mean_w = np.zeros((n_models, q), dtype=np.float64)
    cond_var_w = np.zeros((n_models, q), dtype=np.float64)

    for mask in range(n_models):
        idx = [j for j in range(p) if (mask >> j) & 1]
        k = len(idx)
        sizes[mask] = k
        V = np.hstack([Xc[:, idx], Wc]) if (k + q) else np.empty((n, 0))
        if V.shape[1] == 0:
            ess = 0.0
        else:
            G = V.T @ V
            b = V.T @ yc
            c, low = cho_factor(G, lower=True, check_finite=False)
            theta_hat = cho_solve((c, low), b, check_finite=False)
            ess = float(b @ theta_hat)
            Ginv = cho_solve((c, low), np.eye(V.shape[1]), check_finite=False)
        one_minus_r2 = max((tss_c - ess) / tss_c, tiny)
        log_scores[mask] = (
            0.5 * (df - k - q) * log1pg
            - 0.5 * df * np.log1p(g * one_minus_r2)
            + log_model_prior(k)
        )
        if V.shape[1]:
            e_sigma2 = (tss_c - shrink * ess) / (df - 2)
            theta_mean = shrink * theta_hat
            theta_var = e_sigma2 * shrink * np.diag(Ginv)
            cond_mean[mask, idx] = theta_mean[:k]
            cond_var[mask, idx] = theta_var[:k]
            cond_mean_w[mask, :] = theta_mean[k:]
            cond_var_w[mask, :] = theta_var[k:]

    log_norm = float(logsumexp(log_scores))
    pmp = np.exp(log_scores - log_norm)

    masks = np.arange(n_models, dtype=np.int64)
    pip = np.array([pmp[((masks >> j) & 1).astype(bool)].sum() for j in range(p)])
    pip = np.clip(pip, 0.0, 1.0)

    coef_mean = pmp @ cond_mean
    coef_sd = np.sqrt(np.maximum(pmp @ (cond_var + cond_mean**2) - coef_mean**2, 0.0))
    always_mean = pmp @ cond_mean_w if q else np.zeros(0)
    always_sd = (
        np.sqrt(np.maximum(pmp @ (cond_var_w + cond_mean_w**2) - always_mean**2, 0.0))
        if q else np.zeros(0)
    )

    size_dist = np.bincount(sizes, weights=pmp, minlength=p + 1)
    order = np.argsort(pmp)[::-1][: min(top_k, n_models)]
    return {
        "n_models": n_models,
        "log_scores": log_scores,
        "pmp": pmp,
        "pip": pip,
        "coef_mean": coef_mean,
        "coef_sd": coef_sd,
        "always_mean": always_mean,
        "always_sd": always_sd,
        "mean_model_size": float(sizes @ pmp),
        "size_distribution": size_dist,
        "top_masks": masks[order],
    }
