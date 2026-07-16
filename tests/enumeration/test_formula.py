"""Validate the enumeration core against an independent implementation of
the g-prior marginal likelihood written directly from Liang et al. (2008):

    log BF(M_gamma : M_null) = ((n - 1 - k)/2) log(1+g)
                             - ((n - 1)/2) log(1 + g (1 - R2_gamma))

computed with plain lstsq residual sums of squares (no shared code paths
with gpubma.cpu.enumeration).
"""

import itertools
import math

import numpy as np
import pytest

from gpubma.cpu.enumeration import enumerate_models
from gpubma.priors.model_priors import log_model_prior_function


def independent_log_scores(X, y, g, log_prior):
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    tss = yc @ yc
    scores = np.empty(1 << p)
    for mask in range(1 << p):
        idx = [j for j in range(p) if (mask >> j) & 1]
        k = len(idx)
        if k == 0:
            r2 = 0.0
        else:
            beta, *_ = np.linalg.lstsq(Xc[:, idx], yc, rcond=None)
            rss = float(((yc - Xc[:, idx] @ beta) ** 2).sum())
            r2 = 1.0 - rss / tss
        log_bf = 0.5 * (n - 1 - k) * math.log1p(g) - 0.5 * (n - 1) * math.log1p(g * (1 - r2))
        scores[mask] = log_bf + log_prior(k)
    return scores


@pytest.mark.parametrize("p", [1, 2, 5])
def test_matches_independent_formula(p):
    rng = np.random.default_rng(12345)
    n = 200
    X = rng.standard_normal((n, p))
    y = X @ np.linspace(1.0, 0.2, p) + rng.standard_normal(n)
    g = float(max(n, p * p))
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), p)

    ref = independent_log_scores(X, y, g, log_prior)

    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    out = enumerate_models(Xc, yc, df_resid=n - 1, g=g, log_model_prior=log_prior,
                           compute_coefficients=False)
    np.testing.assert_allclose(out["log_scores"], ref, rtol=0, atol=1e-9)


def test_single_predictor_posterior_by_hand():
    """p=1: two models; verify PMP against a direct two-line computation."""
    rng = np.random.default_rng(7)
    n = 150
    x = rng.standard_normal(n)
    y = 0.5 * x + rng.standard_normal(n)
    xc, yc = x - x.mean(), y - y.mean()
    g = float(n)
    r2 = float((xc @ yc) ** 2 / ((xc @ xc) * (yc @ yc)))
    log_bf1 = 0.5 * (n - 2) * math.log1p(g) - 0.5 * (n - 1) * math.log1p(g * (1 - r2))
    prior = 0.5  # betabinomial(1,1) with p=1: each size has probability 1/2
    z = prior + prior * math.exp(log_bf1)
    pmp1_expected = prior * math.exp(log_bf1) / z

    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), 1)
    out = enumerate_models(xc.reshape(-1, 1), yc, df_resid=n - 1, g=g,
                           log_model_prior=log_prior)
    assert out["pmp"][1] == pytest.approx(pmp1_expected, rel=1e-12)
    assert out["pip"][0] == pytest.approx(pmp1_expected, rel=1e-12)
