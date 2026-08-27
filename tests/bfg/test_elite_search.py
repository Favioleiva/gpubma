"""Unit tests for GPU Elite Search and threshold estimators."""

import math
import numpy as np
import pytest

from gpubma.bfg.elite_search import GPUEliteSearch, ThresholdEstimator, estimate_tail_prevalence
from gpubma.bfg.registry import EliteRegistry
from gpubma.bfg.scorer import BFGScorer
from gpubma.priors.model_priors import log_model_prior_function


def test_threshold_estimators():
    scores = np.linspace(10.0, 100.0, 100)
    q = 0.05

    tau_raw = ThresholdEstimator.raw_empirical_quantile(scores, q)
    tau_finite = ThresholdEstimator.finite_population_quantile(scores, q, N_k=1000)
    tau_cons = ThresholdEstimator.conservative_quantile(scores, q, alpha=0.05)

    assert tau_raw >= 90.0
    assert tau_finite >= 90.0
    assert tau_cons <= tau_raw  # Conservative is lower to capture more


def test_tail_prevalence_estimator():
    H_hat, se_H_hat, total_H, p_hat = estimate_tail_prevalence(
        h_k=10, m_k=100, r_k=50, N_k=1000
    )
    assert p_hat == 0.10
    assert H_hat == 950 * 0.10
    assert se_H_hat > 0.0


def test_gpu_elite_search_execution():
    rng = np.random.default_rng(20260715)
    n, p = 80, 8
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 2.0 + rng.normal(size=n)
    log_prior_fn, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), n_predictors=p)

    scorer = BFGScorer(X_r=X, y_r=y, df_resid=n-1, g=float(n), log_model_prior=log_prior_fn, device="cpu")
    registry = EliteRegistry(p=p)
    elite_search = GPUEliteSearch(scorer=scorer, registry=registry)

    res = elite_search.run_lattice_elite_search(
        k=3, r_k=10, m_k=15, target_q=0.10, rng=rng
    )
    assert res.k == 3
    assert res.r_k == 10
    assert res.m_k <= 15
    assert math.isfinite(res.threshold_tau)
