"""Deterministic seed reproducibility tests for BFG."""

import numpy as np
import pytest

from gpubma.bfg import fit_bfg


def test_deterministic_seed_reproducibility():
    rng = np.random.default_rng(20260715)
    n, p = 80, 8
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 1.5 + X[:, 1] * 1.0 + rng.normal(size=n)

    # Run 1
    res1 = fit_bfg(
        y=y,
        X=X,
        budget_models=1000,
        seed=424242,
        verbose=False,
    )

    # Run 2 with identical seed
    res2 = fit_bfg(
        y=y,
        X=X,
        budget_models=1000,
        seed=424242,
        verbose=False,
    )

    # Verify bit-level reproducibility
    assert res1.log_Z == res2.log_Z
    assert res1.map_model_id == res2.map_model_id
    assert res1.map_log_score == res2.map_log_score
    assert np.allclose(res1.pips.values, res2.pips.values, atol=1e-12)
    assert np.allclose(res1.model_size_posterior.values, res2.model_size_posterior.values, atol=1e-12)
