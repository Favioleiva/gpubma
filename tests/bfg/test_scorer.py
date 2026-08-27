"""Unit tests for BFGScorer, bitmask representations, and cache integrity."""

import math
import numpy as np
import pytest

from gpubma.bfg.scorer import (
    BFGScorer,
    count_set_bits,
    detect_hardware,
    get_immediate_children,
    get_immediate_parents,
    hamming_distance,
    indices_to_model_id,
    is_subset,
    model_id_to_indices,
    model_id_to_vars,
    vars_to_model_id,
)
from gpubma.cpu.enumeration import enumerate_models
from gpubma.priors.model_priors import log_model_prior_function


def test_bitmask_utilities():
    p = 10
    names = [f"x{j + 1}" for j in range(p)]

    # Model containing x1, x3, x5 -> indices [0, 2, 4] -> mask 1 | 4 | 16 = 21
    m = 21
    assert count_set_bits(m) == 3
    assert model_id_to_indices(m, p) == [0, 2, 4]
    assert indices_to_model_id([0, 2, 4]) == 21
    assert model_id_to_vars(m, names) == ["x1", "x3", "x5"]
    assert vars_to_model_id(["x1", "x3", "x5"], names) == 21

    # Hamming distance
    m2 = 23  # indices [0, 1, 2, 4] (added x2)
    assert hamming_distance(m, m2) == 1
    assert is_subset(m, m2)
    assert not is_subset(m2, m)

    # Parents & Children
    parents = get_immediate_parents(m, p)
    assert len(parents) == 3
    for p_id in parents:
        assert count_set_bits(p_id) == 2
        assert is_subset(p_id, m)

    children = get_immediate_children(m, p)
    assert len(children) == p - 3
    for c_id in children:
        assert count_set_bits(c_id) == 4
        assert is_subset(m, c_id)


def test_hardware_detection():
    hw = detect_hardware("cpu")
    assert hw["device_type"] == "cpu"
    assert hw["float64_supported"] is True

    hw_cuda = detect_hardware("cuda")
    assert "cuda_available" in hw_cuda


def test_scorer_exact_parity_with_cpu_enumeration():
    rng = np.random.default_rng(20260715)
    n, p = 100, 6
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 1.8 + X[:, 1] * 1.2 + rng.normal(size=n)
    
    # FWL project on intercept
    X_r = X - X.mean(axis=0)
    y_r = y - y.mean()
    tss_norm = float(y_r @ y_r)
    df_resid = n - 1
    g = float(max(n, p * p))

    log_prior_fn, prior_desc = log_model_prior_function(("betabinomial", 1.0, 1.0), n_predictors=p)

    scorer = BFGScorer(
        X_r=X_r,
        y_r=y_r,
        df_resid=df_resid,
        g=g,
        log_model_prior=log_prior_fn,
        tss_norm=tss_norm,
        k_always=0,
        device="cpu",
    )

    cpu_enum = enumerate_models(
        X=X_r,
        y=y_r,
        df_resid=df_resid,
        g=g,
        log_model_prior=log_prior_fn,
        tss_norm=tss_norm,
        k_always=0,
        compute_coefficients=True,
    )

    all_models = list(range(1 << p))
    score_dict = scorer.score_batch(all_models)

    # Verify log scores match to machine precision (< 1e-12)
    for m in all_models:
        s_bfg = score_dict[m]
        s_cpu = cpu_enum["log_scores"][m]
        assert abs(s_bfg - s_cpu) < 1e-12, f"Score mismatch at model {m}: {s_bfg} vs {s_cpu}"

    # Verify cache hits
    assert scorer.n_unique_evaluated == (1 << p)
    # Scoring again should yield 100% cache hits
    scorer.score_batch(all_models)
    assert scorer.n_cache_hits == (1 << p)
