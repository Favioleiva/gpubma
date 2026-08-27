"""Unit tests for combinatorial sampling and exact wing enumeration."""

import math
import numpy as np
import pytest

from gpubma.bfg.sampling import ExactWingEnumerator, LatticeSampler
from gpubma.bfg.scorer import BFGScorer, count_set_bits
from gpubma.priors.model_priors import log_model_prior_function


def test_lattice_sampler_no_duplicates_and_exclusions():
    p = 15
    sampler = LatticeSampler(p=p)
    rng = np.random.default_rng(42)

    k = 5
    N_k = math.comb(p, k)
    n_sample = 200

    # Test sample without exclusions
    samples = sampler.sample_combinations(k=k, n_samples=n_sample, rng=rng)
    assert len(samples) == n_sample
    assert len(set(samples)) == n_sample
    for m in samples:
        assert count_set_bits(m) == k

    # Test with exclusions
    exclude = set(samples[:50])
    new_samples = sampler.sample_combinations(k=k, n_samples=n_sample, rng=rng, exclude_set=exclude)
    assert len(new_samples) == n_sample
    assert len(set(new_samples)) == n_sample
    assert len(set(new_samples).intersection(exclude)) == 0


def test_exact_wing_enumerator():
    rng = np.random.default_rng(42)
    n, p = 50, 6
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    log_prior_fn, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), n_predictors=p)

    scorer = BFGScorer(
        X_r=X, y_r=y, df_resid=n-1, g=50.0, log_model_prior=log_prior_fn, device="cpu"
    )
    wing_enum = ExactWingEnumerator(scorer, p=p)

    assert wing_enum.is_exact_wing(0, max_wing_size=100)
    assert wing_enum.is_exact_wing(1, max_wing_size=100)

    models, scores, log_Z_k = wing_enum.enumerate_lattice(1)
    assert len(models) == p
    assert len(scores) == p
    assert math.isfinite(log_Z_k)
