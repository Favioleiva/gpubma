"""Unit tests for gpubma.bfg.shell_recovery."""

from itertools import combinations
import math
import numpy as np
import pytest

from gpubma.bfg import shell_recovery
from gpubma.bfg.shell_recovery import (
    RemainderSampler,
    mask_from_rank,
    rank_indices,
    rank_mask,
    reconstruct_shell_mixture,
    sample_size,
    shell_population_size,
    unrank_combination,
    weighted_distances,
    weighted_summary,
)


def test_public_import():
    import gpubma.bfg.shell_recovery as sr
    assert hasattr(sr, "RemainderSampler")
    assert hasattr(sr, "RecoveryScorer")
    assert hasattr(sr, "sample_size")
    assert hasattr(sr, "reconstruct_shell_mixture")


def test_rank_bijection():
    for p in range(1, 9):
        for k in range(p + 1):
            ranks = []
            for c in combinations(range(p), k):
                r = rank_indices(c)
                ranks.append(r)
                assert unrank_combination(r, p, k) == list(c)
                # mask round-trip
                mask = sum(1 << j for j in c)
                assert rank_mask(mask, p) == r
                assert mask_from_rank(r, p, k) == mask
            assert sorted(ranks) == list(range(math.comb(p, k)))


def test_bigint_metadata_only():
    """Verify arbitrary precision integer rank/unrank without scoring."""
    for p in [60, 90, 100]:
        k = p // 2
        N = math.comb(p, k)
        for r in [0, N // 2, N - 1]:
            assert rank_indices(unrank_combination(r, p, k)) == r
        assert sample_size(p, k, 1000, 1_000_000) == 1_000_000


def test_sample_size_logic_and_validation():
    # p=30, k=15: N = 155,117,520. 1% is 1,551,176.
    # With cap=100,000, cap binds
    assert sample_size(30, 15, discovered=100, cap=100_000) == 100_000
    # With cap=10,000, cap binds
    assert sample_size(30, 15, discovered=100, cap=10_000) == 10_000
    # Small shell: p=30, k=3: N = 4060. 1% is 41. Cap is 100,000.
    assert sample_size(30, 3, discovered=0, cap=100_000) == 41
    # If discovered == N, sample size is 0
    assert sample_size(30, 1, discovered=30, cap=100_000) == 0

    # Invalid inputs
    for args in [(5, 2, 11, 10), (5, 2, -1, 10), (5, 2, 0, 0)]:
        with pytest.raises(ValueError):
            sample_size(*args)


def test_remainder_sampler_exclusions_and_nested_determinism():
    p, k = 15, 7
    excluded = [1, 3, 8]
    sampler1 = RemainderSampler(p, k, excluded_ranks=excluded, seed=123)
    chunk1 = sampler1.extend(30)
    chunk2 = sampler1.extend(100)

    sampler2 = RemainderSampler(p, k, excluded_ranks=excluded, seed=123)
    assert np.array_equal(chunk1, sampler2.extend(30))
    assert np.array_equal(chunk2, sampler2.extend(100))

    # All 100 selections must be unique and disjoint from excluded
    assert len(sampler1.selected) == len(set(sampler1.selected)) == 100
    assert not set(sampler1.selected) & set(excluded)

    # Full remainder census
    p_small, k_small = 8, 3
    c = RemainderSampler(p_small, k_small, [1, 2], seed=123)
    c.extend(math.comb(p_small, k_small) - 2)
    assert set(c.selected) == set(range(math.comb(p_small, k_small))) - {1, 2}


def test_reconstruct_shell_mixture_weight_accounting():
    # Shell with N=100, D=20 discovered, n=10 sampled
    disc = [10.0] * 20
    samp = [5.0] * 10
    values, weights = reconstruct_shell_mixture(disc, samp, N_k=100)
    assert len(values) == 30
    assert len(weights) == 30
    # Discovered weights: 1 / 100 = 0.01 each (total 0.20)
    assert np.allclose(weights[:20], 0.01)
    # Remainder weights: (100 - 20) / (100 * 10) = 80 / 1000 = 0.08 each (total 0.80)
    assert np.allclose(weights[20:], 0.08)
    assert np.isclose(weights.sum(), 1.0)


def test_population_weights_not_equal_pooling():
    values = np.array([8.0, 9.0, 0.0, 4.0])
    weights = np.array([0.1, 0.1, 0.4, 0.4])
    v, w, mean, variance, qs = weighted_summary(values, weights)
    # Analytical weighted mean: 8*0.1 + 9*0.1 + 0*0.4 + 4*0.4 = 0.8 + 0.9 + 0 + 1.6 = 3.3
    assert abs(mean - 3.3) < 1e-12
    # Simple average would be 5.25 != 3.3
    assert abs(mean - values.mean()) > 1.0
    assert abs(w[v <= 4].sum() - 0.8) < 1e-12

    # Census of remainder plus known subset recovers entire population
    x = np.arange(10, dtype=float)
    values = np.r_[x[8:], x[:8]]
    weights = np.full(10, 0.1)
    v, w, mean, variance, qs = weighted_summary(values, weights)
    ks, w1 = weighted_distances(x, v, w)
    assert abs(mean - 4.5) < 1e-12
    assert abs(variance - 8.25) < 1e-12
    assert ks < 1e-12 and w1 < 1e-12
