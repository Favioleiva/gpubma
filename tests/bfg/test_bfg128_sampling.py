"""Unit tests for BFG128 DirectUniformShellSampler128."""

import math
import itertools
import pytest
import numpy as np
import numba

from gpubma.bfg.bfg128.types import UINT64_MAX, validate_mask128
from gpubma.bfg.bfg128.bitops import unpack_mask, popcount
from gpubma.bfg.bfg128.sampling import (
    DirectUniformShellSampler128,
    SampledBatch128,
    sample_shell_128,
)
from gpubma.bfg.bfg128.rng import draw_raw_candidates


def test_boundary_dimensions_and_k():
    """Verify sampler across boundary dimensions (61, 67, 100, 128) and extreme k values."""
    for p in [61, 67, 100, 128]:
        k_cases = [0, 1, p // 2, p - 1, p]
        for k in k_cases:
            # Test small sample
            n_samples = 1 if (k == 0 or k == p) else 10
            sampler = DirectUniformShellSampler128(p=p, k=k, seed=42)
            batch = sampler.sample(n_samples=n_samples)

            assert len(batch) == n_samples
            assert batch.idx.shape == (n_samples, k)
            assert batch.mask_lo.shape == (n_samples,)
            assert batch.mask_hi.shape == (n_samples,)

            # Consistency checks
            for i in range(n_samples):
                lo = int(batch.mask_lo[i])
                hi = int(batch.mask_hi[i])
                unp = unpack_mask(lo, hi, p=p)
                assert unp == list(batch.idx[i])
                assert popcount(lo, hi) == k
                validate_mask128(lo, hi, p=p)


def test_mask_consistency_and_uniqueness():
    """Verify mask consistency and zero duplicates on larger sample."""
    p = 100
    k = 25
    n = 2000
    sampler = DirectUniformShellSampler128(p=p, k=k, seed=999)
    batch = sampler.sample(n_samples=n)

    assert len(batch) == n
    struct_arr = batch.to_structured_array()
    unique_models = set(zip(batch.mask_lo.tolist(), batch.mask_hi.tolist()))
    assert len(unique_models) == n, "Duplicate model found in sample batch!"

    for i in range(min(n, 100)):
        lo = int(batch.mask_lo[i])
        hi = int(batch.mask_hi[i])
        assert unpack_mask(lo, hi, p=p) == list(batch.idx[i])
        assert popcount(lo, hi) == k


def test_exclusion_ledger():
    """Verify that models in the exclusion ledger are never returned."""
    p = 67
    k = 5
    seed = 12345

    # First draw 50 models
    sampler1 = DirectUniformShellSampler128(p=p, k=k, seed=seed)
    batch1 = sampler1.sample(n_samples=50)
    excluded = set(zip(batch1.mask_lo.tolist(), batch1.mask_hi.tolist()))
    assert len(excluded) == 50

    # Draw next 100 models with exclusion ledger
    sampler2 = DirectUniformShellSampler128(p=p, k=k, seed=seed, excluded_models=excluded)
    batch2 = sampler2.sample(n_samples=100)

    batch2_models = set(zip(batch2.mask_lo.tolist(), batch2.mask_hi.tolist()))
    overlap = excluded.intersection(batch2_models)
    assert len(overlap) == 0, f"Exclusion failed: found {len(overlap)} excluded models!"


def test_remainder_limit_fail_fast():
    """Verify fail-fast error when requesting more models than available in remainder."""
    p = 10
    k = 2
    N_k = math.comb(p, k)  # 45

    sampler = DirectUniformShellSampler128(p=p, k=k, seed=1)
    with pytest.raises(ValueError, match="exceeds available shell remainder"):
        sampler.sample(n_samples=46)

    # With exclusion
    excl = {(1, 0), (2, 0), (4, 0)}  # 3 models
    sampler_excl = DirectUniformShellSampler128(p=p, k=k, seed=1, excluded_models=excl)
    with pytest.raises(ValueError, match="exceeds available shell remainder"):
        sampler_excl.sample(n_samples=43)  # 45 - 3 = 42 available


def test_thread_invariance():
    """Verify generated raw candidate sequence is bit-for-bit identical regardless of Numba threads."""
    p = 67
    k = 12
    B = 2000
    seed = 88888
    start_r = 1000

    idx1 = np.empty((B, k), dtype=np.int64)
    lo1 = np.empty(B, dtype=np.uint64)
    hi1 = np.empty(B, dtype=np.uint64)

    idx2 = np.empty((B, k), dtype=np.int64)
    lo2 = np.empty(B, dtype=np.uint64)
    hi2 = np.empty(B, dtype=np.uint64)

    default_threads = numba.get_num_threads()
    try:
        # Run with 1 thread
        numba.set_num_threads(1)
        draw_raw_candidates(p, k, B, start_r, seed, idx1, lo1, hi1)

        # Run with max threads
        numba.set_num_threads(default_threads)
        draw_raw_candidates(p, k, B, start_r, seed, idx2, lo2, hi2)

        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_array_equal(lo1, lo2)
        np.testing.assert_array_equal(hi1, hi2)
    finally:
        numba.set_num_threads(default_threads)


def test_chunk_invariance():
    """Verify sample of N in one call equals concatenation of smaller chunks."""
    p = 67
    k = 15
    n_total = 1000
    seed = 777
    excl = {(10, 0), (20, 0)}

    sampler = DirectUniformShellSampler128(p=p, k=k, seed=seed, excluded_models=excl)

    # Draw in single call with large chunk
    batch_large = sampler.sample(n_samples=n_total, chunk_size=1024)

    # Draw with small chunk size
    batch_small = sampler.sample(n_samples=n_total, chunk_size=37)

    np.testing.assert_array_equal(batch_large.idx, batch_small.idx)
    np.testing.assert_array_equal(batch_large.mask_lo, batch_small.mask_lo)
    np.testing.assert_array_equal(batch_large.mask_hi, batch_small.mask_hi)
    assert batch_large.next_raw_index == batch_small.next_raw_index


def test_restart_resume():
    """Verify split sampling from next_raw_index exactly matches uninterrupted run."""
    p = 100
    k = 20
    seed = 42

    sampler = DirectUniformShellSampler128(p=p, k=k, seed=seed)

    # 1. Uninterrupted 1000 models
    full_batch = sampler.sample(n_samples=1000)

    # 2. Split into two 500-model calls
    batch_part1 = sampler.sample(n_samples=500, start_raw_index=0)
    seen_part1 = set(zip(batch_part1.mask_lo.tolist(), batch_part1.mask_hi.tolist()))

    sampler_resumed = DirectUniformShellSampler128(
        p=p,
        k=k,
        seed=seed,
        excluded_models=seen_part1,
    )
    batch_part2 = sampler_resumed.sample(
        n_samples=500,
        start_raw_index=batch_part1.next_raw_index,
    )

    # Concatenate parts
    comb_idx = np.concatenate([batch_part1.idx, batch_part2.idx])
    comb_lo = np.concatenate([batch_part1.mask_lo, batch_part2.mask_lo])
    comb_hi = np.concatenate([batch_part1.mask_hi, batch_part2.mask_hi])

    np.testing.assert_array_equal(comb_idx, full_batch.idx)
    np.testing.assert_array_equal(comb_lo, full_batch.mask_lo)
    np.testing.assert_array_equal(comb_hi, full_batch.mask_hi)
    assert batch_part2.next_raw_index == full_batch.next_raw_index


def test_census_enumeration():
    """Verify deterministic combinations census against itertools.combinations."""
    cases = [(67, 2), (100, 2), (128, 2)]
    for p, k in cases:
        N_k = math.comb(p, k)
        sampler = DirectUniformShellSampler128(p=p, k=k, census_threshold=N_k)
        assert sampler.is_census

        batch = sampler.sample(n_samples=N_k)
        assert len(batch) == N_k
        assert batch.is_census

        # Compare against itertools
        expected_combos = list(itertools.combinations(range(p), k))
        for i, exp in enumerate(expected_combos):
            assert tuple(batch.idx[i]) == exp
            lo = int(batch.mask_lo[i])
            hi = int(batch.mask_hi[i])
            assert unpack_mask(lo, hi, p=p) == list(exp)
