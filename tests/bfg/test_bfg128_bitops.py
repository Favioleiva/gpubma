"""Unit tests for BFG128 Mask128 representation, bit operations, boundaries, and batch kernels."""

import pytest
import numpy as np

from gpubma.bfg.bfg128.types import (
    UINT64_MAX,
    UINT64_BITS,
    MAX_P,
    MIN_WIDE_P,
    DTYPE_MODEL_128,
    validate_mask128,
    mask_to_hex,
    mask_from_hex,
)
from gpubma.bfg.bfg128.bitops import (
    pack_indices,
    unpack_mask,
    popcount,
    contains,
    add_bit,
    remove_bit,
    get_parents,
    get_children,
    popcount_batch,
    pack_indices_batch,
    unpack_mask_batch,
    batch_get_parents,
    batch_get_children,
    to_structured_array,
    from_structured_array,
)


# ==============================================================================
# BOUNDARY AND WORD-CROSSING TESTS
# ==============================================================================

def test_boundary_bits():
    """Verify bit packing, unpacking, and containment at exact 64-bit word boundaries."""
    # Test j = 0 (first bit of mask_lo)
    lo0, hi0 = pack_indices([0], p=128)
    assert lo0 == 1
    assert hi0 == 0
    assert unpack_mask(lo0, hi0, p=128) == [0]
    assert contains(lo0, hi0, 0, p=128)
    assert not contains(lo0, hi0, 1, p=128)

    # Test j = 63 (last bit of mask_lo / MSB of uint64)
    lo63, hi63 = pack_indices([63], p=128)
    assert lo63 == (1 << 63)
    assert hi63 == 0
    assert unpack_mask(lo63, hi63, p=128) == [63]
    assert contains(lo63, hi63, 63, p=128)
    assert not contains(lo63, hi63, 62, p=128)
    assert not contains(lo63, hi63, 64, p=128)

    # Test j = 64 (first bit of mask_hi / LSB of high word)
    lo64, hi64 = pack_indices([64], p=128)
    assert lo64 == 0
    assert hi64 == 1
    assert unpack_mask(lo64, hi64, p=128) == [64]
    assert contains(lo64, hi64, 64, p=128)
    assert not contains(lo64, hi64, 63, p=128)
    assert not contains(lo64, hi64, 65, p=128)

    # Test j = 127 (last bit of mask_hi / MSB of high word)
    lo127, hi127 = pack_indices([127], p=128)
    assert lo127 == 0
    assert hi127 == (1 << 63)
    assert unpack_mask(lo127, hi127, p=128) == [127]
    assert contains(lo127, hi127, 127, p=128)
    assert not contains(lo127, hi127, 126, p=128)


def test_straddling_and_isolated_words():
    """Verify models with bits only in lo, only in hi, or straddling both words."""
    # Bits only in mask_lo
    lo_only_idx = [0, 5, 23, 63]
    lo, hi = pack_indices(lo_only_idx, p=128)
    assert hi == 0
    assert unpack_mask(lo, hi, p=128) == lo_only_idx
    assert popcount(lo, hi) == len(lo_only_idx)

    # Bits only in mask_hi
    hi_only_idx = [64, 70, 100, 127]
    lo, hi = pack_indices(hi_only_idx, p=128)
    assert lo == 0
    assert unpack_mask(lo, hi, p=128) == hi_only_idx
    assert popcount(lo, hi) == len(hi_only_idx)

    # Straddling both words around boundary 63/64
    straddle_idx = [62, 63, 64, 65]
    lo, hi = pack_indices(straddle_idx, p=128)
    assert lo == (1 << 62) | (1 << 63)
    assert hi == (1 << 0) | (1 << 1)
    assert unpack_mask(lo, hi, p=128) == straddle_idx
    assert popcount(lo, hi) == 4


# ==============================================================================
# INVARIANTS AND CORNER CASES
# ==============================================================================

def test_null_and_saturated_models():
    """Verify null model (k=0) and fully saturated models (k=p)."""
    for p in [61, 64, 67, 100, 128]:
        # Null model
        lo_null, hi_null = pack_indices([], p=p)
        assert lo_null == 0 and hi_null == 0
        assert unpack_mask(lo_null, hi_null, p=p) == []
        assert popcount(lo_null, hi_null) == 0
        assert get_parents(lo_null, hi_null, p=p) == []
        children_null = get_children(lo_null, hi_null, p=p)
        assert len(children_null) == p

        # Saturated model
        sat_idx = list(range(p))
        lo_sat, hi_sat = pack_indices(sat_idx, p=p)
        assert popcount(lo_sat, hi_sat) == p
        assert unpack_mask(lo_sat, hi_sat, p=p) == sat_idx
        assert get_children(lo_sat, hi_sat, p=p) == []
        parents_sat = get_parents(lo_sat, hi_sat, p=p)
        assert len(parents_sat) == p


def test_parent_child_invariants():
    """Verify that parents have size k-1 and are subsets; children have size k+1 and are supersets."""
    p = 67
    idx = [0, 62, 63, 64, 66]  # k = 5, straddling boundary
    k = len(idx)
    lo, hi = pack_indices(idx, p=p)

    # Test Parents
    parents = get_parents(lo, hi, p=p)
    assert len(parents) == k
    for plo, phi in parents:
        assert popcount(plo, phi) == k - 1
        # Subset check: all bits in parent must be in original
        assert (plo & lo) == plo
        assert (phi & hi) == phi
        # Hamming distance must be 1
        assert (popcount(lo ^ plo, hi ^ phi)) == 1

    # Test Children
    children = get_children(lo, hi, p=p)
    assert len(children) == p - k
    for clo, chi in children:
        assert popcount(clo, chi) == k + 1
        # Superset check: all bits in original must be in child
        assert (clo & lo) == lo
        assert (chi & hi) == hi
        # Hamming distance must be 1
        assert (popcount(clo ^ lo, chi ^ hi)) == 1
        # No bits >= p
        validate_mask128(clo, chi, p=p)


def test_add_remove_bit():
    """Verify adding and removing individual bits."""
    p = 100
    lo, hi = pack_indices([10, 64], p=p)
    assert popcount(lo, hi) == 2

    # Add bit 63 (at lo boundary)
    lo1, hi1 = add_bit(lo, hi, 63, p=p)
    assert contains(lo1, hi1, 63, p=p)
    assert popcount(lo1, hi1) == 3

    # Add bit 64 (already present, should remain 3)
    lo2, hi2 = add_bit(lo1, hi1, 64, p=p)
    assert popcount(lo2, hi2) == 3
    assert (lo2, hi2) == (lo1, hi1)

    # Remove bit 64
    lo3, hi3 = remove_bit(lo2, hi2, 64, p=p)
    assert not contains(lo3, hi3, 64, p=p)
    assert popcount(lo3, hi3) == 2

    # Remove bit 10
    lo4, hi4 = remove_bit(lo3, hi3, 10, p=p)
    assert not contains(lo4, hi4, 10, p=p)
    assert popcount(lo4, hi4) == 1
    assert unpack_mask(lo4, hi4, p=p) == [63]


def test_validation_and_errors():
    """Verify defensive validation fails closed on illegal inputs."""
    # Out of range index
    with pytest.raises(ValueError):
        pack_indices([128], p=128)
    with pytest.raises(ValueError):
        pack_indices([67], p=67)
    with pytest.raises(ValueError):
        pack_indices([-1], p=67)

    # p out of range
    with pytest.raises(ValueError):
        pack_indices([5], p=129)
    with pytest.raises(ValueError):
        pack_indices([5], p=0)

    # Bits >= p in mask
    with pytest.raises(ValueError):
        # p=67 allows up to variable 66 (bit 2 of hi). Bit 3 of hi is variable 67.
        validate_mask128(0, 1 << 3, p=67)

    with pytest.raises(ValueError):
        # p=60 allows only mask_lo bits 0..59. Bit 60 should fail.
        validate_mask128(1 << 60, 0, p=60)

    with pytest.raises(ValueError):
        # p <= 64 requires mask_hi == 0
        validate_mask128(0, 1, p=64)


# ==============================================================================
# BATCH OPERATIONS AND NUMPY/NUMBA EQUIVALENTS
# ==============================================================================

def test_batch_operations():
    """Verify batch packing, batch popcount, batch unpacking, and parent/child kernels."""
    p = 67
    k = 15
    B = 250

    # Draw random models
    np.random.seed(42)
    idx_batch = np.sort(
        np.array([np.random.choice(p, size=k, replace=False) for _ in range(B)]),
        axis=1,
    )

    # Batch pack
    lo_batch, hi_batch = pack_indices_batch(idx_batch, p=p)
    assert lo_batch.shape == (B,)
    assert hi_batch.shape == (B,)
    assert lo_batch.dtype == np.uint64
    assert hi_batch.dtype == np.uint64

    # Compare batch pack with scalar pack
    for i in range(B):
        slo, shi = pack_indices(idx_batch[i], p=p)
        assert slo == int(lo_batch[i])
        assert shi == int(hi_batch[i])

    # Batch popcount
    counts = popcount_batch(lo_batch, hi_batch)
    assert np.all(counts == k)

    # Batch unpack
    unpacked_batch = unpack_mask_batch(lo_batch, hi_batch, p=p, k=k)
    np.testing.assert_array_equal(unpacked_batch, idx_batch)

    # Batch parents
    parents_lo, parents_hi = batch_get_parents(lo_batch, hi_batch, p=p, k=k)
    assert parents_lo.shape == (B, k)
    assert parents_hi.shape == (B, k)
    # Check parity with scalar get_parents for each model
    for i in range(min(B, 20)):
        scalar_parents = get_parents(int(lo_batch[i]), int(hi_batch[i]), p=p)
        for j, (slo, shi) in enumerate(scalar_parents):
            assert int(parents_lo[i, j]) == slo
            assert int(parents_hi[i, j]) == shi

    # Batch children
    children_lo, children_hi = batch_get_children(lo_batch, hi_batch, p=p, k=k)
    assert children_lo.shape == (B, p - k)
    assert children_hi.shape == (B, p - k)
    for i in range(min(B, 20)):
        scalar_children = get_children(int(lo_batch[i]), int(hi_batch[i]), p=p)
        for j, (slo, shi) in enumerate(scalar_children):
            assert int(children_lo[i, j]) == slo
            assert int(children_hi[i, j]) == shi


def test_structured_array():
    """Verify conversion to and from DTYPE_MODEL_128 structured array."""
    lo = np.array([1, 2, 3], dtype=np.uint64)
    hi = np.array([4, 5, 6], dtype=np.uint64)

    s_arr = to_structured_array(lo, hi)
    assert s_arr.dtype == DTYPE_MODEL_128
    assert len(s_arr) == 3
    assert s_arr["lo"][0] == 1
    assert s_arr["hi"][0] == 4

    lo_out, hi_out = from_structured_array(s_arr)
    np.testing.assert_array_equal(lo_out, lo)
    np.testing.assert_array_equal(hi_out, hi)


def test_hex_formatting():
    """Verify canonical hex formatting and parsing."""
    lo = 0x0123_4567_89AB_CDEF
    hi = 0xFEDC_BA98_7654_3210
    hex_str = mask_to_hex(lo, hi)
    assert hex_str == "0xfedcba9876543210_0123456789abcdef"

    lo_parsed, hi_parsed = mask_from_hex(hex_str)
    assert lo_parsed == lo
    assert hi_parsed == hi
