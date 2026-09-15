"""Core bit operations and batch kernels for 128-bit model representations.

This module provides:
1. Canonical scalar operations:
   - pack_indices(indices, p) -> (mask_lo, mask_hi)
   - unpack_mask(mask_lo, mask_hi, p) -> list of active variable indices
   - popcount(mask_lo, mask_hi) -> k
   - contains(mask_lo, mask_hi, j, p) -> bool
   - add_bit(mask_lo, mask_hi, j, p) -> (mask_lo, mask_hi)
   - remove_bit(mask_lo, mask_hi, j, p) -> (mask_lo, mask_hi)
   - get_parents(mask_lo, mask_hi, p) -> list of parent models (size k-1)
   - get_children(mask_lo, mask_hi, p) -> list of child models (size k+1)

2. High-performance batch equivalents:
   - popcount_batch(mask_lo, mask_hi) -> ndarray
   - pack_indices_batch(idx, p) -> (mask_lo, mask_hi)
   - unpack_mask_batch(mask_lo, mask_hi, p, k) -> ndarray of shape (B, k)
   - batch_get_parents(mask_lo, mask_hi, p, k) -> (parents_lo, parents_hi) of shape (B, k)
   - batch_get_children(mask_lo, mask_hi, p, k) -> (children_lo, children_hi) of shape (B, p - k)
   - to_structured_array / from_structured_array for DTYPE_MODEL_128
"""

from typing import Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    import numba
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

from gpubma.bfg.bfg128.types import (
    DTYPE_MODEL_128,
    MAX_P,
    UINT64_BITS,
    UINT64_MAX,
    Mask128,
    validate_mask128,
)


# ==============================================================================
# SCALAR CORE BIT OPERATIONS
# ==============================================================================

def pack_indices(indices: Iterable[int], p: int = MAX_P) -> Mask128:
    """Pack an iterable of predictor variable indices into a canonical (mask_lo, mask_hi) pair.

    Parameters
    ----------
    indices : Iterable[int]
        Predictor indices, 0-indexed, where 0 <= j < p <= 128.
    p : int, default=128
        Total number of candidate predictors (1 <= p <= 128).

    Returns
    -------
    (mask_lo, mask_hi) : Tuple[int, int]
        Unsigned 64-bit integer words representing active model predictors.

    Raises
    ------
    ValueError
        If p is outside [1, 128], or if any index is outside [0, p - 1].
    """
    if p < 1 or p > MAX_P:
        raise ValueError(f"p must be in [1, {MAX_P}], got {p}")

    mask_lo = 0
    mask_hi = 0

    for j in indices:
        if not isinstance(j, (int, np.integer)):
            raise TypeError(f"Predictor index must be integer, got {type(j).__name__}")
        idx = int(j)
        if idx < 0 or idx >= p:
            raise ValueError(f"Predictor index {idx} out of valid range [0, {p})")

        if idx < UINT64_BITS:
            mask_lo |= (1 << idx)
        else:
            mask_hi |= (1 << (idx - UINT64_BITS))

    return (mask_lo, mask_hi)


def unpack_mask(mask_lo: int, mask_hi: int, p: int = MAX_P) -> List[int]:
    """Unpack a 128-bit model mask into a sorted list of active predictor indices.

    Uses efficient O(k) least-significant-bit extraction.

    Parameters
    ----------
    mask_lo : int
        Low 64-bit word (variables 0..63).
    mask_hi : int
        High 64-bit word (variables 64..127).
    p : int, default=128
        Total number of candidate predictors. Validates that no active bit >= p.

    Returns
    -------
    List[int]
        Sorted list of active predictor indices in [0, p - 1].
    """
    validate_mask128(mask_lo, mask_hi, p)

    indices: List[int] = []

    # Unpack mask_lo
    m_lo = int(mask_lo) & UINT64_MAX
    while m_lo:
        lsb = m_lo & -m_lo
        indices.append(lsb.bit_length() - 1)
        m_lo ^= lsb

    # Unpack mask_hi
    m_hi = int(mask_hi) & UINT64_MAX
    while m_hi:
        lsb = m_hi & -m_hi
        indices.append(UINT64_BITS + (lsb.bit_length() - 1))
        m_hi ^= lsb

    return indices


def popcount(mask_lo: int, mask_hi: int) -> int:
    """Compute the model size k (number of active predictors) in O(1) hardware instructions.

    Parameters
    ----------
    mask_lo : int
        Low 64-bit word.
    mask_hi : int
        High 64-bit word.

    Returns
    -------
    k : int
        Total number of active bits across both words.
    """
    return (int(mask_lo) & UINT64_MAX).bit_count() + (int(mask_hi) & UINT64_MAX).bit_count()


def contains(mask_lo: int, mask_hi: int, j: int, p: int = MAX_P) -> bool:
    """Check whether predictor variable j is included in the model.

    Parameters
    ----------
    mask_lo : int
        Low 64-bit word.
    mask_hi : int
        High 64-bit word.
    j : int
        Predictor index to test (0 <= j < p).
    p : int, default=128
        Total number of candidate predictors.

    Returns
    -------
    bool
        True if predictor j is active, False otherwise.
    """
    if j < 0 or j >= p:
        raise ValueError(f"Predictor index {j} out of range [0, {p})")

    if j < UINT64_BITS:
        return bool((int(mask_lo) >> j) & 1)
    else:
        return bool((int(mask_hi) >> (j - UINT64_BITS)) & 1)


def add_bit(mask_lo: int, mask_hi: int, j: int, p: int = MAX_P) -> Mask128:
    """Return a new model with predictor variable j added (toggled ON).

    Parameters
    ----------
    mask_lo : int
        Low 64-bit word.
    mask_hi : int
        High 64-bit word.
    j : int
        Predictor index to add (0 <= j < p).
    p : int, default=128
        Total number of candidate predictors.

    Returns
    -------
    (new_lo, new_hi) : Mask128
    """
    if j < 0 or j >= p:
        raise ValueError(f"Predictor index {j} out of range [0, {p})")

    lo = int(mask_lo) & UINT64_MAX
    hi = int(mask_hi) & UINT64_MAX

    if j < UINT64_BITS:
        return (lo | (1 << j), hi)
    else:
        return (lo, hi | (1 << (j - UINT64_BITS)))


def remove_bit(mask_lo: int, mask_hi: int, j: int, p: int = MAX_P) -> Mask128:
    """Return a new model with predictor variable j removed (toggled OFF).

    Parameters
    ----------
    mask_lo : int
        Low 64-bit word.
    mask_hi : int
        High 64-bit word.
    j : int
        Predictor index to remove (0 <= j < p).
    p : int, default=128
        Total number of candidate predictors.

    Returns
    -------
    (new_lo, new_hi) : Mask128
    """
    if j < 0 or j >= p:
        raise ValueError(f"Predictor index {j} out of range [0, {p})")

    lo = int(mask_lo) & UINT64_MAX
    hi = int(mask_hi) & UINT64_MAX

    if j < UINT64_BITS:
        return ((lo & ~(1 << j)) & UINT64_MAX, hi)
    else:
        return (lo, (hi & ~(1 << (j - UINT64_BITS))) & UINT64_MAX)


def get_parents(mask_lo: int, mask_hi: int, p: int = MAX_P) -> List[Mask128]:
    """Generate all immediate parent models (size k - 1) by dropping each active predictor.

    Parameters
    ----------
    mask_lo : int
        Low 64-bit word.
    mask_hi : int
        High 64-bit word.
    p : int, default=128
        Total number of candidate predictors.

    Returns
    -------
    List[Mask128]
        List of k parent models, each having popcount k - 1 and contained in this model.
    """
    validate_mask128(mask_lo, mask_hi, p)

    parents: List[Mask128] = []
    lo = int(mask_lo) & UINT64_MAX
    hi = int(mask_hi) & UINT64_MAX

    # Drop bits from mask_lo
    m_lo = lo
    while m_lo:
        lsb = m_lo & -m_lo
        parents.append((lo ^ lsb, hi))
        m_lo ^= lsb

    # Drop bits from mask_hi
    m_hi = hi
    while m_hi:
        lsb = m_hi & -m_hi
        parents.append((lo, hi ^ lsb))
        m_hi ^= lsb

    return parents


def get_children(mask_lo: int, mask_hi: int, p: int) -> List[Mask128]:
    """Generate all immediate child models (size k + 1) by adding each inactive predictor in [0, p-1].

    Parameters
    ----------
    mask_lo : int
        Low 64-bit word.
    mask_hi : int
        High 64-bit word.
    p : int
        Total number of candidate predictors (1 <= p <= 128).

    Returns
    -------
    List[Mask128]
        List of p - k child models, each having popcount k + 1 and containing this model.
    """
    validate_mask128(mask_lo, mask_hi, p)

    children: List[Mask128] = []
    lo = int(mask_lo) & UINT64_MAX
    hi = int(mask_hi) & UINT64_MAX

    # Invert bits in mask_lo up to min(p, 64)
    p_lo = min(p, UINT64_BITS)
    mask_lo_valid = ((1 << p_lo) - 1) if p_lo < UINT64_BITS else UINT64_MAX
    inv_lo = (~lo) & mask_lo_valid

    while inv_lo:
        lsb = inv_lo & -inv_lo
        children.append((lo | lsb, hi))
        inv_lo ^= lsb

    # Invert bits in mask_hi up to (p - 64) if p > 64
    if p > UINT64_BITS:
        p_hi = p - UINT64_BITS
        mask_hi_valid = ((1 << p_hi) - 1) if p_hi < UINT64_BITS else UINT64_MAX
        inv_hi = (~hi) & mask_hi_valid

        while inv_hi:
            lsb = inv_hi & -inv_hi
            children.append((lo, hi | lsb))
            inv_hi ^= lsb

    return children


# ==============================================================================
# VECTORIZED / BATCH KERNELS
# ==============================================================================

def popcount_batch(
    mask_lo: Union[np.ndarray, Sequence[int]],
    mask_hi: Union[np.ndarray, Sequence[int]],
) -> np.ndarray:
    """Compute popcounts (model sizes) for a batch of 128-bit models.

    Parameters
    ----------
    mask_lo : np.ndarray of shape (B,), dtype uint64
    mask_hi : np.ndarray of shape (B,), dtype uint64

    Returns
    -------
    np.ndarray of shape (B,), dtype int64
        Popcount (model size k) for each model in the batch.
    """
    lo = np.asarray(mask_lo, dtype=np.uint64)
    hi = np.asarray(mask_hi, dtype=np.uint64)

    if lo.shape != hi.shape:
        raise ValueError(f"Shape mismatch: mask_lo {lo.shape} != mask_hi {hi.shape}")

    if hasattr(np, "bitwise_count"):
        return (np.bitwise_count(lo) + np.bitwise_count(hi)).astype(np.int64)
    else:  # pragma: no cover
        # Vectorized bit-manipulation fallback
        def _pop64(v):
            v = v - ((v >> np.uint64(1)) & np.uint64(0x5555555555555555))
            v = (v & np.uint64(0x3333333333333333)) + ((v >> np.uint64(2)) & np.uint64(0x3333333333333333))
            v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
            return ((v * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)

        return _pop64(lo) + _pop64(hi)


if _HAS_NUMBA:

    @numba.njit(parallel=True, fastmath=True)
    def _pack_indices_numba(idx: np.ndarray, mask_lo: np.ndarray, mask_hi: np.ndarray) -> None:
        B = idx.shape[0]
        k = idx.shape[1]
        for i in numba.prange(B):
            lo = np.uint64(0)
            hi = np.uint64(0)
            for m in range(k):
                v = idx[i, m]
                if v < 64:
                    lo |= np.uint64(1) << np.uint64(v)
                else:
                    hi |= np.uint64(1) << np.uint64(v - 64)
            mask_lo[i] = lo
            mask_hi[i] = hi

    @numba.njit(parallel=True, fastmath=True)
    def _unpack_mask_numba(
        mask_lo: np.ndarray,
        mask_hi: np.ndarray,
        p: int,
        k: int,
        out: np.ndarray,
    ) -> None:
        B = mask_lo.shape[0]
        p_lo = min(p, 64)
        p_hi = max(0, p - 64)
        for i in numba.prange(B):
            col = 0
            l = mask_lo[i]
            h = mask_hi[i]
            for j in range(p_lo):
                if (l >> np.uint64(j)) & np.uint64(1):
                    out[i, col] = j
                    col += 1
            for j in range(p_hi):
                if (h >> np.uint64(j)) & np.uint64(1):
                    out[i, col] = 64 + j
                    col += 1

    @numba.njit(parallel=True, fastmath=True)
    def _batch_get_parents_numba(
        lo: np.ndarray,
        hi: np.ndarray,
        p: int,
        k: int,
        out_lo: np.ndarray,
        out_hi: np.ndarray,
    ) -> None:
        B = lo.shape[0]
        p_lo = min(p, 64)
        p_hi = max(0, p - 64)
        for i in numba.prange(B):
            l = lo[i]
            h = hi[i]
            col = 0
            for j in range(p_lo):
                bit = np.uint64(1) << np.uint64(j)
                if l & bit:
                    out_lo[i, col] = l ^ bit
                    out_hi[i, col] = h
                    col += 1
            for j in range(p_hi):
                bit = np.uint64(1) << np.uint64(j)
                if h & bit:
                    out_lo[i, col] = l
                    out_hi[i, col] = h ^ bit
                    col += 1

    @numba.njit(parallel=True, fastmath=True)
    def _batch_get_children_numba(
        lo: np.ndarray,
        hi: np.ndarray,
        p: int,
        k: int,
        out_lo: np.ndarray,
        out_hi: np.ndarray,
    ) -> None:
        B = lo.shape[0]
        p_lo = min(p, 64)
        p_hi = max(0, p - 64)
        for i in numba.prange(B):
            l = lo[i]
            h = hi[i]
            col = 0
            for j in range(p_lo):
                bit = np.uint64(1) << np.uint64(j)
                if not (l & bit):
                    out_lo[i, col] = l | bit
                    out_hi[i, col] = h
                    col += 1
            for j in range(p_hi):
                bit = np.uint64(1) << np.uint64(j)
                if not (h & bit):
                    out_lo[i, col] = l
                    out_hi[i, col] = h | bit
                    col += 1


def pack_indices_batch(idx: np.ndarray, p: int = MAX_P) -> Tuple[np.ndarray, np.ndarray]:
    """Pack a batch of model predictor index sets into uint64 word arrays.

    Parameters
    ----------
    idx : np.ndarray of shape (B, k), integer dtype
        Predictor indices for B models of size k.
    p : int, default=128
        Total candidate predictors (1 <= p <= 128).

    Returns
    -------
    (mask_lo, mask_hi) : Tuple[np.ndarray, np.ndarray]
        Two 1D arrays of shape (B,), dtype uint64.
    """
    idx_arr = np.asarray(idx, dtype=np.int64)
    if idx_arr.ndim != 2:
        raise ValueError(f"idx must be a 2D array of shape (B, k), got shape {idx_arr.shape}")

    B, k = idx_arr.shape
    if p < 1 or p > MAX_P:
        raise ValueError(f"p must be in [1, {MAX_P}], got {p}")

    if B == 0 or k == 0:
        return np.zeros(B, dtype=np.uint64), np.zeros(B, dtype=np.uint64)

    if (idx_arr < 0).any() or (idx_arr >= p).any():
        raise ValueError(f"Predictor indices in idx exceed bounds [0, {p})")

    mask_lo = np.empty(B, dtype=np.uint64)
    mask_hi = np.empty(B, dtype=np.uint64)

    if _HAS_NUMBA:
        _pack_indices_numba(np.ascontiguousarray(idx_arr), mask_lo, mask_hi)
    else:  # pragma: no cover
        # Pure NumPy fallback
        is_lo = idx_arr < 64
        is_hi = ~is_lo

        shifts_lo = np.where(is_lo, idx_arr, 0).astype(np.uint64)
        bits_lo = np.where(is_lo, np.uint64(1) << shifts_lo, np.uint64(0))
        mask_lo = np.bitwise_or.reduce(bits_lo, axis=1, dtype=np.uint64)

        shifts_hi = np.where(is_hi, idx_arr - 64, 0).astype(np.uint64)
        bits_hi = np.where(is_hi, np.uint64(1) << shifts_hi, np.uint64(0))
        mask_hi = np.bitwise_or.reduce(bits_hi, axis=1, dtype=np.uint64)

    return mask_lo, mask_hi


def unpack_mask_batch(
    mask_lo: np.ndarray,
    mask_hi: np.ndarray,
    p: int = MAX_P,
    k: Optional[int] = None,
) -> np.ndarray:
    """Unpack a batch of uniform-size 128-bit models into an index array of shape (B, k).

    Parameters
    ----------
    mask_lo : np.ndarray of shape (B,), dtype uint64
    mask_hi : np.ndarray of shape (B,), dtype uint64
    p : int, default=128
        Total candidate predictors (1 <= p <= 128).
    k : int, optional
        Model size. If None, verified to be uniform across all models in batch.

    Returns
    -------
    idx : np.ndarray of shape (B, k), dtype int64
        Sorted predictor indices for each model.
    """
    lo = np.ascontiguousarray(mask_lo, dtype=np.uint64)
    hi = np.ascontiguousarray(mask_hi, dtype=np.uint64)

    if lo.ndim != 1 or hi.ndim != 1 or lo.shape[0] != hi.shape[0]:
        raise ValueError(f"Inputs must be 1D arrays of equal length, got {lo.shape} and {hi.shape}")

    B = lo.shape[0]
    if B == 0:
        return np.empty((0, k if k is not None else 0), dtype=np.int64)

    if k is None:
        counts = popcount_batch(lo, hi)
        k_val = int(counts[0])
        if not np.all(counts == k_val):
            raise ValueError(
                "unpack_mask_batch requires all models in batch to have the same model size k. "
                f"Found varying sizes: min={counts.min()}, max={counts.max()}"
            )
        k = k_val

    if k == 0:
        return np.empty((B, 0), dtype=np.int64)

    out = np.empty((B, k), dtype=np.int64)

    if _HAS_NUMBA:
        _unpack_mask_numba(lo, hi, p, k, out)
    else:  # pragma: no cover
        for i in range(B):
            unp = unpack_mask(int(lo[i]), int(hi[i]), p)
            out[i, :len(unp)] = unp

    return out


def batch_get_parents(
    mask_lo: np.ndarray,
    mask_hi: np.ndarray,
    p: int = MAX_P,
    k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate all immediate parent models for a batch of models of uniform size k.

    Parameters
    ----------
    mask_lo : np.ndarray of shape (B,), dtype uint64
    mask_hi : np.ndarray of shape (B,), dtype uint64
    p : int, default=128
        Total candidate predictors.
    k : int, optional
        Model size. If None, verified to be uniform across the batch.

    Returns
    -------
    (parents_lo, parents_hi) : Tuple[np.ndarray, np.ndarray]
        Arrays of shape (B, k), dtype uint64.
        Row i contains the k parents of model i.
    """
    lo = np.ascontiguousarray(mask_lo, dtype=np.uint64)
    hi = np.ascontiguousarray(mask_hi, dtype=np.uint64)

    if lo.ndim != 1 or hi.ndim != 1 or lo.shape[0] != hi.shape[0]:
        raise ValueError(f"Inputs must be 1D arrays of equal length, got {lo.shape} and {hi.shape}")

    B = lo.shape[0]
    if k is None:
        if B == 0:
            k = 0
        else:
            counts = popcount_batch(lo, hi)
            k_val = int(counts[0])
            if not np.all(counts == k_val):
                raise ValueError("batch_get_parents requires uniform model size k across the batch")
            k = k_val

    if B == 0 or k == 0:
        return np.empty((B, 0), dtype=np.uint64), np.empty((B, 0), dtype=np.uint64)

    out_lo = np.empty((B, k), dtype=np.uint64)
    out_hi = np.empty((B, k), dtype=np.uint64)

    if _HAS_NUMBA:
        _batch_get_parents_numba(lo, hi, p, k, out_lo, out_hi)
    else:  # pragma: no cover
        for i in range(B):
            pars = get_parents(int(lo[i]), int(hi[i]), p)
            for j, (plo, phi) in enumerate(pars):
                out_lo[i, j] = plo
                out_hi[i, j] = phi

    return out_lo, out_hi


def batch_get_children(
    mask_lo: np.ndarray,
    mask_hi: np.ndarray,
    p: int,
    k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate all immediate child models for a batch of models of uniform size k.

    Parameters
    ----------
    mask_lo : np.ndarray of shape (B,), dtype uint64
    mask_hi : np.ndarray of shape (B,), dtype uint64
    p : int
        Total candidate predictors (1 <= p <= 128).
    k : int, optional
        Model size. If None, verified to be uniform across the batch.

    Returns
    -------
    (children_lo, children_hi) : Tuple[np.ndarray, np.ndarray]
        Arrays of shape (B, p - k), dtype uint64.
        Row i contains the (p - k) children of model i.
    """
    lo = np.ascontiguousarray(mask_lo, dtype=np.uint64)
    hi = np.ascontiguousarray(mask_hi, dtype=np.uint64)

    if lo.ndim != 1 or hi.ndim != 1 or lo.shape[0] != hi.shape[0]:
        raise ValueError(f"Inputs must be 1D arrays of equal length, got {lo.shape} and {hi.shape}")

    B = lo.shape[0]
    if k is None:
        if B == 0:
            k = 0
        else:
            counts = popcount_batch(lo, hi)
            k_val = int(counts[0])
            if not np.all(counts == k_val):
                raise ValueError("batch_get_children requires uniform model size k across the batch")
            k = k_val

    num_children = max(0, p - k)
    if B == 0 or num_children == 0:
        return np.empty((B, 0), dtype=np.uint64), np.empty((B, 0), dtype=np.uint64)

    out_lo = np.empty((B, num_children), dtype=np.uint64)
    out_hi = np.empty((B, num_children), dtype=np.uint64)

    if _HAS_NUMBA:
        _batch_get_children_numba(lo, hi, p, k, out_lo, out_hi)
    else:  # pragma: no cover
        for i in range(B):
            children = get_children(int(lo[i]), int(hi[i]), p)
            for j, (clo, chi) in enumerate(children):
                out_lo[i, j] = clo
                out_hi[i, j] = chi

    return out_lo, out_hi


# ==============================================================================
# STRUCTURED ARRAY UTILITIES
# ==============================================================================

def to_structured_array(
    mask_lo: Union[np.ndarray, Sequence[int]],
    mask_hi: Union[np.ndarray, Sequence[int]],
) -> np.ndarray:
    """Combine mask_lo and mask_hi arrays into a single structured numpy array with DTYPE_MODEL_128."""
    lo = np.asarray(mask_lo, dtype=np.uint64)
    hi = np.asarray(mask_hi, dtype=np.uint64)
    if lo.shape != hi.shape:
        raise ValueError(f"Shape mismatch: {lo.shape} != {hi.shape}")

    struct_arr = np.empty(lo.shape, dtype=DTYPE_MODEL_128)
    struct_arr["lo"] = lo
    struct_arr["hi"] = hi
    return struct_arr


def from_structured_array(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (mask_lo, mask_hi) arrays from a structured numpy array with DTYPE_MODEL_128."""
    if arr.dtype != DTYPE_MODEL_128:
        raise TypeError(f"Expected array with dtype {DTYPE_MODEL_128}, got {arr.dtype}")
    return arr["lo"], arr["hi"]
