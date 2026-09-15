"""Deterministic counter-based PRNG streams for BFG128 shell sampling.

This module provides an independently reproducible, stream-based pseudo-random number generator
where each model draw r is a pure mathematical function of:
    (global_seed, p, k, r)

Properties:
1. Thread-invariant: Running with 1 thread, 4 threads, or 20 threads produces the exact same
   sample sequence bit-for-bit.
2. Chunk-invariant: Drawing 1,000,000 models in a single batch vs. 10 x 100,000 chunks produces
   the exact same sample sequence bit-for-bit.
3. Unbiased: Uses Lemire's exact rejection-bounded integer generation over uint64 words,
   ensuring mathematically uniform subset selection with zero modulo bias.
"""

from typing import Tuple
import numpy as np

try:
    import numba
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


# ==============================================================================
# NUMBA JIT-COMPILED CORE KERNELS
# ==============================================================================

if _HAS_NUMBA:

    @numba.njit(inline="always")
    def splitmix64(state: np.ndarray) -> np.uint64:
        """Advance SplitMix64 state and return 64-bit pseudo-random unsigned integer."""
        state[0] = state[0] + np.uint64(0x9E3779B97F4A7C15)
        z = state[0]
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))

    @numba.njit(inline="always")
    def hash_sample(global_seed: int, p: int, k: int, r: int) -> np.uint64:
        """Compute an avalanche-mixed 64-bit seed from (global_seed, p, k, r).

        Uses distinct high-entropy multipliers for each coordinate to prevent cross-stream collisions.
        """
        z = np.uint64(global_seed) + np.uint64(r) * np.uint64(0x9E3779B97F4A7C15)
        z ^= (np.uint64(p) * np.uint64(0x517CC1B727220A95))
        z ^= (np.uint64(k) * np.uint64(0x94D049BB133111EB))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))

    @numba.njit(inline="always")
    def rand_bounded(state: np.ndarray, s: int) -> int:
        """Generate an unbiased uniform integer in [0, s) from 64-bit PRNG word.

        Rejection probability is s / 2^64 < 128 / 2^64 ~ 6.9e-18, eliminating modulo bias.
        """
        u_s = np.uint64(s)
        limit = np.uint64(0xFFFFFFFFFFFFFFFF) - (np.uint64(0xFFFFFFFFFFFFFFFF) % u_s)
        u = splitmix64(state)
        while u >= limit:
            u = splitmix64(state)
        return int(u % u_s)

    @numba.njit(parallel=True, fastmath=True)
    def draw_raw_candidates_numba(
        p: int,
        k: int,
        B: int,
        start_r: int,
        global_seed: int,
        out_idx: np.ndarray,
        out_lo: np.ndarray,
        out_hi: np.ndarray,
    ) -> None:
        """Generate B raw uniform k-subsets from candidate indices [start_r, start_r + B - 1].

        Each iteration i is completely independent and evaluated using stack memory.
        """
        for i in numba.prange(B):
            r = start_r + i
            state = np.empty(1, dtype=np.uint64)
            state[0] = hash_sample(global_seed, p, k, r)

            # 128-element stack buffer for partial Fisher-Yates shuffle
            pool = np.empty(128, dtype=np.int64)
            for j in range(p):
                pool[j] = j

            # Partial Fisher-Yates: exactly k swaps
            for t in range(k):
                offset = rand_bounded(state, p - t)
                j = t + offset
                tmp = pool[t]
                pool[t] = pool[j]
                pool[j] = tmp

            # In-place insertion sort of the k selected elements
            for a in range(1, k):
                key = pool[a]
                b = a - 1
                while b >= 0 and pool[b] > key:
                    pool[b + 1] = pool[b]
                    b -= 1
                pool[b + 1] = key

            # Direct word packing into uint64 mask_lo and mask_hi
            lo = np.uint64(0)
            hi = np.uint64(0)
            for m in range(k):
                val = pool[m]
                out_idx[i, m] = val
                if val < 64:
                    lo |= (np.uint64(1) << np.uint64(val))
                else:
                    hi |= (np.uint64(1) << np.uint64(val - 64))

            out_lo[i] = lo
            out_hi[i] = hi


# ==============================================================================
# PURE PYTHON / NUMPY FALLBACK (If Numba unavailable)
# ==============================================================================

def draw_raw_candidates_fallback(
    p: int,
    k: int,
    B: int,
    start_r: int,
    global_seed: int,
    out_idx: np.ndarray,
    out_lo: np.ndarray,
    out_hi: np.ndarray,
) -> None:  # pragma: no cover
    """Pure Python fallback for draw_raw_candidates when Numba is not installed."""
    UINT64_MAX = 0xFFFFFFFFFFFFFFFF

    def _hash(s, p, k, r):
        z = (int(s) + int(r) * 0x9E3779B97F4A7C15) & UINT64_MAX
        z ^= (int(p) * 0x517CC1B727220A95) & UINT64_MAX
        z ^= (int(k) * 0x94D049BB133111EB) & UINT64_MAX
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MAX
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & UINT64_MAX
        return (z ^ (z >> 31)) & UINT64_MAX

    def _splitmix(state):
        state[0] = (state[0] + 0x9E3779B97F4A7C15) & UINT64_MAX
        z = state[0]
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MAX
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & UINT64_MAX
        return (z ^ (z >> 31)) & UINT64_MAX

    def _rand_bounded(state, s):
        limit = UINT64_MAX - (UINT64_MAX % s)
        u = _splitmix(state)
        while u >= limit:
            u = _splitmix(state)
        return u % s

    for i in range(B):
        r = start_r + i
        state = [_hash(global_seed, p, k, r)]
        pool = list(range(p))
        for t in range(k):
            offset = _rand_bounded(state, p - t)
            j = t + offset
            pool[t], pool[j] = pool[j], pool[t]
        selected = sorted(pool[:k])
        lo = 0
        hi = 0
        for m, val in enumerate(selected):
            out_idx[i, m] = val
            if val < 64:
                lo |= (1 << val)
            else:
                hi |= (1 << (val - 64))
        out_lo[i] = np.uint64(lo)
        out_hi[i] = np.uint64(hi)


def draw_raw_candidates(
    p: int,
    k: int,
    B: int,
    start_r: int,
    global_seed: int,
    out_idx: np.ndarray,
    out_lo: np.ndarray,
    out_hi: np.ndarray,
) -> None:
    """Dispatches raw candidate drawing to Numba parallel kernel or fallback."""
    if _HAS_NUMBA:
        draw_raw_candidates_numba(p, k, B, start_r, global_seed, out_idx, out_lo, out_hi)
    else:  # pragma: no cover
        draw_raw_candidates_fallback(p, k, B, start_r, global_seed, out_idx, out_lo, out_hi)
