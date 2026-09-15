"""Direct Uniform Wide Shell Sampler for BFG128 (61 <= p <= 128).

This module implements a native, direct, uniform k-subset sampler for large predictor spaces.

Key Features:
1. Exact Uniformity: Samples without replacement from the shell { M : |M| = k } via partial
   Fisher-Yates shuffle with zero combinatorial rank/unrank overhead in the hot path.
2. Thread-Invariant Determinism: Counter-based PRNG streams ensure the generated candidate
   sequence is 100% bit-for-bit identical regardless of CPU thread count (1 vs. 20 threads).
3. Chunk-Invariant Resumability: Sample stream semantics are indexed by raw_draw_index r.
   Sampling 1,000,000 models in one call equals calling ten consecutive 100,000 chunks.
4. Exact Duplicate Detection & Exclusion: Excludes previously discovered models and avoids
   intra-sample duplicates, guaranteeing exactly n_samples unique models.
5. Small Census Shell Support: Shells with C(p, k) <= census_threshold are enumerated
   deterministically in canonical lexicographic order through the same unified interface.
"""

from dataclasses import dataclass
import itertools
import math
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from gpubma.bfg.bfg128.bitops import pack_indices_batch, to_structured_array
from gpubma.bfg.bfg128.rng import draw_raw_candidates
from gpubma.bfg.bfg128.types import (
    DTYPE_MODEL_128,
    MAX_P,
    Mask128,
    validate_mask128,
)

try:
    import numba
    from numba.typed import Set as NumbaSet
    from numba import types as numba_types

    _HAS_NUMBA = True
    _MODEL_TUPLE_TYPE = numba_types.Tuple((numba_types.uint64, numba_types.uint64))

    @numba.njit(fastmath=True)
    def _filter_and_copy_numba(
        cand_idx: np.ndarray,
        cand_lo: np.ndarray,
        cand_hi: np.ndarray,
        C: int,
        curr_r: int,
        out_idx: np.ndarray,
        out_lo: np.ndarray,
        out_hi: np.ndarray,
        n_accepted: int,
        n_samples: int,
        seen_set: NumbaSet,
    ) -> Tuple[int, int]:
        k_dim = cand_idx.shape[1]
        for i in range(C):
            key = (cand_lo[i], cand_hi[i])
            if key not in seen_set:
                seen_set.add(key)
                for m in range(k_dim):
                    out_idx[n_accepted, m] = cand_idx[i, m]
                out_lo[n_accepted] = cand_lo[i]
                out_hi[n_accepted] = cand_hi[i]
                n_accepted += 1
                if n_accepted == n_samples:
                    return n_accepted, curr_r + i + 1
        return n_accepted, curr_r + C

except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


@dataclass(frozen=True)
class SampledBatch128:
    """Immutable container holding a batch of sampled 128-bit models."""

    idx: np.ndarray  # shape (B, k), dtype int64
    mask_lo: np.ndarray  # shape (B,), dtype uint64
    mask_hi: np.ndarray  # shape (B,), dtype uint64
    raw_draws_consumed: int
    next_raw_index: int
    is_census: bool

    def __len__(self) -> int:
        return int(self.mask_lo.shape[0])

    def to_structured_array(self) -> np.ndarray:
        """Convert mask_lo and mask_hi to a structured NumPy array with DTYPE_MODEL_128."""
        return to_structured_array(self.mask_lo, self.mask_hi)


class DirectUniformShellSampler128:
    """Direct uniform k-subset sampler for wide model spaces (61 <= p <= 128).

    Abstracts over deterministic census enumeration (when C(p, k) <= census_threshold)
    and stream-based direct uniform pseudo-random sampling with exact exclusion/deduplication.
    """

    def __init__(
        self,
        p: int,
        k: int,
        seed: int = 0,
        excluded_models: Optional[Iterable[Mask128]] = None,
        census_threshold: Optional[int] = None,
    ) -> None:
        """Initialize the wide shell sampler.

        Parameters
        ----------
        p : int
            Total number of candidate predictors (1 <= p <= 128, canonically 61 <= p <= 128).
        k : int
            Model size (0 <= k <= p).
        seed : int, default=0
            Global pseudo-random seed.
        excluded_models : Optional[Iterable[Mask128]]
            Set or collection of (mask_lo, mask_hi) models to exclude from sampling.
        census_threshold : Optional[int]
            Threshold on C(p, k). If C(p, k) <= census_threshold, deterministic
            lexicographic census enumeration is used instead of pseudo-random sampling.
        """
        if p < 1 or p > MAX_P:
            raise ValueError(f"Predictor dimension p must be in [1, {MAX_P}], got {p}")
        if k < 0 or k > p:
            raise ValueError(f"Model size k must be in [0, p={p}], got {k}")

        self.p: int = p
        self.k: int = k
        self.seed: int = int(seed) & 0xFFFFFFFFFFFFFFFF
        self.total_combinations: int = math.comb(p, k)
        self.census_threshold: Optional[int] = census_threshold

        # Normalize and validate exclusion set
        self.excluded_models: Set[Mask128] = set()
        if excluded_models is not None:
            for item in excluded_models:
                m_lo, m_hi = int(item[0]), int(item[1])
                validate_mask128(m_lo, m_hi, p=self.p)
                self.excluded_models.add((m_lo, m_hi))

        # Check remainder limit
        self.total_available: int = self.total_combinations - len(self.excluded_models)
        if self.total_available < 0:
            raise ValueError(
                f"Number of excluded models ({len(self.excluded_models)}) exceeds total "
                f"combinations C({p}, {k}) = {self.total_combinations}"
            )

        # Decide census vs random mode
        self.is_census: bool = (
            census_threshold is not None and self.total_combinations <= census_threshold
        )

    def sample(
        self,
        n_samples: int,
        start_raw_index: int = 0,
        chunk_size: int = 32768,
    ) -> SampledBatch128:
        """Draw n_samples unique, non-excluded models from shell k.

        Parameters
        ----------
        n_samples : int
            Number of accepted unique models required.
        start_raw_index : int, default=0
            Logical starting index in the deterministic raw draw sequence.
        chunk_size : int, default=32768
            Batch size for parallel candidate generation.

        Returns
        -------
        SampledBatch128
            Tuple containing idx, mask_lo, mask_hi, raw_draws_consumed, next_raw_index, is_census.

        Raises
        ------
        ValueError
            If n_samples exceeds total_available models in the shell remainder.
        """
        if n_samples < 0:
            raise ValueError(f"n_samples must be non-negative, got {n_samples}")

        if n_samples > self.total_available:
            raise ValueError(
                f"Requested {n_samples:,} samples exceeds available shell remainder "
                f"{self.total_available:,} (C({self.p},{self.k})={self.total_combinations:,}, "
                f"excluded={len(self.excluded_models):,})"
            )

        if n_samples == 0:
            return SampledBatch128(
                idx=np.empty((0, self.k), dtype=np.int64),
                mask_lo=np.empty(0, dtype=np.uint64),
                mask_hi=np.empty(0, dtype=np.uint64),
                raw_draws_consumed=0,
                next_raw_index=start_raw_index,
                is_census=self.is_census,
            )

        # Deterministic census dispatch
        if self.is_census:
            return self.enumerate_census(start_index=start_raw_index, count=n_samples)

        # Stream-based pseudo-random sampling with exact rejection
        out_idx = np.empty((n_samples, self.k), dtype=np.int64)
        out_lo = np.empty(n_samples, dtype=np.uint64)
        out_hi = np.empty(n_samples, dtype=np.uint64)
        n_accepted = 0
        curr_r = int(start_raw_index)
        cand_idx_buf = np.empty((chunk_size, self.k), dtype=np.int64)
        cand_lo_buf = np.empty(chunk_size, dtype=np.uint64)
        cand_hi_buf = np.empty(chunk_size, dtype=np.uint64)

        if _HAS_NUMBA:
            seen_numba = NumbaSet.empty(_MODEL_TUPLE_TYPE)
            for m_lo, m_hi in self.excluded_models:
                seen_numba.add((np.uint64(m_lo), np.uint64(m_hi)))

            while n_accepted < n_samples:
                needed = n_samples - n_accepted
                C = min(needed, chunk_size)

                draw_raw_candidates(
                    self.p,
                    self.k,
                    C,
                    curr_r,
                    self.seed,
                    cand_idx_buf[:C],
                    cand_lo_buf[:C],
                    cand_hi_buf[:C],
                )

                n_accepted, next_r = _filter_and_copy_numba(
                    cand_idx_buf[:C],
                    cand_lo_buf[:C],
                    cand_hi_buf[:C],
                    C,
                    curr_r,
                    out_idx,
                    out_lo,
                    out_hi,
                    n_accepted,
                    n_samples,
                    seen_numba,
                )
                curr_r = next_r

                if n_accepted == n_samples:
                    draws_consumed = next_r - start_raw_index
                    return SampledBatch128(
                        idx=out_idx,
                        mask_lo=out_lo,
                        mask_hi=out_hi,
                        raw_draws_consumed=draws_consumed,
                        next_raw_index=next_r,
                        is_census=False,
                    )
        else:
            seen = set(self.excluded_models)
            while n_accepted < n_samples:
                needed = n_samples - n_accepted
                C = min(needed, chunk_size)

                draw_raw_candidates(
                    self.p,
                    self.k,
                    C,
                    curr_r,
                    self.seed,
                    cand_idx_buf[:C],
                    cand_lo_buf[:C],
                    cand_hi_buf[:C],
                )

                for i in range(C):
                    r_item = curr_r + i
                    key = (int(cand_lo_buf[i]), int(cand_hi_buf[i]))
                    if key not in seen:
                        seen.add(key)
                        out_idx[n_accepted] = cand_idx_buf[i]
                        out_lo[n_accepted] = cand_lo_buf[i]
                        out_hi[n_accepted] = cand_hi_buf[i]
                        n_accepted += 1

                        if n_accepted == n_samples:
                            next_r = r_item + 1
                            draws_consumed = next_r - start_raw_index
                            return SampledBatch128(
                                idx=out_idx,
                                mask_lo=out_lo,
                                mask_hi=out_hi,
                                raw_draws_consumed=draws_consumed,
                                next_raw_index=next_r,
                                is_census=False,
                            )

                curr_r += C

        # Fallback if loop finishes exactly at chunk boundary
        draws_consumed = curr_r - start_raw_index
        return SampledBatch128(
            idx=out_idx[:n_accepted],
            mask_lo=out_lo[:n_accepted],
            mask_hi=out_hi[:n_accepted],
            raw_draws_consumed=draws_consumed,
            next_raw_index=curr_r,
            is_census=False,
        )

    def enumerate_census(
        self,
        start_index: int = 0,
        count: Optional[int] = None,
    ) -> SampledBatch128:
        """Enumerate combinations in canonical lexicographic order.

        Parameters
        ----------
        start_index : int, default=0
            Starting offset among non-excluded combinations.
        count : Optional[int]
            Number of models to return. If None, returns all remaining combinations.

        Returns
        -------
        SampledBatch128
        """
        if self.k == 0:
            all_combos = [()]
        elif self.k == self.p:
            all_combos = [tuple(range(self.p))]
        else:
            all_combos = list(itertools.combinations(range(self.p), self.k))

        # Convert to numpy index array
        combos_arr = np.array(all_combos, dtype=np.int64).reshape(-1, self.k)
        lo_arr, hi_arr = pack_indices_batch(combos_arr, p=self.p)

        # Filter excluded models
        if self.excluded_models:
            valid_indices = []
            for i in range(len(lo_arr)):
                key = (int(lo_arr[i]), int(hi_arr[i]))
                if key not in self.excluded_models:
                    valid_indices.append(i)
            combos_arr = combos_arr[valid_indices]
            lo_arr = lo_arr[valid_indices]
            hi_arr = hi_arr[valid_indices]

        # Slice by start_index and count
        end_index = len(lo_arr) if count is None else start_index + count
        sliced_idx = combos_arr[start_index:end_index]
        sliced_lo = lo_arr[start_index:end_index]
        sliced_hi = hi_arr[start_index:end_index]

        return SampledBatch128(
            idx=sliced_idx,
            mask_lo=sliced_lo,
            mask_hi=sliced_hi,
            raw_draws_consumed=0,
            next_raw_index=start_index + len(sliced_lo),
            is_census=True,
        )


def sample_shell_128(
    p: int,
    k: int,
    n_samples: int,
    seed: int = 0,
    excluded_models: Optional[Iterable[Mask128]] = None,
    start_raw_index: int = 0,
    census_threshold: Optional[int] = None,
) -> SampledBatch128:
    """Convenience functional API for sampling shell k in BFG128."""
    sampler = DirectUniformShellSampler128(
        p=p,
        k=k,
        seed=seed,
        excluded_models=excluded_models,
        census_threshold=census_threshold,
    )
    return sampler.sample(n_samples=n_samples, start_raw_index=start_raw_index)
