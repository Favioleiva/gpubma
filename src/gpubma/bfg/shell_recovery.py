"""Validated post-BFG stratified random shell recovery.

Architecture:
    Targeted BFG discovery (focused on high-evidence modes and champions)
    +
    Representative uniform random shell sampling (for reticular score distributions).

Critical rule:
    Targeted BFG discoveries must NOT be pooled as if they were random draws when
    estimating reticular score distributions.

    For shell k with population N_k = C(p, k):
      - Discovered models (D_k) receive weight 1 / N_k each.
      - Random remainder models (n_k) receive weight (N_k - D_k) / (N_k * n_k) each.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from gpubma.gpu.enumerator import binomial_table, unrank_combinations


def shell_population_size(p: int, k: int) -> int:
    """Return the total number of models in shell k of a p-variable space: C(p, k)."""
    return math.comb(p, k)


def sample_size(
    p: int,
    k: int,
    discovered: int = 0,
    cap: int = 100_000,
    rule_fraction: float = 0.01,
) -> int:
    """Compute required random sample size for shell k under rule fraction and cap.

    Default rule is 1% of shell population, capped at cap (default 100,000),
    bounded by the undiscovered remainder N_k - D_k.
    """
    N = math.comb(p, k)
    if not 0 <= discovered <= N or cap <= 0:
        raise ValueError(f"Invalid population (N={N}, discovered={discovered}) or cap={cap}")
    rule_target = (N + int(1.0 / rule_fraction) - 1) // int(1.0 / rule_fraction) if rule_fraction > 0 else 0
    return min(rule_target, cap, N - discovered)


def rank_indices(indices: Sequence[int]) -> int:
    """Compute co-lexicographic rank of ascending variable indices."""
    return sum(math.comb(int(v), j + 1) for j, v in enumerate(indices))


def rank_mask(mask: int, p: int) -> int:
    """Convert a model bitmask to its co-lexicographic rank within its shell."""
    if mask < 0 or mask >= (1 << p):
        raise ValueError(f"Invalid model mask {mask} for p={p}")
    return rank_indices([j for j in range(p) if (mask >> j) & 1])


def unrank_combination(rank: int, p: int, k: int) -> List[int]:
    """Unrank a co-lexicographic rank into ascending variable indices.

    Supports arbitrary precision integers (safe for p=60, 90, 100).
    """
    N = math.comb(p, k)
    if not 0 <= rank < N:
        raise ValueError(f"Invalid rank {rank} for C({p},{k})={N}")
    out = [0] * k
    upper = p - 1
    for j in range(k, 0, -1):
        lo = j - 1
        hi = upper
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if math.comb(mid, j) <= rank:
                lo = mid
            else:
                hi = mid - 1
        out[j - 1] = lo
        rank -= math.comb(lo, j)
        upper = lo - 1
    return out


def mask_from_rank(rank: int, p: int, k: int) -> int:
    """Convert a co-lexicographic rank in shell k to an integer bitmask."""
    indices = unrank_combination(rank, p, k)
    mask = 0
    for idx in indices:
        mask |= (1 << idx)
    return mask


class RemainderSampler:
    """Uniform without-replacement sampler for the undiscovered remainder of shell k.

    Accepts first distinct non-excluded values from uniform pseudo-random ranks.
    Deterministic nested sampling: repeated calls to extend(target) extend the
    sample prefix without modifying previously chosen ranks.
    """

    def __init__(
        self,
        p: int,
        k: int,
        excluded_ranks: Iterable[int] = (),
        seed: int = 2026091401,
    ):
        self.p = p
        self.k = k
        self.N = math.comb(p, k)
        self.excluded = set(map(int, excluded_ranks))
        self.seen = set(self.excluded)
        if any(r < 0 or r >= self.N for r in self.excluded):
            raise ValueError(f"Excluded rank out of range [0, {self.N})")

        # Deterministic per-shell seed derivation
        self.seed = int.from_bytes(
            hashlib.sha256(f"{seed}:shell:{p}:{k}".encode()).digest()[:16],
            "little",
        )
        self.large = self.N > np.iinfo(np.int64).max
        self.rng = (
            random.Random(self.seed)
            if self.large
            else np.random.Generator(np.random.PCG64(self.seed))
        )
        self.selected: List[int] = []

    def extend(self, target: int) -> np.ndarray:
        """Extend sample to target count, returning only the newly added ranks."""
        max_available = self.N - len(self.excluded)
        if target < len(self.selected) or target > max_available:
            raise ValueError(
                f"Invalid target {target}; current={len(self.selected)}, max={max_available}"
            )
        needed = target - len(self.selected)
        if needed == 0:
            return np.array([], dtype=object if self.large else np.int64)

        if self.large:
            new: List[int] = []
            while len(new) < needed:
                r = self.rng.randrange(self.N)
                if r not in self.seen:
                    self.seen.add(r)
                    new.append(r)
        else:
            chunks = []
            remain = needed
            forbidden = np.array(sorted(self.seen), dtype=np.int64)
            while remain > 0:
                batch_size = min(max(1024, int(remain * 1.03) + 32), 1_100_000)
                proposal = self.rng.integers(0, self.N, size=batch_size, dtype=np.int64)
                # Stable first occurrence; never reorder accepted ranks by value
                _, first = np.unique(proposal, return_index=True)
                candidate = proposal[np.sort(first)]
                candidate = candidate[~np.isin(candidate, forbidden, assume_unique=True, kind="sort")][:remain]
                chunks.append(candidate)
                remain -= len(candidate)
                forbidden = np.union1d(forbidden, candidate)
            new = np.concatenate(chunks).tolist() if chunks else []
            self.seen.update(new)

        self.selected.extend(new)
        return np.array(new, dtype=object if self.large else np.int64)


class RecoveryScorer:
    """High-throughput model scorer for stratified shell recovery.

    Evaluates models without adding them to BFGScorer's internal search cache,
    preserving search discovery state and avoiding dictionary overhead.
    """

    def __init__(self, scorer: Any, batch_size: int = 65536):
        self.scorer = scorer
        self.batch_size = batch_size
        self.device = scorer.device
        self.p = scorer.p
        self.backend = getattr(scorer, "backend", "gpu" if scorer.device.type == "cuda" else "cpu")

        if self.backend == "gpu":
            self.table = torch.from_numpy(binomial_table(self.p)).to(self.device)

    def score_ranks(self, ranks: Sequence[int], k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Score a sequence of co-lexicographic ranks for shell k.

        Returns
        -------
        scores : np.ndarray
            Canonical log scores (float64).
        masks : np.ndarray
            Model bitmasks (int64).
        """
        if len(ranks) == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

        s = self.scorer
        scores_out: List[np.ndarray] = []
        masks_out: List[np.ndarray] = []

        if k == 0:
            one_minus_r2 = max(s.tss / s.tss_norm, s.tiny)
            s0 = float(
                0.5 * (s.df - s.k_always) * s.log1pg
                - 0.5 * s.df * math.log1p(s.g * one_minus_r2)
                + s.lp_size_cpu[0]
            )
            n = len(ranks)
            return np.full(n, s0, dtype=np.float64), np.zeros(n, dtype=np.int64)

        if self.backend == "gpu":
            for start in range(0, len(ranks), self.batch_size):
                chunk = np.ascontiguousarray(ranks[start : start + self.batch_size], dtype=np.int64)
                r = torch.from_numpy(chunk).to(self.device)
                idx = unrank_combinations(r, k, self.table, torch)
                mask = (torch.ones_like(idx) << idx).sum(dim=1)

                Z = s.Zxx[idx.unsqueeze(2), idx.unsqueeze(1)]
                b = s.Zxy[idx].unsqueeze(-1)
                L = torch.linalg.cholesky(Z)
                u = torch.linalg.solve_triangular(L, b, upper=False)
                ess = (u.squeeze(-1) ** 2).sum(dim=1)
                one_minus_r2 = torch.clamp((s.tss - ess) / s.tss_norm, min=s.tiny)

                scores = (
                    0.5 * (s.df - k - s.k_always) * s.log1pg
                    - 0.5 * s.df * torch.log1p(s.g * one_minus_r2)
                    + s.lp_size[k]
                )
                ss = scores.cpu().numpy()
                mm = mask.cpu().numpy()
                if not np.isfinite(ss).all():
                    raise ValueError("Non-finite canonical score encountered in recovery")
                scores_out.append(ss)
                masks_out.append(mm)
        else:
            from scipy.linalg import cho_factor, cho_solve

            for start in range(0, len(ranks), self.batch_size):
                chunk = ranks[start : start + self.batch_size]
                chunk_scores = []
                chunk_masks = []
                for r in chunk:
                    idx = np.array(unrank_combination(int(r), self.p, k), dtype=np.intp)
                    mask = int(np.sum(1 << idx))
                    Z_sub = s.Zxx_cpu[np.ix_(idx, idx)]
                    b_sub = s.Zxy_cpu[idx]
                    c, low = cho_factor(Z_sub, lower=True, check_finite=False)
                    w = cho_solve((c, low), b_sub, check_finite=False)
                    ess = float(b_sub @ w)
                    one_minus_r2 = max((s.tss - ess) / s.tss_norm, s.tiny)
                    sc = (
                        0.5 * (s.df - k - s.k_always) * s.log1pg
                        - 0.5 * s.df * math.log1p(s.g * one_minus_r2)
                        + s.lp_size_cpu[k]
                    )
                    chunk_scores.append(float(sc))
                    chunk_masks.append(mask)
                scores_out.append(np.array(chunk_scores, dtype=np.float64))
                masks_out.append(np.array(chunk_masks, dtype=np.int64))

        return np.concatenate(scores_out), np.concatenate(masks_out)


def reconstruct_shell_mixture(
    discovered_scores: Sequence[float],
    sample_scores: Sequence[float],
    N_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct representative full-shell score distribution from BFG discoveries and random remainder.

    Targeted BFG discoveries are NOT pooled equally with random draws.
    Discovered models receive weight 1 / N_k each.
    Random remainder models receive weight (N_k - D_k) / (N_k * n_k) each.

    Returns
    -------
    values : np.ndarray
        Concatenated scores [discovered_scores, sample_scores].
    weights : np.ndarray
        Representative shell weights summing to 1.0.
    """
    disc = np.asarray(discovered_scores, dtype=np.float64)
    samp = np.asarray(sample_scores, dtype=np.float64)
    D = len(disc)
    n = len(samp)
    M = N_k - D
    if M < 0:
        raise ValueError(f"Discovered count D={D} exceeds shell population N_k={N_k}")
    if n > M:
        raise ValueError(f"Sample count n={n} exceeds undiscovered remainder M={M}")

    if D == N_k:
        weights = np.full(D, 1.0 / N_k, dtype=np.float64)
        return disc, weights
    if n == 0:
        if D == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        weights = np.full(D, 1.0 / D, dtype=np.float64)
        return disc, weights

    values = np.concatenate([disc, samp])
    w_disc = np.full(D, 1.0 / N_k, dtype=np.float64)
    w_samp = np.full(n, float(M) / (float(N_k) * float(n)), dtype=np.float64)
    weights = np.concatenate([w_disc, w_samp])
    return values, weights


def weighted_summary(
    values: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Compute weighted summary statistics: sorted values, weights, mean, variance, and quantiles."""
    v_arr = np.asarray(values, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    order = np.argsort(v_arr, kind="stable")
    v = v_arr[order]
    w = w_arr[order]
    mu = float(np.dot(v, w))
    variance = float(np.dot((v - mu) ** 2, w))
    cdf = np.cumsum(w)
    if len(cdf) > 0:
        cdf[-1] = 1.0
        qs = v[np.searchsorted(cdf, [0.05, 0.25, 0.5, 0.75, 0.95], side="left")]
    else:
        qs = np.array([], dtype=np.float64)
    return v, w, mu, variance, qs


def _weighted_distances_python(
    exact: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float]:
    """Exact two discrete-CDF Kolmogorov-Smirnov and Wasserstein-1 distances."""
    i = 0
    j = 0
    n = len(exact)
    m = len(values)
    fy = 0.0
    previous = min(exact[0], values[0])
    ks = 0.0
    w1 = 0.0
    while i < n or j < m:
        x = exact[i] if i < n else float("inf")
        y = values[j] if j < m else float("inf")
        z = min(x, y)
        diff = i / n - fy
        w1 += abs(diff) * (z - previous)
        ks = max(ks, abs(diff))
        while i < n and exact[i] == z:
            i += 1
        while j < m and values[j] == z:
            fy += weights[j]
            j += 1
        ks = max(ks, abs(i / n - fy))
        previous = z
    return float(ks), float(w1)


try:
    from numba import njit
    _weighted_distances_jit = njit(cache=False)(_weighted_distances_python)

    def weighted_distances(
        exact: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute exact KS distance and Wasserstein-1 distance between exact reference and weighted sample."""
        return _weighted_distances_jit(exact, values, weights)

except ImportError:
    def weighted_distances(
        exact: np.ndarray,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute exact KS distance and Wasserstein-1 distance between exact reference and weighted sample."""
        return _weighted_distances_python(exact, values, weights)
