"""Combinatorial sampling and exact boundary wing enumeration for BFG."""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.special import logsumexp

from gpubma.bfg.scorer import BFGScorer


class ExactWingEnumerator:
    """Exhaustive enumerator for small boundary wings of the Boolean lattice.

    Used when comb(p, k) is small (e.g. k <= 3 or k >= p - 3) so that
    exact enumeration is faster and more accurate than statistical approximation.
    """

    def __init__(self, scorer: BFGScorer, p: int):
        self.scorer = scorer
        self.p = p

    def is_exact_wing(self, k: int, max_wing_size: int = 4096) -> bool:
        """Return True if lattice size comb(p, k) <= max_wing_size."""
        return math.comb(self.p, k) <= max_wing_size

    def enumerate_lattice(self, k: int) -> Tuple[List[int], List[float], float]:
        """Exhaustively enumerate and score all models in lattice k.

        Returns
        -------
        Tuple[List[int], List[float], float]
            (model_ids, scores, log_Z_k)
        """
        if k == 0:
            models = [0]
        elif k == self.p:
            models = [(1 << self.p) - 1]
        else:
            models = []
            for combo in itertools.combinations(range(self.p), k):
                mask = 0
                for bit in combo:
                    mask |= (1 << bit)
                models.append(mask)

        scores_dict = self.scorer.score_batch(models)
        scores = [scores_dict[m] for m in models]
        log_Z_k = float(logsumexp(scores))
        return models, scores, log_Z_k


class LatticeSampler:
    """Fast uniform random sampling of k-combinations without replacement."""

    def __init__(self, p: int):
        self.p = p

    def sample_combinations(
        self,
        k: int,
        n_samples: int,
        rng: np.random.Generator,
        exclude_set: Optional[Set[int]] = None,
    ) -> List[int]:
        """Draw n_samples distinct k-combinations uniformly without replacement.

        Parameters
        ----------
        k : int
            Lattice model size (number of active predictors).
        n_samples : int
            Number of distinct models to sample.
        rng : np.random.Generator
            Numpy random generator with deterministic seed.
        exclude_set : Optional[Set[int]]
            Set of model IDs that must be excluded (e.g., known elites).

        Returns
        -------
        List[int]
            List of distinct model IDs of length min(n_samples, available).
        """
        if k < 0 or k > self.p:
            return []

        total_in_lattice = math.comb(self.p, k)
        n_excluded = len(exclude_set) if exclude_set else 0
        total_available = max(0, total_in_lattice - n_excluded)

        if total_available == 0 or n_samples <= 0:
            return []

        target_count = min(n_samples, total_available)

        # Boundary cases k=0 and k=p
        if k == 0:
            return [0] if (exclude_set is None or 0 not in exclude_set) else []
        if k == self.p:
            full_mask = (1 << self.p) - 1
            return [full_mask] if (exclude_set is None or full_mask not in exclude_set) else []

        seen: Set[int] = set(exclude_set) if exclude_set is not None else set()
        samples: List[int] = []

        # If requesting most of the lattice, generate exhaustive combinations and permute
        if total_in_lattice <= 10000 and target_count >= 0.5 * total_available:
            all_combos = []
            for combo in itertools.combinations(range(self.p), k):
                m = 0
                for bit in combo:
                    m |= (1 << bit)
                if m not in seen:
                    all_combos.append(m)
            rng.shuffle(all_combos)
            return all_combos[:target_count]

        # Fast rejection sampling with vectorized batch draws
        while len(samples) < target_count:
            needed = target_count - len(samples)
            batch_size = max(needed * 2, 1000)
            
            if self.p <= 64:
                # Vectorized batch draws via argpartition
                keys = rng.random(size=(batch_size, self.p))
                perms = np.argpartition(keys, k, axis=1)[:, :k]
                bool_mat = np.zeros((batch_size, self.p), dtype=bool)
                np.put_along_axis(bool_mat, perms, True, axis=1)
                powers = (1 << np.arange(self.p, dtype=np.int64))
                masks = (bool_mat * powers).sum(axis=1)

                for mask in masks:
                    m = int(mask)
                    if m not in seen:
                        seen.add(m)
                        samples.append(m)
                        if len(samples) == target_count:
                            break
            else:
                for _ in range(batch_size):
                    perm = rng.choice(self.p, size=k, replace=False)
                    mask = 0
                    for bit in perm:
                        mask |= (1 << int(bit))
                    if mask not in seen:
                        seen.add(mask)
                        samples.append(mask)
                        if len(samples) == target_count:
                            break

        return samples
