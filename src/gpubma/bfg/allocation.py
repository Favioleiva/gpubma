"""Adaptive budget allocation strategies for BFG model-space search."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set

import numpy as np


class BudgetAllocator:
    """Allocates model evaluation budgets adaptively across Boolean lattice levels."""

    @staticmethod
    def allocate(
        total_budget: int,
        p: int,
        exact_wings: Set[int],
        strategy: str = "adaptive",
        P_k_hat: Optional[np.ndarray] = None,
        frontier_scores: Optional[Dict[int, float]] = None,
        min_per_lattice: int = 500,
    ) -> Dict[int, int]:
        """Allocate evaluation counts B_k for each non-wing lattice.

        Parameters
        ----------
        total_budget : int
            Total budget available for statistical sampling across non-wing lattices.
        p : int
            Number of candidate predictors.
        exact_wings : Set[int]
            Lattices evaluated exhaustively (zero sampling budget needed).
        strategy : str, default="adaptive"
            Allocation strategy: "uniform", "posterior", or "adaptive".
        P_k_hat : Optional[np.ndarray]
            Current estimate of posterior model-size distribution P(k|y).
        frontier_scores : Optional[Dict[int, float]]
            Current upper evidence frontier scores U(k).
        min_per_lattice : int, default=500
            Minimum budget assigned to any non-wing lattice.
        """
        non_wings = [k for k in range(p + 1) if k not in exact_wings]
        n_non_wings = len(non_wings)

        if n_non_wings == 0:
            return {}

        min_total = min_per_lattice * n_non_wings
        budget_to_distribute = max(0, total_budget - min_total)

        if strategy == "uniform" or P_k_hat is None:
            # Uniform allocation across non-wings
            extra_per_k = budget_to_distribute // n_non_wings
            allocations = {k: min_per_lattice + extra_per_k for k in non_wings}
            rem = budget_to_distribute % n_non_wings
            for k in non_wings[:rem]:
                allocations[k] += 1
            return allocations

        if strategy == "posterior":
            # Proportional to P_hat(k|y)
            weights = np.array([max(P_k_hat[k], 1e-6) for k in non_wings], dtype=np.float64)
            weights = weights / np.sum(weights)

        elif strategy == "adaptive":
            # Proportional to A_k = P_hat(k|y) * log(N_k) or frontier relevance
            pk_sub = np.array([max(P_k_hat[k], 1e-8) for k in non_wings], dtype=np.float64)
            log_Nk = np.array([math.log(max(math.comb(p, k), 2)) for k in non_wings], dtype=np.float64)
            weights = pk_sub * log_Nk
            weights = weights / np.sum(weights)

        else:
            raise ValueError(f"Unknown allocation strategy '{strategy}'")

        allocations = {}
        distributed = 0
        for idx, k in enumerate(non_wings):
            extra = int(math.floor(weights[idx] * budget_to_distribute))
            allocations[k] = min_per_lattice + extra
            distributed += extra

        # Distribute remaining rounding difference
        remainder = budget_to_distribute - distributed
        sort_indices = np.argsort(weights)[::-1]
        for idx in sort_indices[:remainder]:
            allocations[non_wings[idx]] += 1

        # Cap by available lattice size comb(p, k)
        for k in non_wings:
            N_k = math.comb(p, k)
            if allocations[k] > N_k:
                allocations[k] = N_k

        return allocations
