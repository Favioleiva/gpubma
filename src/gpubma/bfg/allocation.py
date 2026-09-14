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
        remaining_by_k: Optional[Dict[int, int]] = None,
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
        if not isinstance(total_budget, (int, np.integer)) or total_budget < 0:
            raise ValueError("total_budget must be a nonnegative integer")
        if not isinstance(min_per_lattice, (int, np.integer)) or min_per_lattice < 0:
            raise ValueError("min_per_lattice must be a nonnegative integer")
        if p < 0 or strategy not in ("uniform", "posterior", "adaptive"):
            raise ValueError("Invalid lattice dimension or allocation strategy")
        if P_k_hat is not None:
            P_k_hat = np.asarray(P_k_hat, dtype=np.float64)
            if P_k_hat.shape != (p + 1,) or not np.isfinite(P_k_hat).all() or np.any(P_k_hat < 0):
                raise ValueError("P_k_hat must contain p+1 finite nonnegative values")
        non_wings = [k for k in range(p + 1) if k not in exact_wings]
        n_non_wings = len(non_wings)

        if n_non_wings == 0:
            return {}

        capacities = {k: math.comb(p, k) for k in non_wings}
        if remaining_by_k is not None:
            for k in non_wings:
                value = remaining_by_k.get(k, capacities[k])
                if not isinstance(value, (int, np.integer)) or value < 0:
                    raise ValueError("remaining_by_k must contain nonnegative integer capacities")
                capacities[k] = min(capacities[k], int(value))
        target = min(int(total_budget), sum(capacities.values()))
        allocations = {k: 0 for k in non_wings}
        if target == 0:
            return allocations
        if strategy == "uniform" or P_k_hat is None:
            weights = np.ones(n_non_wings, dtype=np.float64)
        elif strategy == "posterior":
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

        # Minima are best-effort insurance, never a promise exceeding the budget.
        # Water-fill both rounds to redistribute capacity-clipped allocations.
        def distribute(amount, limits, relevance):
            while amount > 0:
                active = [i for i, k in enumerate(non_wings) if allocations[k] < limits[k]]
                if not active:
                    break
                w = relevance[active]
                w = w / w.sum()
                quotas = amount * w
                spent = 0
                for i, quota in zip(active, quotas):
                    k = non_wings[i]
                    take = min(limits[k] - allocations[k], int(math.floor(quota)))
                    allocations[k] += take
                    spent += take
                amount -= spent
                if amount:
                    order = sorted(range(len(active)), key=lambda j: (-(quotas[j] % 1), active[j]))
                    for j in order:
                        k = non_wings[active[j]]
                        if amount and allocations[k] < limits[k]:
                            allocations[k] += 1
                            amount -= 1
            return amount
        floor_limits = {k: min(min_per_lattice, capacities[k]) for k in non_wings}
        floor_budget = min(target, sum(floor_limits.values()))
        distribute(floor_budget, floor_limits, np.ones(n_non_wings))
        distribute(target - sum(allocations.values()), capacities, weights)
        return allocations
