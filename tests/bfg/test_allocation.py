"""Unit tests for BudgetAllocator."""

import numpy as np
import pytest

from gpubma.bfg.allocation import BudgetAllocator


def test_budget_allocation_strategies():
    p = 30
    exact_wings = {0, 1, 2, 3, 27, 28, 29, 30}
    total_budget = 50000

    # 1. Uniform
    alloc_u = BudgetAllocator.allocate(
        total_budget=total_budget,
        p=p,
        exact_wings=exact_wings,
        strategy="uniform",
    )
    assert sum(alloc_u.values()) == total_budget
    for k in alloc_u:
        assert k not in exact_wings
        assert alloc_u[k] >= 500

    # 2. Posterior
    P_hat_k = np.zeros(p + 1)
    P_hat_k[15] = 0.8
    P_hat_k[14] = 0.1
    P_hat_k[16] = 0.1

    alloc_p = BudgetAllocator.allocate(
        total_budget=total_budget,
        p=p,
        exact_wings=exact_wings,
        strategy="posterior",
        P_k_hat=P_hat_k,
    )
    assert sum(alloc_p.values()) == total_budget
    assert alloc_p[15] > alloc_p[5]  # Mass concentration gets more budget

    # 3. Adaptive
    alloc_a = BudgetAllocator.allocate(
        total_budget=total_budget,
        p=p,
        exact_wings=exact_wings,
        strategy="adaptive",
        P_k_hat=P_hat_k,
    )
    assert sum(alloc_a.values()) == total_budget
    assert alloc_a[15] > alloc_a[5]
