"""Zero-leakage test verifying that BFG operates with complete air-gap from truth files."""

import math
import sys
import numpy as np
import pytest

from gpubma.bfg import fit_bfg


def test_zero_leakage_air_gapped_execution():
    """Verify that BFG executes and recovers posterior distributions without any ground truth files."""
    rng = np.random.default_rng(20260715)
    n, p = 100, 10
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 2.0 + X[:, 1] * 1.5 + rng.normal(size=n)

    # Verify no truth module is imported before run
    assert "contracts.Contract1" not in sys.modules

    # Execute BFG
    result = fit_bfg(
        y=y,
        X=X,
        budget_models=3000,
        seed=20260715,
        verbose=False,
    )

    # Verify results are valid and finite
    assert math.isfinite(result.log_Z)
    assert len(result.pips) == p
    assert result.pips["x1"] > 0.80
    assert result.pips["x2"] > 0.80

    # Ensure no truth files were loaded into global registry
    assert "contracts.Contract1" not in sys.modules
