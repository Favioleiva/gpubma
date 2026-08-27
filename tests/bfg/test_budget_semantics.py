"""Tests for strict hard-ceiling budget compliance in BFG."""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from gpubma.bfg import fit_bfg, BFGConfig


def test_hard_budget_compliance():
    """Verify that total unique evaluated models never exceeds the requested budget_models."""
    data_path = Path("data/synthetic/panel_30_center15.parquet")
    if not data_path.exists():
        pytest.skip("Benchmark data not found.")

    df = pd.read_parquet(data_path)
    candidate_cols = [f"x{j}" for j in range(1, 31)]
    control_cols = ["w1", "w2"]

    y = df["y"]
    X = df[candidate_cols]
    always_in = df[control_cols]

    for requested_budget in [5_000, 15_000, 30_000, 50_000]:
        result = fit_bfg(
            y=y,
            X=X,
            candidate_names=candidate_cols,
            always_in=always_in,
            budget_models=requested_budget,
            budget_semantics="hard",
            device="cuda",
            seed=20260715,
            verbose=False,
        )

        assert result.n_models_evaluated <= requested_budget, (
            f"Hard budget violation: Evaluated {result.n_models_evaluated:,} > Budget {requested_budget:,}"
        )
        assert math.isfinite(result.log_Z)
        assert len(result.pips) == 30
