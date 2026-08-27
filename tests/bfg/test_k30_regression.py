"""K=30 benchmark regression test verifying reproduction of Contract 5 metrics."""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from gpubma.bfg import fit_bfg


def test_k30_canonical_benchmark_regression():
    data_path = Path("data/synthetic/panel_30_center15.parquet")
    if not data_path.exists():
        pytest.skip(f"Benchmark file {data_path} not found.")

    df = pd.read_parquet(data_path)
    candidate_cols = [f"x{j}" for j in range(1, 31)]
    control_cols = ["w1", "w2"]

    y = df["y"]
    X = df[candidate_cols]
    always_in = df[control_cols]

    # Target certified values from Contract 5
    exact_log_Z_true = 1119.273576922141
    exact_map_pmp_true = 0.4715
    exact_map_log_score = 1118.521781

    # Execute BFG
    result = fit_bfg(
        y=y,
        X=X,
        candidate_names=candidate_cols,
        always_in=always_in,
        budget_models=50_000,
        recon_sample_per_lattice=2500,
        acesm_beta=3.5,
        beam_width=15,
        seed=20260715,
        device="cuda",
        verbose=False,
    )

    # 1. Global log Z error: |Delta log Z| < 0.01
    delta_log_Z = abs(result.log_Z - exact_log_Z_true)
    assert delta_log_Z < 0.010, f"Global denominator error exceeds tolerance: |Delta log Z| = {delta_log_Z:.6f} >= 0.010"

    # 2. MAP model verification
    expected_map = [f"x{j}" for j in range(1, 15)] + ["x30"]
    assert sorted(result.map_model) == sorted(expected_map), (
        f"MAP model mismatch: {result.map_model} vs expected {expected_map}"
    )
    assert abs(result.map_log_score - exact_map_log_score) < 1e-4

    # 3. MAP PMP verification
    pmp_err = abs(result.map_pmp - exact_map_pmp_true)
    assert pmp_err < 0.015, f"MAP PMP discrepancy: {result.map_pmp:.4f} vs {exact_map_pmp_true:.4f}"

    # 4. PIPs: x1-x13 close to 1.0, x14 high, x30 high, x15 low
    pips = result.pips
    for j in range(1, 14):
        assert pips[f"x{j}"] > 0.95, f"PIP of x{j} too low: {pips[f'x{j}']:.4f}"
    assert pips["x30"] > 0.85
