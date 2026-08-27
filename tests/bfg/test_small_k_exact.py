"""Small-K exact integration test comparing BFG against exhaustive CPU enumeration."""

import math
import numpy as np
import pandas as pd
import pytest

from gpubma.api import bma_regress
from gpubma.bfg import fit_bfg


def test_small_k_exact_validation():
    rng = np.random.default_rng(20260715)
    n, p = 120, 8
    X = rng.normal(size=(n, p))
    # True signal on x1, x2, x3
    y = X[:, 0] * 2.0 + X[:, 1] * 1.5 + X[:, 2] * 1.0 + rng.normal(scale=0.8, size=n)

    var_names = [f"x{j + 1}" for j in range(p)]
    df = pd.DataFrame(X, columns=var_names)
    df["y"] = y

    # 1. Exact Enumeration Reference
    exact_res = bma_regress(
        data=df,
        outcome="y",
        predictors=var_names,
        always_prior="shrink",
        backend="cpu",
    )

    # 2. BFG Execution
    bfg_res = fit_bfg(
        y=y,
        X=X,
        candidate_names=var_names,
        budget_models=2000,
        seed=20260715,
        always_prior="shrink",
        verbose=False,
    )

    # Log denominator comparison
    exact_log_Z = float(exact_res.log_scores.max() + math.log(np.sum(np.exp(exact_res.log_scores - exact_res.log_scores.max()))))
    delta_log_Z = abs(bfg_res.log_Z - exact_log_Z)
    assert delta_log_Z < 0.05, f"Global denominator discrepancy: BFG {bfg_res.log_Z:.6f} vs Exact {exact_log_Z:.6f}"

    # Posterior model size distribution TVD
    exact_pk = exact_res.size_distribution
    bfg_pk = bfg_res.model_size_posterior.values
    tvd_pk = float(0.5 * np.sum(np.abs(bfg_pk - exact_pk)))
    assert tvd_pk < 0.05, f"Posterior model size TVD too high: {tvd_pk:.6f}"

    # PIPs comparison
    exact_pips = exact_res.pip
    bfg_pips = bfg_res.pips.values
    max_pip_diff = float(np.max(np.abs(bfg_pips - exact_pips)))
    assert max_pip_diff < 0.05, f"Max PIP difference too high: {max_pip_diff:.6f}"

    # Global MAP model identity
    exact_top = exact_res.top_models(1).iloc[0]
    exact_map_vars = exact_top["predictors"].split()
    assert sorted(bfg_res.map_model) == sorted(exact_map_vars), (
        f"MAP model mismatch: BFG {bfg_res.map_model} vs Exact {exact_map_vars}"
    )
