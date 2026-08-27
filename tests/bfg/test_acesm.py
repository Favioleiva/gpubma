"""Unit tests for ACESM denominator reconstruction and cumulative curves."""

import math
import numpy as np
import pytest

from gpubma.bfg.acesm import ACESMReconstructor, CumulativeCurveBuilder, RestrictedShapeWeibullModel


def test_cumulative_curve_builder():
    k = 5
    N_k = 1000
    disc_scores = [100.0, 95.0, 92.0]
    samp_scores = [80.0, 75.0, 70.0, 65.0]

    curve = CumulativeCurveBuilder.build_empirical_curve(
        k=k,
        discovered_scores=disc_scores,
        sampled_scores=samp_scores,
        N_k=N_k,
        n_grid_points=50,
    )

    assert curve.k == k
    assert curve.U_obs == 100.0
    assert curve.n_elites == 3
    assert curve.n_samples == 4
    assert len(curve.d_grid) == 50
    assert len(curve.C_rel_obs) == 50

    # Strict monotonicity check
    for j in range(len(curve.C_rel_obs) - 1):
        assert curve.C_rel_obs[j+1] >= curve.C_rel_obs[j]


def test_acesm_reconstruction():
    k = 5
    N_k = 2000
    disc_scores = [50.0, 48.0]
    samp_scores = [35.0, 32.0, 30.0, 28.0, 25.0]

    curve = CumulativeCurveBuilder.build_empirical_curve(
        k=k,
        discovered_scores=disc_scores,
        sampled_scores=samp_scores,
        N_k=N_k,
        n_grid_points=60,
    )

    fit_res = ACESMReconstructor.fit_lattice(curve=curve, beta=3.5)
    assert math.isfinite(fit_res.log_Z_hat)
    assert fit_res.log_Z_hat >= curve.Z_known_log
    assert fit_res.alpha_hat > 0
    assert fit_res.beta_hat == 3.5
