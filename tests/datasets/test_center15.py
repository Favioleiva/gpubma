"""Focused tests for the frozen panel_30_center15 benchmark.

Covers: deterministic regeneration, schema and column ordering, structural
truth (15 nonzero + 15 exact-zero coefficients, proxy mapping), R^2
tolerance, design and residualized rank, condition-number bounds, Parquet
reload parity, metadata consistency and SHA-256 recording, and protection
of the pre-existing frozen panel_30 artifacts.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gpubma.cpu.enumeration import enumerate_models
from gpubma.datasets.center15 import (
    BETA_TRUE, K_TRUE, N_OBS, P, PROXY_PARTNER, PROXY_PRIMARY, SEED,
    generate_panel_30_center15, structural_beta,
)
from gpubma.datasets.io_utils import compare_frames_exact, file_sha256
from gpubma.priors.model_priors import log_model_prior_function

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "synthetic"
PARQUET = DATA / "panel_30_center15.parquet"
META_JSON = DATA / "panel_30_center15_metadata.json"

EXPECTED_COLUMNS = (["individual_id", "period", "y"]
                    + [f"x{j}" for j in range(1, 31)] + ["w1", "w2"])


@pytest.fixture(scope="module")
def frozen() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)


@pytest.fixture(scope="module")
def meta() -> dict:
    return json.loads(META_JSON.read_text())


def test_deterministic_regeneration():
    df1, _ = generate_panel_30_center15(seed=SEED)
    df2, _ = generate_panel_30_center15(seed=SEED)
    pd.testing.assert_frame_equal(df1, df2)


def test_schema_and_column_ordering(frozen):
    assert list(frozen.columns) == EXPECTED_COLUMNS
    assert frozen.shape == (N_OBS, len(EXPECTED_COLUMNS)) == (2000, 35)
    assert frozen["individual_id"].dtype == np.int32
    assert frozen["period"].dtype == np.int32
    for col in EXPECTED_COLUMNS[2:]:
        assert frozen[col].dtype == np.float64
    assert not frozen.isna().any().any()
    assert np.isfinite(frozen.to_numpy(np.float64)).all()


def test_candidate_variable_count(frozen):
    xcols = [c for c in frozen.columns if c.startswith("x")]
    assert len(xcols) == P == 30
    assert xcols == [f"x{j}" for j in range(1, 31)]


def test_structural_beta_counts():
    beta = structural_beta()
    assert beta.shape == (30,)
    assert int(np.count_nonzero(beta)) == K_TRUE == 15
    assert np.array_equal(beta[:15], BETA_TRUE)
    # proxies x16-x30 have structural coefficients EXACTLY zero
    assert np.array_equal(beta[15:], np.zeros(15))


def test_proxy_mapping(meta):
    assert PROXY_PRIMARY == list(range(1, 16))
    for i, entry in enumerate(meta["proxy_mappings"]):
        assert entry["proxy"] == f"x{16 + i}"
        assert entry["primary_true_variable"] == f"x{PROXY_PRIMARY[i]}"
        assert entry["secondary_partner"] == f"x{PROXY_PARTNER[i]}"
        # partner is a different true variable from the same block of five
        assert entry["secondary_partner"] != entry["primary_true_variable"]
        assert (PROXY_PARTNER[i] - 1) // 5 == (PROXY_PRIMARY[i] - 1) // 5
        assert 0.60 < entry["realized_correlation_with_primary"] < 0.995


def test_r2_within_tolerance(frozen, meta):
    y = frozen["y"].to_numpy(np.float64)
    D = np.column_stack([np.ones(len(frozen)),
                         frozen[["w1", "w2"]].to_numpy(np.float64),
                         frozen[[f"x{j}" for j in range(1, 16)]].to_numpy(np.float64)])
    Q, _ = np.linalg.qr(D)
    resid = y - Q @ (Q.T @ y)
    y_c = y - y.mean()
    r2 = 1.0 - (resid @ resid) / (y_c @ y_c)
    assert 0.695 <= r2 <= 0.705
    assert r2 < 1.0
    assert abs(r2 - meta["noise_calibration"]["realized_r2"]) < 1e-12


def test_design_and_residualized_rank(frozen):
    X = frozen[[f"x{j}" for j in range(1, 31)]].to_numpy(np.float64)
    assert np.linalg.matrix_rank(X) == 30
    A = np.column_stack([np.ones(len(frozen)),
                         frozen[["w1", "w2"]].to_numpy(np.float64)])
    Q, _ = np.linalg.qr(A)
    X_r = X - Q @ (Q.T @ X)
    assert np.linalg.matrix_rank(X_r) == 30
    eig = np.linalg.eigvalsh(X_r.T @ X_r)
    assert eig[0] > 0
    assert eig[-1] / eig[0] < 1e6  # float64 Cholesky headroom
    np.linalg.cholesky(X_r.T @ X_r)


def test_parquet_reload_parity_with_regeneration(frozen):
    df, _ = generate_panel_30_center15(seed=SEED)
    rep = compare_frames_exact(df, frozen, float_atol=0.0)
    assert rep["pass"], rep


def test_metadata_consistency(frozen, meta):
    assert meta["dataset_name"] == "panel_30_center15"
    assert meta["seed"] == SEED == 20260724
    assert meta["n_observations"] == len(frozen) == 2000
    assert meta["n_candidate_regressors"] == 30
    assert meta["true_model_size"] == 15
    assert meta["true_variables"] == [f"x{j}" for j in range(1, 16)]
    assert meta["proxy_variables"] == [f"x{j}" for j in range(16, 31)]
    assert meta["structural_beta"] == structural_beta().tolist()
    assert meta["canonical_format"] == "parquet"
    assert meta["column_order"] == EXPECTED_COLUMNS
    assert meta["model_space_statements"]["total_models"] == 2**30
    assert meta["model_space_statements"]["central_layer_size"] == 155117520
    assert meta["always_included_controls"]["delta"] == [0.8, -0.6]


def test_sha256_recorded_and_matches(meta):
    assert meta["parquet_sha256"] == file_sha256(PARQUET)
    gen = ROOT / "scripts" / "generate_panel_30_center15.py"
    assert meta["generator_sha256"] == file_sha256(gen)
    assert len(meta["parquet_sha256"]) == 64
    int(meta["parquet_sha256"], 16)  # valid hex


def test_no_csv_or_dta_artifact_exists():
    assert not (DATA / "panel_30_center15.csv").exists()
    assert not (DATA / "panel_30_center15.dta").exists()


def test_existing_panel_30_artifacts_unchanged():
    """The new benchmark must never overwrite the frozen panel_* artifacts:
    every legacy file must still match the SHA-256 in its own metadata."""
    for stem in ("panel_8", "panel_12", "panel_30"):
        legacy = json.loads((DATA / f"{stem}_metadata.json").read_text())
        for fmt, entry in legacy["file_checksums"].items():
            path = DATA / f"{stem}.{fmt}"
            assert path.exists(), path
            assert file_sha256(path) == entry["sha256"], f"{stem}.{fmt} changed"


def test_sufficient_statistics_compatibility_smoke(frozen):
    """Minimal load + sufficient-statistics compatibility check: the new
    Parquet artifact feeds the existing enumeration machinery (subset of
    12 predictors -> 4,096 models on CPU; NOT a 2^30 run)."""
    y = frozen["y"].to_numpy(np.float64)
    n = len(y)
    X = frozen[[f"x{j}" for j in range(1, 13)]].to_numpy(np.float64)
    A = np.column_stack([np.ones(n), frozen[["w1", "w2"]].to_numpy(np.float64)])
    Q, _ = np.linalg.qr(A)
    y_r = y - Q @ (Q.T @ y)
    X_r = X - Q @ (Q.T @ X)
    yc = y - y.mean()
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), 12)
    res = enumerate_models(
        X_r, y_r, df_resid=n - 1, g=float(max(n, 12 * 12)),
        log_model_prior=log_prior, tss_norm=float(yc @ yc), k_always=2,
        compute_coefficients=False,
    )
    assert res["n_models_evaluated"] == 4096
    assert np.isfinite(res["log_normalizer"])
    assert (res["pip"] >= 0).all() and (res["pip"] <= 1).all()
    assert abs(res["size_distribution"].sum() - 1.0) < 1e-9
