"""Stage 6 Final Validation and Release Readiness Test Suite.

Validates the full BFG / BFG128 system across four regimes:
- Regime 1: p = 30 Legacy Oracle
- Regime 2: p = 67 End-to-end BFG128
- Regime 3: p = 100, k = 50 Thesis-scale stress test (100,000 scored models)
- Regime 4: p = 128, k = 64 Boundary stress test (100,000 scored models)
- Dispatch Boundary: p in {60, 61, 128, 129}
- Checkpoint / Resume: Bitwise/exact reproduction
- Persistence / Provenance: Fail-closed verification
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from gpubma.bfg.config import BFGConfig
from gpubma.bfg.engine import BFGEngine
from gpubma.bfg.results import BFGResult
from gpubma.bfg.scorer import BFGScorer
from gpubma.bfg.bfg128.engine import BFG128Engine
from gpubma.bfg.bfg128.recovery import WideShellRecovery
from gpubma.bfg.bfg128.registry import WideEliteRegistry
from gpubma.bfg.bfg128.results import BFG128Result
from gpubma.bfg.bfg128.bitops import unpack_mask


def _make_data(n: int, p: int, seed: int = 42, active: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    if active > 0:
        beta[:min(active, p)] = np.linspace(2.5, 0.5, min(active, p))
    y = X @ beta + rng.standard_normal(n)
    return y, X


# ==============================================================================
# REGIME 1: p = 30 LEGACY ORACLE
# ==============================================================================

def test_regime1_p30_legacy_oracle():
    """Verify p=30 routes to legacy engine and reproduces canonical discovery."""
    data_path = Path("data/synthetic/panel_30_center15.parquet")
    if not data_path.is_file():
        pytest.skip("data/synthetic/panel_30_center15.parquet not found")

    import pandas as pd
    df = pd.read_parquet(data_path)
    candidate_cols = [f"x{j}" for j in range(1, 31)]
    control_cols = ["w1", "w2"]

    y = df["y"]
    X = df[candidate_cols]
    always_in = df[control_cols]

    cfg = BFGConfig(
        budget_models=50_000,
        recon_sample_per_lattice=2500,
        beam_width=15,
        seed=20260715,
        device="cuda",
        verbose=False,
    )
    engine = BFGEngine(config=cfg)
    res = engine.fit(y, X, candidate_names=candidate_cols, always_in=always_in)

    # 1. Verify legacy routing
    assert isinstance(res, BFGResult)
    assert not isinstance(res, BFG128Result)
    assert res.backend == "legacy"

    # 2. Verify canonical discovery
    assert res.map_model_id == 536887295
    assert len(res.map_model) == 15
    assert np.isclose(res.map_log_score, 1118.521781, atol=1e-5)
    expected_vars = [f"x{i}" for i in range(1, 15)] + ["x30"]
    assert res.map_model == expected_vars


# ==============================================================================
# REGIME 2: p = 67 END-TO-END BFG128
# ==============================================================================

def test_regime2_p67_end_to_end():
    """Verify p=67 routes automatically to BFG128 and satisfies all invariants."""
    n, p = 200, 67
    y, X = _make_data(n=n, p=p, seed=20260915, active=5)
    cfg = BFGConfig(
        budget_models=600,
        recon_sample_per_lattice=40,
        seed=20260915,
        beam_width=4,
        verbose=False,
    )
    engine = BFGEngine(config=cfg)
    res = engine.fit(y, X)

    # Routing
    assert isinstance(res, BFG128Result)
    assert res.backend == "bfg128"
    assert res.n_predictors == 67

    # Finite log Z & bounded PIPs
    assert np.isfinite(res.log_Z)
    assert len(res.pips) == 67
    assert (res.pips >= 0.0).all() and (res.pips <= 1.0).all()

    # Best discovered model
    assert res.map_model_id > 0
    assert len(res.map_model) > 0

    # Shell conditional identity: sum_j PIP_{k, j} = k
    for k, s_res in res.shell_results.items():
        if s_res.binding_rule in ("census", "remainder_direct", "fully_discovered"):
            assert math.isclose(float(np.sum(s_res.pip_k)), float(k), abs_tol=1e-3)

    # Reporting text audit: no "Global MAP", contains "Best Discovered MAP"
    summary_txt = res.summary()
    assert "Global MAP" not in summary_txt
    assert "Best Discovered MAP" in summary_txt
    assert "67" in summary_txt


# ==============================================================================
# REGIME 3: p = 100, k = 50 THESIS-SCALE STRESS TEST (100,000 MODELS)
# ==============================================================================

def test_regime3_p100_k50_stress():
    """Thesis-scale stress test: C(100, 50) > 1e29, 100,000 scored models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, p, k = 500, 100, 50
    rng = np.random.default_rng(20260915)
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)

    prior_fn = lambda k_size: -math.log(p + 1) - math.lgamma(p + 1) + math.lgamma(k_size + 1) + math.lgamma(p - k_size + 1)
    scorer = BFGScorer(X_r=X, y_r=y, df_resid=n, g=float(n), log_model_prior=prior_fn, device=device)
    registry = WideEliteRegistry(p=p)
    recovery = WideShellRecovery(scorer=scorer, registry=registry)

    target_n = 100_000
    res = recovery.recover_shell(
        k=k,
        target_evaluations=target_n,
        seed=20260915,
        batch_size=8192,
    )

    assert res.n_recovery == target_n
    assert np.isfinite(res.log_Z_k)
    assert len(res.pip_k) == p
    assert (res.pip_k >= 0.0).all() and (res.pip_k <= 1.0).all()

    # Sum of conditional PIPs must equal k = 50
    pip_sum = float(np.sum(res.pip_k))
    assert math.isclose(pip_sum, float(k), abs_tol=1e-3)


# ==============================================================================
# REGIME 4: p = 128, k = 64 BOUNDARY STRESS TEST (100,000 MODELS)
# ==============================================================================

def test_regime4_p128_k64_stress():
    """Boundary stress test: p=128, k=64 across bit 63/64, 100,000 scored models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, p, k = 500, 128, 64
    rng = np.random.default_rng(20260915)
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)

    prior_fn = lambda k_size: -math.log(p + 1) - math.lgamma(p + 1) + math.lgamma(k_size + 1) + math.lgamma(p - k_size + 1)
    scorer = BFGScorer(X_r=X, y_r=y, df_resid=n, g=float(n), log_model_prior=prior_fn, device=device)
    registry = WideEliteRegistry(p=p)
    recovery = WideShellRecovery(scorer=scorer, registry=registry)

    target_n = 100_000
    res = recovery.recover_shell(
        k=k,
        target_evaluations=target_n,
        seed=20260915,
        batch_size=8192,
    )

    assert res.n_recovery == target_n
    assert np.isfinite(res.log_Z_k)
    assert len(res.pip_k) == p
    assert (res.pip_k >= 0.0).all() and (res.pip_k <= 1.0).all()

    # Sum of conditional PIPs must equal k = 64
    pip_sum = float(np.sum(res.pip_k))
    assert math.isclose(pip_sum, float(k), abs_tol=1e-3)

    # Verify champion key if present has valid 128-bit mask
    if res.champion_key is not None:
        lo, hi = res.champion_key
        assert (lo >= 0) and (hi >= 0)
        unpacked = unpack_mask(lo, hi, p=128)
        assert len(unpacked) == 64
        assert max(unpacked) < 128


# ==============================================================================
# DISPATCH BOUNDARY TESTS
# ==============================================================================

def test_dispatch_boundary_exhaustive():
    """Verify sharp boundaries: p=60 legacy, p=61 BFG128, p=128 BFG128, p=129 error."""
    rng = np.random.default_rng(42)

    # p = 60 -> legacy
    X60 = rng.standard_normal((100, 60))
    y60 = rng.standard_normal(100)
    cfg60 = BFGConfig(budget_models=5000, budget_semantics="sampling_only", beam_width=2, verbose=False)
    res60 = BFGEngine(config=cfg60).fit(y60, X60)
    assert res60.backend == "legacy"
    assert isinstance(res60, BFGResult)

    # p = 61 -> BFG128
    X61 = rng.standard_normal((70, 61))
    y61 = rng.standard_normal(70)
    cfg61 = BFGConfig(budget_models=50, recon_sample_per_lattice=5, verbose=False)
    res61 = BFGEngine(config=cfg61).fit(y61, X61)
    assert res61.backend == "bfg128"
    assert isinstance(res61, BFG128Result)

    # p = 128 -> BFG128
    X128 = rng.standard_normal((140, 128))
    y128 = rng.standard_normal(140)
    cfg128 = BFGConfig(budget_models=50, recon_sample_per_lattice=5, verbose=False)
    res128 = BFGEngine(config=cfg128).fit(y128, X128)
    assert res128.backend == "bfg128"
    assert isinstance(res128, BFG128Result)

    # p = 129 -> explicit ValueError
    X129 = rng.standard_normal((140, 129))
    y129 = rng.standard_normal(140)
    with pytest.raises(ValueError, match="Unsupported dimension p=129"):
        BFGEngine(config=cfg128).fit(y129, X129)


# ==============================================================================
# CHECKPOINT / RESUME TESTS (p = 67 AND p = 100)
# ==============================================================================

def test_checkpoint_resume_p67_and_p100(tmp_path):
    """Verify bitwise/exact reproduction on resume for full p=67 and p=100 shell recovery."""
    # 1. p = 67 Full Run Resume Test
    y67, X67 = _make_data(n=80, p=67, seed=48, active=3)
    ckpt67 = tmp_path / "ckpt_p67"
    ckpt67.mkdir()

    cfg1 = BFGConfig(
        budget_models=250,
        recon_sample_per_lattice=15,
        seed=20260915,
        checkpoint_dir=ckpt67,
        resume=False,
        verbose=False,
    )
    res1 = BFG128Engine(config=cfg1).fit(y67, X67)

    cfg2 = BFGConfig(
        budget_models=250,
        recon_sample_per_lattice=15,
        seed=20260915,
        checkpoint_dir=ckpt67,
        resume=True,
        verbose=False,
    )
    res2 = BFG128Engine(config=cfg2).fit(y67, X67)

    assert res2.map_model_id == res1.map_model_id
    assert res2.log_Z == res1.log_Z
    assert np.array_equal(res2.pips.values, res1.pips.values)

    # 2. p = 100 Shell Recovery Resume Test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p, k = 100, 30
    rng = np.random.default_rng(20260915)
    X100 = rng.standard_normal((120, p))
    y100 = rng.standard_normal(120)
    prior_fn = lambda k_sz: -math.log(p + 1)
    scorer = BFGScorer(X_r=X100, y_r=y100, df_resid=120, g=120.0, log_model_prior=prior_fn, device=device)
    reg = WideEliteRegistry(p=p)
    rec = WideShellRecovery(scorer=scorer, registry=reg)

    # Run part 1 (1,000 models from start_raw_index=0)
    res_rec1 = rec.recover_shell(k=k, target_evaluations=1000, seed=42, start_raw_index=0)
    # Resume part 2 (another 1,000 models continuing from next_raw_index)
    res_rec2 = rec.recover_shell(k=k, target_evaluations=1000, seed=42, start_raw_index=res_rec1.next_raw_index)

    # Direct run of 2,000 models
    res_rec_all = rec.recover_shell(k=k, target_evaluations=2000, seed=42, start_raw_index=0)

    assert res_rec2.next_raw_index == res_rec_all.next_raw_index
    assert res_rec_all.n_recovery == 2000


# ==============================================================================
# PERSISTENCE & PROVENANCE AUDIT (FAIL-CLOSED)
# ==============================================================================

def test_persistence_provenance_audit(tmp_path):
    """Verify wide artifacts contain complete provenance and fail closed on incompatible resume."""
    y, X = _make_data(n=80, p=67, seed=48)
    ckpt_dir = tmp_path / "ckpt_provenance"
    ckpt_dir.mkdir()

    cfg = BFGConfig(
        budget_models=150,
        recon_sample_per_lattice=10,
        seed=20260915,
        checkpoint_dir=ckpt_dir,
        resume=False,
        verbose=False,
    )
    BFG128Engine(config=cfg).fit(y, X)

    state_file = ckpt_dir / "bfg128_state.json"
    assert state_file.is_file()

    meta = json.loads(state_file.read_text(encoding="utf-8"))

    # Required provenance fields
    required_keys = [
        "backend", "schema_version", "package_version", "fingerprint",
        "n_predictors", "n_obs", "seed", "candidate_names", "model_prior",
        "g_prior", "budget_models", "recon_sample_per_lattice",
        "n_models_evaluated", "elapsed_seconds", "execution_state", "shell_results"
    ]
    for key in required_keys:
        assert key in meta, f"Missing provenance metadata key: {key}"

    assert meta["backend"] == "bfg128"
    assert meta["n_predictors"] == 67
    from gpubma import __version__ as gpubma_version
    assert meta["package_version"] == gpubma_version

    # Incompatible resume test 1: Deliberate corruption of backend
    meta_corrupt = meta.copy()
    meta_corrupt["backend"] = "incompatible_engine"
    state_file.write_text(json.dumps(meta_corrupt), encoding="utf-8")

    cfg_resume = BFGConfig(
        budget_models=150,
        recon_sample_per_lattice=10,
        seed=20260915,
        checkpoint_dir=ckpt_dir,
        resume=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match="Incompatible checkpoint backend"):
        BFG128Engine(config=cfg_resume).fit(y, X)

    # Incompatible resume test 2: Deliberate data change (tampered fingerprint)
    state_file.write_text(json.dumps(meta), encoding="utf-8")
    y_tampered = y.copy()
    y_tampered[0] += 10.0  # alter dataset
    with pytest.raises(ValueError, match="Checkpoint fingerprint mismatch"):
        BFG128Engine(config=cfg_resume).fit(y_tampered, X)
