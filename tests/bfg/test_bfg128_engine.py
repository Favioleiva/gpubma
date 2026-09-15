"""Tests for native BFG128 engine, registry, genealogy, elite search, recovery, and dispatch."""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from gpubma.bfg.config import BFGConfig
from gpubma.bfg.engine import BFGEngine, fit_bfg
from gpubma.bfg.registry import ModelProvenance
from gpubma.bfg.scorer import BFGScorer, count_set_bits
from gpubma.priors.gpriors import resolve_g
from gpubma.priors.model_priors import log_model_prior_function

from gpubma.bfg.bfg128.bitops import pack_indices, popcount, unpack_mask
from gpubma.bfg.bfg128.elite_search import WideEliteSearch, WideEliteSearchResult
from gpubma.bfg.bfg128.engine import BFG128Engine
from gpubma.bfg.bfg128.genealogy import WideGenealogicalSearch
from gpubma.bfg.bfg128.recovery import ShellAccumulator128, WideShellRecovery
from gpubma.bfg.bfg128.registry import WideEliteRegistry, WideModelRecord
from gpubma.bfg.bfg128.results import BFG128Result


def _make_synthetic_data(n: int = 150, p: int = 67, seed: int = 20260915):
    """Generate synthetic linear regression data with known sparse signals."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    # Signals on x1, x2, x3, x65
    beta = np.zeros(p)
    beta[0] = 2.5
    beta[1] = -2.0
    beta[2] = 1.8
    if p > 64:
        beta[64] = 2.2
    y = X @ beta + rng.standard_normal(n) * 1.0
    controls = rng.standard_normal((n, 2))
    return y, X, controls


def _make_scorer(y, X, controls=None, device="auto"):
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device

    n, p = X.shape
    if controls is not None:
        A = np.column_stack([np.ones(n, dtype=np.float64), controls])
        k_always = controls.shape[1]
    else:
        A = np.ones((n, 1), dtype=np.float64)
        k_always = 0

    Q, _ = np.linalg.qr(A)
    y_r = y - Q @ (Q.T @ y)
    X_r = X - Q @ (Q.T @ X)
    y_c = y - y.mean()
    tss_norm = float(y_c @ y_c)
    df_resid = n - 1

    g_spec = resolve_g("benchmark", n_obs=n, n_predictors=p)
    log_prior_fn, prior_desc = log_model_prior_function(("betabinomial", 1.0, 1.0), n_predictors=p)

    return BFGScorer(
        X_r=X_r,
        y_r=y_r,
        df_resid=df_resid,
        g=g_spec.g,
        log_model_prior=log_prior_fn,
        prior_description=prior_desc,
        tss_norm=tss_norm,
        k_always=k_always,
        device=dev,
        max_eval_budget=None,
    )


# ==============================================================================
# 1. REGISTRY TESTS
# ==============================================================================

def test_wide_elite_registry_basic():
    p = 67
    reg = WideEliteRegistry(p=p)
    assert reg.total_registered() == 0

    # Register single model
    m1 = pack_indices([0, 1, 65], p=p)
    added = reg.register(
        mask_lo=m1[0],
        mask_hi=m1[1],
        log_score=100.5,
        provenance=ModelProvenance.FORWARD_GENEALOGY,
        source_tag="test1",
    )
    assert added is True
    assert reg.total_registered() == 1
    assert len(reg.by_k[3]) == 1

    # Duplicate registration returns False and updates source tags
    added2 = reg.register(
        mask_lo=m1[0],
        mask_hi=m1[1],
        log_score=100.5,
        provenance=ModelProvenance.ELITE_HIT,
        source_tag="test2",
    )
    assert added2 is False
    assert reg.total_registered() == 1
    rec = reg.records[m1]
    assert rec.provenance == ModelProvenance.MULTIPLE_SOURCE
    assert "test1" in rec.source_tags and "test2" in rec.source_tags

    # Batch registration
    m2 = pack_indices([2, 3], p=p)
    m3 = pack_indices([4, 66], p=p)
    n_added = reg.register_batch(
        mask_lo=[m2[0], m3[0], m1[0]],
        mask_hi=[m2[1], m3[1], m1[1]],
        scores=[95.0, 110.0, 100.5],
        provenance=ModelProvenance.RANDOM_BULK,
    )
    assert n_added == 2
    assert reg.total_registered() == 3

    # Champion
    champ = reg.get_champion()
    assert champ is not None
    assert champ.key == m3
    assert champ.log_score == 110.0

    # DataFrame export
    df = reg.to_dataframe()
    assert len(df) == 3
    assert "mask_lo" in df.columns
    assert "mask_hi" in df.columns
    assert "model_size" in df.columns


# ==============================================================================
# 2. GENEALOGY TESTS
# ==============================================================================

def test_wide_genealogical_search():
    y, X, controls = _make_synthetic_data(n=100, p=67, seed=42)
    scorer = _make_scorer(y, X, controls)
    registry = WideEliteRegistry(p=scorer.p)
    searcher = WideGenealogicalSearch(scorer=scorer, registry=registry)

    # Forward greedy from null model
    fwd = searcher.forward_greedy(start_key=(0, 0), target_k=5)
    assert len(fwd.path) == 6
    assert fwd.path[0][1] == 0  # k=0
    assert fwd.path[-1][1] == 5  # k=5
    assert registry.total_registered() > 0

    # Backward greedy from full model
    bwd = searcher.backward_greedy(target_k=62)
    assert len(bwd.path) == 6
    assert bwd.path[0][1] == 67
    assert bwd.path[-1][1] == 62

    # Forward beam search
    beam = searcher.forward_beam(start_seeds=[fwd.best_model_key], beam_width=3, target_k=6)
    assert beam.best_k in range(fwd.best_k, 7)


# ==============================================================================
# 3. ELITE SEARCH TESTS
# ==============================================================================

def test_wide_elite_search():
    y, X, controls = _make_synthetic_data(n=100, p=67, seed=43)
    scorer = _make_scorer(y, X, controls)
    registry = WideEliteRegistry(p=scorer.p)
    elite = WideEliteSearch(scorer=scorer, registry=registry)

    res = elite.run_lattice_elite_search(k=4, r_k=20, m_k=50, target_q=0.1)
    assert res.k == 4
    assert res.r_k == 20
    assert res.m_k == 50
    assert np.isfinite(res.threshold_tau)
    assert len(res.calibration_scores) == 20
    assert registry.total_registered() == 70


# ==============================================================================
# 4. RECOVERY & ONLINE ACCUMULATOR TESTS
# ==============================================================================

def test_shell_accumulator_online_properties():
    p = 67
    k = 3
    accum = ShellAccumulator128(p=p, k=k, N_k=1000, D_k=10)

    # Add batch 1
    scores1 = np.array([10.0, 12.0, 11.0], dtype=np.float64)
    idx1 = np.array([[0, 1, 2], [0, 2, 64], [1, 2, 65]], dtype=np.int64)
    accum.update_batch(scores=scores1, idx_batch=idx1, weight=1.0)

    # Add batch 2 with different expansion weight
    scores2 = np.array([9.0, 8.5], dtype=np.float64)
    idx2 = np.array([[0, 1, 3], [2, 4, 65]], dtype=np.int64)
    accum.update_batch(scores=scores2, idx_batch=idx2, weight=10.0)

    # Verify log_Z_k is finite
    assert np.isfinite(accum.log_Z_k)

    # Verify pip_k
    pips = accum.pip_k
    assert len(pips) == p
    assert (pips >= 0.0).all() and (pips <= 1.0).all()
    # In shell k, sum of PIPs must equal k exactly
    assert np.isclose(np.sum(pips), k, atol=1e-10)


def test_wide_shell_recovery_census_and_sampling():
    y, X, controls = _make_synthetic_data(n=80, p=67, seed=44)
    scorer = _make_scorer(y, X, controls)
    registry = WideEliteRegistry(p=scorer.p)
    recovery = WideShellRecovery(scorer=scorer, registry=registry, census_max_combinations=100)

    # Census shell k=0
    r0 = recovery.enumerate_census_shell(0)
    assert r0.k == 0
    assert r0.N_k == 1
    assert r0.D_k == 1
    assert np.isfinite(r0.log_Z_k)

    # Census shell k=1
    r1 = recovery.enumerate_census_shell(1)
    assert r1.k == 1
    assert r1.N_k == 67
    assert r1.D_k == 67
    assert np.isclose(np.sum(r1.pip_k), 1.0, atol=1e-10)

    # Sampling shell k=10
    r10 = recovery.recover_shell(k=10, target_evaluations=50, seed=123)
    assert r10.k == 10
    assert r10.n_recovery == 50
    assert np.isfinite(r10.log_Z_k)
    assert np.isclose(np.sum(r10.pip_k), 10.0, atol=1e-10)


# ==============================================================================
# 5. DISPATCH AND BOUNDS TESTS
# ==============================================================================

def test_dispatch_unsupported_p_above_128():
    rng = np.random.default_rng(45)
    X_large = rng.standard_normal((50, 129))
    y = rng.standard_normal(50)

    engine = BFGEngine()
    with pytest.raises(ValueError, match="Unsupported dimension p=129"):
        engine.fit(y, X_large)


def test_dispatch_p_less_than_or_equal_60():
    # p=10 should dispatch to legacy BFG engine and succeed
    rng = np.random.default_rng(46)
    X_small = rng.standard_normal((40, 10))
    y = rng.standard_normal(40)

    res = fit_bfg(y, X_small, budget_models=2000, seed=46, verbose=False)
    # Legacy engine produces BFGResult
    assert hasattr(res, "candidate_names")
    assert len(res.candidate_names) == 10


def test_dispatch_61_to_128_end_to_end():
    # Test p=67 full end-to-end fit via fit_bfg dispatch
    y, X, controls = _make_synthetic_data(n=100, p=67, seed=47)
    cfg = BFGConfig(
        budget_models=500,
        recon_sample_per_lattice=50,
        seed=20260915,
        beam_width=3,
        verbose=False,
    )
    engine = BFGEngine(config=cfg)
    res = engine.fit(y, X, always_in=controls)

    assert isinstance(res, BFG128Result)
    assert res.n_predictors == 67
    assert np.isfinite(res.log_Z)
    assert len(res.pips) == 67
    assert (res.pips >= 0.0).all() and (res.pips <= 1.0).all()
    assert res.map_model_id > 0
    assert len(res.map_model) > 0
    # Top models summary should work
    top_df = res.top_models(3)
    assert len(top_df) <= 3
    # Summary string formatting
    summary_txt = res.summary()
    assert "BFG128" in summary_txt
    assert "67" in summary_txt


# ==============================================================================
# 6. CHECKPOINT AND RESUME TESTS
# ==============================================================================

def test_bfg128_checkpoint_resume(tmp_path):
    y, X, controls = _make_synthetic_data(n=80, p=67, seed=48)
    ckpt_dir = tmp_path / "test_ckpt"
    ckpt_dir.mkdir()

    cfg1 = BFGConfig(
        budget_models=300,
        recon_sample_per_lattice=20,
        seed=20260915,
        checkpoint_dir=ckpt_dir,
        resume=False,
        verbose=False,
    )
    engine1 = BFG128Engine(config=cfg1)
    res1 = engine1.fit(y, X, always_in=controls)

    assert (ckpt_dir / "bfg128_state.json").is_file()
    assert (ckpt_dir / "bfg128_registry.parquet").is_file()

    # Resume with identical configuration
    cfg2 = BFGConfig(
        budget_models=300,
        recon_sample_per_lattice=20,
        seed=20260915,
        checkpoint_dir=ckpt_dir,
        resume=True,
        verbose=False,
    )
    engine2 = BFG128Engine(config=cfg2)
    res2 = engine2.fit(y, X, always_in=controls)

    assert res2.map_model_id == res1.map_model_id
    # Exact bit-level float64 equality on resume
    assert res2.log_Z == res1.log_Z
    assert np.array_equal(res2.pips.values, res1.pips.values)


# ==============================================================================
# 7. HYBRID ESTIMATOR AUDIT UNIT TEST
# ==============================================================================

def test_hybrid_recovery_estimator_hand_check():
    """Verify wide recovery estimator matches exact hand-calculated hybrid formula."""
    from gpubma.bfg.bfg128.recovery import ShellAccumulator128

    p = 4
    k = 2
    N_2 = math.comb(p, k)  # 6
    D_2 = 2  # Discovered B_2 = {M1={0,1}, M2={0,2}}
    remainder_2 = N_2 - D_2  # 4
    n_2 = 2  # Sampled from remainder S_2 = {M4={1,2}, M6={2,3}}
    expansion_weight = remainder_2 / n_2  # 2.0

    accum = ShellAccumulator128(p=p, k=k, N_k=N_2, D_k=D_2)

    # 1. Accumulate B_2 = {M1, M2} with weight 1.0
    disc_scores = np.array([1.0, 2.0], dtype=np.float64)
    disc_idx = np.array([[0, 1], [0, 2]], dtype=np.int64)
    accum.update_batch(disc_scores, disc_idx, weight=1.0)

    # 2. Accumulate S_2 = {M4, M6} with weight (N_k - D_k) / n_k
    rem_scores = np.array([1.5, 0.0], dtype=np.float64)
    rem_idx = np.array([[1, 2], [2, 3]], dtype=np.int64)
    accum.update_batch(rem_scores, rem_idx, weight=expansion_weight)

    # Theoretical hand calculations:
    # Z_2 = sum_{M in B_2} exp(s_M) + (N_2 - D_2)/n_2 * sum_{M in S_2} exp(s_M)
    e1, e2 = math.exp(1.0), math.exp(2.0)
    e15, e0 = math.exp(1.5), math.exp(0.0)
    Z_expected = (e1 + e2) + expansion_weight * (e15 + e0)
    log_Z_expected = math.log(Z_expected)

    # PIP numerators:
    # Z_j2+ = sum_{M in B_2, j in M} exp(s_M) + (N_2 - D_2)/n_2 * sum_{M in S_2, j in M} exp(s_M)
    Z0 = (e1 + e2) + expansion_weight * 0.0
    Z1 = e1 + expansion_weight * e15
    Z2 = e2 + expansion_weight * (e15 + e0)
    Z3 = 0.0 + expansion_weight * e0

    pips_expected = np.array([Z0, Z1, Z2, Z3]) / Z_expected

    assert math.isclose(accum.log_Z_k, log_Z_expected, rel_tol=1e-15, abs_tol=1e-15)
    assert np.allclose(accum.pip_k, pips_expected, atol=1e-15)
    assert math.isclose(float(np.sum(accum.pip_k)), float(k), rel_tol=1e-14)
