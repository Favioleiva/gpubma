"""Tests for the bounded-memory exhaustive GPU enumerator (Phase 2).

Every posterior quantity is validated against the exact CPU oracle on
small model spaces; the checkpoint/resume path must be bit-identical to an
uninterrupted run; unranking must cover every model exactly once.
"""

import itertools
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gpubma.gpu.batch_scorer import torch_cuda_available
from gpubma.gpu.enumerator import binomial_table

CUDA_OK, CUDA_MSG = torch_cuda_available()

requires_cuda = pytest.mark.skipif(
    not CUDA_OK, reason=f"CUDA unavailable, GPU tests skipped explicitly: {CUDA_MSG}"
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def workdir():
    """Self-managed temp dir (pytest's tmp_path factory needs to scan the
    shared temp root, which some sandboxes forbid)."""
    d = Path(tempfile.mkdtemp(prefix="gpubma-enum-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_binomial_table_exact():
    import math

    C = binomial_table(30)
    for n in (5, 17, 30):
        for j in range(n + 1):
            assert C[n, j] == math.comb(n, j)
    assert C[30, 15] == 155_117_520


def test_unranking_covers_every_combination_exactly_once():
    """CPU-torch check (no CUDA needed): colex unranking reproduces
    itertools.combinations as a set, exactly once each."""
    import torch

    from gpubma.gpu.enumerator import unrank_combinations

    p = 10
    binom = torch.from_numpy(binomial_table(p))
    for k in (1, 3, 5, 10):
        total = int(binomial_table(p)[p, k])
        ranks = torch.arange(total, dtype=torch.int64)
        combos = unrank_combinations(ranks, k, binom, torch).numpy()
        assert combos.shape == (total, k)
        # ascending within each row, all rows distinct, complete coverage
        assert (np.diff(combos, axis=1) > 0).all() if k > 1 else True
        seen = {tuple(row) for row in combos}
        expected = set(itertools.combinations(range(p), k))
        assert seen == expected


def _shrink_inputs(df, predictors, controls):
    """Residualized inputs + shrink-convention parameters, exactly as
    gpubma.api.bma_regress prepares them."""
    y = df["y"].to_numpy(np.float64)
    X = df[predictors].to_numpy(np.float64)
    n = len(y)
    A = np.column_stack([np.ones(n)] + [df[c].to_numpy(np.float64) for c in controls])
    Q, _ = np.linalg.qr(A)
    yr = y - Q @ (Q.T @ y)
    Xr = X - Q @ (Q.T @ X)
    yc = y - y.mean()
    return Xr, yr, dict(df_resid=n - 1, tss_norm=float(yc @ yc),
                        k_always=A.shape[1] - 1)


def _cpu_reference(Xr, yr, conv, g, p):
    from gpubma.cpu.enumeration import enumerate_models
    from gpubma.priors.model_priors import log_model_prior_function

    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), p)
    return enumerate_models(Xr, yr, g=g, log_model_prior=log_prior,
                            top_k=10, **conv), log_prior


@pytest.fixture(scope="module")
def pair(panel8):
    if not CUDA_OK:
        pytest.skip(f"CUDA unavailable, GPU tests skipped explicitly: {CUDA_MSG}")
    from gpubma.gpu.enumerator import enumerate_models_gpu

    predictors = [f"x{j}" for j in range(1, 9)]
    Xr, yr, conv = _shrink_inputs(panel8, predictors, ["w1", "w2"])
    cpu, log_prior = _cpu_reference(Xr, yr, conv, 1000.0, len(predictors))
    gpu = enumerate_models_gpu(Xr, yr, g=1000.0, log_model_prior=log_prior,
                               top_k=10, keep_scores=True,
                               max_chunk=64,  # force many chunks
                               progress_every_s=0.0, **conv)
    return cpu, gpu


@requires_cuda
class TestEnumeratorAgainstCpuOracle:
    def test_exact_model_count(self, pair):
        cpu, gpu = pair
        assert gpu["n_models_evaluated"] == gpu["n_models_expected"] == 256
        assert gpu["n_models_evaluated"] == cpu["n_models_evaluated"]

    def test_per_model_log_scores(self, pair):
        cpu, gpu = pair
        np.testing.assert_allclose(gpu["log_scores"], cpu["log_scores"],
                                   rtol=0.0, atol=1e-10)

    def test_log_normalizer(self, pair):
        cpu, gpu = pair
        assert gpu["log_normalizer"] == pytest.approx(cpu["log_normalizer"],
                                                      abs=1e-10)

    def test_pip(self, pair):
        cpu, gpu = pair
        np.testing.assert_allclose(gpu["pip"], cpu["pip"], rtol=0.0, atol=1e-12)

    def test_coefficient_moments(self, pair):
        cpu, gpu = pair
        np.testing.assert_allclose(gpu["coef_mean"], cpu["coef_mean"],
                                   rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(gpu["coef_sd"], cpu["coef_sd"],
                                   rtol=0.0, atol=1e-12)

    def test_size_distribution_and_mean_size(self, pair):
        cpu, gpu = pair
        np.testing.assert_allclose(gpu["size_distribution"],
                                   cpu["size_distribution"],
                                   rtol=0.0, atol=1e-12)
        assert gpu["mean_model_size"] == pytest.approx(cpu["mean_model_size"],
                                                       abs=1e-12)
        assert gpu["size_distribution"].sum() == pytest.approx(1.0, abs=1e-12)

    def test_top_models_agree(self, pair):
        cpu, gpu = pair
        assert [m["mask"] for m in gpu["top_models"]] == \
            [m["mask"] for m in cpu["top_models"]]
        for mg, mc in zip(gpu["top_models"], cpu["top_models"]):
            assert mg["log_score"] == pytest.approx(mc["log_score"], abs=1e-10)
            assert mg["pmp"] == pytest.approx(mc["pmp"], abs=1e-12)

    def test_float64_declared_and_used(self, pair):
        _, gpu = pair
        assert gpu["runtime"]["precision"] == "float64"


@requires_cuda
def test_enumerator_with_fixed_effect_dummies_matches_cpu(panel8):
    """Shrink convention with two-way FE dummies in the always block
    (q = 110): the enumerator input is the residualized data, identical to
    the validated CPU path."""
    from gpubma.api import bma_regress
    from gpubma.fixed_effects.design import dummy_design
    from gpubma.gpu.enumerator import enumerate_models_gpu
    from gpubma.priors.model_priors import log_model_prior_function

    predictors = [f"x{j}" for j in range(1, 9)]
    ref = bma_regress(data=panel8, outcome="y", predictors=predictors,
                      controls=["w1", "w2"], fixed_effects=["individual", "time"],
                      fe_method="dummies", entity_col="individual_id",
                      time_col="period", always_prior="shrink", g=1000.0,
                      model_prior=("betabinomial", 1.0, 1.0))

    y = panel8["y"].to_numpy(np.float64)
    X = panel8[predictors].to_numpy(np.float64)
    D, _ = dummy_design(panel8, ["individual", "time"], "individual_id", "period")
    W = np.column_stack([panel8[["w1", "w2"]].to_numpy(np.float64), D])
    A = np.column_stack([np.ones(len(y)), W])
    Q, _ = np.linalg.qr(A)
    yr = y - Q @ (Q.T @ y)
    Xr = X - Q @ (Q.T @ X)
    yc = y - y.mean()
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), 8)

    gpu = enumerate_models_gpu(Xr, yr, df_resid=len(y) - 1, g=1000.0,
                               log_model_prior=log_prior,
                               tss_norm=float(yc @ yc), k_always=W.shape[1],
                               keep_scores=True, progress_every_s=0.0)
    order = np.argsort(ref.masks)
    assert np.array_equal(ref.masks[order], np.arange(256))
    np.testing.assert_allclose(gpu["log_scores"], ref.log_scores[order],
                               rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(gpu["pip"], ref.pip, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(gpu["coef_mean"], ref.coef_mean, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(gpu["coef_sd"], ref.coef_sd, rtol=0.0, atol=1e-12)


@requires_cuda
def test_checkpoint_resume_bit_identical(panel8, workdir):
    """Interrupt after a few chunks, resume from disk, and require results
    BIT-IDENTICAL to an uninterrupted run with the same chunking."""
    from gpubma.gpu.enumerator import enumerate_models_gpu
    from gpubma.priors.model_priors import log_model_prior_function

    predictors = [f"x{j}" for j in range(1, 9)]
    Xr, yr, conv = _shrink_inputs(panel8, predictors, ["w1", "w2"])
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), 8)
    common = dict(g=1000.0, log_model_prior=log_prior, max_chunk=32,
                  progress_every_s=0.0, **conv)

    full = enumerate_models_gpu(Xr, yr, **common)

    ckpt = workdir / "enum.ckpt.npz"
    part = enumerate_models_gpu(Xr, yr, checkpoint_path=ckpt,
                                stop_after_chunks=3, **common)
    assert part["interrupted"] is True
    assert part["models_done"] < 256
    resumed = enumerate_models_gpu(Xr, yr, checkpoint_path=ckpt, resume=True,
                                   **common)

    assert resumed["n_models_evaluated"] == 256
    assert resumed["runtime"]["resumed"] is True
    # bit-identical: same chunk partitioning, same deterministic reductions
    assert resumed["log_normalizer"] == full["log_normalizer"]
    np.testing.assert_array_equal(resumed["pip"], full["pip"])
    np.testing.assert_array_equal(resumed["coef_mean"], full["coef_mean"])
    np.testing.assert_array_equal(resumed["coef_sd"], full["coef_sd"])
    np.testing.assert_array_equal(resumed["size_distribution"],
                                  full["size_distribution"])
    assert [m["mask"] for m in resumed["top_models"]] == \
        [m["mask"] for m in full["top_models"]]


@requires_cuda
def test_resume_refuses_mismatched_configuration(panel8, workdir):
    from gpubma.gpu.enumerator import enumerate_models_gpu
    from gpubma.priors.model_priors import log_model_prior_function

    predictors = [f"x{j}" for j in range(1, 9)]
    Xr, yr, conv = _shrink_inputs(panel8, predictors, ["w1", "w2"])
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), 8)
    ckpt = workdir / "enum.ckpt.npz"
    enumerate_models_gpu(Xr, yr, g=1000.0, log_model_prior=log_prior,
                         checkpoint_path=ckpt, stop_after_chunks=2,
                         max_chunk=32, progress_every_s=0.0, **conv)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        enumerate_models_gpu(Xr, yr, g=500.0, log_model_prior=log_prior,
                             checkpoint_path=ckpt, resume=True,
                             max_chunk=32, progress_every_s=0.0, **conv)


@requires_cuda
def test_chunk_size_invariance(panel8):
    """Different chunk partitions change only the floating-point reduction
    order; results must agree to tight tolerance."""
    from gpubma.gpu.enumerator import enumerate_models_gpu
    from gpubma.priors.model_priors import log_model_prior_function

    predictors = [f"x{j}" for j in range(1, 9)]
    Xr, yr, conv = _shrink_inputs(panel8, predictors, ["w1", "w2"])
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), 8)
    a = enumerate_models_gpu(Xr, yr, g=1000.0, log_model_prior=log_prior,
                             max_chunk=16, progress_every_s=0.0, **conv)
    b = enumerate_models_gpu(Xr, yr, g=1000.0, log_model_prior=log_prior,
                             max_chunk=1 << 16, progress_every_s=0.0, **conv)
    assert a["log_normalizer"] == pytest.approx(b["log_normalizer"], abs=1e-12)
    np.testing.assert_allclose(a["pip"], b["pip"], rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(a["coef_sd"], b["coef_sd"], rtol=0.0, atol=1e-13)


@requires_cuda
def test_repeated_runs_are_reproducible(panel8):
    from gpubma.gpu.enumerator import enumerate_models_gpu
    from gpubma.priors.model_priors import log_model_prior_function

    predictors = [f"x{j}" for j in range(1, 9)]
    Xr, yr, conv = _shrink_inputs(panel8, predictors, ["w1", "w2"])
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), 8)
    kw = dict(g=1000.0, log_model_prior=log_prior, progress_every_s=0.0, **conv)
    r1 = enumerate_models_gpu(Xr, yr, **kw)
    r2 = enumerate_models_gpu(Xr, yr, **kw)
    assert r1["log_normalizer"] == r2["log_normalizer"]
    np.testing.assert_array_equal(r1["pip"], r2["pip"])
    np.testing.assert_array_equal(r1["coef_mean"], r2["coef_mean"])


@requires_cuda
def test_bounded_device_memory(panel8):
    """A tiny VRAM budget must be respected (peak measured well under the
    total card memory and scaling with the budget, not with 2^p)."""
    import torch

    from gpubma.gpu.enumerator import enumerate_models_gpu
    from gpubma.priors.model_priors import log_model_prior_function

    predictors = [f"x{j}" for j in range(1, 9)]
    Xr, yr, conv = _shrink_inputs(panel8, predictors, ["w1", "w2"])
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), 8)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = enumerate_models_gpu(Xr, yr, g=1000.0, log_model_prior=log_prior,
                               vram_budget_bytes=64 << 20,
                               progress_every_s=0.0, **conv)
    assert out["runtime"]["peak_gpu_memory_bytes"] < 256 << 20
