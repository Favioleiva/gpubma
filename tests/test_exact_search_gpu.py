"""Bounded CUDA validation: never enumerate a large model space in tests."""
import math

import numpy as np
import pytest
import torch

from gpubma import bma_regress
from gpubma.gpu.enumerator import (
    binomial_table, enumerate_models_gpu, unrank_combinations, validate_cursor,
)
from gpubma.priors.model_priors import log_model_prior_function
from test_search_strategy import frame

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("p", [1, 30, 31, 32])
def test_wide_counts_and_checkpoint_cursor(p, tmp_path):
    table = binomial_table(p)
    assert table.dtype == np.int64
    assert sum(int(v) for v in table[p]) == 1 << p
    for k in range(p + 1):
        assert int(table[p, k]) == math.comb(p, k)
        done = sum(math.comb(p, j) for j in range(k))
        validate_cursor(p, done, k, 0)
        validate_cursor(p, done + math.comb(p, k) - 1, k, math.comb(p, k) - 1)
    validate_cursor(p, 1 << p, p + 1, 0)
    dest = tmp_path / "wide_counters.npz"
    np.savez(dest, done=np.int64(1 << p), mask=np.int64((1 << p) - 1))
    with np.load(dest) as saved:
        assert int(saved["done"]) == 1 << p
        assert int(saved["mask"]) == (1 << p) - 1
    with pytest.raises(ValueError, match="model count"):
        validate_cursor(p, -1, 0, 0)
    with pytest.raises(ValueError, match="combination rank"):
        validate_cursor(p, 0, 0, 1)
    with pytest.raises(ValueError, match="does not match"):
        validate_cursor(p, 0, 1, 0)


def test_exact_rejects_large_space_before_cuda(monkeypatch):
    monkeypatch.setattr("gpubma.gpu.batch_scorer.torch_cuda_available",
                        lambda: pytest.fail("should validate dimension before CUDA"))
    with pytest.raises(ValueError, match="Received p=33"):
        enumerate_models_gpu(np.zeros((40, 33)), np.ones(40), df_resid=39,
                             g=40., log_model_prior=lambda k: 0.)


def test_auto_exact_has_no_cpu_fallback(monkeypatch):
    monkeypatch.setattr("gpubma.gpu.batch_scorer.torch_cuda_available",
                        lambda: (False, "simulated CUDA absence"))
    data, names = frame(1)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        bma_regress(data, "y", names, search="auto")


@cuda
@pytest.mark.parametrize("p", [1, 30, 31, 32])
def test_gpu_unranking_every_size_and_boundary_ids(p):
    table = torch.from_numpy(binomial_table(p)).cuda()
    for k in range(p + 1):
        count = math.comb(p, k)
        ranks = sorted(set([0, count // 2, count - 1]))
        indices = unrank_combinations(torch.tensor(ranks, dtype=torch.int64, device="cuda"),
                                      k, table, torch)
        masks = (torch.ones_like(indices) << indices).sum(dim=1).cpu().tolist()
        for rank, row, mask in zip(ranks, indices.cpu().tolist(), masks):
            assert row == sorted(set(row)) and len(row) == k
            assert all(0 <= j < p for j in row)
            assert sum(math.comb(j, pos + 1) for pos, j in enumerate(row)) == rank
            assert mask == sum(1 << j for j in row)
        assert masks[0] == (1 << k) - 1
        assert masks[-1] == ((1 << k) - 1) << (p - k)


@cuda
@pytest.mark.parametrize("p", [1, 5])
@pytest.mark.parametrize("prior", ["shrink", "flat"])
def test_exact_numerical_reference_with_controls_and_fe(p, prior):
    data, names = frame(p)
    options = dict(controls=["control"], fixed_effects=["individual"],
                   entity_col="district", always_prior=prior, top_k=1 << p)
    cpu = bma_regress(data, "y", names, **options)
    gpu = bma_regress(data, "y", names, search="auto", **options,
                      exact_options={"keep_scores": True, "progress_every_s": 0})
    assert gpu.n_models_evaluated == cpu.n_models_evaluated == 1 << p
    assert cpu.backend == "cpu" and gpu.backend == "gpu"
    for field in ("log_scores", "pip", "coef_mean", "coef_sd", "size_distribution"):
        np.testing.assert_allclose(getattr(gpu, field), getattr(cpu, field), rtol=0, atol=1e-10)
    np.testing.assert_allclose(gpu.top_models(1 << p).pmp.sum(), 1., atol=1e-12)
    np.testing.assert_array_equal(gpu.top_models(1 << p)["mask"], cpu.top_models(1 << p)["mask"])
    assert gpu.df_resid == cpu.df_resid
    assert gpu.pmp is None and gpu.masks is None  # streaming, not 2**p-sized storage


@cuda
def test_checkpoint_resume_numerical_parity(tmp_path):
    data, names = frame(5)
    X = data[names].to_numpy(copy=True); X -= X.mean(axis=0)
    y = data.y.to_numpy().copy(); y -= y.mean()
    prior, _ = log_model_prior_function(("betabinomial", 1., 1.), 5)
    args = dict(df_resid=79, g=80., log_model_prior=prior, progress_every_s=0)
    whole = enumerate_models_gpu(X, y, **args)
    path = tmp_path / "checkpoint.npz"
    part = enumerate_models_gpu(X, y, **args, checkpoint_path=path, stop_after_chunks=3)
    assert part["interrupted"] and part["models_done"] == 16
    resumed = enumerate_models_gpu(X, y, **args, checkpoint_path=path, resume=True)
    for field in ("pip", "coef_mean", "coef_sd", "size_distribution", "log_normalizer"):
        np.testing.assert_array_equal(resumed[field], whole[field])
    assert resumed["top_models"] == whole["top_models"]


@cuda
def test_p32_only_null_and_singletons(tmp_path):
    data, names = frame(32)
    X = data[names].to_numpy(copy=True); X -= X.mean(axis=0)
    y = data.y.to_numpy().copy(); y -= y.mean()
    prior, _ = log_model_prior_function(("betabinomial", 1., 1.), 32)
    path = tmp_path / "p32_bounded.npz"
    out = enumerate_models_gpu(X, y, df_resid=79, g=1024., log_model_prior=prior,
                               checkpoint_path=path, stop_after_chunks=2, progress_every_s=0)
    assert out["interrupted"] and out["models_done"] == 33
    assert out["n_models_expected"] == 4294967296
    with np.load(path) as state:
        assert state["top_masks"].dtype == np.int64
        assert int(state["models_done"]) == 33
        validate_cursor(32, int(state["models_done"]), int(state["next_k"]), int(state["next_rank"]))
