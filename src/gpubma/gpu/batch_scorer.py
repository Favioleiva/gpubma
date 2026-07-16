"""GPU feasibility baseline: batched float64 scoring of all 2^p models.

This is NOT the production enumerator. It scores a modest model space
(p <= 16) by grouping models by size and running batched Cholesky solves on
the sufficient statistics with PyTorch. Its purpose is to demonstrate genuine
float64 GPU execution, validate results against the CPU reference, and give
an honest measured models/second figure for this hardware.
"""

from __future__ import annotations

import itertools
import time

import numpy as np

_MAX_GPU_PREDICTORS = 16


def torch_cuda_available() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False (no usable CUDA device)"
    return True, f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}"


def gpu_hardware_info() -> dict:
    ok, msg = torch_cuda_available()
    if not ok:
        return {"cuda": False, "reason": msg}
    import torch

    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "multiprocessors": props.multi_processor_count,
        "total_vram_bytes": props.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
    }


def gpu_score_all_models(X, y, *, df_resid: int, g: float, log_model_prior,
                         device: str = "cuda", tss_norm: float | None = None,
                         k_always: int = 0):
    """Score all 2^p models on the GPU in float64. Returns dict with
    ``log_scores`` ordered by mask and a runtime breakdown (cold vs warm
    timing is the caller's responsibility; this reports phase timings)."""
    ok, msg = torch_cuda_available()
    if not ok:
        raise RuntimeError(f"CUDA unavailable: {msg}")
    import torch

    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n, p = X.shape
    if p > _MAX_GPU_PREDICTORS:
        raise ValueError(
            f"feasibility scorer caps p at {_MAX_GPU_PREDICTORS} (got {p}); "
            "the production enumerator is out of scope in Phase 1"
        )
    n_models = 1 << p

    t0 = time.perf_counter()
    Zxx_np = X.T @ X
    Zxy_np = X.T @ y
    tss = float(y @ y)
    if tss_norm is None:
        tss_norm = tss
        if k_always:
            raise ValueError("k_always requires tss_norm (shrink convention)")
    t_suff = time.perf_counter() - t0

    t1 = time.perf_counter()
    dev = torch.device(device)
    Zxx = torch.from_numpy(Zxx_np).to(dev)
    Zxy = torch.from_numpy(Zxy_np).to(dev)
    torch.cuda.synchronize()
    t_transfer = time.perf_counter() - t1

    log1pg = float(np.log1p(g))
    log_scores = np.empty(n_models, dtype=np.float64)
    log_scores[0] = (0.5 * (df_resid - k_always) * log1pg
                     - 0.5 * df_resid * np.log1p(g * max(tss / tss_norm, np.finfo(np.float64).tiny))
                     + log_model_prior(0))

    t2 = time.perf_counter()
    tiny = float(np.finfo(np.float64).tiny)
    for k in range(1, p + 1):
        combos = np.fromiter(
            itertools.chain.from_iterable(itertools.combinations(range(p), k)),
            dtype=np.int64,
        ).reshape(-1, k)
        idx = torch.from_numpy(combos).to(dev)  # (m, k)
        Z = Zxx[idx.unsqueeze(2), idx.unsqueeze(1)]          # (m, k, k)
        b = Zxy[idx].unsqueeze(-1)                            # (m, k, 1)
        L = torch.linalg.cholesky(Z)
        u = torch.linalg.solve_triangular(L, b, upper=False)  # (m, k, 1)
        ess = (u.squeeze(-1) ** 2).sum(dim=1)                 # (m,)
        one_minus_r2 = torch.clamp((tss - ess) / tss_norm, min=tiny)
        log_ml = (0.5 * (df_resid - k - k_always) * log1pg
                  - 0.5 * df_resid * torch.log1p(g * one_minus_r2))
        scores_k = (log_ml + log_model_prior(k)).cpu().numpy()
        # map combos back to mask positions
        masks_k = np.zeros(len(combos), dtype=np.int64)
        for col in range(k):
            masks_k |= np.int64(1) << combos[:, col]
        log_scores[masks_k] = scores_k
    torch.cuda.synchronize()
    t_score = time.perf_counter() - t2

    peak_mem = int(torch.cuda.max_memory_allocated(dev))
    total = time.perf_counter() - t0
    return {
        "log_scores": log_scores,
        "n_models": n_models,
        "runtime": {
            "backend": "gpu",
            "precision": "float64",
            "sufficient_statistics_s": t_suff,
            "transfer_s": t_transfer,
            "scoring_s": t_score,
            "total_s": total,
            "models_per_second_scoring": n_models / t_score if t_score > 0 else float("nan"),
            "peak_gpu_memory_bytes": peak_mem,
        },
    }
