"""GPU feasibility check (Phase 1, section 10 of the task spec).

Steps: detect GPU -> transfer sufficient statistics -> genuine float64
operation -> synchronize -> cold/warm timing -> compare with CPU -> verdict.

Run:  python -m gpubma.gpu.feasibility
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gpubma.cpu.enumeration import enumerate_models
from gpubma.gpu.batch_scorer import gpu_hardware_info, gpu_score_all_models, torch_cuda_available
from gpubma.priors.model_priors import log_model_prior_function


def run_feasibility(n_obs: int = 1000, p: int = 8, seed: int = 20260715,
                    warm_repeats: int = 5) -> dict:
    report = {"hardware": gpu_hardware_info()}
    ok, msg = torch_cuda_available()
    report["cuda_available"] = ok
    if not ok:
        report["verdict"] = f"NOT SUITABLE for GPU kernel development here: {msg}"
        return report
    import torch

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_obs, p))
    beta = np.zeros(p); beta[:3] = (1.0, -0.5, 0.25)
    y = X @ beta + rng.standard_normal(n_obs)
    df_resid = n_obs - 1
    Xc = X - X.mean(axis=0); yc = y - y.mean()
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), p)
    g = float(max(n_obs, p * p))

    # 1) basic float64 op validated against CPU
    a = torch.from_numpy(rng.standard_normal((256, 256))).cuda()
    bmat = torch.from_numpy(rng.standard_normal((256, 256))).cuda()
    prod_gpu = (a @ bmat).sum(); torch.cuda.synchronize()
    prod_cpu = float((a.cpu().numpy() @ bmat.cpu().numpy()).sum())
    fp64_diff = abs(float(prod_gpu.item()) - prod_cpu)
    report["float64_matmul"] = {
        "dtype": str(a.dtype), "gpu": float(prod_gpu.item()), "cpu": prod_cpu,
        "abs_diff": fp64_diff, "pass": bool(fp64_diff < 1e-8),
    }

    # 2) CPU reference scores
    cpu = enumerate_models(Xc, yc, df_resid=df_resid, g=g, log_model_prior=log_prior,
                           compute_coefficients=False)

    # 3) GPU batch scoring: cold then warm
    t0 = time.perf_counter()
    gpu_cold = gpu_score_all_models(Xc, yc, df_resid=df_resid, g=g, log_model_prior=log_prior)
    cold_s = time.perf_counter() - t0
    warm_times = []
    for _ in range(warm_repeats):
        t = time.perf_counter()
        gpu_warm = gpu_score_all_models(Xc, yc, df_resid=df_resid, g=g, log_model_prior=log_prior)
        warm_times.append(time.perf_counter() - t)
    max_diff = float(np.max(np.abs(gpu_warm["log_scores"] - cpu["log_scores"])))
    report["model_scoring"] = {
        "n_obs": n_obs, "predictors": p, "n_models": 1 << p, "g": g,
        "cold_total_s": cold_s,
        "warm_total_s_median": float(np.median(warm_times)),
        "warm_total_s_min": float(np.min(warm_times)),
        "warm_total_s_max": float(np.max(warm_times)),
        "warm_repeats": warm_repeats,
        "gpu_vs_cpu_max_abs_logscore_diff": max_diff,
        "scores_match_cpu": bool(max_diff < 1e-9),
        "peak_gpu_memory_bytes": gpu_warm["runtime"]["peak_gpu_memory_bytes"],
    }

    suitable = report["float64_matmul"]["pass"] and report["model_scoring"]["scores_match_cpu"]
    report["verdict"] = (
        "SUITABLE for later CUDA kernel development: genuine float64 execution "
        "verified and batched model scores match the CPU reference."
        if suitable
        else "NOT SUITABLE: float64 execution or CPU agreement failed; investigate before Phase 2."
    )
    return report


def main() -> None:
    report = run_feasibility()
    out = Path(__file__).resolve().parents[3] / "reports" / "gpu_feasibility.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
