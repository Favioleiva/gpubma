"""Structured & Strong-Heredity GPU BMA Enumerator.

Evaluates partitioned model universes where structural polynomial/interaction terms
follow heredity constraints (e.g., Translog production functions) combined with
unconstrained subsets of linear auxiliary controls.

Precision: IEEE 754 Float64 Double Precision on NVIDIA CUDA GPU.
"""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from gpubma.gpu.batch_scorer import torch_cuda_available, gpu_hardware_info
from gpubma.gpu.enumerator import binomial_table
from gpubma.result import BMAResult


def get_translog_heredity_masks() -> tuple[dict[int, list[list[int]]], int]:
    """Generates all 1,337 valid factor masks satisfying strong heredity for a 4-factor Translog.
    
    Factors order:
      0: P, 1: E, 2: L, 3: R
      4: P^2, 5: E^2, 6: L^2, 7: R^2
      8: PE, 9: PL, 10: PR, 11: EL, 12: ER, 13: LR
    """
    factor_masks_by_size = {k: [] for k in range(15)}
    valid_count = 0
    for mask in range(1 << 14):
        P  = (mask >> 0) & 1; E  = (mask >> 1) & 1; L  = (mask >> 2) & 1; R  = (mask >> 3) & 1
        P2 = (mask >> 4) & 1; E2 = (mask >> 5) & 1; L2 = (mask >> 6) & 1; R2 = (mask >> 7) & 1
        PE = (mask >> 8) & 1; PL = (mask >> 9) & 1; PR = (mask >> 10) & 1
        EL = (mask >> 11) & 1; ER = (mask >> 12) & 1; LR = (mask >> 13) & 1

        if P2 and not P: continue
        if E2 and not E: continue
        if L2 and not L: continue
        if R2 and not R: continue

        if PE and not (P and E): continue
        if PL and not (P and L): continue
        if PR and not (P and R): continue
        if EL and not (E and L): continue
        if ER and not (E and R): continue
        if LR and not (L and R): continue

        indices = [bit for bit in range(14) if (mask & (1 << bit))]
        factor_masks_by_size[len(indices)].append(indices)
        valid_count += 1
    return factor_masks_by_size, valid_count


def enumerate_structured_models_gpu(
    X: np.ndarray,
    y: np.ndarray,
    *,
    df_resid: int,
    g: float,
    log_model_prior_by_size: np.ndarray | list[float],
    structural_masks_by_size: dict[int, list[list[int]]],
    n_structural: int = 14,
    n_controls: int = 18,
    predictor_names: list[str] | None = None,
    outcome_name: str = "y",
    device: str = "cuda",
    batch_size: int = 32768,
    top_k: int = 100,
) -> BMAResult:
    """Exhaustively score a structured/heredity model space on one GPU in float64.
    
    Evaluates all configurations formed by combining structural masks with all 2^n_controls
    subsets of linear controls.
    """
    ok, msg = torch_cuda_available()
    if not ok:
        raise RuntimeError(f"CUDA unavailable: {msg}")

    t_start = time.perf_counter()
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n, p = X.shape
    assert p == n_structural + n_controls, f"Expected {n_structural + n_controls} predictors, got {p}"
    
    if predictor_names is None:
        predictor_names = [f"x{i}" for i in range(p)]

    # Sufficient statistics
    t0_gram = time.perf_counter()
    Zxx_np = X.T @ X
    Zxy_np = X.T @ y
    yty_val = float(y @ y)
    t_gram = time.perf_counter() - t0_gram

    dev = torch.device(device)
    XtX_gpu = torch.from_numpy(Zxx_np).to(device=dev, dtype=torch.float64)
    Xty_gpu = torch.from_numpy(Zxy_np).to(device=dev, dtype=torch.float64)

    factor_g = g / (1.0 + g)
    log_1p_g = math.log(1.0 + g)
    df_factor = float(df_resid) / 2.0

    binom_total_np = binomial_table(p)
    log_prior_by_k = np.asarray(log_model_prior_by_size, dtype=np.float64)

    # Precompute control index tensors on device
    control_combos_by_k = {}
    for k_c in range(n_controls + 1):
        if k_c == 0:
            control_combos_by_k[0] = torch.empty((1, 0), dtype=torch.int64, device=dev)
        else:
            combos = list(itertools.combinations(range(n_controls), k_c))
            control_combos_by_k[k_c] = torch.tensor(combos, dtype=torch.int64, device=dev) + n_structural

    # Pass 1: Global Maximum Score Search
    t0_p1 = time.perf_counter()
    global_max_score = -1e300
    n_eval = 0

    for k_f, masks_f in structural_masks_by_size.items():
        if not masks_f: continue
        for f_idx in masks_f:
            f_tensor = torch.tensor(f_idx, dtype=torch.int64, device=dev) if k_f > 0 else torch.empty((0,), dtype=torch.int64, device=dev)
            for k_c in range(n_controls + 1):
                k_tot = k_f + k_c
                if k_tot == 0: continue
                c_tensor = control_combos_by_k[k_c]
                n_c = c_tensor.shape[0]
                log_prior_m = float(log_prior_by_k[k_tot] - math.log(float(binom_total_np[p, k_tot])))
                penalty = -0.5 * k_tot * log_1p_g + log_prior_m

                for s in range(0, n_c, batch_size):
                    e = min(s + batch_size, n_c)
                    B = e - s
                    n_eval += B
                    combos_c = c_tensor[s:e]
                    if k_f > 0 and k_c > 0:
                        combos = torch.cat([f_tensor.unsqueeze(0).expand(B, k_f), combos_c], dim=1)
                    elif k_f > 0:
                        combos = f_tensor.unsqueeze(0).expand(B, k_f)
                    else:
                        combos = combos_c

                    b_idx = combos.unsqueeze(2).expand(B, k_tot, k_tot)
                    b_col = combos.unsqueeze(1).expand(B, k_tot, k_tot)
                    A = XtX_gpu[b_idx, b_col]
                    b = Xty_gpu[combos].unsqueeze(2)

                    L = torch.linalg.cholesky(A)
                    v = torch.linalg.solve_triangular(L, b, upper=False)
                    quad = torch.sum(v.squeeze(2)**2, dim=1)
                    sse = torch.clamp(yty_val - factor_g * quad, min=1e-13)
                    score = -df_factor * torch.log(sse) + penalty
                    m = float(score.max().item())
                    if m > global_max_score: global_max_score = m

    t_p1 = time.perf_counter() - t0_p1

    # Pass 2: Exact Posterior Accumulation
    t0_p2 = time.perf_counter()
    pip_accum = torch.zeros(p, dtype=torch.float64, device=dev)
    b_mean_accum = torch.zeros(p, dtype=torch.float64, device=dev)
    b_sq_accum = torch.zeros(p, dtype=torch.float64, device=dev)
    msize_accum = torch.zeros(p + 1, dtype=torch.float64, device=dev)

    sum_w = 0.0
    cobb_douglas_mass = 0.0
    translog_terms_mass = 0.0
    top_records = []

    for k_f, masks_f in structural_masks_by_size.items():
        if not masks_f: continue
        for f_idx in masks_f:
            f_tensor = torch.tensor(f_idx, dtype=torch.int64, device=dev) if k_f > 0 else torch.empty((0,), dtype=torch.int64, device=dev)
            for k_c in range(n_controls + 1):
                k_tot = k_f + k_c
                if k_tot == 0: continue
                c_tensor = control_combos_by_k[k_c]
                n_c = c_tensor.shape[0]
                log_prior_m = float(log_prior_by_k[k_tot] - math.log(float(binom_total_np[p, k_tot])))
                penalty = -0.5 * k_tot * log_1p_g + log_prior_m

                for s in range(0, n_c, batch_size):
                    e = min(s + batch_size, n_c)
                    B = e - s
                    combos_c = c_tensor[s:e]
                    if k_f > 0 and k_c > 0:
                        combos = torch.cat([f_tensor.unsqueeze(0).expand(B, k_f), combos_c], dim=1)
                    elif k_f > 0:
                        combos = f_tensor.unsqueeze(0).expand(B, k_f)
                    else:
                        combos = combos_c

                    b_idx = combos.unsqueeze(2).expand(B, k_tot, k_tot)
                    b_col = combos.unsqueeze(1).expand(B, k_tot, k_tot)
                    A = XtX_gpu[b_idx, b_col]
                    b = Xty_gpu[combos].unsqueeze(2)

                    L = torch.linalg.cholesky(A)
                    v = torch.linalg.solve_triangular(L, b, upper=False)
                    quad = torch.sum(v.squeeze(2)**2, dim=1)
                    sse = torch.clamp(yty_val - factor_g * quad, min=1e-13)
                    score = -df_factor * torch.log(sse) + penalty
                    diff = score - global_max_score
                    w = torch.exp(diff)

                    b_w = float(w.sum().item())
                    sum_w += b_w
                    msize_accum[k_tot] += b_w

                    if (k_f <= 4) and (len(f_idx) == 0 or all(idx < 4 for idx in f_idx)):
                        cobb_douglas_mass += b_w

                    n_2nd = sum(1 for idx in f_idx if idx >= 4)
                    translog_terms_mass += n_2nd * b_w

                    for j in range(k_tot):
                        pip_accum.scatter_add_(0, combos[:, j], w)

                    beta_hat = factor_g * torch.linalg.solve_triangular(L.transpose(1, 2), v, upper=True).squeeze(2)
                    w_exp = w.unsqueeze(1)
                    for j in range(k_tot):
                        b_mean_accum.scatter_add_(0, combos[:, j], (w_exp * beta_hat)[:, j])
                        b_sq_accum.scatter_add_(0, combos[:, j], (w_exp * (beta_hat**2))[:, j])

                    if torch.any(w > 1e-4):
                        for h_idx in torch.where(w > 1e-4)[0]:
                            top_records.append({
                                "mask": int(sum(1 << int(idx) for idx in combos[h_idx].tolist())),
                                "included": combos[h_idx].tolist(),
                                "size": k_tot,
                                "pmp": float(w[h_idx].item()),
                                "log_score": float(score[h_idx].item()),
                            })

    t_p2 = time.perf_counter() - t0_p2
    total_t = t_p1 + t_p2

    pip_f = (pip_accum / sum_w).cpu().numpy()
    b_mean_u = (b_mean_accum / sum_w).cpu().numpy()
    b_sq_u = (b_sq_accum / sum_w).cpu().numpy()
    b_mean_c = np.divide(b_mean_u, pip_f, out=np.zeros_like(b_mean_u), where=pip_f > 1e-12)
    b_sq_c = np.divide(b_sq_u, pip_f, out=np.zeros_like(b_sq_u), where=pip_f > 1e-12)
    b_sd_c = np.sqrt(np.maximum(0.0, b_sq_c - b_mean_c**2))

    msize_dist = (msize_accum / sum_w).cpu().numpy()
    e_k = float(np.sum(np.arange(p + 1) * msize_dist))
    log_marg_lik = global_max_score + math.log(sum_w)

    for rec in top_records:
        rec["pmp"] /= sum_w
    top_records.sort(key=lambda r: r["pmp"], reverse=True)

    class FixedEffectsSpec:
        def describe(self): return f"FWL Absorbed (df_resid={df_resid})"

    class GSpec:
        def __init__(self, g_val): self.g_val = g_val
        def describe(self): return f"Zellner g = {self.g_val:,.0f}"

    runtime_dict = {
        "time_gram_s": t_gram,
        "time_pass1_s": t_p1,
        "time_pass2_s": t_p2,
        "time_total_s": total_t,
        "throughput_models_per_s": n_eval / total_t,
    }

    hw_info = gpu_hardware_info()

    return BMAResult(
        outcome=outcome_name,
        predictor_names=predictor_names,
        n_obs=n,
        n_predictors=p,
        n_models_expected=n_eval,
        n_models_evaluated=n_eval,
        df_resid=df_resid,
        g_spec=GSpec(g),
        model_prior_description=f"Beta-Binomial(1, 1) over {p} predictors",
        fixed_effects_info={"fe_method": "fwl_alternating", "df_resid": df_resid},
        backend="gpu",
        precision="float64",
        pip=pip_f,
        coef_mean=b_mean_c,
        coef_sd=b_sd_c,
        mean_model_size=e_k,
        size_distribution=msize_dist,
        _top_models=top_records[:top_k],
        runtime=runtime_dict,
        hardware=hw_info,
        notes=[
            f"Pure Cobb-Douglas Mass: {cobb_douglas_mass / sum_w:.6f}",
            f"Expected Translog Second-Order Terms: {translog_terms_mass / sum_w:.4f}",
            f"Integrated Log Marginal Likelihood: {log_marg_lik:.6f}",
        ],
    )
