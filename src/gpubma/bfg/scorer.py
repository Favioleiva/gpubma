"""High-throughput float64 PyTorch GPU/CPU bitmask scorer with evaluation caching."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from gpubma.gpu.batch_scorer import torch_cuda_available
from gpubma.priors.model_priors import log_model_prior_function


# --------------------------------------------------------------------------
# Bitwise & Model-ID Representation Utilities
# --------------------------------------------------------------------------

def count_set_bits(model_id: int) -> int:
    """Return the number of active predictors (model size k) in integer bitmask."""
    return int(model_id).bit_count()


def hamming_distance(m1: int, m2: int) -> int:
    """Compute Hamming distance between two models: d_H(m1, m2) = |m1 ^ m2|."""
    return int(m1 ^ m2).bit_count()


def is_subset(m1: int, m2: int) -> bool:
    """Return True if model m1 is a subset of m2 (m1 <= m2 in Boolean lattice)."""
    return (m1 & m2) == m1


def model_id_to_indices(model_id: int, p: int) -> List[int]:
    """Extract zero-indexed active variable indices from a model bitmask."""
    return [j for j in range(p) if (model_id & (1 << j))]


def indices_to_model_id(indices: Sequence[int]) -> int:
    """Pack variable indices into a unique integer bitmask."""
    mask = 0
    for idx in indices:
        mask |= (1 << int(idx))
    return mask


def model_id_to_vars(model_id: int, candidate_names: Sequence[str]) -> List[str]:
    """Convert integer bitmask to list of candidate variable names."""
    p = len(candidate_names)
    return [candidate_names[j] for j in range(p) if (model_id & (1 << j))]


def vars_to_model_id(var_list: Sequence[str], candidate_names: Sequence[str]) -> int:
    """Convert list of candidate variable names to integer bitmask."""
    name_map = {name: idx for idx, name in enumerate(candidate_names)}
    mask = 0
    for v in var_list:
        if v in name_map:
            mask |= (1 << name_map[v])
        else:
            raise KeyError(f"Variable '{v}' not found in candidate names.")
    return mask


def get_immediate_parents(model_id: int, p: int) -> List[int]:
    """Return all immediate parent model IDs (removing one active variable)."""
    parents = []
    for j in range(p):
        bit = 1 << j
        if model_id & bit:
            parents.append(model_id ^ bit)
    return parents


def get_immediate_children(model_id: int, p: int) -> List[int]:
    """Return all immediate child model IDs (adding one inactive variable)."""
    children = []
    for j in range(p):
        bit = 1 << j
        if not (model_id & bit):
            children.append(model_id | bit)
    return children


# --------------------------------------------------------------------------
# Hardware Detection
# --------------------------------------------------------------------------

def detect_hardware(requested_device: str = "cuda") -> Dict[str, Any]:
    """Detect available hardware capabilities and select execution device.

    Returns
    -------
    dict with device info: device_type, device_name, vram_total_gb, vram_free_gb, cuda_available.
    """
    cuda_ok, cuda_reason = torch_cuda_available()
    info = {
        "cuda_available": cuda_ok,
        "cuda_reason": cuda_reason,
        "device_type": "cpu",
        "device_name": "CPU",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "float64_supported": True,
    }

    if requested_device.startswith("cuda"):
        if cuda_ok:
            dev_idx = 0
            if ":" in requested_device:
                try:
                    dev_idx = int(requested_device.split(":")[-1])
                except ValueError:
                    dev_idx = 0
            dev_name = torch.cuda.get_device_name(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            total_mem = props.total_memory / (1024 ** 3)
            free_mem = (props.total_memory - torch.cuda.memory_allocated(dev_idx)) / (1024 ** 3)
            info.update({
                "device_type": f"cuda:{dev_idx}",
                "device_name": dev_name,
                "vram_total_gb": round(total_mem, 2),
                "vram_free_gb": round(free_mem, 2),
            })
        else:
            # If explicit cuda requested but unavailable
            info["device_type"] = "cpu"
    else:
        info["device_type"] = "cpu"

    return info


# --------------------------------------------------------------------------
# High-Throughput Scorer with Transparent Global Evaluation Cache
# --------------------------------------------------------------------------

class BFGScorer:
    """Float64 PyTorch GPU/CPU bitmask model scorer with transparent evaluation cache.

    Guarantees:
    - Exact float64 reference scoring matching canonical GPUBMA backend.
    - Zero redundant evaluations via transparent in-memory evaluation cache.
    - Auditable call and cache-hit counters.
    - Safe execution across GPU (batched Cholesky) and CPU fallback.
    """

    def __init__(
        self,
        X_r: np.ndarray,
        y_r: np.ndarray,
        df_resid: int,
        g: float,
        log_model_prior: Callable[[int], float],
        prior_description: str = "",
        tss_norm: Optional[float] = None,
        k_always: int = 0,
        device: str = "cuda",
        max_eval_budget: Optional[int] = None,
    ):
        self.n, self.p = X_r.shape
        self.k_always = int(k_always)
        self.df = float(df_resid)
        self.tss = float(y_r @ y_r)
        self.tss_norm = float(tss_norm) if tss_norm is not None else self.tss
        self.g = float(g)
        self.log1pg = float(np.log1p(self.g))
        self.shrink = float(self.g / (1.0 + self.g))
        self.tiny = float(np.finfo(np.float64).tiny)
        self.prior_description = prior_description
        self.max_eval_budget: Optional[int] = int(max_eval_budget) if max_eval_budget is not None else None

        # Determine device
        hw = detect_hardware(requested_device=device)
        if device.startswith("cuda") and hw["cuda_available"]:
            self.device = torch.device(hw["device_type"])
            self.device_name = hw["device_name"]
            self.backend = "gpu"
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU"
            self.backend = "cpu"

        # Precompute sufficient statistics on device in strict float64
        Zxx_np = np.ascontiguousarray(X_r.T @ X_r, dtype=np.float64)
        Zxy_np = np.ascontiguousarray(X_r.T @ y_r, dtype=np.float64)
        self.Zxx = torch.from_numpy(Zxx_np).to(self.device)
        self.Zxy = torch.from_numpy(Zxy_np).to(self.device)
        self.Zxx_cpu = Zxx_np
        self.Zxy_cpu = Zxy_np

        # Precompute prior by model size k in float64
        lp_size = np.array([log_model_prior(k) for k in range(self.p + 1)], dtype=np.float64)
        self.lp_size = torch.from_numpy(lp_size).to(self.device)
        self.lp_size_cpu = lp_size

        # Evaluation Cache & Diagnostics
        self.cache: Dict[int, float] = {}
        self.eval_order: List[int] = []
        self.n_eval_calls: int = 0
        self.n_cache_hits: int = 0

    def reset_cache(self) -> None:
        """Clear the evaluation cache and reset counters."""
        self.cache.clear()
        self.eval_order.clear()
        self.n_eval_calls = 0
        self.n_cache_hits = 0

    @property
    def n_unique_evaluated(self) -> int:
        """Total number of distinct models evaluated and cached."""
        return len(self.cache)

    def score_single(self, model_id: int) -> float:
        """Score a single model bitmask, returning its canonical log score."""
        self.n_eval_calls += 1
        if model_id in self.cache:
            self.n_cache_hits += 1
            return self.cache[model_id]

        if self.max_eval_budget is not None and len(self.cache) >= self.max_eval_budget:
            return float("-inf")

        scores = self.score_batch([model_id])
        return scores.get(model_id, float("-inf"))

    def score_batch(self, model_ids: Sequence[int], chunk_size: int = 65536) -> Dict[int, float]:
        """Score a batch of model bitmasks, evaluating only previously unseen models.

        Parameters
        ----------
        model_ids : Sequence[int]
            List or array of integer model bitmasks.
        chunk_size : int, default=65536
            Maximum models processed per GPU sub-batch to respect VRAM limits.

        Returns
        -------
        Dict[int, float]
            Mapping from model_id to canonical log score log z_M.
        """
        n_incoming = len(model_ids)
        self.n_eval_calls += n_incoming
        
        # Check cache
        already_cached = [m for m in model_ids if m in self.cache]
        unseen = [m for m in model_ids if m not in self.cache]
        unique_unseen = list(dict.fromkeys(unseen))
        intra_batch_dups = len(unseen) - len(unique_unseen)
        self.n_cache_hits += (len(already_cached) + intra_batch_dups)

        if not unique_unseen:
            return {m: self.cache[m] for m in model_ids}

        # Enforce hard evaluation budget cap
        if self.max_eval_budget is not None:
            rem = max(0, self.max_eval_budget - len(self.cache))
            if len(unique_unseen) > rem:
                unique_unseen = unique_unseen[:rem]
            if not unique_unseen:
                return {m: self.cache.get(m, float("-inf")) for m in model_ids}

        # Group unique unseen by model size k
        by_k: Dict[int, List[int]] = {}
        for m in unique_unseen:
            k = count_set_bits(m)
            if k not in by_k:
                by_k[k] = []
            by_k[k].append(m)

        # Evaluate by lattice size
        for k, mlist in by_k.items():
            if k == 0:
                one_minus_r2 = max(self.tss / self.tss_norm, self.tiny)
                s0 = float(
                    0.5 * (self.df - self.k_always) * self.log1pg
                    - 0.5 * self.df * math.log1p(self.g * one_minus_r2)
                    + self.lp_size_cpu[0]
                )
                for m in mlist:
                    self.cache[m] = s0
                    self.eval_order.append(m)
                continue

            if self.backend == "gpu":
                for c_start in range(0, len(mlist), chunk_size):
                    chunk = mlist[c_start : c_start + chunk_size]
                    b_size = len(chunk)
                    
                    if self.p <= 64:
                        # Vectorized NumPy bit unpacking
                        arr = np.array(chunk, dtype=np.int64)
                        bool_mask = ((arr[:, None] >> np.arange(self.p, dtype=np.int64)) & 1) == 1
                        idx_arr = np.nonzero(bool_mask)[1].reshape(b_size, k)
                    else:
                        idx_arr = np.zeros((b_size, k), dtype=np.int64)
                        for i, m in enumerate(chunk):
                            idx_arr[i] = [j for j in range(self.p) if (m & (1 << j))]

                    idx = torch.from_numpy(idx_arr).to(self.device)
                    # Batch slice submatrices: (B, k, k) and (B, k, 1)
                    Z = self.Zxx[idx.unsqueeze(2), idx.unsqueeze(1)]
                    b = self.Zxy[idx].unsqueeze(-1)

                    L = torch.linalg.cholesky(Z)
                    u = torch.linalg.solve_triangular(L, b, upper=False)
                    ess = (u.squeeze(-1) ** 2).sum(dim=1)
                    one_minus_r2 = torch.clamp((self.tss - ess) / self.tss_norm, min=self.tiny)

                    scores = (
                        0.5 * (self.df - k - self.k_always) * self.log1pg
                        - 0.5 * self.df * torch.log1p(self.g * one_minus_r2)
                        + self.lp_size[k]
                    )
                    scores_list = scores.cpu().tolist()
                    for m, s in zip(chunk, scores_list):
                        self.cache[m] = s
                    self.eval_order.extend(chunk)
            else:
                # CPU Cholesky fallback
                from scipy.linalg import cho_factor, cho_solve
                for m in mlist:
                    idx = np.array([j for j in range(self.p) if (m & (1 << j))], dtype=np.intp)
                    Z_sub = self.Zxx_cpu[np.ix_(idx, idx)]
                    b_sub = self.Zxy_cpu[idx]
                    c, low = cho_factor(Z_sub, lower=True, check_finite=False)
                    w = cho_solve((c, low), b_sub, check_finite=False)
                    ess = float(b_sub @ w)
                    one_minus_r2 = max((self.tss - ess) / self.tss_norm, self.tiny)
                    s = (
                        0.5 * (self.df - k - self.k_always) * self.log1pg
                        - 0.5 * self.df * math.log1p(self.g * one_minus_r2)
                        + self.lp_size_cpu[k]
                    )
                    self.cache[m] = float(s)
                    self.eval_order.append(m)

        return {m: self.cache.get(m, float("-inf")) for m in model_ids}

    def compute_model_coefficients(self, model_id: int) -> Dict[str, Any]:
        """Compute posterior coefficient mean and standard deviation conditional on model_id.

        Returns
        -------
        Dict with 'indices', 'coef_mean', 'coef_sd', 'ess', 'r2'.
        """
        from scipy.linalg import cho_factor, cho_solve
        k = count_set_bits(model_id)
        if k == 0:
            return {
                "indices": [],
                "coef_mean": np.array([], dtype=np.float64),
                "coef_sd": np.array([], dtype=np.float64),
                "ess": 0.0,
                "r2": 0.0,
            }

        idx = np.array(model_id_to_indices(model_id, self.p), dtype=np.intp)
        Z_sub = self.Zxx_cpu[np.ix_(idx, idx)]
        b_sub = self.Zxy_cpu[idx]
        c, low = cho_factor(Z_sub, lower=True, check_finite=False)
        beta_hat = cho_solve((c, low), b_sub, check_finite=False)
        zinv_diag = np.diag(cho_solve((c, low), np.eye(len(idx)), check_finite=False))
        ess = float(b_sub @ beta_hat)

        # Stata shrink convention moments
        ess_always = self.tss_norm - self.tss
        e_sigma2 = (self.tss_norm - self.shrink * (ess_always + ess)) / max(self.df - 2.0, 1.0)
        cond_mean = self.shrink * beta_hat
        cond_var = e_sigma2 * self.shrink * zinv_diag
        cond_sd = np.sqrt(np.maximum(cond_var, 0.0))

        return {
            "indices": idx.tolist(),
            "coef_mean": cond_mean,
            "coef_sd": cond_sd,
            "ess": ess,
            "r2": ess / self.tss_norm,
        }
