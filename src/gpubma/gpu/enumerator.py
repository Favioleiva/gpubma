"""Bounded-memory exhaustive single-GPU BMA enumerator (Phase 2).

Architecture per docs/ADR_0001_GPU_ENUMERATOR.md (streamed direct batching):

- models are enumerated size-by-size; within size k the C(p, k)
  combinations are walked in colexicographic rank order in chunks, so
  every model is visited exactly once and a chunk is fully determined by
  ``(k, first_rank, chunk_size)`` — stateless, checkpointable, and (later)
  shardable by rank range;
- ranks are unranked to predictor-index tensors ON DEVICE via the
  combinatorial number system with an exact int64 binomial table;
- each chunk gathers (B, k, k) submatrices of the precomputed residualized
  Gram and runs batched Cholesky + triangular solves — the verified
  Stata-compatible FWL formulation (docs/FWL_BLOCK_FORMULATION.md); the
  statistical formula is identical to the CPU oracle's;
- posterior quantities are accumulated with a numerically stable running
  rescale (streaming log-sum-exp): PIP numerators, coefficient moment
  numerators, model-size mass, the normalizer, and a running top-K —
  all in float64 on device, O(p) state, never 2^p storage;
- deterministic checkpoint/resume: accumulator state + (k, next_rank) are
  written atomically; resume verifies a SHA-256 of the sufficient
  statistics and prior configuration before continuing.

float64 only (CLAUDE.md rule 6). Enumeration only — no sampling (rule 4).
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path

import numpy as np

CHECKPOINT_VERSION = 1


# --------------------------------------------------------------------------
# exact combinatorics
# --------------------------------------------------------------------------

def binomial_table(p: int) -> np.ndarray:
    """(p+1) x (p+1) table of C(n, j) as exact int64 (p <= 62 is safe;
    we use p <= 30 where C(30, 15) = 155,117,520)."""
    C = np.zeros((p + 1, p + 1), dtype=np.int64)
    C[:, 0] = 1
    for n in range(1, p + 1):
        for j in range(1, n + 1):
            C[n, j] = C[n - 1, j - 1] + C[n - 1, j]
    return C


def unrank_combinations(ranks, k: int, binom, torch):
    """Unrank colex ranks -> (B, k) ascending predictor indices, on device.

    Combinatorial number system: for rank r, the largest index c_k is the
    greatest c with C(c, k) <= r; recurse on r - C(c_k, k) with k - 1.
    Column C(:, j) of the binomial table is non-decreasing, so each step is
    a batched searchsorted. Exact in int64 throughout.
    """
    B = ranks.shape[0]
    out = torch.empty((B, k), dtype=torch.int64, device=ranks.device)
    r = ranks.clone()
    for j in range(k, 0, -1):
        col = binom[:, j].contiguous()  # C(c, j), non-decreasing in c
        c = torch.searchsorted(col, r, right=True) - 1
        out[:, j - 1] = c
        r = r - col[c]
    return out  # ascending along dim 1 by construction


# --------------------------------------------------------------------------
# checkpoint helpers
# --------------------------------------------------------------------------

def _config_digest(Zxx, Zxy, tss, tss_norm, df_resid, g, k_always,
                   log_prior_by_size, top_k) -> str:
    h = hashlib.sha256()
    for arr in (Zxx, Zxy, log_prior_by_size):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(np.array([tss, tss_norm, float(df_resid), float(g),
                       float(k_always), float(top_k)], dtype=np.float64).tobytes())
    return h.hexdigest()


def _save_checkpoint(path: Path, state: dict) -> None:
    # np.savez appends .npz when missing, so keep the temp name canonical
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **state)
    os.replace(str(tmp), str(path))


def _load_checkpoint(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


# --------------------------------------------------------------------------
# the enumerator
# --------------------------------------------------------------------------

def enumerate_models_gpu(
    X: np.ndarray,
    y: np.ndarray,
    *,
    df_resid: int,
    g: float,
    log_model_prior,
    tss_norm: float | None = None,
    k_always: int = 0,
    top_k: int = 10,
    compute_coefficients: bool = True,
    device: str = "cuda",
    vram_budget_bytes: int = 2 << 30,
    max_chunk: int = 1 << 16,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_s: float = 60.0,
    resume: bool = False,
    stop_after_chunks: int | None = None,
    progress_every_s: float = 10.0,
    progress: "callable | None" = None,
    keep_scores: bool = False,
):
    """Exhaustively score all 2^p models on one GPU in float64.

    X, y must already be residualized on the always block; the convention
    parameters (``df_resid``, ``tss_norm``, ``k_always``) work exactly as in
    :func:`gpubma.cpu.enumeration.enumerate_models` (shrink or flat).

    Returns a dict of posterior quantities and a runtime report, or — if
    ``stop_after_chunks`` interrupted the run — a dict with
    ``"interrupted": True`` after writing a checkpoint.

    ``keep_scores`` (small p only) additionally returns per-model unnormalized
    log scores and masks in mask order for validation against the CPU oracle.
    """
    from gpubma.gpu.batch_scorer import torch_cuda_available

    ok, msg = torch_cuda_available()
    if not ok:
        raise RuntimeError(f"CUDA unavailable: {msg}")
    import torch

    t_start = time.perf_counter()
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    n, p = X.shape
    if p > 30:
        raise ValueError(f"p = {p} exceeds the designed maximum of 30")
    if df_resid <= 2:
        raise ValueError(f"effective degrees of freedom too small: {df_resid}")
    n_models_expected = 1 << p
    if keep_scores and p > 20:
        raise ValueError("keep_scores stores 2^p values; limited to p <= 20")

    # ---- sufficient statistics (identical to the CPU oracle) -------------
    Zxx_np = X.T @ X
    Zxy_np = X.T @ y
    tss = float(y @ y)
    if tss <= 0:
        raise ValueError("residualized outcome has zero variation")
    if tss_norm is None:
        tss_norm = tss
        if k_always:
            raise ValueError("k_always requires tss_norm (shrink convention)")
    ess_always = tss_norm - tss
    if ess_always < -1e-9 * tss_norm:
        raise ValueError("tss_norm must be >= residualized y'y")
    log_prior_by_size = np.array([log_model_prior(k) for k in range(p + 1)],
                                 dtype=np.float64)

    digest = _config_digest(Zxx_np, Zxy_np, tss, tss_norm, df_resid, g,
                            k_always, log_prior_by_size, top_k)

    dev = torch.device(device)
    Zxx = torch.from_numpy(Zxx_np).to(dev)
    Zxy = torch.from_numpy(Zxy_np).to(dev)
    assert Zxx.dtype == torch.float64 and Zxy.dtype == torch.float64
    binom_np = binomial_table(p)
    binom = torch.from_numpy(binom_np).to(dev)

    log1pg = float(np.log1p(g))
    shrink = g / (1.0 + g)
    df = float(df_resid)
    tiny = float(np.finfo(np.float64).tiny)
    lp_size = torch.from_numpy(log_prior_by_size).to(dev)

    # ---- accumulator state (float64, O(p), on device) --------------------
    M = torch.tensor(-math.inf, dtype=torch.float64, device=dev)  # running max
    sum_w = torch.zeros((), dtype=torch.float64, device=dev)
    pip_num = torch.zeros(p, dtype=torch.float64, device=dev)
    mean_num = torch.zeros(p, dtype=torch.float64, device=dev)
    m2_num = torch.zeros(p, dtype=torch.float64, device=dev)
    size_num = torch.zeros(p + 1, dtype=torch.float64, device=dev)
    top_scores = torch.full((top_k,), -math.inf, dtype=torch.float64, device=dev)
    top_masks = torch.zeros(top_k, dtype=torch.int64, device=dev)
    models_done = 0
    start_k, start_rank = 0, 0
    elapsed_prior = 0.0  # from previous (resumed) sessions

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    if resume:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError("resume=True but no checkpoint found at "
                                    f"{checkpoint_path}")
        st = _load_checkpoint(checkpoint_path)
        if int(st["version"]) != CHECKPOINT_VERSION:
            raise ValueError("checkpoint version mismatch")
        if str(st["digest"]) != digest:
            raise ValueError(
                "checkpoint does not match the current data/prior "
                "configuration (SHA-256 mismatch); refusing to resume")
        M = torch.tensor(float(st["M"]), dtype=torch.float64, device=dev)
        sum_w = torch.tensor(float(st["sum_w"]), dtype=torch.float64, device=dev)
        pip_num = torch.from_numpy(st["pip_num"]).to(dev)
        mean_num = torch.from_numpy(st["mean_num"]).to(dev)
        m2_num = torch.from_numpy(st["m2_num"]).to(dev)
        size_num = torch.from_numpy(st["size_num"]).to(dev)
        top_scores = torch.from_numpy(st["top_scores"]).to(dev)
        top_masks = torch.from_numpy(st["top_masks"]).to(dev)
        models_done = int(st["models_done"])
        start_k = int(st["next_k"])
        start_rank = int(st["next_rank"])
        elapsed_prior = float(st["elapsed_s"])
        if keep_scores:
            raise ValueError("keep_scores cannot be combined with resume")

    kept_scores: list = []
    kept_masks: list = []

    def _chunk_size_for(k: int) -> int:
        if k == 0:
            return 1
        # measured ~5 float64 buffers of k^2 per model in flight (ADR JSON)
        by_mem = int(vram_budget_bytes // (8 * 5 * max(k * k, 8)))
        return max(1024, min(max_chunk, by_mem))

    def _save(next_k: int, next_rank: int) -> None:
        if checkpoint_path is None:
            return
        state = dict(
            version=np.int64(CHECKPOINT_VERSION), digest=np.str_(digest),
            M=M.cpu().numpy(), sum_w=sum_w.cpu().numpy(),
            pip_num=pip_num.cpu().numpy(), mean_num=mean_num.cpu().numpy(),
            m2_num=m2_num.cpu().numpy(), size_num=size_num.cpu().numpy(),
            top_scores=top_scores.cpu().numpy(), top_masks=top_masks.cpu().numpy(),
            models_done=np.int64(models_done), next_k=np.int64(next_k),
            next_rank=np.int64(next_rank),
            elapsed_s=np.float64(elapsed_prior + (time.perf_counter() - t_start)),
        )
        _save_checkpoint(checkpoint_path, state)

    def _rescale_to(new_max) -> None:
        nonlocal M, sum_w
        scale = torch.exp(M - new_max)  # <= 1; exp(-inf - x) = 0 handled
        if torch.isinf(M):
            scale = torch.zeros_like(scale)
        sum_w.mul_(scale)
        pip_num.mul_(scale)
        mean_num.mul_(scale)
        m2_num.mul_(scale)
        size_num.mul_(scale)
        M = new_max.clone()

    # ---- main loop --------------------------------------------------------
    chunks_run = 0
    interrupted = False
    last_ckpt = time.perf_counter()
    last_prog = time.perf_counter()
    t_score_total = 0.0

    for k in range(start_k, p + 1):
        total_k = int(binom_np[p, k])
        rank0 = start_rank if k == start_k else 0
        while rank0 < total_k:
            B = min(_chunk_size_for(k), total_k - rank0)
            t0 = time.perf_counter()
            if k == 0:
                one_minus_r2 = max(tss / tss_norm, tiny)
                score0 = (0.5 * (df - k_always) * log1pg
                          - 0.5 * df * math.log1p(g * one_minus_r2)
                          + float(log_prior_by_size[0]))
                scores = torch.tensor([score0], dtype=torch.float64, device=dev)
                masks = torch.zeros(1, dtype=torch.int64, device=dev)
                idx = None
                cond_mean = cond_var = None
            else:
                ranks = torch.arange(rank0, rank0 + B, dtype=torch.int64,
                                     device=dev)
                idx = unrank_combinations(ranks, k, binom, torch)
                Z = Zxx[idx.unsqueeze(2), idx.unsqueeze(1)]     # (B, k, k)
                b = Zxy[idx].unsqueeze(-1)                       # (B, k, 1)
                L = torch.linalg.cholesky(Z)
                u = torch.linalg.solve_triangular(L, b, upper=False)
                ess = (u.squeeze(-1) ** 2).sum(dim=1)            # (B,)
                one_minus_r2 = torch.clamp((tss - ess) / tss_norm, min=tiny)
                scores = (0.5 * (df - k - k_always) * log1pg
                          - 0.5 * df * torch.log1p(g * one_minus_r2)
                          + lp_size[k])
                masks = (torch.ones_like(idx) << idx).sum(dim=1)
                if compute_coefficients:
                    beta_hat = torch.linalg.solve_triangular(
                        L.transpose(1, 2), u, upper=True).squeeze(-1)  # (B, k)
                    zinv_diag = torch.cholesky_inverse(L).diagonal(
                        dim1=-2, dim2=-1)                        # (B, k)
                    e_sigma2 = (tss_norm - shrink * (ess_always + ess)) / (df - 2)
                    cond_mean = shrink * beta_hat
                    cond_var = e_sigma2.unsqueeze(1) * shrink * zinv_diag
                else:
                    cond_mean = cond_var = None

            # ---- stable streaming reduction --------------------------------
            chunk_max = scores.max()
            if bool(chunk_max > M):
                _rescale_to(chunk_max)
            w = torch.exp(scores - M)                            # (B,)
            sum_w.add_(w.sum())
            size_num[k] += w.sum()
            if idx is not None:
                # per-row scatter into dense (B, p) then a fixed-shape column
                # sum: deterministic on CUDA, unlike index_add_ (atomics).
                def _scatter_colsum(values):                     # values (B, k)
                    dense = torch.zeros(values.shape[0], p,
                                        dtype=torch.float64, device=dev)
                    dense.scatter_(1, idx, values)
                    return dense.sum(dim=0)

                w_col = w.unsqueeze(1).expand_as(idx).contiguous()
                pip_num += _scatter_colsum(w_col)
                if compute_coefficients:
                    mean_num += _scatter_colsum(w.unsqueeze(1) * cond_mean)
                    m2_num += _scatter_colsum(
                        w.unsqueeze(1) * (cond_var + cond_mean ** 2))

            # ---- running top-K ---------------------------------------------
            cat_scores = torch.cat([top_scores, scores])
            cat_masks = torch.cat([top_masks, masks])
            best = torch.topk(cat_scores, k=min(top_k, cat_scores.numel()))
            top_scores = best.values
            top_masks = cat_masks[best.indices]

            if keep_scores:
                kept_scores.append(scores.cpu().numpy())
                kept_masks.append(masks.cpu().numpy())

            torch.cuda.synchronize()
            t_score_total += time.perf_counter() - t0
            models_done += B
            rank0 += B
            chunks_run += 1

            now = time.perf_counter()
            if progress_every_s and now - last_prog >= progress_every_s:
                info = {
                    "models_done": models_done,
                    "models_total": n_models_expected,
                    "fraction": models_done / n_models_expected,
                    "current_size": k,
                    "elapsed_s": elapsed_prior + (now - t_start),
                    "models_per_second": models_done / max(now - t_start, 1e-9),
                }
                if progress:
                    progress(info)
                else:
                    print(f"  [{info['fraction']:7.2%}] k={k:2d} "
                          f"{info['models_done']:,}/{info['models_total']:,} "
                          f"({info['models_per_second']:,.0f} models/s)")
                last_prog = now
            if checkpoint_path and now - last_ckpt >= checkpoint_every_s:
                next_k, next_rank = (k, rank0) if rank0 < total_k else (k + 1, 0)
                _save(next_k, next_rank)
                last_ckpt = now
            if stop_after_chunks is not None and chunks_run >= stop_after_chunks:
                next_k, next_rank = (k, rank0) if rank0 < total_k else (k + 1, 0)
                _save(next_k, next_rank)
                interrupted = True
                break
        if interrupted:
            break

    peak_gpu = int(torch.cuda.max_memory_allocated(dev))
    elapsed = elapsed_prior + (time.perf_counter() - t_start)

    if interrupted:
        return {
            "interrupted": True,
            "models_done": models_done,
            "n_models_expected": n_models_expected,
            "checkpoint": str(checkpoint_path),
            "runtime": {"elapsed_s": elapsed,
                        "peak_gpu_memory_bytes": peak_gpu},
        }

    # ---- exact count and finalization -------------------------------------
    if models_done != n_models_expected:
        raise AssertionError(
            f"enumeration incomplete: {models_done} != {n_models_expected}")

    log_norm = float((M + torch.log(sum_w)).cpu())
    pip = (pip_num / sum_w).cpu().numpy()
    overshoot = float(pip.max() - 1.0) if p else 0.0
    if overshoot > 1e-12:
        raise FloatingPointError(
            f"PIP exceeds 1 beyond numerical noise: 1 + {overshoot:.3e}")
    pip = np.clip(pip, 0.0, 1.0)
    size_dist = (size_num / sum_w).cpu().numpy()
    mean_size = float(np.arange(p + 1) @ size_dist)
    coef_mean = coef_sd = None
    if compute_coefficients:
        coef_mean = (mean_num / sum_w).cpu().numpy()
        m2 = (m2_num / sum_w).cpu().numpy()
        coef_sd = np.sqrt(np.maximum(m2 - coef_mean ** 2, 0.0))

    order = torch.argsort(top_scores, descending=True)
    top_scores_np = top_scores[order].cpu().numpy()
    top_masks_np = top_masks[order].cpu().numpy()
    top_models = [
        {
            "mask": int(m),
            "included": [j for j in range(p) if (int(m) >> j) & 1],
            "size": int(bin(int(m)).count("1")),
            "pmp": float(np.exp(s - log_norm)),
            "log_score": float(s),
        }
        for s, m in zip(top_scores_np, top_masks_np)
        if np.isfinite(s)
    ]

    out = {
        "n_models_expected": n_models_expected,
        "n_models_evaluated": models_done,
        "log_normalizer": log_norm,
        "pip": pip,
        "mean_model_size": mean_size,
        "size_distribution": size_dist,
        "coef_mean": coef_mean,
        "coef_sd": coef_sd,
        "top_models": top_models,
        "tss_residualized": tss,
        "normalization_check": {
            "size_distribution_sum": float(size_dist.sum()),
            "pip_max_overshoot": overshoot,
        },
        "runtime": {
            "backend": "gpu-enumerator",
            "precision": "float64",
            "device": torch.cuda.get_device_name(0),
            "elapsed_s": elapsed,
            "scoring_s": t_score_total,
            "models_per_second": models_done / elapsed if elapsed > 0 else float("nan"),
            "peak_gpu_memory_bytes": peak_gpu,
            "chunks": chunks_run,
            "resumed": bool(resume),
        },
    }
    if keep_scores:
        all_scores = np.concatenate(kept_scores)
        all_masks = np.concatenate(kept_masks)
        order = np.argsort(all_masks)
        assert np.array_equal(all_masks[order], np.arange(n_models_expected)), \
            "each model must be enumerated exactly once"
        out["log_scores"] = all_scores[order]  # mask order, like the CPU oracle
    if checkpoint_path:
        _save(p + 1, 0)
    return out
