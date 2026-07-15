"""Measured microbenchmarks for the GPU enumerator architecture ADR.

Compares, on the ACTUAL detected GPU in float64, the per-model cost of the
candidate inner loops:

  A. streamed direct batching — batched Cholesky factorization + triangular
     solve on (B, k, k) Gram submatrices (one independent model per batch
     element), including the gather that builds the submatrices and the
     log-score arithmetic;
  B. Gray-code rank-1 Cholesky update/downdate — one Gray step advances B
     independent lanes by one model each: a batched rank-1 update (or
     downdate) of the maintained factors followed by a batched triangular
     solve for the ESS;
  C. component costs — gather alone, cholesky alone, solve alone — so the
     ADR can attribute where time goes.

Every number this script prints is MEASURED on the detected device (warm
medians over repeats, synchronized timing). It projects nothing.

Usage: python scripts/benchmark_enumerator_candidates.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPEATS = 5


def _sync():
    import torch
    torch.cuda.synchronize()


def _timed(fn, repeats=REPEATS):
    """Warm-up once, then median wall time over `repeats` synchronized runs."""
    fn()
    _sync()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def real_gram(p=30):
    """Residualized Gram matrix from the frozen panel_30 dataset (real data,
    shrink convention preprocessing: residualize on [1, w1, w2])."""
    import pandas as pd

    df = pd.read_parquet(ROOT / "data" / "synthetic" / "panel_30.parquet")
    y = df["y"].to_numpy(np.float64)
    X = df[[f"x{j}" for j in range(1, p + 1)]].to_numpy(np.float64)
    A = np.column_stack([np.ones(len(df)), df[["w1", "w2"]].to_numpy(np.float64)])
    Q, _ = np.linalg.qr(A)
    Xr = X - Q @ (Q.T @ X)
    yr = y - Q @ (Q.T @ y)
    return Xr.T @ Xr, Xr.T @ yr, float(yr @ yr)


def bench_direct(torch, dev, Zxx, Zxy, k, batch, rng):
    """Candidate A: direct batched Cholesky scoring of `batch` models of
    size k, including gather and score arithmetic."""
    p = Zxx.shape[0]
    combos = np.argsort(rng.random((batch, p)), axis=1)[:, :k].astype(np.int64)
    combos.sort(axis=1)
    idx = torch.from_numpy(combos).to(dev)
    tiny = float(np.finfo(np.float64).tiny)

    def run():
        Z = Zxx[idx.unsqueeze(2), idx.unsqueeze(1)]
        b = Zxy[idx].unsqueeze(-1)
        L = torch.linalg.cholesky(Z)
        u = torch.linalg.solve_triangular(L, b, upper=False)
        ess = (u.squeeze(-1) ** 2).sum(dim=1)
        one_minus_r2 = torch.clamp(1.0 - ess / 1000.0, min=tiny)
        _ = -0.5 * 999 * torch.log1p(1000.0 * one_minus_r2)

    t = _timed(run)
    # components, measured separately
    t_gather = _timed(lambda: Zxx[idx.unsqueeze(2), idx.unsqueeze(1)])
    Z = Zxx[idx.unsqueeze(2), idx.unsqueeze(1)]
    t_chol = _timed(lambda: torch.linalg.cholesky(Z))
    L = torch.linalg.cholesky(Z)
    b = Zxy[idx].unsqueeze(-1)
    t_solve = _timed(lambda: torch.linalg.solve_triangular(L, b, upper=False))
    return {
        "k": k, "batch": batch, "seconds_per_batch": t,
        "models_per_second": batch / t,
        "gather_s": t_gather, "cholesky_s": t_chol, "solve_s": t_solve,
    }


def bench_gray_step(torch, dev, k, lanes):
    """Candidate B: one Gray-code step across `lanes` independent lanes:
    batched rank-1 Cholesky UPDATE (the algorithmically favourable case;
    downdates cost the same flops but subtract) + batched triangular solve.

    The update is the textbook algorithm; its outer loop over the k rows is
    inherently sequential, so a float64 batched implementation issues ~6
    vector kernels per row. This measures the whole step honestly.
    """
    rng = np.random.default_rng(0)
    Amat = rng.standard_normal((k, 3 * k))
    spd = Amat @ Amat.T + k * np.eye(k)
    L0 = np.linalg.cholesky(spd)
    L = torch.from_numpy(np.broadcast_to(L0, (lanes, k, k)).copy()).to(dev)
    x0 = torch.from_numpy(rng.standard_normal((lanes, k)) * 0.01).to(dev)
    b = torch.from_numpy(rng.standard_normal((lanes, k, 1))).to(dev)

    def run():
        Lw = L.clone()
        x = x0.clone()
        for j in range(k):
            ljj = Lw[:, j, j]
            xj = x[:, j]
            r = torch.sqrt(ljj * ljj + xj * xj)
            c = r / ljj
            s = xj / ljj
            Lw[:, j, j] = r
            if j + 1 < k:
                col = Lw[:, j + 1:, j]
                Lw[:, j + 1:, j] = (col + s.unsqueeze(1) * x[:, j + 1:]) / c.unsqueeze(1)
                x[:, j + 1:] = c.unsqueeze(1) * x[:, j + 1:] - s.unsqueeze(1) * Lw[:, j + 1:, j]
        u = torch.linalg.solve_triangular(Lw, b, upper=False)
        _ = (u.squeeze(-1) ** 2).sum(dim=1)

    t = _timed(run)
    return {
        "k": k, "lanes": lanes, "seconds_per_gray_step": t,
        "models_per_second": lanes / t,  # one model per lane per step
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable; the ADR microbenchmarks require the GPU")
        return 1
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} (CC {props.major}.{props.minor}, "
          f"{props.total_memory / 2**30:.1f} GiB), torch {torch.__version__}, "
          f"float64 throughout")

    Zxx_np, Zxy_np, _ = real_gram(30)
    Zxx = torch.from_numpy(Zxx_np).to(dev)
    Zxy = torch.from_numpy(Zxy_np).to(dev)
    rng = np.random.default_rng(20260715)

    out = {"device": props.name, "torch": torch.__version__,
           "precision": "float64", "repeats": REPEATS,
           "direct": [], "gray": []}

    print("\nA. streamed direct batching (gather + batched Cholesky + solve + score)")
    print(f"{'k':>3} {'batch':>7} {'models/s':>12} {'gather%':>8} {'chol%':>7} {'solve%':>7}")
    for k, batch in [(4, 1 << 16), (8, 1 << 16), (12, 1 << 16), (15, 1 << 16),
                     (18, 1 << 15), (22, 1 << 15), (26, 1 << 14), (30, 1 << 14)]:
        r = bench_direct(torch, dev, Zxx, Zxy, k, batch, rng)
        out["direct"].append(r)
        tot = r["seconds_per_batch"]
        print(f"{k:>3} {batch:>7} {r['models_per_second']:>12,.0f} "
              f"{100 * r['gather_s'] / tot:>7.1f}% {100 * r['cholesky_s'] / tot:>6.1f}% "
              f"{100 * r['solve_s'] / tot:>6.1f}%")
        peak = torch.cuda.max_memory_allocated() / 2**20
        r["peak_gpu_mib"] = peak
        torch.cuda.reset_peak_memory_stats()

    print("\nB. Gray-code rank-1 update step (batched update + triangular solve)")
    print(f"{'k':>3} {'lanes':>7} {'models/s':>12} {'ms/step':>9}")
    for k, lanes in [(8, 1 << 16), (15, 1 << 16), (22, 1 << 15), (30, 1 << 14)]:
        r = bench_gray_step(torch, dev, k, lanes)
        out["gray"].append(r)
        print(f"{k:>3} {lanes:>7} {r['models_per_second']:>12,.0f} "
              f"{1e3 * r['seconds_per_gray_step']:>9.2f}")
        r["peak_gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
