"""Phase 1 benchmark ladder: honest, measured-only timings plus clearly
labelled projections. ``python -m gpubma.benchmark --max-predictors 15``.

Every ladder row is labelled Measured / Projected / Not evaluated. Projections
are computed only from a real measured model-scoring benchmark, state their
source and scaling assumption, and list reasons they may fail.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gpubma.cpu.enumeration import enumerate_models
from gpubma.datasets.synthetic import DEFAULT_SEED, generate_panel
from gpubma.gpu.batch_scorer import gpu_score_all_models, gpu_hardware_info, torch_cuda_available
from gpubma.priors.model_priors import log_model_prior_function

LADDER = [8, 10, 12, 15, 18, 20, 24, 26, 28, 30]
_GPU_CAP = 16   # feasibility scorer limit
_CPU_SAFE_DEFAULT = 15


def _bench_cpu(X, y, df_resid, g, log_prior, repeats):
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = enumerate_models(X, y, df_resid=df_resid, g=g, log_model_prior=log_prior,
                               compute_coefficients=False)
        times.append(time.perf_counter() - t0)
    return out, times


def _bench_gpu(X, y, df_resid, g, log_prior, repeats):
    t0 = time.perf_counter()
    out = gpu_score_all_models(X, y, df_resid=df_resid, g=g, log_model_prior=log_prior)
    cold = time.perf_counter() - t0
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = gpu_score_all_models(X, y, df_resid=df_resid, g=g, log_model_prior=log_prior)
        times.append(time.perf_counter() - t0)
    return out, cold, times


def run_ladder(max_predictors: int = _CPU_SAFE_DEFAULT, repeats: int = 3,
               seed: int = DEFAULT_SEED) -> dict:
    cuda_ok, cuda_msg = torch_cuda_available()
    hw = gpu_hardware_info()
    df_panel, _ = generate_panel(100, 10, max(p for p in LADDER if p <= 30), seed=seed)
    n = len(df_panel)
    y_full = df_panel["y"].to_numpy(np.float64)
    yc = y_full - y_full.mean()

    rows = []
    measured_gpu = measured_cpu = None
    for p in LADDER:
        n_models = 1 << p
        row = {"predictors": p, "n_models": n_models, "n_obs": n}
        if p > max_predictors:
            row["status"] = "Not evaluated"
            row["reason"] = f"beyond --max-predictors {max_predictors} (Phase 1 safe limit)"
            rows.append(row)
            continue
        X = df_panel[[f"x{j}" for j in range(1, p + 1)]].to_numpy(np.float64)
        Xc = X - X.mean(axis=0)
        df_resid = n - 1
        g = float(max(n, p * p))
        log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), p)

        cpu_out, cpu_times = _bench_cpu(Xc, yc, df_resid, g, log_prior, repeats)
        row["status"] = "Measured"
        row["cpu"] = {
            "backend": "cpu (numpy/scipy, float64)",
            "repeats": repeats,
            "median_s": statistics.median(cpu_times),
            "min_s": min(cpu_times), "max_s": max(cpu_times),
            "models_per_second": n_models / statistics.median(cpu_times),
            "sufficient_statistics_s": cpu_out["runtime"]["sufficient_statistics_s"],
            "enumeration_s": cpu_out["runtime"]["enumeration_s"],
            "reduction_s": cpu_out["runtime"]["reduction_s"],
        }
        measured_cpu = (p, row["cpu"]["models_per_second"])

        if cuda_ok and p <= _GPU_CAP:
            gpu_out, cold, gpu_times = _bench_gpu(Xc, yc, df_resid, g, log_prior, repeats)
            diff = float(np.max(np.abs(gpu_out["log_scores"] - cpu_out["log_scores"])))
            row["gpu"] = {
                "backend": "gpu (torch batched cholesky, float64)",
                "device": hw.get("device_name"),
                "precision": "float64",
                "cold_s": cold,
                "repeats": repeats,
                "median_s": statistics.median(gpu_times),
                "min_s": min(gpu_times), "max_s": max(gpu_times),
                "models_per_second": n_models / statistics.median(gpu_times),
                "peak_gpu_memory_bytes": gpu_out["runtime"]["peak_gpu_memory_bytes"],
                "max_abs_logscore_diff_vs_cpu": diff,
            }
            measured_gpu = (p, row["gpu"]["models_per_second"])
        elif cuda_ok:
            row["gpu"] = {"status": "Not evaluated",
                          "reason": f"feasibility scorer caps p at {_GPU_CAP} in Phase 1"}
        else:
            row["gpu"] = {"status": "Not evaluated", "reason": f"CUDA unavailable: {cuda_msg}"}
        rows.append(row)

    # honest projections from the largest measured run
    projections = []
    for src_name, src in (("cpu", measured_cpu), ("gpu", measured_gpu)):
        if src is None:
            continue
        p_src, mps = src
        for p in (24, 26, 28, 30):
            projections.append({
                "status": "Projected",
                "target_predictors": p,
                "target_models": 1 << p,
                "backend": src_name,
                "source_benchmark": f"measured {src_name} run at p={p_src} on this machine",
                "scaling_assumption": "constant models/second (linear scaling in model count)",
                "projected_seconds": (1 << p) / mps,
                "throughput_stability": "single-size extrapolation; stability across sizes NOT established",
                "confidence": "LOW",
                "reasons_projection_may_fail": [
                    "per-model cost grows with model size k (larger Cholesky factors dominate at larger p)",
                    "memory pressure and batching behaviour differ at larger p",
                    "the Phase 1 scorers are not the production enumerator",
                    "reduction/normalization costs are excluded from the scaling model",
                ],
            })

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hw,
        "seed": seed,
        "note": "Phase 1 feasibility benchmark; NOT the production CUDA enumerator.",
        "ladder": rows,
        "projections": projections,
    }


def format_markdown(report: dict) -> str:
    hw = report["hardware"]
    lines = [
        "# GPUBMA Phase 1 benchmark report",
        "",
        f"- Generated (UTC): {report['timestamp_utc']}",
        f"- GPU: {hw.get('device_name', 'none detected')} "
        f"(CC {hw.get('compute_capability', 'n/a')}, {hw.get('multiprocessors', 'n/a')} SMs)",
        f"- Seed: {report['seed']}",
        f"- {report['note']}",
        "",
        "## Ladder (Measured / Not evaluated)",
        "",
        "| p | models | status | CPU median s | CPU models/s | GPU cold s | GPU warm median s | GPU models/s | GPU==CPU max diff |",
        "|---|--------|--------|--------------|--------------|------------|-------------------|--------------|-------------------|",
    ]
    for r in report["ladder"]:
        if r["status"] != "Measured":
            lines.append(f"| {r['predictors']} | {r['n_models']:,} | {r['status']} | — | — | — | — | — | — |")
            continue
        c = r["cpu"]; gpu = r.get("gpu", {})
        has_gpu = "median_s" in gpu
        lines.append(
            f"| {r['predictors']} | {r['n_models']:,} | Measured "
            f"| {c['median_s']:.4f} | {c['models_per_second']:,.0f} "
            + (f"| {gpu['cold_s']:.4f} | {gpu['median_s']:.4f} | {gpu['models_per_second']:,.0f} "
               f"| {gpu['max_abs_logscore_diff_vs_cpu']:.2e} |" if has_gpu
               else f"| — | — | — | — ({gpu.get('reason', 'n/a')}) |")
        )
    lines += ["", "## Projections (clearly labelled, LOW confidence)", ""]
    if not report["projections"]:
        lines.append("No projections: no measured source benchmark available.")
    for pr in report["projections"]:
        lines.append(
            f"- **Projected** {pr['backend']} p={pr['target_predictors']} "
            f"({pr['target_models']:,} models): ~{pr['projected_seconds']:,.1f} s "
            f"— source: {pr['source_benchmark']}; assumption: {pr['scaling_assumption']}; "
            f"confidence: {pr['confidence']}. May fail because: "
            + "; ".join(pr["reasons_projection_may_fail"]) + "."
        )
    lines += ["", "All numbers above are labelled Measured, Projected, or Not evaluated (task rule 7).", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gpubma benchmark")
    parser.add_argument("--max-predictors", type=int, default=_CPU_SAFE_DEFAULT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    report = run_ladder(max_predictors=args.max_predictors, repeats=args.repeats)
    out_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2))
    md = format_markdown(report)
    (out_dir / "benchmark_report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"saved: {out_dir / 'benchmark_report.md'} and .json")
    return 0
