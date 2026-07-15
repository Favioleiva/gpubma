"""Progressive validation ladder for the GPU enumerator (Phase 2).

For each level p the script runs the exhaustive GPU enumerator on real
frozen data (panel_12 at p = 12 — the Stata-validated dataset — and
panel_30 predictor subsets above) and records, ALL MEASURED:

  - models expected and processed (exact counts);
  - elapsed wall time and models/second;
  - peak GPU memory and peak host working set;
  - posterior normalization checks (size-distribution sum, PIP overshoot);
  - reproducibility (second full run must be bit-identical);
  - checkpoint/resume correctness (interrupt + resume must be bit-identical
    to the uninterrupted run);
  - CPU-GPU discrepancies where CPU validation is practical:
      p <= 20: full CPU reference incl. coefficient moments and per-model
               log scores;  p in (22, 24): scores-only CPU reference
               (aggregates: normalizer, PIP, size distribution, mean size);
  - at p = 12: complete CPU-GPU-Stata parity (per-model normalized log
    PMPs against the executed bmaregress saving() export, PIP/means/sds
    against e() exports).

p = 30 is NOT run by this script (requires explicit user authorization).

Usage: python scripts/run_enumeration_ladder.py [--levels 12 18 20 22 24]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpubma.cpu.enumeration import enumerate_models  # noqa: E402
from gpubma.gpu.enumerator import enumerate_models_gpu  # noqa: E402
from gpubma.priors.model_priors import log_model_prior_function  # noqa: E402

STATA_OUT = ROOT / "validation" / "stata" / "output"
G_FIXED = 1000.0  # = max(n, p^2) for every ladder level (n = 1000, p <= 24)
CPU_FULL_MAX_P = 20
CPU_SCORES_MAX_P = 24


def shrink_inputs(df: pd.DataFrame, p: int):
    predictors = [f"x{j}" for j in range(1, p + 1)]
    y = df["y"].to_numpy(np.float64)
    X = df[predictors].to_numpy(np.float64)
    n = len(y)
    A = np.column_stack([np.ones(n), df[["w1", "w2"]].to_numpy(np.float64)])
    Q, _ = np.linalg.qr(A)
    yr = y - Q @ (Q.T @ y)
    Xr = X - Q @ (Q.T @ X)
    yc = y - y.mean()
    conv = dict(df_resid=n - 1, tss_norm=float(yc @ yc), k_always=2)
    return Xr, yr, conv


def peak_host_bytes() -> int | None:
    try:
        import psutil

        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", None) or info.rss
    except ImportError:
        return None


def compare_stata_p12(gpu, log_prior) -> dict:
    """Complete Stata parity at p = 12 against the executed oracle."""
    out = {}
    models_path = STATA_OUT / "medium_no_fe_models.dta"
    st = pd.read_stata(models_path)
    states = st[[f"state_eq1_p{j}" for j in range(1, 13)]].to_numpy(np.int64)
    masks = (states << np.arange(12, dtype=np.int64)).sum(axis=1)
    order = np.argsort(masks)
    st_logpost = st["_logposterior"].to_numpy(np.float64)[order]
    st_norm = st_logpost - np.logaddexp.reduce(np.sort(st_logpost))
    gpu_norm = gpu["log_scores"] - np.logaddexp.reduce(np.sort(gpu["log_scores"]))
    out["max_abs_diff_normalized_logpmp_vs_stata"] = float(
        np.max(np.abs(st_norm - gpu_norm)))

    names = (STATA_OUT / "medium_no_fe_colnames.txt").read_text().split()
    idx = {n: i for i, n in enumerate(names)}
    read = lambda f: pd.read_csv(f, float_precision="round_trip").iloc[0].to_numpy(float)
    st_pip = read(STATA_OUT / "medium_no_fe_pip.csv")
    st_b = read(STATA_OUT / "medium_no_fe_b_bma.csv")
    st_sd = np.sqrt(read(STATA_OUT / "medium_no_fe_v_diag.csv"))
    cols = [idx[f"x{j}"] for j in range(1, 13)]
    out["max_abs_diff_pip_vs_stata"] = float(
        np.max(np.abs(st_pip[cols] - gpu["pip"])))
    out["max_abs_diff_coef_mean_vs_stata"] = float(
        np.max(np.abs(st_b[cols] - gpu["coef_mean"])))
    out["max_abs_diff_coef_sd_vs_stata"] = float(
        np.max(np.abs(st_sd[cols] - gpu["coef_sd"])))
    scalars = pd.read_csv(STATA_OUT / "medium_no_fe_scalars.csv",
                          float_precision="round_trip").iloc[0]
    out["abs_diff_mean_model_size_vs_stata"] = float(abs(
        (scalars["msize_mean"] - scalars["p_always"]) - gpu["mean_model_size"]))
    out["stata_k_models_equal"] = bool(
        int(scalars["k_models"]) == gpu["n_models_evaluated"])
    return out


def run_level(p: int, df: pd.DataFrame, dataset_name: str, ckpt_dir: Path) -> dict:
    print(f"\n=== p = {p} ({dataset_name}, 2^{p} = {1 << p:,} models) ===")
    Xr, yr, conv = shrink_inputs(df, p)
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), p)
    keep = p <= CPU_FULL_MAX_P
    kw = dict(g=G_FIXED, log_model_prior=log_prior, top_k=10,
              progress_every_s=30.0, **conv)

    rec: dict = {
        "p": p, "dataset": dataset_name, "g": G_FIXED,
        "n_models_expected": 1 << p, "label": "Measured",
    }

    # --- run 1 (the measured run) -----------------------------------------
    t0 = time.perf_counter()
    r1 = enumerate_models_gpu(Xr, yr, keep_scores=keep, **kw)
    elapsed = time.perf_counter() - t0
    assert r1["n_models_evaluated"] == 1 << p
    rec.update(
        n_models_processed=r1["n_models_evaluated"],
        elapsed_s=elapsed,
        models_per_second=r1["n_models_evaluated"] / elapsed,
        peak_gpu_memory_bytes=r1["runtime"]["peak_gpu_memory_bytes"],
        peak_host_memory_bytes=peak_host_bytes(),
        chunks=r1["runtime"]["chunks"],
        device=r1["runtime"]["device"],
        normalization={
            "size_distribution_sum": r1["normalization_check"]["size_distribution_sum"],
            "pip_max_overshoot": r1["normalization_check"]["pip_max_overshoot"],
        },
        mean_model_size=r1["mean_model_size"],
        log_normalizer=r1["log_normalizer"],
        top_models=[{"mask": m["mask"], "size": m["size"], "pmp": m["pmp"]}
                    for m in r1["top_models"][:5]],
    )
    print(f"  run 1: {elapsed:.2f} s, {rec['models_per_second']:,.0f} models/s, "
          f"peak GPU {rec['peak_gpu_memory_bytes'] / 2**20:.0f} MiB")

    # --- reproducibility: run 2 must be bit-identical -----------------------
    r2 = enumerate_models_gpu(Xr, yr, **kw)
    rec["reproducible_bit_identical"] = bool(
        r1["log_normalizer"] == r2["log_normalizer"]
        and np.array_equal(r1["pip"], r2["pip"])
        and np.array_equal(r1["coef_mean"], r2["coef_mean"])
        and np.array_equal(r1["size_distribution"], r2["size_distribution"]))
    print(f"  reproducibility (bit-identical): {rec['reproducible_bit_identical']}")

    # --- checkpoint/resume correctness --------------------------------------
    ckpt = ckpt_dir / f"ladder_p{p}.ckpt.npz"
    part = enumerate_models_gpu(Xr, yr, checkpoint_path=ckpt,
                                stop_after_chunks=max(3, rec["chunks"] // 3), **kw)
    resumed = enumerate_models_gpu(Xr, yr, checkpoint_path=ckpt, resume=True, **kw)
    rec["checkpoint_resume_bit_identical"] = bool(
        part.get("interrupted") is True
        and resumed["log_normalizer"] == r1["log_normalizer"]
        and np.array_equal(resumed["pip"], r1["pip"])
        and np.array_equal(resumed["coef_mean"], r1["coef_mean"])
        and np.array_equal(resumed["size_distribution"], r1["size_distribution"]))
    rec["checkpoint_interrupted_at_models"] = int(part.get("models_done", 0))
    ckpt.unlink(missing_ok=True)
    print(f"  checkpoint/resume (bit-identical after interrupt at "
          f"{rec['checkpoint_interrupted_at_models']:,} models): "
          f"{rec['checkpoint_resume_bit_identical']}")

    # --- CPU validation where practical -------------------------------------
    if p <= CPU_SCORES_MAX_P:
        full = p <= CPU_FULL_MAX_P
        t0 = time.perf_counter()
        cpu = enumerate_models(Xr, yr, g=G_FIXED, log_model_prior=log_prior,
                               compute_coefficients=full, top_k=10,
                               keep_masks=False, **conv)
        cpu_s = time.perf_counter() - t0
        d = {
            "cpu_seconds": cpu_s,
            "cpu_mode": "full (scores + coefficient moments)" if full
                        else "scores-only (coefficient moments impractical)",
            "abs_diff_log_normalizer": abs(cpu["log_normalizer"] - r1["log_normalizer"]),
            "max_abs_diff_pip": float(np.max(np.abs(cpu["pip"] - r1["pip"]))),
            "max_abs_diff_size_distribution": float(np.max(np.abs(
                cpu["size_distribution"] - r1["size_distribution"]))),
            "abs_diff_mean_model_size": abs(cpu["mean_model_size"] - r1["mean_model_size"]),
            "top_masks_equal": [m["mask"] for m in cpu["top_models"]]
                               == [m["mask"] for m in r1["top_models"]],
        }
        if full:
            d["max_abs_diff_per_model_log_score"] = float(np.max(np.abs(
                cpu["log_scores"] - r1["log_scores"])))
            d["max_abs_diff_coef_mean"] = float(np.max(np.abs(
                cpu["coef_mean"] - r1["coef_mean"])))
            d["max_abs_diff_coef_sd"] = float(np.max(np.abs(
                cpu["coef_sd"] - r1["coef_sd"])))
        rec["cpu_validation"] = d
        print(f"  CPU check ({d['cpu_mode']}, {cpu_s:.1f} s): "
              f"max |dPIP| = {d['max_abs_diff_pip']:.3e}")
    else:
        rec["cpu_validation"] = {"cpu_mode": "Not evaluated"}

    # --- Stata parity at p = 12 ---------------------------------------------
    if p == 12 and dataset_name == "panel_12":
        rec["stata_validation"] = compare_stata_p12(r1, log_prior)
        print(f"  Stata parity: max |d normalized logPMP| = "
              f"{rec['stata_validation']['max_abs_diff_normalized_logpmp_vs_stata']:.3e}")
    return rec


def to_markdown(records: list[dict], device: str) -> str:
    lines = [
        "# GPU enumerator — progressive validation ladder",
        "",
        f"Device: {device}; float64 throughout; g = {G_FIXED:.0f} = max(n, p^2) "
        "for every level; beta-binomial(1,1) model prior; shrink (Stata) "
        "convention with always block [1, w1, w2].",
        "",
        "Every number below is **Measured** (never projected). p = 30 was NOT "
        "run (requires explicit authorization).",
        "",
        "| p | models | elapsed s | models/s | peak GPU MiB | peak host MiB | "
        "size-dist sum | reproducible | ckpt/resume | CPU check | Stata check |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        cpu = r["cpu_validation"]
        if cpu.get("cpu_mode") == "Not evaluated":
            cpu_cell = "Not evaluated"
        else:
            worst = max(v for k, v in cpu.items()
                        if k.startswith(("max_abs_diff", "abs_diff")))
            cpu_cell = f"max diff {worst:.1e}"
        stata = r.get("stata_validation")
        stata_cell = (f"max diff {max(v for k, v in stata.items() if 'diff' in k):.1e}"
                      if stata else "—")
        host = r["peak_host_memory_bytes"]
        host_cell = f"{host / 2**20:.0f}" if host else "n/a"
        lines.append(
            f"| {r['p']} | {r['n_models_processed']:,} | {r['elapsed_s']:.2f} "
            f"| {r['models_per_second']:,.0f} "
            f"| {r['peak_gpu_memory_bytes'] / 2**20:.0f} "
            f"| {host_cell} "
            f"| {r['normalization']['size_distribution_sum']:.15f} "
            f"| {'yes' if r['reproducible_bit_identical'] else 'NO'} "
            f"| {'yes' if r['checkpoint_resume_bit_identical'] else 'NO'} "
            f"| {cpu_cell} | {stata_cell} |"
        )
    lines += [
        "",
        "Notes:",
        "- 'reproducible' and 'ckpt/resume' assert BIT-IDENTICAL results",
        "  (same chunk partitioning, deterministic reductions).",
        "- CPU check compares against the exact CPU oracle: full (per-model",
        "  scores + moments) for p <= 20, scores-only aggregates for p = 22, 24.",
        "- peak host MiB is the process-lifetime peak working set at the time",
        "  the level finished (cumulative across levels).",
        "- Stata check (p = 12 only): executed bmaregress oracle, all 4,096",
        "  models (normalized log PMPs) plus PIPs/means/sds/model size.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+",
                        default=[12, 18, 20, 22, 24])
    parser.add_argument("--out-prefix", type=Path,
                        default=ROOT / "reports" / "enumeration_ladder")
    args = parser.parse_args()
    if any(p > 24 for p in args.levels):
        print("levels above 24 (in particular p = 30) require explicit "
              "user authorization; refusing")
        return 1

    panel_12 = pd.read_parquet(ROOT / "data" / "synthetic" / "panel_12.parquet")
    panel_30 = pd.read_parquet(ROOT / "data" / "synthetic" / "panel_30.parquet")

    import torch

    device = torch.cuda.get_device_name(0)
    ckpt_dir = Path(tempfile.mkdtemp(prefix="gpubma-ladder-"))
    records = []
    for p in args.levels:
        df, name = (panel_12, "panel_12") if p == 12 else (panel_30, "panel_30")
        records.append(run_level(p, df, name, ckpt_dir))

    args.out_prefix.with_suffix(".json").write_text(json.dumps(
        {"device": device, "records": records}, indent=2))
    args.out_prefix.with_suffix(".md").write_text(
        to_markdown(records, device), encoding="utf-8")
    print(f"\nwrote {args.out_prefix.with_suffix('.md')} and .json")
    ok = all(r["reproducible_bit_identical"] and r["checkpoint_resume_bit_identical"]
             for r in records)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
