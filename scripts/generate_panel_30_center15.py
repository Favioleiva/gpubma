"""Generate and validate the frozen panel_30_center15 benchmark artifact.

Deterministic: seed 20260724, 200 individuals x 10 periods = 2,000 rows,
p = 30 candidates, true model size 15 (central Pascal layer), correlated
latent-factor blocks, 15 structural-zero proxies, realized R^2 calibrated
analytically to 0.700. DGP: src/gpubma/datasets/center15.py and
docs/DATA_GENERATING_PROCESS.md.

Writes ONLY (Parquet is the canonical format — no CSV, no .dta):

    data/synthetic/panel_30_center15.parquet
    data/synthetic/panel_30_center15_metadata.json

Never touches the frozen panel_8/panel_12/panel_30 artifacts; a guard
verifies their checksums are unchanged afterwards. Fails loudly (non-zero
exit) if any validation check fails.

Byte-stability note: Parquet numerical content is exact float64, but the
physical bytes embed the writer library version (pyarrow "created_by"
footer field) and may legally differ across pyarrow/pandas versions even
when the numerical dataset is bit-identical. The metadata therefore records
both the file SHA-256 (this environment's bytes) and the environment.

Usage:  python scripts/generate_panel_30_center15.py
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpubma.datasets.io_utils import compare_frames_exact, file_sha256  # noqa: E402
from gpubma.datasets.center15 import (  # noqa: E402
    BETA_TRUE, BLOCKS, K_TRUE, N_OBS, P, SEED, TARGET_R2,
    generate_panel_30_center15, structural_beta,
)

OUT_DIR = ROOT / "data" / "synthetic"
STEM = "panel_30_center15"
PARQUET = OUT_DIR / f"{STEM}.parquet"
META_JSON = OUT_DIR / f"{STEM}_metadata.json"
FROZEN_STEMS = ("panel_8", "panel_12", "panel_30")

EXPECTED_COLUMNS = (["individual_id", "period", "y"]
                    + [f"x{j}" for j in range(1, 31)] + ["w1", "w2"])
R2_TOL = (0.695, 0.705)
MAX_CONDITION_NUMBER = 1e6      # residualized Gram, float64 Cholesky headroom
NEAR_DUPLICATE_CORR = 0.995

# representative model index sets (0-based) for float64 Cholesky checks
CHOLESKY_MODELS = {
    "true_15 (x1-x15)": list(range(15)),
    "sparse_1 (x1)": [0],
    "sparse_3 (x1,x6,x11)": [0, 5, 10],
    "central_true+proxyA (x1-x10, x16-x20)": list(range(10)) + list(range(15, 20)),
    "central_all_proxies (x16-x30)": list(range(15, 30)),
    "dense_25": list(range(25)),
    "dense_30 (all)": list(range(30)),
}


def _residualize_on_controls(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[[f"x{j}" for j in range(1, 31)]].to_numpy(np.float64)
    A = np.column_stack([np.ones(len(df)),
                         df[["w1", "w2"]].to_numpy(np.float64)])
    Q, _ = np.linalg.qr(A)
    X_r = X - Q @ (Q.T @ X)
    y_r = df["y"].to_numpy(np.float64) - Q @ (Q.T @ df["y"].to_numpy(np.float64))
    return X_r, y_r


def _true_spec_r2(df: pd.DataFrame) -> float:
    y = df["y"].to_numpy(np.float64)
    D = np.column_stack([np.ones(len(df)),
                         df[["w1", "w2"]].to_numpy(np.float64),
                         df[[f"x{j}" for j in range(1, 16)]].to_numpy(np.float64)])
    Q, _ = np.linalg.qr(D)
    resid = y - Q @ (Q.T @ y)
    y_c = y - y.mean()
    return float(1.0 - (resid @ resid) / (y_c @ y_c))


def validate(df: pd.DataFrame, where: str) -> dict:
    """All required checks on a dataframe; returns the diagnostics dict."""
    # A. shape and schema
    assert df.shape == (N_OBS, len(EXPECTED_COLUMNS)), f"{where}: shape {df.shape}"
    assert list(df.columns) == EXPECTED_COLUMNS, f"{where}: column order"
    assert len(set(df.columns)) == len(df.columns), f"{where}: duplicate columns"
    assert not df.isna().any().any(), f"{where}: missing values"
    num = df.to_numpy(np.float64)
    assert np.isfinite(num).all(), f"{where}: non-finite values"
    assert df["individual_id"].dtype == np.int32 and df["period"].dtype == np.int32
    for col in EXPECTED_COLUMNS[2:]:
        assert df[col].dtype == np.float64, f"{where}: {col} dtype {df[col].dtype}"

    # B. structural truth
    beta = structural_beta()
    assert beta.shape == (P,)
    assert int(np.count_nonzero(beta)) == K_TRUE == 15
    assert np.count_nonzero(beta[15:]) == 0, "proxy coefficient leaked into DGP"
    assert np.array_equal(beta[:15], BETA_TRUE)

    # C. linear algebra
    X = df[[f"x{j}" for j in range(1, 31)]].to_numpy(np.float64)
    assert np.linalg.matrix_rank(X) == P, f"{where}: candidate design rank"
    X_r, _ = _residualize_on_controls(df)
    assert np.linalg.matrix_rank(X_r) == P, f"{where}: residualized rank"
    G = X_r.T @ X_r
    eigvals = np.linalg.eigvalsh(G)
    cond = float(eigvals[-1] / eigvals[0])
    assert eigvals[0] > 0, f"{where}: residualized Gram not PD"
    assert cond < MAX_CONDITION_NUMBER, f"{where}: condition number {cond:.3e}"
    for label, idx in CHOLESKY_MODELS.items():
        np.linalg.cholesky(G[np.ix_(idx, idx)])  # raises LinAlgError on failure

    # D. correlation design
    C = np.corrcoef(X, rowvar=False)
    blocks0 = {k: [j - 1 for j in v] for k, v in BLOCKS.items()}
    within = {}
    for name, ix in blocks0.items():
        vals = [abs(C[a, b]) for a in ix for b in ix if a < b]
        within[name] = {"mean_abs": float(np.mean(vals)),
                        "min_abs": float(np.min(vals)),
                        "max_abs": float(np.max(vals))}
    across_vals = []
    names = list(blocks0)
    for i1 in range(len(names)):
        for i2 in range(i1 + 1, len(names)):
            across_vals += [abs(C[a, b]) for a in blocks0[names[i1]]
                            for b in blocks0[names[i2]]]
    across = {"mean_abs": float(np.mean(across_vals)),
              "max_abs": float(np.max(across_vals))}
    assert across["mean_abs"] < min(w["mean_abs"] for w in within.values()) / 2, (
        f"{where}: across-block correlation not substantially smaller")
    proxy_primary_corr = [float(C[15 + i, i]) for i in range(15)]
    assert min(proxy_primary_corr) > 0.60, f"{where}: weak proxy"
    off = np.abs(C - np.eye(P))
    assert off.max() < NEAR_DUPLICATE_CORR, (
        f"{where}: near-duplicate columns, |corr| = {off.max():.4f}")
    iu = np.triu_indices(P, k=1)
    order = np.argsort(off[iu])[::-1][:5]
    strongest = [{"pair": [f"x{iu[0][o] + 1}", f"x{iu[1][o] + 1}"],
                  "abs_corr": float(off[iu][o])} for o in order]

    # E. outcome
    r2 = _true_spec_r2(df)
    assert R2_TOL[0] <= r2 <= R2_TOL[1], f"{where}: realized R2 {r2:.6f}"
    assert r2 < 1.0

    return {
        "validated_on": where,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "design_rank": P,
        "residualized_design_rank": P,
        "gram_min_eigenvalue": float(eigvals[0]),
        "gram_max_eigenvalue": float(eigvals[-1]),
        "gram_condition_number": cond,
        "cholesky_float64_models_checked": list(CHOLESKY_MODELS),
        "within_block_abs_correlation": within,
        "across_block_abs_correlation": across,
        "proxy_primary_correlation_min": float(min(proxy_primary_corr)),
        "proxy_primary_correlation_max": float(max(proxy_primary_corr)),
        "strongest_absolute_pairwise_correlations": strongest,
        "max_offdiagonal_abs_correlation": float(off.max()),
        "realized_r2_true_specification": r2,
        "y_mean": float(df["y"].mean()),
        "y_sd": float(df["y"].std(ddof=1)),
        "y_variance": float(df["y"].var(ddof=1)),
    }


def main() -> int:
    frozen_before = {
        stem: {ext: file_sha256(OUT_DIR / f"{stem}.{ext}")
               for ext in ("csv", "parquet", "dta")}
        for stem in FROZEN_STEMS
    }

    print("proposed structural beta vector (x1..x30):")
    print(structural_beta().tolist())

    df, meta = generate_panel_30_center15(seed=SEED)
    diag_mem = validate(df, "in-memory dataframe")
    print(f"in-memory validation OK: R2 = {diag_mem['realized_r2_true_specification']:.6f}, "
          f"cond = {diag_mem['gram_condition_number']:.1f}")

    # canonical artifact: Parquet only, written from the validated frame
    df.to_parquet(PARQUET, index=False)
    back = pd.read_parquet(PARQUET)
    rep = compare_frames_exact(df, back, float_atol=0.0)
    assert rep["pass"], f"parquet reload not bit-identical: {rep}"
    diag_file = validate(back, "reloaded parquet file")
    for key in ("realized_r2_true_specification", "gram_condition_number",
                "gram_min_eigenvalue", "gram_max_eigenvalue"):
        assert diag_file[key] == diag_mem[key], f"reload diagnostic drift: {key}"
    print("parquet reload validation OK (bit-identical, diagnostics reproduced)")

    # guard: frozen artifacts untouched
    for stem, sums in frozen_before.items():
        for ext, sha in sums.items():
            assert file_sha256(OUT_DIR / f"{stem}.{ext}") == sha, (
                f"FROZEN ARTIFACT MODIFIED: {stem}.{ext}")
    print("frozen panel_8/panel_12/panel_30 artifacts verified unchanged")

    meta.update({
        "validation": {"in_memory": diag_mem, "reloaded_parquet": diag_file},
        "parquet_file": PARQUET.name,
        "parquet_sha256": file_sha256(PARQUET),
        "generator_sha256": file_sha256(Path(__file__)),
        "parquet_byte_stability_note": (
            "Parquet stores the float64 values exactly, but physical bytes "
            "embed the writer version (pyarrow 'created_by' footer) and may "
            "differ across pyarrow/pandas versions even when the numerical "
            "dataset is bit-identical; the numerical content is what the "
            "tests compare."),
        "creation_environment": {
            "python_version": sys.version.split()[0],
            "package_versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "pyarrow")
            },
        },
    })
    META_JSON.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {PARQUET.name} (sha256 {meta['parquet_sha256'][:16]}…) "
          f"and {META_JSON.name}")
    print(f"metadata json sha256 (self-hash, reported only): "
          f"{file_sha256(META_JSON)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
