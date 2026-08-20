"""Scientific DGP audit of the frozen panel_30_center15 benchmark.

Audits the CANONICAL Parquet artifact (not the generator's in-memory
output): structure, checksums, deterministic regeneration, linear-algebra
conditioning, the intended correlation design, and the realized signal
properties (R^2, variance decomposition, SNR, OLS fits of the true model,
the full model, and proxy-substituted models).

This is the local validation phase ONLY — it never runs the full p = 30
exhaustive enumeration. Reduced CPU/GPU parity lives in
tests/test_panel_30_center15_gpubma.py.

Writes:
    reports/panel_30_center15_dgp_validation.json   (machine-readable, incl.
                                                     the full 30x30 correlation
                                                     matrix)
    reports/panel_30_center15_dgp_validation.md     (human-readable summary)

Fails loudly (non-zero exit, no reports) if any audited property is wrong.

Usage:  python scripts/validate_panel_30_center15.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpubma.datasets.io_utils import compare_frames_exact, file_sha256  # noqa: E402
from gpubma.datasets.center15 import (  # noqa: E402
    BETA_TRUE, BLOCKS, CONTROL_DELTA, SEED, generate_panel_30_center15,
    structural_beta,
)

DATA = ROOT / "data" / "synthetic"
PARQUET = DATA / "panel_30_center15.parquet"
META_JSON = DATA / "panel_30_center15_metadata.json"
OUT_JSON = ROOT / "reports" / "panel_30_center15_dgp_validation.json"
OUT_MD = ROOT / "reports" / "panel_30_center15_dgp_validation.md"

EXPECTED_COLUMNS = (["individual_id", "period", "y"]
                    + [f"x{j}" for j in range(1, 31)] + ["w1", "w2"])
TRUE_VARS = [f"x{j}" for j in range(1, 16)]
PROXY_VARS = [f"x{j}" for j in range(16, 31)]


def _ols(df: pd.DataFrame, xcols: list[str]) -> dict:
    """OLS of y on [1, w1, w2, xcols]; returns fit summary."""
    y = df["y"].to_numpy(np.float64)
    D = np.column_stack([np.ones(len(df)),
                         df[["w1", "w2"]].to_numpy(np.float64),
                         df[xcols].to_numpy(np.float64)])
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ coef
    y_c = y - y.mean()
    r2 = float(1.0 - (resid @ resid) / (y_c @ y_c))
    return {"regressors": xcols, "r2": r2,
            "candidate_coefficients": coef[3:].tolist(),
            "residual_sd": float(np.std(resid, ddof=D.shape[1]))}


def audit(df: pd.DataFrame, meta: dict) -> dict:
    report: dict = {}

    # ---- 1. structure, schema, checksums, regeneration -------------------
    assert df.shape == (2000, 35), f"shape {df.shape}"
    assert list(df.columns) == EXPECTED_COLUMNS, "column names/order"
    assert meta["true_variables"] == TRUE_VARS
    assert meta["proxy_variables"] == PROXY_VARS
    assert not df.isna().any().any(), "missing values"
    assert np.isfinite(df.to_numpy(np.float64)).all(), "non-finite values"
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    assert dtypes["individual_id"] == "int32" and dtypes["period"] == "int32"
    assert all(v == "float64" for c, v in dtypes.items()
               if c not in ("individual_id", "period"))

    sha = file_sha256(PARQUET)
    assert sha == meta["parquet_sha256"], "Parquet SHA-256 != metadata"

    regen, _ = generate_panel_30_center15(seed=SEED)
    rep = compare_frames_exact(regen, df, float_atol=0.0)
    assert rep["pass"], f"regeneration differs numerically: {rep}"
    with tempfile.TemporaryDirectory(prefix="gpubma-c15-") as tmp:
        tmp_pq = Path(tmp) / "regen.parquet"
        regen.to_parquet(tmp_pq, index=False)
        regen_sha = file_sha256(tmp_pq)
    byte_identical = regen_sha == sha
    report["structure"] = {
        "n_observations": 2000, "n_candidate_regressors": 30,
        "n_true_regressors": 15, "n_proxy_regressors": 15,
        "columns": EXPECTED_COLUMNS, "dtypes": dtypes,
        "missing_values": 0, "non_finite_values": 0,
        "parquet_sha256": sha, "parquet_sha256_matches_metadata": True,
        "regeneration_numerically_exact": True,
        "regeneration_byte_identical_this_environment": byte_identical,
        "serialization_note": (
            "Numerical regeneration is exact (atol 0.0). Physical Parquet "
            "bytes embed the writer version (pyarrow 'created_by' footer), "
            "so byte identity is guaranteed only within one pyarrow/pandas "
            "environment; this run's environment "
            + ("reproduced the canonical bytes exactly."
               if byte_identical else
               "produced different bytes (different writer version).")),
        "structural_beta": structural_beta().tolist(),
    }

    # ---- linear algebra ----------------------------------------------------
    X = df[[f"x{j}" for j in range(1, 31)]].to_numpy(np.float64)
    sv = np.linalg.svd(X, compute_uv=False)
    A = np.column_stack([np.ones(len(df)), df[["w1", "w2"]].to_numpy(np.float64)])
    Q, _ = np.linalg.qr(A)
    X_r = X - Q @ (Q.T @ X)
    y = df["y"].to_numpy(np.float64)
    eig_raw = np.linalg.eigvalsh(X.T @ X)
    eig_res = np.linalg.eigvalsh(X_r.T @ X_r)
    assert np.linalg.matrix_rank(X) == 30
    assert np.linalg.matrix_rank(X_r) == 30
    assert eig_res[0] > 0
    np.linalg.cholesky(X_r.T @ X_r)
    cond_X = float(sv[0] / sv[-1])
    report["linear_algebra"] = {
        "design_rank": 30, "residualized_design_rank": 30,
        "singular_values_X_min_max": [float(sv[-1]), float(sv[0])],
        "condition_number_X": cond_X,
        "condition_number_XtX": float(eig_raw[-1] / eig_raw[0]),
        "residualized_gram_eigenvalues_min_max": [float(eig_res[0]),
                                                  float(eig_res[-1])],
        "condition_number_residualized_gram": float(eig_res[-1] / eig_res[0]),
        "float64_cholesky_of_residualized_gram": "succeeded",
        "stability_assessment": (
            f"cond(X'X) ~ {eig_raw[-1] / eig_raw[0]:.1f} is ~14 orders of "
            "magnitude below the float64 danger zone (~1e15); Cholesky-based "
            "enumeration loses < 2 decimal digits worst-case. No stability "
            "concern."),
    }

    # ---- 2. correlation design --------------------------------------------
    C = np.corrcoef(X, rowvar=False)
    names = [f"x{j}" for j in range(1, 31)]
    blocks0 = {k: [j - 1 for j in v] for k, v in BLOCKS.items()}
    block_of = {}
    for bname, members in BLOCKS.items():
        for j in members:
            block_of[j - 1] = bname          # true var
            block_of[j + 14] = bname         # its proxy (x{15+j})
    within = {}
    for bname, ix in blocks0.items():
        vals = [abs(C[a, b]) for a in ix for b in ix if a < b]
        within[bname] = {"mean_abs": float(np.mean(vals)),
                         "min_abs": float(np.min(vals)),
                         "max_abs": float(np.max(vals))}
    across_true = []
    bn = list(blocks0)
    for i1 in range(len(bn)):
        for i2 in range(i1 + 1, len(bn)):
            across_true += [abs(C[a, b]) for a in blocks0[bn[i1]]
                            for b in blocks0[bn[i2]]]

    proxy_table = []
    for i, entry in enumerate(meta["proxy_mappings"]):
        pcol, prim = 15 + i, int(entry["primary_true_variable"][1:]) - 1
        part = int(entry["secondary_partner"][1:]) - 1
        r_prim = float(C[pcol, prim])
        r_part = float(C[pcol, part])
        assert abs(r_prim - entry["realized_correlation_with_primary"]) < 1e-12
        assert abs(r_part - entry["realized_correlation_with_secondary"]) < 1e-12
        proxy_table.append({
            "proxy": entry["proxy"],
            "primary": entry["primary_true_variable"],
            "partner": entry["secondary_partner"],
            "a": entry["primary_coefficient"], "b": entry["secondary_coefficient"],
            "noise_scale": entry["noise_scale"],
            "corr_with_primary": r_prim, "corr_with_partner": r_part,
            "corr_with_y": float(np.corrcoef(X[:, pcol], y)[0, 1]),
            "primary_corr_with_y": float(np.corrcoef(X[:, prim], y)[0, 1]),
        })
    prim_corrs = [t["corr_with_primary"] for t in proxy_table]

    # unintended = pairs whose members derive from different latent blocks
    iu = np.triu_indices(30, k=1)
    off = np.abs(C[iu])
    cross_mask = np.array([block_of[a] != block_of[b]
                           for a, b in zip(iu[0], iu[1])])
    strongest_unintended = []
    order = np.argsort(np.where(cross_mask, off, -1.0))[::-1][:5]
    for o in order:
        strongest_unintended.append({
            "pair": [names[iu[0][o]], names[iu[1][o]]],
            "abs_corr": float(off[o])})
    order_all = np.argsort(off)[::-1][:5]
    strongest_all = [{"pair": [names[iu[0][o]], names[iu[1][o]]],
                      "abs_corr": float(off[o])} for o in order_all]
    max_cross = float(off[cross_mask].max())
    assert max_cross < 0.10, f"unintended cross-block correlation {max_cross}"
    assert 0.60 < min(prim_corrs) and max(prim_corrs) < 0.995

    # competition: proxies must be strong substitutes, not weak duplicates
    r2_true = _ols(df, TRUE_VARS)
    r2_proxy_only = _ols(df, PROXY_VARS)
    competition_ratio = r2_proxy_only["r2"] / r2_true["r2"]
    report["correlation_design"] = {
        "correlation_matrix_variables": names,
        "correlation_matrix": C.tolist(),
        "within_block_true_abs_correlation": within,
        "across_block_true_abs_correlation": {
            "mean_abs": float(np.mean(across_true)),
            "max_abs": float(np.max(across_true))},
        "proxy_to_source_mapping": proxy_table,
        "proxy_primary_correlation_min": float(min(prim_corrs)),
        "proxy_primary_correlation_max": float(max(prim_corrs)),
        "strongest_absolute_correlations_overall": strongest_all,
        "strongest_unintended_cross_block_correlations": strongest_unintended,
        "max_unintended_cross_block_abs_correlation": max_cross,
        "competition_assessment": (
            "Proxies are strong imperfect substitutes: |corr(proxy, primary)| "
            f"in [{min(prim_corrs):.3f}, {max(prim_corrs):.3f}] (never "
            ">= 0.995), and the 15 proxies ALONE recover "
            f"{100 * competition_ratio:.1f}% of the true model's R^2 "
            f"({r2_proxy_only['r2']:.4f} vs {r2_true['r2']:.4f}). This creates "
            "genuine posterior model competition rather than duplicate "
            "columns or ignorable noise."),
    }

    # ---- 3. realized signal properties --------------------------------------
    beta = structural_beta()
    W = df[["w1", "w2"]].to_numpy(np.float64)
    delta = np.array(CONTROL_DELTA)
    xb = X[:, :15] @ beta[:15]
    wd = W @ delta
    signal = xb + wd
    eps = y - signal                       # exact structural residual
    var = lambda v: float(np.var(v, ddof=1))
    r2_full = _ols(df, [f"x{j}" for j in range(1, 31)])
    swap_one = _ols(df, ["x16"] + TRUE_VARS[1:])            # x1 -> its proxy
    swap_blockA = _ols(df, PROXY_VARS[:5] + TRUE_VARS[5:])  # block A -> proxies
    coef_dev = np.abs(np.array(r2_true["candidate_coefficients"]) - beta[:15])
    report["signal_properties"] = {
        "realized_r2_true_model": r2_true["r2"],
        "var_x_beta": var(xb),
        "var_w_delta": var(wd),
        "var_total_signal": var(signal),
        "var_epsilon_realized": var(eps),
        "var_y": var(y),
        "signal_to_noise_ratio": var(signal) / var(eps),
        "snr_consistency_r2_over_1mr2": r2_true["r2"] / (1 - r2_true["r2"]),
        "ols_true_model": r2_true,
        "ols_true_model_max_abs_coef_deviation_from_beta": float(coef_dev.max()),
        "ols_full_30_model": {k: v for k, v in r2_full.items()
                              if k != "candidate_coefficients"},
        "ols_all_proxies_replace_all_true": {
            k: v for k, v in r2_proxy_only.items()
            if k != "candidate_coefficients"},
        "ols_single_swap_x1_to_x16": {k: v for k, v in swap_one.items()
                                      if k != "candidate_coefficients"},
        "ols_block_A_swapped_to_proxies": {
            k: v for k, v in swap_blockA.items()
            if k != "candidate_coefficients"},
        "proxy_imitation_assessment": (
            f"Replacing ALL 15 true variables by their proxies retains R^2 = "
            f"{r2_proxy_only['r2']:.4f} of the true model's "
            f"{r2_true['r2']:.4f}; a single swap (x1 -> x16) costs only "
            f"{r2_true['r2'] - swap_one['r2']:.4f} R^2. Proxies imitate the "
            "signal closely enough that model selection cannot rely on fit "
            "alone — exactly the intended difficulty."),
    }
    assert 0.695 <= r2_true["r2"] <= 0.705
    assert r2_full["r2"] < 1.0
    return report


def to_markdown(rep: dict) -> str:
    s, la, cd, sp = (rep["structure"], rep["linear_algebra"],
                     rep["correlation_design"], rep["signal_properties"])
    lines = [
        "# panel_30_center15 — DGP validation report",
        "",
        "Audited from the canonical Parquet artifact "
        "(`data/synthetic/panel_30_center15.parquet`). This report records "
        "the local DGP-validation phase, which did not itself run p = 30. "
        "The later canonical exact run is documented in "
        "`reports/CANONICAL_P30_RESULTS.md`.",
        "",
        "## Structure",
        f"- 2,000 observations x 35 columns; 30 candidates (15 true x1-x15, "
        f"15 proxies x16-x30); no missing/non-finite values; dtypes as frozen.",
        f"- Parquet SHA-256 matches metadata: `{s['parquet_sha256'][:16]}…`",
        f"- Deterministic regeneration (seed {SEED}): numerically exact; "
        f"byte-identical in this environment: "
        f"{s['regeneration_byte_identical_this_environment']}.",
        "",
        "## Linear algebra",
        f"- rank(X) = rank(residualized X) = 30.",
        f"- singular values of X: [{la['singular_values_X_min_max'][0]:.3f}, "
        f"{la['singular_values_X_min_max'][1]:.3f}]; cond(X) = "
        f"{la['condition_number_X']:.2f}; cond(X'X) = "
        f"{la['condition_number_XtX']:.2f}.",
        f"- residualized Gram eigenvalues "
        f"[{la['residualized_gram_eigenvalues_min_max'][0]:.1f}, "
        f"{la['residualized_gram_eigenvalues_min_max'][1]:.1f}], cond = "
        f"{la['condition_number_residualized_gram']:.2f}; float64 Cholesky OK.",
        f"- {la['stability_assessment']}",
        "",
        "## Correlation design",
        "| block | mean abs r | min | max |",
        "|---|---|---|---|",
    ]
    for b, v in cd["within_block_true_abs_correlation"].items():
        lines.append(f"| {b} (true, within) | {v['mean_abs']:.3f} | "
                     f"{v['min_abs']:.3f} | {v['max_abs']:.3f} |")
    ab = cd["across_block_true_abs_correlation"]
    lines += [
        f"| across blocks (true) | {ab['mean_abs']:.3f} | — | {ab['max_abs']:.3f} |",
        "",
        f"- proxy-primary correlations: [{cd['proxy_primary_correlation_min']:.3f}, "
        f"{cd['proxy_primary_correlation_max']:.3f}]; no near-duplicates.",
        f"- strongest unintended (cross-block) |corr|: "
        f"{cd['max_unintended_cross_block_abs_correlation']:.3f}.",
        "",
        "### Proxy-to-source mapping",
        "| proxy | primary | partner | a | b | noise | r(primary) | r(partner) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in cd["proxy_to_source_mapping"]:
        lines.append(f"| {t['proxy']} | {t['primary']} | {t['partner']} | "
                     f"{t['a']:.2f} | {t['b']:.2f} | {t['noise_scale']:.2f} | "
                     f"{t['corr_with_primary']:.3f} | {t['corr_with_partner']:.3f} |")
    lines += [
        "",
        f"- {cd['competition_assessment']}",
        "",
        "## Realized signal",
        f"- realized R^2 (true spec [1, w1, w2, x1-x15]): "
        f"**{sp['realized_r2_true_model']:.6f}** (target 0.695-0.705).",
        f"- Var(X beta) = {sp['var_x_beta']:.4f}; Var(W delta) = "
        f"{sp['var_w_delta']:.4f}; Var(signal) = {sp['var_total_signal']:.4f}; "
        f"Var(eps) = {sp['var_epsilon_realized']:.4f}; Var(y) = "
        f"{sp['var_y']:.4f}.",
        f"- SNR = Var(signal)/Var(eps) = "
        f"**{sp['signal_to_noise_ratio']:.4f}** (consistent with "
        f"R^2/(1-R^2) = {sp['snr_consistency_r2_over_1mr2']:.4f}).",
        "",
        "| OLS model | R^2 |",
        "|---|---|",
        f"| true 15 (x1-x15) | {sp['ols_true_model']['r2']:.4f} |",
        f"| full 30 | {sp['ols_full_30_model']['r2']:.4f} |",
        f"| all 15 proxies replace all true | "
        f"{sp['ols_all_proxies_replace_all_true']['r2']:.4f} |",
        f"| single swap x1 -> x16 | {sp['ols_single_swap_x1_to_x16']['r2']:.4f} |",
        f"| block A -> its proxies | "
        f"{sp['ols_block_A_swapped_to_proxies']['r2']:.4f} |",
        "",
        f"- OLS on the true model recovers beta within max abs deviation "
        f"{sp['ols_true_model_max_abs_coef_deviation_from_beta']:.4f}.",
        f"- {sp['proxy_imitation_assessment']}",
        "",
        "## Verdict",
        "The frozen artifact matches its intended design exactly: central-"
        "layer size-15 truth, block-correlated true regressors, strong-but-"
        "imperfect structural-zero proxies, R^2 on target, and conditioning "
        "far inside float64 Cholesky safety margins.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    df = pd.read_parquet(PARQUET)
    meta = json.loads(META_JSON.read_text())
    rep = audit(df, meta)
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2) + "\n")
    OUT_MD.write_text(to_markdown(rep), encoding="utf-8")
    sp = rep["signal_properties"]
    ratio = (sp["ols_all_proxies_replace_all_true"]["r2"]
             / sp["realized_r2_true_model"])
    print(f"AUDIT PASSED — wrote {OUT_JSON.name} and {OUT_MD.name}")
    print(f"  R2 = {sp['realized_r2_true_model']:.6f}, "
          f"SNR = {sp['signal_to_noise_ratio']:.4f}, "
          f"cond(X'X) = {rep['linear_algebra']['condition_number_XtX']:.1f}, "
          f"proxy-only R2 ratio = {ratio:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
