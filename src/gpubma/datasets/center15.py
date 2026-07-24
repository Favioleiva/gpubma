"""Deterministic generator for the panel_30_center15 benchmark.

A second, harder p = 30 synthetic benchmark, distinct from the frozen
``panel_30`` artifact (which has a sparse size-5 truth and independent
predictors). This one places the true model in the CENTRAL Pascal layer
and forces posterior competition through collinearity and proxies:

- p = 30 candidate regressors, true model size k_true = 15
  (the central layer C(30,15) = 155,117,520 of the 2^30 = 1,073,741,824
  model space);
- true regressors x1-x15 drawn from three latent-factor blocks
  (x1-x5 <- F_A, x6-x10 <- F_B, x11-x15 <- F_C) with heterogeneous
  deterministic loadings in [0.60, 0.85]:

      x_j_raw = loading_j * F_block + sqrt(1 - loading_j^2) * eta_j,
      F_block, eta_j ~ N(0, 1)

  so variables are meaningfully correlated within blocks and nearly
  uncorrelated across blocks;
- proxy regressors x16-x30 with structural coefficients EXACTLY zero:
  each proxy is a noisy linear combination of its primary true regressor
  (one-to-one map x{15+j} <-> x{j}), a same-block partner, and noise:

      proxy_raw = a * true_primary_raw + b * true_partner_raw
                + sigma_proxy * nu,          nu ~ N(0, 1)

  with heterogeneous deterministic a in [0.70, 0.90], b in [0.10, 0.35];
- all 30 candidate columns standardized with the sample mean and sample
  standard deviation (ddof = 1) AFTER generation;
- outcome  y = X_std[:, :15] @ beta_true + W @ delta + sigma_eps * eps,
  eps ~ N(0, 1), with sigma_eps calibrated ANALYTICALLY (closed form, one
  fixed seed, no seed search) so the realized in-sample OLS R^2 of the
  true specification [1, w1, w2, x1..x15] equals the 0.70 target.

Schema compatibility: identical column names, ordering, and dtypes as
``panel_30`` (individual_id, period, y, x1..x30, w1, w2; a balanced
200 x 10 panel = 2,000 rows). Unlike ``panel_30`` there are NO individual
or time effects — the identifiers exist purely for schema compatibility,
so the benchmark isolates the correlated-proxy challenge (documented).

Determinism: every random column has its own
``numpy.random.default_rng([seed, n_individuals, n_periods, stream_tag])``
stream, the same convention as :mod:`gpubma.datasets.synthetic`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATASET_NAME = "panel_30_center15"
DATASET_VERSION = 1
SEED = 20260724  # distinct from the panel_8/12/30 seed (20260715)

N_INDIVIDUALS = 200
N_PERIODS = 10
N_OBS = N_INDIVIDUALS * N_PERIODS  # 2,000
P = 30
K_TRUE = 15
TARGET_R2 = 0.700

# --- latent-factor blocks for the true regressors x1..x15 -------------------
BLOCKS = {"A": list(range(1, 6)), "B": list(range(6, 11)), "C": list(range(11, 16))}

# heterogeneous deterministic loadings in [0.60, 0.85], one per true regressor
LOADINGS = np.array([
    0.85, 0.72, 0.66, 0.78, 0.61,   # block A: x1-x5
    0.83, 0.64, 0.75, 0.69, 0.80,   # block B: x6-x10
    0.62, 0.84, 0.71, 0.77, 0.65,   # block C: x11-x15
], dtype=np.float64)

# --- true coefficients (prompt-specified starting vector, used unchanged) ---
BETA_TRUE = np.array([
    1.20, -1.10, 1.00, -0.90, 0.80,
    0.75, -0.70, 0.65, -0.60, 0.55,
    0.50, -0.45, 0.40, -0.35, 0.30,
], dtype=np.float64)

CONTROL_DELTA = (0.8, -0.6)  # same always-included controls as panel_30

# --- proxy construction x16..x30 --------------------------------------------
# primary map is one-to-one: x{15+j} <-> x{j}; the partner is the next true
# regressor within the same block (cyclic within the block of five).
PROXY_PRIMARY = list(range(1, 16))                     # x16->x1, ..., x30->x15
PROXY_PARTNER = [
    ((j - 1) % 5 + 1) % 5 + 1 + 5 * ((j - 1) // 5)     # cyclic same-block next
    for j in PROXY_PRIMARY
]
PROXY_A = np.array([
    0.88, 0.74, 0.81, 0.70, 0.85,
    0.77, 0.90, 0.72, 0.79, 0.86,
    0.71, 0.83, 0.76, 0.89, 0.73,
], dtype=np.float64)
PROXY_B = np.array([
    0.15, 0.32, 0.10, 0.27, 0.20,
    0.34, 0.12, 0.25, 0.18, 0.30,
    0.14, 0.22, 0.35, 0.11, 0.28,
], dtype=np.float64)
PROXY_SIGMA = np.array([
    0.50, 0.62, 0.45, 0.58, 0.52,
    0.47, 0.60, 0.55, 0.49, 0.63,
    0.53, 0.46, 0.59, 0.51, 0.56,
], dtype=np.float64)

# RNG stream tags (one independent stream per random column)
_STREAM_EPS = 3
_STREAM_FACTOR_BASE = 10      # F_A = 11, F_B = 12, F_C = 13
_STREAM_ETA_BASE = 1000       # eta for true regressor j uses 1000 + j
_STREAM_NU_BASE = 3000        # nu  for proxy x{15+j}   uses 3000 + j
_STREAM_W_BASE = 2000         # control c uses 2000 + c


def structural_beta() -> np.ndarray:
    """The full 30-element structural coefficient vector (15 nonzero + 15 zeros)."""
    beta = np.zeros(P, dtype=np.float64)
    beta[:K_TRUE] = BETA_TRUE
    return beta


def _standardize(M: np.ndarray) -> np.ndarray:
    """Column-wise (x - sample mean) / sample sd, ddof = 1."""
    return (M - M.mean(axis=0)) / M.std(axis=0, ddof=1)


def generate_panel_30_center15(seed: int = SEED):
    """Generate the benchmark deterministically. Returns ``(df, meta)``.

    ``meta`` records the complete structural truth (blocks, loadings, proxy
    mappings and coefficients, beta, controls, calibration); file-level
    checksums and environment facts are added by the generator script.
    """
    def rng(tag: int) -> np.random.Generator:
        return np.random.default_rng([seed, N_INDIVIDUALS, N_PERIODS, tag])

    # latent factors and true regressors (raw)
    F = {name: rng(_STREAM_FACTOR_BASE + i + 1).standard_normal(N_OBS)
         for i, name in enumerate(("A", "B", "C"))}
    X_raw = np.empty((N_OBS, P), dtype=np.float64)
    for name, members in BLOCKS.items():
        for j in members:
            lam = LOADINGS[j - 1]
            eta = rng(_STREAM_ETA_BASE + j).standard_normal(N_OBS)
            X_raw[:, j - 1] = lam * F[name] + np.sqrt(1.0 - lam * lam) * eta

    # proxies (raw), built from the raw true regressors
    for i, j in enumerate(PROXY_PRIMARY):
        col = K_TRUE + i  # 0-based column of x{16+i}
        nu = rng(_STREAM_NU_BASE + j).standard_normal(N_OBS)
        X_raw[:, col] = (PROXY_A[i] * X_raw[:, j - 1]
                         + PROXY_B[i] * X_raw[:, PROXY_PARTNER[i] - 1]
                         + PROXY_SIGMA[i] * nu)

    # standardize ALL 30 candidates with sample statistics
    X = _standardize(X_raw)

    # controls: independent of every candidate regressor
    W = np.column_stack([rng(_STREAM_W_BASE + c).standard_normal(N_OBS)
                         for c in (1, 2)])
    delta = np.array(CONTROL_DELTA, dtype=np.float64)

    # outcome: analytic sigma calibration to the realized-R^2 target.
    # signal lies exactly in span([1, w1, w2, x1..x15]) (standardization is
    # affine), so with the design annihilator M and centering M1:
    #   RSS(sigma) = sigma^2 ||M e||^2,
    #   TSS(sigma) = ||M1 s||^2 + 2 sigma <M1 s, M1 e> + sigma^2 ||M1 e||^2,
    # and R2 = 1 - RSS/TSS = TARGET_R2 is a quadratic in sigma:
    #   sigma^2 (||M e||^2 - (1-R2*)||M1 e||^2)
    #     - 2 (1-R2*) sigma <M1 s, M1 e> - (1-R2*) ||M1 s||^2 = 0.
    # One fixed seed; the positive root is used. No seed search.
    signal = X[:, :K_TRUE] @ BETA_TRUE + W @ delta
    eps_unit = rng(_STREAM_EPS).standard_normal(N_OBS)

    design = np.column_stack([np.ones(N_OBS), W, X[:, :K_TRUE]])
    Q, _ = np.linalg.qr(design)
    Me = eps_unit - Q @ (Q.T @ eps_unit)
    s_c = signal - signal.mean()
    e_c = eps_unit - eps_unit.mean()
    r = 1.0 - TARGET_R2
    a_coef = float(Me @ Me - r * (e_c @ e_c))
    b_coef = float(-2.0 * r * (s_c @ e_c))
    c_coef = float(-r * (s_c @ s_c))
    disc = b_coef * b_coef - 4.0 * a_coef * c_coef
    sigma_eps = float((-b_coef + np.sqrt(disc)) / (2.0 * a_coef))
    assert sigma_eps > 0.0

    y = signal + sigma_eps * eps_unit

    ind = np.repeat(np.arange(1, N_INDIVIDUALS + 1), N_PERIODS)
    per = np.tile(np.arange(1, N_PERIODS + 1), N_INDIVIDUALS)
    df = pd.DataFrame({"individual_id": ind.astype(np.int32),
                       "period": per.astype(np.int32),
                       "y": y})
    for j in range(1, P + 1):
        df[f"x{j}"] = X[:, j - 1]
    df["w1"] = W[:, 0]
    df["w2"] = W[:, 1]

    # realized quantities recorded from the actual frozen sample
    resid = y - Q @ (Q.T @ y)
    y_c = y - y.mean()
    realized_r2 = float(1.0 - (resid @ resid) / (y_c @ y_c))

    proxy_map = []
    for i, j in enumerate(PROXY_PRIMARY):
        name = f"x{16 + i}"
        proxy_map.append({
            "proxy": name,
            "primary_true_variable": f"x{j}",
            "secondary_partner": f"x{PROXY_PARTNER[i]}",
            "primary_coefficient": float(PROXY_A[i]),
            "secondary_coefficient": float(PROXY_B[i]),
            "noise_scale": float(PROXY_SIGMA[i]),
            "realized_correlation_with_primary": float(
                np.corrcoef(df[name], df[f"x{j}"])[0, 1]),
            "realized_correlation_with_secondary": float(
                np.corrcoef(df[name], df[f"x{PROXY_PARTNER[i]}"])[0, 1]),
        })

    meta = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "purpose": (
            "Central-layer correlated-proxy benchmark: tests exact exhaustive "
            "BMA posterior recovery under collinearity and proxy substitution "
            "(true model of size 15 inside the central Pascal layer), not "
            "merely sparse variable selection."),
        "distinction_from_panel_30": (
            "panel_30 (unchanged) has a sparse size-5 truth with independent "
            "i.i.d. predictors and unmodeled individual/time effects; "
            "panel_30_center15 has k_true = 15, three correlated latent-factor "
            "blocks, 15 structural-zero proxies, no individual or time "
            "effects, and n = 2,000."),
        "canonical_format": "parquet",
        "no_csv_or_dta": (
            "No CSV or Stata .dta copy is provided: GPUBMA consumes the data "
            "in Python, Parquet stores float64 exactly, and duplicate formats "
            "would add storage and validation complexity."),
        "generator_path": "scripts/generate_panel_30_center15.py",
        "generator_module": "gpubma.datasets.center15.generate_panel_30_center15",
        "seed": seed,
        "rng": ("numpy.random.default_rng([seed, n_individuals, n_periods, "
                "stream_tag]), one independent stream per random column"),
        "n_individuals": N_INDIVIDUALS,
        "n_periods": N_PERIODS,
        "n_observations": N_OBS,
        "panel_effects": (
            "NONE by design — individual_id/period exist only for schema "
            "compatibility with panel_30; the benchmark isolates the "
            "correlated-predictor and proxy challenge."),
        "n_candidate_regressors": P,
        "true_model_size": K_TRUE,
        "true_variables": [f"x{j}" for j in range(1, 16)],
        "proxy_variables": [f"x{j}" for j in range(16, 31)],
        "structural_beta": structural_beta().tolist(),
        "structural_zero_proxies": (
            "x16-x30 have structural coefficients exactly 0 in the outcome "
            "equation."),
        "always_included_controls": {"names": ["w1", "w2"],
                                     "delta": list(CONTROL_DELTA),
                                     "distribution": "i.i.d. N(0,1), independent "
                                                     "of all candidate regressors"},
        "latent_factor_blocks": {k: [f"x{j}" for j in v] for k, v in BLOCKS.items()},
        "factor_loadings": LOADINGS.tolist(),
        "true_regressor_equation": (
            "x_j_raw = loading_j * F_block + sqrt(1 - loading_j^2) * eta_j; "
            "F_block, eta_j ~ N(0,1)"),
        "proxy_equation": (
            "proxy_raw = a * primary_true_raw + b * partner_true_raw "
            "+ noise_scale * nu; nu ~ N(0,1)"),
        "proxy_mappings": proxy_map,
        "standardization": (
            "all 30 candidate columns standardized AFTER generation: "
            "(x - sample mean) / sample sd with ddof = 1; controls w1, w2 "
            "are left as drawn (raw N(0,1))"),
        "outcome_equation": (
            "y = sum_{j=1..15} beta_j * x_j_std + 0.8*w1 - 0.6*w2 "
            "+ sigma_eps * eps; eps ~ N(0,1); beta_j = 0 for j = 16..30"),
        "noise_calibration": {
            "procedure": (
                "analytic closed form, single fixed seed (no seed search): "
                "signal lies in span([1, w1, w2, x1..x15]), hence "
                "RSS(sigma) = sigma^2 ||M eps||^2 with M the design "
                "annihilator; setting R2(sigma) = 1 - RSS/TSS = 0.700 gives a "
                "quadratic in sigma whose positive root is sigma_eps"),
            "r2_regression_variables": ["intercept", "w1", "w2"]
                                       + [f"x{j}" for j in range(1, 16)],
            "theoretical_target_r2": TARGET_R2,
            "realized_r2": realized_r2,
            "sigma_eps": sigma_eps,
            "signal_variance": float(s_c @ s_c / (N_OBS - 1)),
            "target_noise_variance": sigma_eps**2,
            "realized_noise_variance": float(np.var(sigma_eps * eps_unit, ddof=1)),
        },
        "model_space_statements": {
            "total_models": 2**P,
            "total_models_note": "the complete model space contains 2^30 = "
                                 "1,073,741,824 models",
            "central_layer_size": 155117520,
            "central_layer_note": "the true model lies in the central layer "
                                  "C(30,15) = 155,117,520",
        },
        "row_order": "sorted by (individual_id, period)",
        "column_order": (["individual_id", "period", "y"]
                         + [f"x{j}" for j in range(1, 31)] + ["w1", "w2"]),
    }
    return df, meta
