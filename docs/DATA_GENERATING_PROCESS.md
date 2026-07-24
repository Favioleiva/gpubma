# Synthetic data-generating process

Two distinct p = 30 benchmarks exist:

1. **panel_30** (below, with panel_8/panel_12) — the original frozen sparse
   benchmark: k_true = 5, independent i.i.d. predictors, individual/time
   effects, multi-format artifact (CSV/Parquet/DTA). Unchanged.
2. **panel_30_center15** (last section) — the central-layer correlated-proxy
   benchmark: k_true = 15 in the central Pascal layer C(30,15), three
   correlated latent-factor blocks, 15 structural-zero proxy regressors,
   Parquet-only.

Generator: `gpubma.datasets.synthetic.generate_panel`
Regeneration: `python scripts/generate_synthetic_panels.py`
Default seed: **20260715**

## Model

For individual i = 1..N and period t = 1..T:

    y_it = alpha_i + lambda_t + sum_{j=1}^{p} beta_j x_{j,it}
         + sum_{c=1}^{2} delta_c w_{c,it} + eps_it

| component | distribution | default scale |
|---|---|---|
| alpha_i (individual effect) | N(0, sigma_alpha^2) | sigma_alpha = 1.0 |
| lambda_t (time effect) | N(0, sigma_lambda^2) | sigma_lambda = 0.5 |
| x_{j,it} (optional predictors) | i.i.d. N(0, 1), identity covariance | — |
| w_{c,it} (always-included controls) | i.i.d. N(0, 1) | — |
| eps_it | N(0, sigma_eps^2) | sigma_eps = 1.0 |

True coefficients:

    beta = (1.5, -1.0, 0.75, 0.5, 0.25, 0, 0, ..., 0)   # only x1..x5 matter
    delta = (0.8, -0.6)

## Determinism and cross-size nesting

Each random column has its own RNG stream:
`numpy.random.default_rng([seed, n_individuals, n_periods, stream_tag])`.
Streams depend on the seed and panel dimensions but **not** on the number of
predictors. Consequences (all covered by tests):

- Regeneration is bit-identical for a given (seed, N, T, p).
- The first 8 predictor columns of `panel_12` and `panel_30` are bit-identical
  to `panel_8`'s.
- `y` is identical across the three datasets because beta_j = 0 for j > 5.

## Frozen datasets

| dataset | N × T | obs | optional p | models 2^p |
|---|---|---|---|---|
| panel_8  | 100 × 10 | 1,000 | 8  | 256 |
| panel_12 | 100 × 10 | 1,000 | 12 | 4,096 |
| panel_30 | 100 × 10 | 1,000 | 30 | 1,073,741,824 (data artifact only in Phase 1) |

Each is written as CSV (`float_format="%.17g"`, exact float64 round trip via
`float_precision="round_trip"` on read), Parquet (exact), and Stata `.dta`
version 118 (float64 = Stata double, exact). Metadata JSON files record the
seed, DGP equation, true coefficients, individual/time effects, covariance
structure, error variance, dimensions, checksums (file SHA-256 and a
content hash), expected model counts, generation timestamp, and package
versions.

## Column roles

| role | columns |
|---|---|
| outcome | `y` |
| optional BMA predictors | `x1` … `xp` (define the 2^p model space) |
| always-included controls | `w1`, `w2` (never change the model count) |
| individual identifier / FE | `individual_id` |
| time identifier / FE | `period` |

## panel_30_center15 — central-layer correlated-proxy benchmark

Generator: `gpubma.datasets.center15.generate_panel_30_center15`
Regeneration: `python scripts/generate_panel_30_center15.py`
Seed: **20260724** (distinct from the panel_8/12/30 seed)
Canonical artifact: **Parquet only** (`data/synthetic/panel_30_center15.parquet`
plus `panel_30_center15_metadata.json`; no CSV or `.dta` copy — GPUBMA
consumes the data in Python and Parquet stores float64 exactly).

A deliberately harder p = 30 benchmark: the true model has **k_true = 15**
and lies in the central Pascal layer (C(30,15) = 155,117,520 of the
2^30 = 1,073,741,824 models). It tests posterior recovery under
collinearity and proxy substitution, not sparse selection.

Design (2,000 rows as a balanced 200 × 10 panel; identifiers exist only
for schema compatibility — there are **no** individual or time effects):

- **True regressors x1–x15**: three latent-factor blocks
  (x1–x5 ← F_A, x6–x10 ← F_B, x11–x15 ← F_C),
  `x_j_raw = loading_j · F_block + sqrt(1 − loading_j²) · eta_j` with
  heterogeneous deterministic loadings in [0.60, 0.85] — meaningful
  within-block correlation (mean |r| ≈ 0.51–0.55), near-zero across blocks
  (mean |r| ≈ 0.02).
- **Proxy regressors x16–x30**: structural coefficients exactly zero.
  One-to-one primary map x{15+j} ↔ x{j} plus a smaller same-block partner
  loading and independent noise:
  `proxy_raw = a·true_primary + b·true_partner + sigma_proxy·nu`
  (a ∈ [0.70, 0.90], b ∈ [0.10, 0.35]); realized proxy–primary
  correlations ≈ 0.79–0.89, never near-duplicates.
- All 30 candidates standardized after generation (sample mean/sd, ddof=1).
- **Truth**: `beta_true = (1.20, −1.10, 1.00, −0.90, 0.80, 0.75, −0.70,
  0.65, −0.60, 0.55, 0.50, −0.45, 0.40, −0.35, 0.30)` on x1–x15; zeros on
  x16–x30. Controls `w1, w2 ~ N(0,1)` independent of all candidates with
  delta = (0.8, −0.6).
- **Noise calibration**: sigma_eps solved **analytically** (closed-form
  quadratic, single fixed seed, no seed search) so the realized in-sample
  OLS R² of the true specification [1, w1, w2, x1–x15] is 0.700
  (tolerance 0.695–0.705).

The metadata JSON records the complete structural truth, factor loadings,
proxy mappings with realized correlations, calibration details, rank /
eigenvalue / condition-number diagnostics (residualized Gram condition
number ≈ 67), and SHA-256 checksums of the Parquet artifact and generator.
Tests: `tests/datasets/test_center15.py`.
