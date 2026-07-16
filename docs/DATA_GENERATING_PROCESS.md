# Synthetic data-generating process

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
