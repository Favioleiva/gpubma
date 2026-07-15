# Statistical specification (Phase 1 CPU reference)

## Model space

Optional predictors x_1..x_p define 2^p candidate models (validated
explicitly at run time). Always-included controls, the intercept, and fixed
effects form the "always block" A and appear in every candidate model without
enlarging the model space.

## Priors

- **Slopes (Zellner g-prior, VERIFIED vs Stata):** `p(sigma^2) ∝ 1/sigma^2`,
  flat prior on the intercept, and `beta | sigma^2 ~ N(0, g sigma^2 (X'X)^{-1})`
  on slopes. Default fixed g = max(n, p^2) — the FLS (2001) benchmark and
  Stata's documented default (confirmed on executed runs: `e(g)` matched).
  The formulation lives in `gpubma/priors/gpriors.py`.
- **Model prior:** beta-binomial(a, b) over the OPTIONAL predictors only,
  default a = b = 1 (uniform over model size); uniform over models also
  available (`gpubma/priors/model_priors.py`). Stata's
  `mprior(betabinomial 1 1)` matches (confirmed: `e(msize_mean_prior)` =
  p/2 + n_always).

## Always-included blocks: two conventions

### `always_prior="shrink"` — Stata's convention (default, VERIFIED)

The g-prior covers the optional predictors AND the always-included slopes
(controls, fixed-effect dummies) **jointly**; only the intercept is flat.
For model gamma with k optional predictors, q always slopes, joint R² from
the centered regression on [X_gamma, W], and df = n − 1:

    log m(gamma) = ((n−1−k−q)/2)·log(1+g) − ((n−1)/2)·log(1 + g(1 − R²_joint)) + C

Verified on 2026-07-15 against executed StataNow/SE 19.5 `bmaregress`
exports on six designs (no FE; individual, time, two-way FE dummies;
Grunfeld with and without company FE): worst absolute difference across all
264 compared quantities (PIPs, posterior means, posterior sds, mean model
size) was **1.8e-12** (`reports/comparison_report.md`).

Implementation note (FWL): with X̃, ỹ residualized on [1, W],
ESS_joint = ESS_W + ESS_gamma(residualized), the x-block of the joint OLS
solution equals the residualized solution, and the x-block of the inverse
joint Gram equals (X̃'X̃)^{-1} — so the residualized fast path computes the
joint formula exactly. E[sigma^2|gamma] = (TSS_c − s·ESS_joint)/(n−3).

### `always_prior="flat"` — conditional convention (gpubma-specific)

Flat (improper) priors on the always block; A has total effective rank r
(explicit columns plus any rank absorbed by a within transform); df = n − r;
R² on residualized data:

    log m(gamma) = ((df − k)/2)·log(1+g) − (df/2)·log(1 + g(1 − R²_gamma)) + C

With r = 1 (intercept only) both conventions coincide and reduce to the
null-based Bayes factor of Liang et al. (2008, JASA 103:410-423). This
convention is NOT Stata's when q > 0; it is retained because it is the only
coherent treatment for absorbed (within-transformed) fixed effects.

## Posterior quantities

- Normalization by log-sum-exp over all 2^p log scores (stable).
- PMP: p(gamma|y) = exp(log score − logsumexp). Verified to sum to 1.
- PIP_j = Σ_{gamma ∋ j} p(gamma|y), clamped to [0,1] only after verifying any
  overshoot is < 1e-12 (pure summation noise).
- Conditional moments: E[beta|gamma] = s·beta_hat_gamma;
  Var[beta|gamma] = E[sigma^2|gamma] · s · (X̃'X̃)^{-1} with
  E[sigma^2|gamma] = (TSS_norm − s·ESS_total)/(df − 2), where TSS_norm and
  ESS_total follow the active convention above.
- BMA moments by the laws of total expectation/variance with excluded
  coefficients exactly 0. The **coefficient standard deviation formula is
  validated both internally and externally**: under `always_prior="shrink"`
  it reproduces sqrt(diag(e(V_bma))) from executed Stata runs to ~1e-12.

## Numerical policy

- float64 everywhere; per-model Cholesky on the Gram submatrix.
- 1 − R² floored at the smallest positive normal only to protect log1p;
  an R² > 1 would indicate a real numerical failure and is not silently fixed.
- No rounding before comparisons (validation module enforces this).

## Resolved and remaining statistical questions

Resolved on 2026-07-15 by executing the Stata oracle (see STATUS.md for the
full log): Stata's always-block parameterization (joint shrinkage, df = n−1),
the PIP/coefficient export names (e(pip), e(b_bma), e(V_bma)), model-size
accounting (always variables counted; subtract e(p_always)), and posterior-sd
parity. Remaining: dummies-vs-absorption equivalence under the shrink
convention does NOT hold (documented, by design); unbalanced two-way FE
unsupported; MC3 out of scope.
