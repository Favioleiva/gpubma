# Statistical specification (Phase 1 CPU reference)

## Model space

Optional predictors x_1..x_p define 2^p candidate models (validated
explicitly at run time). Always-included controls, the intercept, and fixed
effects form the "always block" A and appear in every candidate model without
enlarging the model space.

## Priors

- **Slopes (g-prior, PROVISIONAL):** after residualizing y and X_gamma on A,
  `beta_gamma | sigma^2 ~ N(0, g sigma^2 (X_gamma' X_gamma)^{-1})` with flat
  priors on the A-coefficients and `p(sigma^2) ∝ 1/sigma^2`.
  Default fixed g = max(n, p^2) — the FLS (2001) benchmark; Stata's
  `bmaregress` documents the same default, but parity is UNVERIFIED
  (see STATUS.md). The formulation is isolated in `gpubma/priors/gpriors.py`.
- **Model prior:** beta-binomial(a, b) with default a = b = 1 (uniform over
  model size); uniform over models also available
  (`gpubma/priors/model_priors.py`).

## Marginal likelihood

Let A have total effective rank r (explicit columns plus any rank absorbed by
a within transform), define df = n − r, and let ỹ, X̃ be the residualized
data. For model gamma with k predictors and R²_gamma computed on the
residualized data:

    log m(gamma) = ((df − k)/2) · log(1+g) − (df/2) · log(1 + g(1 − R²_gamma)) + C

with C common to all models (dropped). With r = 1 (intercept only) this is
exactly the null-based Bayes factor of Liang et al. (2008, JASA 103:410-423).
The generalization replaces n − 1 by df, which is the standard conditioning
of the g-prior on a larger always block; whether Stata makes the same choice
is an open validation question (STATUS.md).

Derivation sketch: integrating the A-coefficients (flat) and sigma^2
(Jeffreys) gives m(y|gamma) ∝ (1+g)^{−k/2} [ỹ'ỹ − s·ỹ'P_gamma ỹ]^{−df/2}
with s = g/(1+g); factoring ỹ'ỹ and rearranging yields the boxed formula.

## Posterior quantities

- Normalization by log-sum-exp over all 2^p log scores (stable).
- PMP: p(gamma|y) = exp(log score − logsumexp). Verified to sum to 1.
- PIP_j = Σ_{gamma ∋ j} p(gamma|y), clamped to [0,1] only after verifying any
  overshoot is < 1e-12 (pure summation noise).
- Conditional moments: E[beta|gamma] = s·beta_hat_gamma;
  Var[beta|gamma] = E[sigma^2|gamma] · s · (X̃'X̃)^{-1} with
  E[sigma^2|gamma] = (ỹ'ỹ − s·ESS_gamma)/(df − 2).
- BMA moments by the laws of total expectation/variance with excluded
  coefficients exactly 0. The **coefficient standard deviation formula is
  internally validated** (independent-formula tests) but **not yet validated
  against Stata**; treat cross-package sd comparisons as pending.

## Numerical policy

- float64 everywhere; per-model Cholesky on the Gram submatrix.
- 1 − R² floored at the smallest positive normal only to protect log1p;
  an R² > 1 would indicate a real numerical failure and is not silently fixed.
- No rounding before comparisons (validation module enforces this).

## Known unresolved statistical questions

Recorded in STATUS.md; chiefly: exact Stata parameterization (constant terms,
df convention with always-included blocks, PIP export names), g-prior
behaviour under absorbed fixed effects, and sd parity.
