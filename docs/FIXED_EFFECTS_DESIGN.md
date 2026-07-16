# Fixed-effects design

Target API (long term):

```python
fixed_effects=["individual"]
fixed_effects=["time"]
fixed_effects=["individual", "time"]
fixed_effects=["region", "industry", "year"]   # later phase
```

Phase 1 implements `individual`, `time`, and their combination, through two
approaches that are compared quantitatively.

## Reference approach: explicit dummies (`fe_method="dummies"`)

- Base category: the **first level in sorted order** of each factor is
  dropped (documented, deterministic).
- The intercept is always present; the design `[1, D_ind, D_time, W]` is
  full-rank by construction and its rank is verified at run time (no
  dummy-variable trap).
- Fixed effects live in the always-included block: they are in **every**
  candidate model and never change the 2^p model count.

## Candidate production approach: residualization (`fe_method="within"`)

- One-way: subtract group means (individual or time).
- Two-way, balanced panels: `x − mean_i − mean_t + grand mean` (the exact
  projection for balanced panels). Unbalanced two-way panels raise an
  explicit error in Phase 1 (iterative demeaning is a later-phase item).
- The absorbed rank is reported: N_i, N_t, or N_i + N_t − 1, matching the
  rank of `[1, dummies]`, so the effective df is aligned with the dummy
  approach.

## Validation performed (scripts/compare_fixed_effects.py, tests)

- Effective always-block rank equal across approaches.
- OLS slopes equal (Frisch–Waugh–Lovell), atol 1e-9/1e-10.
- With identical fixed g and aligned df, all 256 BMA log scores, PMPs, PIPs,
  and coefficient moments coincide (see `reports/fixed_effects_comparison.md`).

## The statistical caveat — now settled by real Stata output

Equality of OLS slopes does **not** automatically establish equality of
Bayesian model scores. Under a g-prior, absorbing fixed effects may affect
prior parameterization, effective sample size, degrees of freedom, marginal
likelihood constants, and model comparison unless the always-included block
is treated consistently.

Executing the oracle (StataNow/SE 19.5, 2026-07-15) settled the question:

- **Stata's convention** (`always_prior="shrink"`, gpubma's default): the
  g-prior shrinks the fixed-effect dummies jointly with the optional
  predictors; only the intercept is flat and df = n − 1. gpubma reproduces
  Stata's PIPs, posterior means, and posterior sds to ~1e-12 for individual,
  time, and two-way FE dummies (`reports/comparison_report.md`).
- **Consequence:** under this convention, dummies-vs-absorption equivalence
  does NOT hold — absorption treats the fixed effects as flat, which is a
  different prior. `bma_regress` therefore refuses
  `always_prior="shrink"` with `fe_method="within"`.
- **gpubma's conditional convention** (`always_prior="flat"`): flat always
  block, df = n − rank(always). Under this convention (and only this one)
  explicit dummies and within absorption produce identical scores, PIPs and
  moments — verified in `scripts/compare_fixed_effects.py` and the test
  suite. This convention is NOT Stata-compatible when always variables are
  present.

For the eventual 2^30 production run this matters: absorption is the cheap
path computationally, but reproducing Stata requires either explicit
(g-shrunk) dummies or a block-inverse implementation of the joint formula
(the FWL identity in docs/STATISTICAL_SPECIFICATION.md gives exactly that:
joint scores are computable from residualized sufficient statistics plus
ESS_W, at absorption-like cost).

Phase 2 update (2026-07-16): the full derivation, including posterior
moments through the block inverse and the proof that flat residualization
is insufficient, is in docs/FWL_BLOCK_FORMULATION.md; numerical
equivalence (explicit joint design == FWL fast path == executed Stata,
q up to 110) is proven in tests/enumeration/test_fwl_block_equivalence.py.
The GPU enumerator (src/gpubma/gpu/enumerator.py) implements this
formulation.
