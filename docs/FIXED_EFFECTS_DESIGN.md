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

## The statistical caveat (do not skip)

Equality of OLS slopes does **not** automatically establish equality of
Bayesian model scores. Under a g-prior, absorbing fixed effects may affect
prior parameterization, effective sample size, degrees of freedom, marginal
likelihood constants, and model comparison unless the always-included block
is treated consistently.

In gpubma the two approaches agree **because we constructed them to**: both
use residualized data, the same fixed g, and df = n − (rank of the full
always block, explicit or absorbed). These are modelling choices. An
implementation that, e.g., keeps df = n − 1 − k after absorption, or rescales
g by the post-absorption sample information, would produce different scores
with the same OLS slopes. Whether Stata's `bmaregress` with
`(i.individual_id, always)` matches our convention is an **open question**
that only real Stata output can settle (STATUS.md item; the prepared .do
scripts exist precisely for this).

Therefore: gpubma documents its convention, tests its internal consistency,
and does not claim Stata-equivalence for fixed-effects BMA scores yet.
