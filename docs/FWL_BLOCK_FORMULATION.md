# The Stata-compatible FWL / block-matrix formulation

This document derives the exact formulation the production enumerator uses to
handle always-included controls and fixed effects **without rebuilding the
joint design matrix for any candidate model**, while reproducing Stata
`bmaregress`'s joint shrinkage (`always_prior="shrink"`) bit-for-bit in
exact arithmetic. Equivalence is proven numerically in
`tests/enumeration/test_fwl_block_equivalence.py` against (a) the explicit
joint-design scorer (`gpubma/validation/joint_reference.py`), (b) the CPU
oracle (`gpubma/cpu/enumeration.py`), and (c) executed Stata exports.

## 1. The joint model and prior (Stata convention, verified 2026-07-15)

For candidate model γ with k = |γ| optional predictors X_γ (n × k) and the
always-included slope block W (n × q; continuous controls **and** explicit
fixed-effect dummies, base categories dropped):

    y = α·1ₙ + X_γ β_γ + W δ + ε,     ε ~ N(0, σ² Iₙ)

with priors

    p(α, σ²) ∝ 1/σ²                                  (flat intercept)
    (β_γ, δ) | σ² ~ N( 0, g σ² (V_γ' M₁ V_γ)⁻¹ )      (JOINT g-prior)

where V_γ = [X_γ, W] and M₁ = Iₙ − 1ₙ1ₙ'/n centers a column. The g-prior
covers optional **and** always slopes jointly; only the intercept is flat.
Integrating α, (β_γ, δ), σ² gives the standard closed form

    log m(y|γ) = ((n−1−k−q)/2)·log(1+g)
               − ((n−1)/2)·log(1 + g·(1 − R²_γ))  + C            (1)

with

    TSS_c = y' M₁ y,     ESS_joint(γ) = ESS of OLS of M₁y on M₁V_γ,
    R²_γ  = ESS_joint(γ) / TSS_c,

and C a constant common to all models (it cancels in normalized posterior
model probabilities). Degrees of freedom are n − 1 regardless of q.

Naively, evaluating (1) requires an OLS solve of size (k+q) per model. With
fixed effects, q can be large (panel_8 two-way FE: q = 110; a production
panel with thousands of individuals: q in the thousands), which would
dominate the per-model cost and force the enumerator to carry an n × q
dummy block on the GPU.

## 2. Block-matrix reduction (Frisch–Waugh–Lovell / Schur complement)

Let A = [1ₙ, W] (rank 1 + q, validated at run time) with thin-QR A = QR,
and define the residualized data

    X̃ = X − Q(Q'X),    ỹ = y − Q(Q'y)        (projection out of span(A))

and the **precomputed** p × p sufficient statistics

    Z̃ = X̃'X̃,    b̃ = X̃'ỹ,    RSS_W = ỹ'ỹ,    ESS_W = TSS_c − RSS_W.

ESS_W is the explained sum of squares of the always block alone (centered).
For a model γ, write the centered joint Gram in block form:

    G_γ = V_γ' M₁ V_γ = [ X_γ'M₁X_γ   X_γ'M₁W ]
                        [ W'M₁X_γ     W'M₁W   ]

Because span(A) = span(1ₙ, W), the Schur complement of the W-block in G_γ
is exactly the residualized Gram:

    S_γ = X_γ'M₁X_γ − X_γ'M₁W (W'M₁W)⁻¹ W'M₁X_γ = X_γ' M_A X_γ = Z̃_γγ   (2)

with M_A = Iₙ − QQ'. Three identities follow from partitioned regression
(FWL) and block inversion:

**(I1) Explained sum of squares decomposes.**

    ESS_joint(γ) = ESS_W + b̃_γ' Z̃_γγ⁻¹ b̃_γ                       (3)

Proof: P_{[A,X_γ]} = P_A + P_{M_A X_γ}; apply to y and take squared norms
(the two projections are orthogonal by construction).

**(I2) The x-block of the joint OLS solution is the residualized solution.**

    β̂_γ (from OLS of y on [1, X_γ, W]) = Z̃_γγ⁻¹ b̃_γ               (4)

**(I3) The x-block of the inverse joint Gram is the inverse Schur
complement.**

    [ G_γ⁻¹ ]_xx = Z̃_γγ⁻¹                                          (5)

by the block-inverse formula, using (2).

Substituting (3) into (1): with ESS_γ̃ := b̃_γ' Z̃_γγ⁻¹ b̃_γ,

    log m(y|γ) = ((n−1−k−q)/2)·log(1+g)
               − ((n−1)/2)·log(1 + g·(TSS_c − ESS_W − ESS_γ̃)/TSS_c) + C   (6)

Every model-dependent quantity in (6) involves only the k × k submatrix
Z̃_γγ and subvector b̃_γ of the **precomputed** residualized statistics; W
never appears again. This is the formula implemented by
`enumerate_models(..., tss_norm=TSS_c, k_always=q, df_resid=n−1)` and the
GPU scorer: note TSS_c − ESS_W = RSS_W = ỹ'ỹ, so `(tss − ess)/tss_norm`
in the code is exactly `(TSS_c − ESS_joint)/TSS_c` here.

## 3. Posterior moments through the same blocks

With s = g/(1+g), the conditional posteriors given γ are

    E[β_γ | y, γ]   = s · β̂_γ = s · Z̃_γγ⁻¹ b̃_γ                     by (4)
    E[σ² | y, γ]    = (TSS_c − s·ESS_joint(γ)) / (n − 3)             (7)
    Var[β_γ | y, γ] = E[σ²|y,γ] · s · Z̃_γγ⁻¹                        by (5)

(7) uses the Inverse-Gamma((n−1)/2, ·) posterior mean, hence the n − 3
divisor. BMA moments follow by total expectation/variance with excluded
coefficients exactly zero — all computable from residualized statistics.

The always-block moments (Stata reports them too) come from the remaining
blocks of G_γ⁻¹:

    δ̂_γ = (W'M₁W)⁻¹ W'M₁ (y − X_γ β̂_γ)
    [G_γ⁻¹]_ww = (W'M₁W)⁻¹ + (W'M₁W)⁻¹ W'M₁X_γ · Z̃_γγ⁻¹ · X_γ'M₁W (W'M₁W)⁻¹

which cost O(kq) extra per model using the precomputed q × p matrix
(W'M₁W)⁻¹W'M₁X. The Phase-2 enumerator aggregates optional-predictor
moments; always-block aggregation is available at this documented cost.

## 4. Why flat residualization is NOT sufficient

The conventional shortcut — residualize on [1, W] and score with a flat
always block — computes

    log m_flat(γ) = ((n−1−q−k)/2)·log(1+g)
                  − ((n−1−q)/2)·log(1 + g·(RSS_W − ESS_γ̃)/RSS_W) + C'

which differs from (6) in the second term twice: the exponent is
(n−1−q)/2 instead of (n−1)/2, and the normalizer is RSS_W instead of
TSS_c (equivalently, 1−R² is measured after versus before removing W's
explained variation). These differences depend on γ through ESS_γ̃ and do
NOT cancel in posterior odds; they change Bayes factors across model sizes
and therefore PIPs and moments. The test
`test_flat_residualization_is_not_sufficient` demonstrates the divergence
on real data (panel_8, individual FE: PIPs differ by more than 10⁻²).
Both conventions coincide exactly when q = 0 (intercept-only always block).

## 5. Cost and memory of the block formulation

One-time (CPU, float64): thin-QR of A — O(n(q+1)²); residualization —
O(npq); Gram accumulation — O(np²). Per model: a k × k Cholesky solve,
independent of q and of n. Enumerator state: Z̃ (p × p), b̃ (p,), plus
three scalars (TSS_c, RSS_W ≡ ỹ'ỹ, and g). For p = 30 that is a 30 × 30
matrix — trivially resident on any GPU; the n × q dummy block never
leaves the host and is never materialized per model.

## 6. Numerical equivalence evidence (executed)

`tests/enumeration/test_fwl_block_equivalence.py` verifies, in float64
without rounding:

1. explicit joint scorer ≡ FWL fast path — log scores, PMPs, PIPs,
   optional-coefficient moments — on panel_12 (no FE, q = 2, 4,096 models)
   and panel_8 (two-way FE dummies, q = 110, 256 models);
2. explicit joint scorer ≡ executed Stata per-model export (panel_12,
   normalized log PMPs for all 4,096 models);
3. explicit joint scorer ≡ executed Stata e() exports (panel_8 two-way FE:
   PIPs, posterior means, posterior sds, including always-block columns);
4. flat residualization ≠ joint shrinkage (documented divergence, §4).

The CPU oracle's own Stata parity (7 designs, 264+ quantities, worst
|diff| ≈ 3.4e-12) is reported in `reports/comparison_report.md`.
