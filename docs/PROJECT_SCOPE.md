# GPUBMA — Project scope

## Long-term objective

An open-source Python package (`gpubma`, BSD-3-Clause) that performs
**exhaustive** Bayesian Model Averaging for Gaussian linear regression on one
NVIDIA GPU. The production experiment targets 30 optional predictors —
2^30 = 1,073,741,824 candidate models — enumerated exhaustively (no MC3) on a
local NVIDIA GPU or a Google Colab A100.

Stata is never a runtime dependency; it serves only as an external validation
oracle on small frozen datasets.

## Phase 1 (this phase) — delivered

1. Environment/GPU diagnostics with a real, validated float64 GPU test
   (`gpubma doctor`, `reports/environment_report.*`).
2. Reproducible public benchmark data (Grunfeld, Stata Press original bytes)
   with provenance and checksums.
3. Deterministic synthetic panels (8/12/30 predictors, seed 20260715) in
   CSV/Parquet/DTA with metadata, checksums, and exact round-trip validation.
4. An exact float64 CPU reference BMA (`bma_regress`), complete enumeration,
   fixed g-prior + beta-binomial model prior, PIPs, model-size distribution,
   coefficient moments, top models.
5. Fixed effects via explicit dummies (reference) and within residualization
   (candidate), with OLS and score-level comparisons.
6. Stata validation `.do` scripts — EXECUTED on StataNow/SE 19.5; Python
   matches the exported oracle on all six designs (worst |diff| 1.8e-12).
7. A GPU feasibility layer (batched float64 scoring validated against CPU)
   and an honest benchmark ladder (Measured / Projected / Not evaluated).

## Explicitly out of scope in Phase 1

- The 2^30 production run (the 30-predictor dataset exists as a data artifact
  only).
- The final CUDA exhaustive enumerator and any custom CUDA extension.
- MC3 or any sampling-based model search (requires explicit authorization).
- Claiming Stata compatibility before real Stata outputs exist.

## Phase 2 candidates (require Phase 1 acceptance + user authorization)

- ~~Execute the Stata oracle and close the parity loop~~ — DONE 2026-07-15
  (StataNow/SE 19.5; all six designs match, worst |diff| 1.8e-12).
- ~~Resolve the g-prior parameterization questions~~ — DONE: Stata uses joint
  g-shrinkage over optional + always slopes with df = n − 1, implemented as
  `always_prior="shrink"` (default).
- Design the production GPU enumerator (bit-ordered model traversal, tiled
  sufficient statistics, streaming reductions, checkpointing) and grow the
  benchmark ladder to 2^18–2^24 before attempting 2^30.
