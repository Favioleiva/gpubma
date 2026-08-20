# GPUBMA — STATUS

Last updated: 2026-08-20. Controlled Phase 2 and the canonical
`panel_30_center15` exact run are complete.

## Canonical exact p = 30 run (measured)

The frozen `data/synthetic/panel_30_center15.parquet` benchmark was executed
on an NVIDIA A100-SXM4-80GB using commit
`ef9d68af462564b5cbdce268be1ff2b43317dd24`:

- 30 optional predictors and exactly 2^30 = 1,073,741,824 models;
- exhaustive enumeration in float64, with no sampling, pruning, truncation,
  approximation, or rejected models;
- beta-binomial(1,1) model prior, `g = 2000`, and the validated Stata-compatible
  shrink convention;
- PASS1: 115.7 s cumulative, 9,277,815 models/s, 1.93 GiB peak GPU memory;
- exact PASS2: 896 s, including top-model ranks, family posteriors and
  coefficient-density grids;
- exact model count, posterior normalization, finite scores and zero Cholesky
  failures verified.

The executed notebook with retained scientific outputs is
`notebooks/GPUBMA_A100_p30_middle15_stata_figures.ipynb`. The compact archive
is `reports/artifacts/panel_30_center15_exact_results.zip`; its inventory and
hashes are documented in `reports/CANONICAL_P30_RESULTS.md`.

## Scientific result

The relevant recovery criterion is signal and family-level uncertainty, not
whether the single data-generating model ranks first. Variables x1–x13 have
PIP approximately 1, x14 has PIP 0.9235, and the only material substitution is
the deliberately difficult x15/x30 family:

- x15 PIP = 0.0905 and its correlated proxy x30 PIP = 0.9638;
- the MAP model replaces x15 with x30 and has PMP 0.4715;
- the exact generating model remains plausible, ranking 8th with PMP 0.0188;
- the x15/x30 family has posterior probability 0.99998 of including at least
  one member;
- the top 10 and top 100 models hold 0.7267 and 0.9370 posterior mass;
- posterior model size is concentrated at 15–17 (mean 15.637, mode 15).

This is scientifically reasonable BMA behavior under strong collinearity:
the signal is recovered while posterior mass records uncertainty between
observationally similar substitutes.

## Validation chain

- Complete Stata oracle parity at p = 12: all 4,096 models, worst absolute
  difference 3.4e-12.
- Progressive RTX 3060 ladder at p = 12/18/20/22/24: exact counts,
  bit-identical repeated runs and checkpoint/resume, CPU parity through the
  documented feasible levels.
- Mandatory p = 15 CPU/GPU smoke test on the canonical data before p = 30:
  all scores within 9.095e-13; PIPs within 5.329e-15; coefficient SDs within
  6.113e-14; identical top-10 masks.
- Current local suite: 147 passed, 1 intentionally skipped, 0 failed. The only
  warning is inability to write pytest's cache in the sandbox.

## Implementation state

- The bounded-memory exact enumerator is implemented in
  `src/gpubma/gpu/enumerator.py` using size-grouped streamed batches,
  on-device combinadic unranking, batched Cholesky, deterministic reductions,
  float64-only arithmetic and atomic checkpoint/resume.
- The FWL/block derivation and Stata-compatible shrink convention remain
  documented and tested in `docs/FWL_BLOCK_FORMULATION.md`.
- The p = 30 enumerator remains standalone; integration into the high-level
  `bma_regress` API has not been performed.

## Remaining caveats and hard limits

- Always-block BMA moments are documented but are not aggregated by the GPU
  enumerator.
- CPU coefficient-moment validation stops at p = 20; p = 22/24 use the
  documented scores-only check, and p = 30 relies on the validated ladder plus
  the mandatory p = 15 smoke test.
- Multi-GPU sharding, MC3/MCMC, Tournament GPUBMA and mixed precision remain
  out of scope. Never report projections as measurements or silently use
  float32.
