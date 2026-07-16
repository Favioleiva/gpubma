# GPUBMA — STATUS

Last updated: 2026-07-16 (Controlled Phase 2: exhaustive GPU enumerator
implemented and validated through p = 24; p = 30 NOT run — awaits explicit
user authorization).

## Phase 2 state (all Measured, never projected)

- **panel_12 Stata oracle EXECUTED** (StataNow/SE 19.5, bmaregress 1.0.2,
  batch mode): 12 optional predictors, 4,096 models enumerated. New:
  `bmaregress, saving()` exports the complete per-model dataset (inclusion
  states, log posterior, log model prior, conditional moments) —
  `validation/stata/output/medium_no_fe_models.dta`. Compared per model:
  normalized log PMPs, PMPs, exact once-only mask coverage, full
  model-size distribution, mean model size, plus PIPs/means/sds from e().
  Worst |diff| 3.4e-12 at tolerance 1e-9. Total Stata designs: 7/7 PASS
  (`reports/comparison_report.md`). `bmastats models, top()` cannot export
  the full space (numlist cap ~2,500 / 50-row r(summary)) — documented in
  the .do file.
- **FWL/block-matrix formulation derived and proven**
  (`docs/FWL_BLOCK_FORMULATION.md`): the joint (Stata "shrink") score and
  posterior moments are computed exactly from residualized sufficient
  statistics via Schur-complement identities; per-model cost is a k × k
  solve independent of the always-block size q and of n. Numerical chain
  proven in `tests/enumeration/test_fwl_block_equivalence.py`:
  explicit joint design == FWL fast path == executed Stata (q up to 110,
  including all 110 always-block posterior means/sds by name); flat
  residualization demonstrably NOT sufficient (executable test).
- **ADR 0001 accepted on measured evidence**
  (`docs/ADR_0001_GPU_ENUMERATOR.md`,
  `reports/enumerator_candidates_bench.json`): Gray-code rank-1
  update/downdate is launch-bound and NOT faster than batched direct
  Cholesky on this hardware (768k vs 947k models/s at k = 30, favourable
  Gray case) → **streamed direct batching** with combinadic unranking.
- **Bounded-memory exhaustive GPU enumerator implemented**
  (`src/gpubma/gpu/enumerator.py`): float64 only; size-grouped chunks over
  colex combination ranks unranked ON DEVICE (exact int64 combinadics);
  batched Cholesky on gathered Gram submatrices (validated formula,
  unchanged); streaming running-rescale log-sum-exp reductions (O(p)
  device state — normalizer, PIPs, moment numerators, size distribution,
  top-K); deterministic reductions (scatter + fixed-shape sums, no
  atomics); atomic checkpoint/resume with SHA-256 config verification;
  VRAM budget caps chunk size; exact model counting asserted.

## Progressive validation ladder (Measured, RTX 3060, float64)

Full table: `reports/enumeration_ladder.md` (+ .json). Summary:

| p | models | elapsed | models/s | peak GPU | CPU check | Stata check |
|---|---|---|---|---|---|---|
| 12 | 4,096 | 0.34 s | 12,110 | 10 MiB | 1.5e-12 (full) | 3.4e-12 (complete) |
| 18 | 262,144 | 0.19 s | 1.35M | 227 MiB | 9.1e-13 (full) | — |
| 20 | 1,048,576 | 0.80 s | 1.32M | 507 MiB | 2.5e-12 (full) | — |
| 22 | 4,194,304 | 2.20 s | 1.90M | 740 MiB | 1.1e-13 (scores) | — |
| 24 | 16,777,216 | 8.96 s | 1.87M | 1,048 MiB | 2.7e-13 (scores) | — |

At every level: exact model counts verified, size distribution sums to 1
within 1e-15, PIP overshoot < 1e-12, second run BIT-IDENTICAL, and
interrupt + resume from checkpoint BIT-IDENTICAL to the uninterrupted run.
CPU checks: full per-model scores + coefficient moments up to p = 20;
scores-only aggregates at p = 22/24 (CPU coefficient pass impractical).
Host peak working set grew from 943 MiB to 1,282 MiB across the whole
ladder (process-lifetime, cumulative).

## Tests

`python -m pytest`: **97 passed, 0 skipped, 0 failed** (5.9 s). GPU tests
execute real float64 work on the detected RTX 3060; they skip with
explicit reasons when CUDA is absent.

## Resolved statistical questions (Phase 1, unchanged)

1. Stata `bmaregress` applies the Zellner g-prior JOINTLY to optional and
   always-included slopes (flat intercept, df = n − 1) —
   `always_prior="shrink"`, gpubma default, verified to ~1e-12.
2. Export names: `e(b_bma)`, `e(V_bma)`, `e(pip)`; no `e(b)`. Stata counts
   always variables in model size (subtract `e(p_always)`).
3. Posterior sd = sqrt(diag(e(V_bma))) reproduced by law-of-total-variance.
4. Default g = max(n, p²) (= `e(g)`); `mprior(betabinomial 1 1)` matches.

## Remaining caveats (honest list)

- Dummies-vs-absorption equivalence does NOT hold under "shrink" (different
  prior on the FE); `bma_regress` rejects shrink+within. The FWL path gives
  Stata-exact results at absorption-like cost (now proven and implemented).
- Unbalanced two-way within transform unsupported (explicit error).
- The enumerator aggregates optional-predictor moments; always-block BMA
  moments are available via the documented O(kq)-per-model extension
  (docs/FWL_BLOCK_FORMULATION.md §3) but not yet implemented on GPU.
- CPU coefficient-moment validation stops at p = 20 (pass-2 loop cost);
  p = 22/24 validated on scores-only aggregates.
- Multi-GPU sharding is designed for (stateless rank-range chunks) but NOT
  implemented (out of scope).
- `e(rngstate)` note from Phase 1 stands (irrelevant for enumeration).

## Hard limits still in force

- **p = 30 / 2^30 NOT run.** All Phase-2 gates through p = 24 now pass;
  the production run still requires explicit user authorization after
  reviewing `reports/enumeration_ladder.md`.
- No MC3/MCMC, no Tournament GPUBMA, no multi-GPU, no mixed precision,
  no silent float32, projections always labelled.
