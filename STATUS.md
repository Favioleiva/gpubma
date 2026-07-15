# GPUBMA — STATUS

Last updated: 2026-07-15 (Phase 1 complete; Stata oracle EXECUTED and parity closed).

## Stata validation — executed, exact results

Oracle: `C:\Program Files\StataNow19\StataSE-64.exe` (Stata 19.5 StataNow, SE,
born 08 Apr 2025), `bmaregress` 1.0.2 (16jul2023), batch mode `/e do <script>`.
All five `.do` scripts exited on their own with exit code 0 and no `r(###)`
errors. Exports (full double precision): `e(b_bma)`, `e(pip)`,
`vecdiag(e(V_bma))`, key scalars → `validation/stata/output/`.
Logs sanitized: no serial number or license-holder information anywhere in the
repository (enforced by `test_stata_exports_exist_and_are_license_free`;
banner logs at the repo root are deleted and git-ignored).

`python scripts/compare_stata_python.py` — comparison per design, tolerance
1e-9, values never rounded before comparison:

| design | data | always block | result | worst abs diff |
|---|---|---|---|---|
| small_no_fe | panel_8 | w1 w2 | **PASS** | ~1e-12 |
| small_individual_fe | panel_8 | w1 w2 + 99 individual dummies | **PASS** | 1.8e-12 (worst overall, coef sd[x1]) |
| small_time_fe | panel_8 | w1 w2 + 9 period dummies | **PASS** | ~1e-12 |
| small_two_way_fe | panel_8 | w1 w2 + 108 dummies | **PASS** | ~1e-12 |
| grunfeld_no_fe | grunfeld | intercept only | **PASS** | ~1e-13 |
| grunfeld_company_fe | grunfeld | 9 company dummies | **PASS** | ~1e-12 |

Compared per design: PIP, posterior mean, and posterior sd for every optional
predictor, plus model count and mean model size (Stata counts always
variables in model size; `e(msize_mean) − e(p_always)` matches Python).
264 quantities total, **all PASS**. Full tables: `reports/comparison_report.md`.

## Resolved statistical questions (were open this morning)

1. **Stata `bmaregress` parameterization — RESOLVED.** Stata applies the
   Zellner g-prior JOINTLY to optional predictors and always-included slopes
   (only the intercept is flat), with df = n − 1 and E[σ²] divisor n − 3.
   Implemented as `always_prior="shrink"` (now the gpubma default) and
   verified to ~1e-12. gpubma's previous conditional convention
   (flat always block, df = n − rank) remains as `always_prior="flat"`.
2. **PIP / coefficient export names — RESOLVED.** No `e(b)`; the matrices are
   `e(b_bma)`, `e(V_bma)` (+ `_c` conditional variants), `e(pip)`, `e(group)`.
3. **Degrees of freedom — RESOLVED.** df = n − 1 under the Stata convention
   regardless of always-block size; the always block enters through the joint
   R² and the (1+g) exponent (k + q), not through df.
4. **Posterior sd formula — RESOLVED.** gpubma's law-of-total-variance sds
   reproduce sqrt(diag(e(V_bma))) to ~1e-12 on all six designs.
5. **Option keywords — RESOLVED.** `enumeration`, `gprior(fixed #)`,
   `mprior(betabinomial 1 1)`, `(varlist, always)` all work as written;
   factor-variable `always` groups use base level = lowest value, matching
   Python's dummy convention.
6. **Default g confirmed:** `e(g)` = 1000 = max(n, p²) on panel_8, matching
   gpubma's benchmark default.

## Remaining caveats (honest list)

- Dummies-vs-absorption equivalence does NOT hold under the Stata "shrink"
  convention (different prior on the fixed effects); it holds only under
  `always_prior="flat"`. `bma_regress` rejects shrink+within explicitly.
  The FWL identity makes the joint (Stata) formula computable from
  residualized statistics at absorption-like cost — the Phase 2 GPU design
  should use it (docs/FIXED_EFFECTS_DESIGN.md).
- Unbalanced two-way within transform unsupported (explicit error).
- Stata comparisons cover p = 8 (256 models) and Grunfeld (4 models);
  larger-p oracle runs are cheap and can be added in Phase 2.
- The `e(rngstate)` in enumeration output suggests bmaregress still seeds an
  RNG; irrelevant for enumeration but worth remembering if sampling is ever
  compared.

## Phase 1 acceptance criteria — final verification

| # | criterion | status |
|---|---|---|
| 1 | actual GPU identified | DONE — NVIDIA GeForce RTX 3060, 12 GiB, CC 8.6, 28 SMs, driver 591.86 |
| 2 | reproducible hardware report | DONE — `reports/environment_report.{md,json}`, `reports/gpu_doctor.json` |
| 3 | real float64 GPU op validated | DONE — matmul diff ≈ 1.8e-12; batched scoring diff 4.5e-13 vs CPU |
| 4 | Grunfeld reproducible with provenance | DONE — original Stata Press bytes + SHA-256, `grunfeld_provenance.json` |
| 5 | 8/12/30 synthetic datasets | DONE — seed 20260715, CSV+Parquet+DTA + metadata |
| 6 | checksums and validation reports | DONE — exact round trips, SHA-256 everywhere |
| 7 | small CPU exhaustive BMA runs | DONE |
| 8 | exactly 256 models for panel_8 | DONE — asserted in results, tests, and Stata (`e(k_models)` = 256) |
| 9 | posterior/PIP consistency checks | DONE — plus external: matches Stata to ~1e-12 |
| 10 | FE dummy vs absorption comparisons | DONE — equivalence proven under "flat"; non-equivalence under "shrink" documented |
| 11 | Stata `.do` files exist | DONE — and EXECUTED (StataNow/SE 19.5) |
| 12 | actual Stata outputs vs prepared scripts distinguished | DONE — real exports in `validation/stata/output/`, headers updated to EXECUTED |
| 13 | benchmark report labelled | DONE — Measured / Projected(LOW) / Not evaluated |
| 14 | tests and exact results recorded | DONE — **73 passed, 0 skipped, 0 failed** (`reports/test_results.md`) |
| 15 | STATUS.md describes next phase without claiming the CUDA enumerator exists | this file — it does **not** exist yet |

## Measured snapshot (RTX 3060, float64)

- CPU reference: ~33,000 models/s (flat, 256 → 32,768 models).
- GPU batched scorer: up to 1.74M models/s at p = 15; max |Δ log-score| vs
  CPU 4.5e-13; still rising with batch size.
- Projection (LOW confidence, labelled, from measured p=15): 2^30 scoring
  ≈ 620 s on this GPU. NOT a measurement.

## Next phase (Phase 2 — requires user authorization)

1. Extend the measured ladder (CPU p=18–20; GPU beyond p=16 with streamed
   size-grouped batches) to establish throughput stability and honestly
   upgrade projection confidence.
2. Add larger-p Stata oracle runs (p = 12 on panel_12 is a cheap extra
   cross-check) using the now-verified export pipeline.
3. Design the production GPU enumerator around the verified "shrink" formula
   via the FWL block-inverse identity: model traversal order, tiled Gram
   submatrices, on-device log-sum-exp, checkpoint/resume, VRAM budget for
   2^30, and an FP64 strategy for GeForce-class hardware (FP64 = 1/64 FP32);
   any deviation from full float64 requires explicit user approval.
4. Only after design review: implement and validate the enumerator at
   2^18–2^24 against the CPU reference before attempting 2^30.
