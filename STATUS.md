# GPUBMA — STATUS

Last updated: 2026-07-15 (Phase 1 completed on the development machine).

## Phase 1 acceptance criteria — verification

| # | criterion | status |
|---|---|---|
| 1 | actual local GPU identified | DONE — NVIDIA GeForce RTX 3060, 12 GiB, CC 8.6, 28 SMs, driver 591.86 (never assumed; detected via nvidia-smi + torch) |
| 2 | reproducible hardware report | DONE — `reports/environment_report.{md,json}`, `reports/gpu_doctor.json`, regenerable via `python -m gpubma.doctor` / `scripts/make_environment_report.py` |
| 3 | GPU executed and validated a real float64 op | DONE — 256×256 float64 matmul, GPU vs CPU abs diff ≈ 1.8e-12; plus batched float64 model scoring, max Δ log-score 4.5e-13 (`reports/gpu_feasibility.json`) |
| 4 | Grunfeld obtained reproducibly with provenance | DONE — original Stata Press bytes (`webuse grunfeld`, r18), 200 rows × 10 companies × 20 years, SHA-256 recorded, `data/public/grunfeld_provenance.json`, rerunnable via `scripts/download_grunfeld.py` |
| 5 | 8/12/30-predictor synthetic datasets generated | DONE — seed 20260715, 100×10 balanced panel, CSV+Parquet+DTA each, `data/synthetic/` |
| 6 | all formats have checksums and validation reports | DONE — SHA-256 per file + content hash in `panel_*_metadata.json`; exact round-trip checks pass |
| 7 | small CPU exhaustive BMA runs | DONE — `bma_regress` on panel_8 |
| 8 | exactly 256 models evaluated for panel_8 | DONE — asserted in results and tests |
| 9 | posterior probs and PIPs pass consistency checks | DONE — sum=1, bounds, size-distribution, independent-formula tests |
| 10 | FE explicit-dummy vs absorption comparisons exist | DONE — `reports/fixed_effects_comparison.{md,json}`, all PASS with aligned df and g |
| 11 | Stata validation .do files exist | DONE — 5 scripts under `validation/stata/` |
| 12 | actual Stata outputs distinguished from prepared scripts | DONE — no Stata outputs exist; scripts explicitly marked PREPARED, NOT EXECUTED; parity test skips with that reason |
| 13 | benchmark report has only measured values or labelled projections | DONE — `reports/benchmark_report.md` (Measured / Projected LOW-confidence / Not evaluated) |
| 14 | executed tests and exact results recorded | DONE — `reports/test_results.md` + raw log: 65 passed, 1 skipped, 0 failed |
| 15 | STATUS.md describes next phase without claiming the CUDA enumerator exists | this file — the production CUDA enumerator does **not** exist yet |

## Measured snapshot (this machine, RTX 3060, float64)

- CPU reference: ~33,000 models/s, flat from 256 to 32,768 models.
- GPU batched scorer: 43k → 1.74M models/s (p=8→15), still rising with batch
  size; max |Δ log-score| vs CPU 4.5e-13; peak GPU memory ≈ 10 MiB at p=8 run.
- Projection (LOW confidence, from measured p=15 GPU run, constant-throughput
  assumption): 2^30 scoring ≈ 620 s on this GPU. NOT a measurement; the
  production enumerator does not exist and FP64 on GeForce Ampere runs at
  1/64 of FP32 throughput.

## Unresolved statistical questions (honest list)

1. **Stata `bmaregress` parameterization is unverified.** Our g-prior
   (fixed g = max(n, p²), flat always-block prior, Jeffreys σ², df = n − rank
   of always block) is labelled PROVISIONAL in `gpubma/priors/gpriors.py`.
   Sources: FLS (2001), Liang et al. (2008), the [BMA] bmaregress manual.
   Output must not be called Stata-compatible until the prepared .do scripts
   run on a real Stata 18+ and `scripts/compare_stata_python.py` passes.
2. **Always-included blocks and df conventions.** With controls/FE dummies in
   every model we use df = n − rank(always block). Whether Stata conditions
   the same way (vs df = n − 1, or g rescaling after absorption) is open;
   equality of OLS slopes does not settle Bayesian score equality
   (docs/FIXED_EFFECTS_DESIGN.md).
3. **Coefficient posterior sd formula** is internally validated but has no
   external oracle yet; cross-package sd comparisons are pending.
4. **Stata export names** (e.g. the e() matrix holding PIPs) unconfirmed;
   scripts log `ereturn list` to resolve this on first execution.
5. **Unbalanced two-way FE** within transform intentionally unsupported
   (raises); iterative demeaning is a later-phase feature.
6. **Grunfeld variant**: the Stata Press file (10 firms, 200 obs, incl. a
   `time` column) matches Stata's `webuse grunfeld`; other distributions
   (statsmodels: 11 firms) differ — ours is pinned by checksum.

## Blocked items

- Executing the Stata oracle: no callable Stata on this machine (only renamed
  `StataSE-64_old.exe` leftovers in `C:\Program Files\Stata17/18`; batch
  execution attempted and produced nothing).

## Next phase (Phase 2 — requires user authorization)

1. Run the five .do scripts on a working Stata 18+, confirm option keywords
   and e() names, export real oracle numbers, close the parity loop
   (tolerance 1e-6, never loosened merely to pass).
2. Resolve unresolved items 1–4 above from the executed logs.
3. Extend the measured benchmark ladder (p = 18, 20 CPU; GPU batching beyond
   p = 16 with size-grouped streaming) to establish throughput stability and
   upgrade projection confidence honestly.
4. Design the production GPU enumerator (model-space traversal order, tiled
   Gram-submatrix batching, on-device log-sum-exp reduction, checkpoint and
   resume, VRAM budget for 2^30) — design first, then implement. The CUDA
   enumerator is NOT implemented today.
5. Decide FP64 strategy for GeForce-class FP64 throughput (1/64): pure FP64
   correctness-first vs mixed-precision with FP64 verification sampling —
   any deviation from full float64 requires explicit, documented user
   approval (CLAUDE.md rule 6).
