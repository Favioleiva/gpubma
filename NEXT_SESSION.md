# NEXT_SESSION — handoff for Phase 2

Phase 1 is COMPLETE (commit `240dacd`, 2026-07-15). Do not re-derive anything
below; verify against the cited files if in doubt.

## Verified statistical conventions

- **`always_prior="shrink"` (default, Stata-verified):** Zellner g-prior
  applied JOINTLY to optional predictors and always-included slopes
  (controls + FE dummies); flat intercept only; df = n − 1; E[σ²] divisor
  n − 3. Matches executed `bmaregress` (StataNow/SE 19.5, bmaregress 1.0.2).
- **`always_prior="flat"`:** gpubma's conditional convention (flat always
  block, df = n − rank). NOT Stata's. Required for `fe_method="within"`;
  dummies ≡ within holds ONLY under this convention. shrink+within raises.
- g default = max(n, p²) (FLS benchmark; equals Stata's `e(g)`).
  Model prior beta-binomial(1,1) over optional predictors only.
- FWL identity (docs/STATISTICAL_SPECIFICATION.md): the joint (Stata) score
  is computable from residualized sufficient statistics + ESS_W — this is
  the formula the GPU enumerator should implement.
- Stata e() names: `e(b_bma)`, `e(V_bma)`, `e(pip)` (no `e(b)`); Stata model
  size counts always vars → subtract `e(p_always)`.

## Completed Stata parity (all PASS, tol 1e-9, worst |diff| 1.8e-12)

small_no_fe, small_individual_fe, small_time_fe, small_two_way_fe (panel_8,
g=1000), grunfeld_no_fe, grunfeld_company_fe (g=200). 264 quantities: PIPs,
posterior means, posterior sds, mean model size, model counts.
Details: `reports/comparison_report.md`; exports: `validation/stata/output/`.
Stata launcher: `C:\Program Files\StataNow19\StataSE-64.exe` `/e do <script>`
from repo root; delete the banner `*.log` files it drops at the root
(license info; git-ignored). Never commit serial/license text.

## Tests and benchmarks (exact, as of `240dacd`)

- `python -m pytest`: **73 passed, 0 skipped, 0 failed** (~3 s).
  Raw log: `reports/test_results_raw.txt`; map: `reports/test_results.md`.
- Benchmark (RTX 3060, float64, 3 repeats, `reports/benchmark_report.md`):
  CPU ~33k models/s flat (p=8..15); GPU batched scorer 43k → 1.74M models/s
  (p=8→15, warm median 18.9 ms at p=15), GPU-vs-CPU max |Δ log-score|
  4.5e-13, cold start ~0.2 s. p ≥ 18 Not evaluated.
  Projection (LOW confidence, labelled): 2^30 ≈ 620 s GPU — NOT a measurement.
- Hardware: RTX 3060 12 GiB, CC 8.6, 28 SMs, driver 591.86; torch
  2.5.1+cu121; FP64 = 1/64 FP32 on GeForce Ampere; `nvcc` NOT installed.

## Unresolved engineering decisions (decide before coding)

1. FP64 strategy for GeForce (1/64 throughput): pure float64 vs mixed
   precision with float64 verification — any deviation from full float64
   needs explicit user approval (CLAUDE.md rule 6).
2. Enumerator architecture: torch batched Cholesky vs custom CUDA kernels
   (would require installing a CUDA toolkit — nvcc absent today).
3. Model traversal/batching for 2^30: size-grouped combinations vs Gray-code
   updates; streaming log-sum-exp reduction; checkpoint/resume format.
4. VRAM and host-memory budget at p=30 (cannot store 2^30 log-scores in
   12 GiB as float64 alongside work buffers — needs streamed reduction).
5. Whether coefficient moments are computed on GPU or CPU-post-pass at scale.
6. Unbalanced two-way FE (iterative demeaning) — deferred feature.

## Controlled Phase 2 milestones (in order, each gated on the previous)

1. Extend MEASURED CPU ladder to p = 18, 20; check throughput stability.
2. Extend GPU scorer beyond p = 16 via streamed size-grouped batches;
   measure p = 18, 20, 24; re-derive projections with honest confidence.
3. Cheap extra oracle run: Stata on panel_12 (4,096 models) via the existing
   export pipeline; parity test at p = 12.
4. Design document for the production enumerator (traversal, reduction,
   checkpointing, memory budget, FP64 policy) — review with user BEFORE
   implementation.
5. Implement and validate at 2^18–2^24 against the CPU reference.

## Hard limits

- **Do NOT run the p=30 / 2^30 production enumeration yet** — not until
  milestones 1–5 pass and the user explicitly authorizes it.
- Never substitute MC3 for enumeration; never silently use float32;
  never report projections as measurements (CLAUDE.md).
