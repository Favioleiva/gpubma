# NEXT_SESSION — handoff after Controlled Phase 2

Phase 2 core is COMPLETE (2026-07-16). The exhaustive single-GPU
enumerator exists, is validated through p = 24, and **p = 30 has NOT been
run** — it requires explicit user authorization after reviewing
`reports/enumeration_ladder.md`.

## What exists and is verified (do not re-derive)

- **Enumerator:** `src/gpubma/gpu/enumerator.py::enumerate_models_gpu`.
  Streamed direct batching (ADR 0001): size-grouped chunks over colex
  combination ranks, unranked on device via exact int64 combinadics;
  batched Cholesky on gathered Gram submatrices; float64 only; O(p)
  streaming reductions (running-rescale log-sum-exp); deterministic
  (no atomics); atomic npz checkpoint/resume gated on a SHA-256 of the
  sufficient statistics + prior config; VRAM budget caps chunk size.
  Inputs are the residualized X, y plus the same convention parameters as
  the CPU oracle (`df_resid`, `tss_norm`, `k_always` — shrink or flat).
- **Formulation:** `docs/FWL_BLOCK_FORMULATION.md` — joint (Stata) score
  and moments from residualized statistics via Schur identities; proven
  explicit-joint == FWL == Stata in
  `tests/enumeration/test_fwl_block_equivalence.py` (q up to 110).
- **Stata oracle:** 7 designs PASS incl. panel_12 per-model export
  (all 4,096 models, `medium_no_fe_models.dta`, worst |diff| 3.4e-12).
  `bmastats models, top()` cannot export > ~2,500 ranks; use
  `bmaregress, saving()`.
- **Ladder (all Measured, RTX 3060):** p = 12/18/20/22/24 —
  16.8M models in 8.96 s at p = 24 (1.87M models/s, 1.0 GiB peak GPU);
  bit-identical reproducibility and interrupt/resume at every level;
  CPU parity ≤ 2.5e-12 (full to p = 20, scores-only at 22/24);
  complete CPU–GPU–Stata parity at p = 12.
- Tests: **97 passed, 0 skipped, 0 failed** (5.9 s).

## Ready next steps (in order)

1. **p = 30 production run — ONLY with explicit user authorization.**
   Everything is in place: `enumerate_models_gpu` accepts p = 30
   (2^30 = 1,073,741,824 models). Honest expectation from the measured
   p = 22/24 sustained rate (~1.9M models/s): roughly 9–10 minutes —
   this is a PROJECTION from measured throughput, not a measurement.
   Use a checkpoint path and `progress_every_s`; validate normalization
   checks, exact count, and (optionally) a scores-only CPU spot-check of
   selected sizes is NOT practical at p = 30 — rely on the p ≤ 24 chain.
   Run on panel_30 (x1–x30, controls w1 w2, g = max(1000, 900) = 1000).
2. Optional hardening before/after: always-block BMA moments on GPU
   (documented O(kq) extension, FWL doc §3); a `bma_regress`
   integration path (`backend="gpu"` currently uses the Phase-1 scorer +
   CPU moments; the enumerator is standalone); wire the enumerator into
   `gpubma.benchmark`.
3. Multi-GPU sharding stays OUT OF SCOPE until authorized (chunks are
   already stateless rank ranges, so the design is shard-ready).

## Operational notes

- Stata launcher: `C:\Program Files\StataNow19\StataSE-64.exe` `/e do
  <script>` from repo root; delete the banner `*.log` files it drops at
  the root (license info; never commit serial/license text — a test
  scans `validation/stata/output/`).
- Ladder rerun: `python scripts/run_enumeration_ladder.py` (refuses
  p > 24 by design). ADR microbench:
  `python scripts/benchmark_enumerator_candidates.py`.
- pytest tmp_path is blocked in some sandboxes on this machine; the GPU
  tests use a self-managed `workdir` fixture (tempfile.mkdtemp).

## Hard limits (unchanged)

- Never substitute MC3/sampling for enumeration; never silently float32;
  never report projections as measurements; no Tournament GPUBMA; no
  multi-GPU; no mixed precision (CLAUDE.md).
