# GPUBMA — Non-negotiable project rules

1. `gpubma` is a Python open-source package (BSD-3-Clause). Stata is **never** a runtime dependency.
2. Stata is validation-only: an external oracle run on small frozen datasets.
3. Exhaustive enumeration of all `2^p` models is the long-term target.
4. Never replace enumeration with MC3 (or any sampler) without explicit authorization from the user.
5. Never assume the GPU model. Detect it (`gpubma doctor`) and record the actual device.
6. Never silently use float32. Precision must be explicit; float64 is the default reference precision.
7. Never report projections as measurements. Benchmark reports must label every number as
   `Measured`, `Projected`, or `Not evaluated`.
8. Never optimize an unvalidated formula. The CPU reference must be correct before GPU work.
9. Never loosen tolerances merely to pass tests. Investigate the discrepancy instead.
10. Keep generated data reproducible from scripts and fixed seeds (default seed: `20260715`).
11. Preserve checksums and provenance for every frozen data artifact.
12. Record unresolved statistical questions honestly in `STATUS.md` (e.g., the exact Stata
    `bmaregress` prior parameterization is provisional until validated against real Stata output).

## Practical notes

- Package layout: `src/gpubma/`; tests under `tests/`; frozen data under `data/`.
- Regenerate synthetic data: `python scripts/generate_synthetic_panels.py`.
- Regenerate Grunfeld snapshot: `python scripts/download_grunfeld.py`.
- Diagnostics: `python -m gpubma.doctor [--json reports/gpu_doctor.json]`.
- Benchmarks: `python -m gpubma.benchmark --max-predictors 15`.
- Tests: `python -m pytest` (GPU tests skip cleanly when CUDA is unavailable, with the skip reason printed).
- Stata oracle (validation only): `STATA_EXE = C:\Program Files\StataNow19\StataSE-64.exe`,
  batch mode `/e do <script>` from the repo root. After running, delete the
  banner logs Stata drops at the repo root and never commit serial-number or
  license-holder text (scan before committing; a test enforces this for
  `validation/stata/output/`).
- The `2^30` production run and the final CUDA enumerator are **out of scope** until Phase 1
  acceptance criteria pass and the user authorizes Phase 2.
