# Benchmark protocol

## Command

```bash
python -m gpubma.benchmark --max-predictors 15 --repeats 3
gpubma benchmark --max-predictors 15
```

Outputs `reports/benchmark_report.md` and `.json`.

## Ladder

8, 10, 12, 15, 18, 20, 24, 26, 28, 30 predictors (256 → 2^30 models).
Phase 1 runs only sizes that finish safely and quickly (default cap p = 15;
the GPU feasibility scorer additionally caps at p = 16). Everything else is
labelled **Not evaluated**.

## Labelling rules (non-negotiable)

Every number is one of:

- **Measured** — actually executed on this machine; the report records the
  exact GPU, backend, precision, observations, predictors, model count,
  cold/initialization time, sufficient-statistics time, enumeration/scoring
  time, reduction time, total time, models/second, peak GPU memory (torch
  allocator), repeat count, and median/min/max runtimes.
- **Projected** — computed from a named measured source benchmark with a
  stated scaling assumption (constant models/second), a stated throughput-
  stability caveat, a confidence tag (LOW until multi-size stability is
  established), and explicit reasons the projection may fail.
- **Not evaluated** — with the reason.

Projections are only produced when a real model-evaluation benchmark exists;
they are never derived from trivial matrix operations, and they are never
reported as measurements (CLAUDE.md rule 7).

## Timing methodology

- CPU: `time.perf_counter` around the full enumeration call;
  coefficient-moment computation excluded (scores only) so CPU and GPU
  measure the same work.
- GPU: explicit `torch.cuda.synchronize()` before stopping any timer; cold
  (first call, includes context/library initialization) reported separately
  from warm repeats; peak memory via `torch.cuda.max_memory_allocated`.
- Repeats: default 3 warm repeats; median is the headline number, min/max
  reported.

## Phase 1 measured snapshot (2026-07-15, RTX 3060, float64)

See `reports/benchmark_report.md` for the authoritative table. Headline:
CPU reference ≈ 33k models/s (flat), GPU batched scorer ≈ 1.74M models/s at
p = 15 with max |Δ log-score| vs CPU 4.5e-13. Throughput was still rising
with batch size, so the p=15 figure is a lower bound on saturated GPU
throughput — another reason the p→30 projections carry LOW confidence.
