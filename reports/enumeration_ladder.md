# GPU enumerator — progressive validation ladder

Device: NVIDIA GeForce RTX 3060; float64 throughout; g = 1000 = max(n, p^2) for every level; beta-binomial(1,1) model prior; shrink (Stata) convention with always block [1, w1, w2].

Every number below is **Measured** (never projected). This ladder stops at p = 24 by design; the later standalone p = 30 canonical run is complete and documented separately in `reports/CANONICAL_P30_RESULTS.md`.

| p | models | elapsed s | models/s | peak GPU MiB | peak host MiB | size-dist sum | reproducible | ckpt/resume | CPU check | Stata check |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 4,096 | 0.34 | 12,110 | 10 | 943 | 1.000000000000000 | yes | yes | max diff 1.5e-12 | max diff 3.4e-12 |
| 18 | 262,144 | 0.19 | 1,345,736 | 227 | 964 | 1.000000000000000 | yes | yes | max diff 9.1e-13 | — |
| 20 | 1,048,576 | 0.80 | 1,315,761 | 507 | 1010 | 1.000000000000000 | yes | yes | max diff 2.5e-12 | — |
| 22 | 4,194,304 | 2.20 | 1,902,505 | 740 | 1058 | 1.000000000000001 | yes | yes | max diff 1.1e-13 | — |
| 24 | 16,777,216 | 8.96 | 1,871,591 | 1048 | 1282 | 1.000000000000000 | yes | yes | max diff 2.7e-13 | — |

Notes:
- 'reproducible' and 'ckpt/resume' assert BIT-IDENTICAL results
  (same chunk partitioning, deterministic reductions).
- CPU check compares against the exact CPU oracle: full (per-model
  scores + moments) for p <= 20, scores-only aggregates for p = 22, 24.
- peak host MiB is the process-lifetime peak working set at the time
  the level finished (cumulative across levels).
- Stata check (p = 12 only): executed bmaregress oracle, all 4,096
  models (normalized log PMPs) plus PIPs/means/sds/model size.
