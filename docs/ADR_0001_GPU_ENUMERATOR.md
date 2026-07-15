# ADR 0001 — Architecture of the exhaustive single-GPU enumerator

- Status: ACCEPTED (2026-07-16)
- Decision: **streamed direct batching** with size-grouped chunks generated
  by combinadic unranking; Gray-code variants REJECTED on measured evidence.
- Hardware evidence measured on: NVIDIA GeForce RTX 3060 (CC 8.6, 12 GiB,
  FP64 = 1/64 FP32), torch 2.5.1+cu121, float64 throughout.
  Script: `scripts/benchmark_enumerator_candidates.py`;
  raw numbers: `reports/enumerator_candidates_bench.json`.

## Context

The enumerator must score every one of 2^p models (p ≤ 30) under the
verified Stata-compatible FWL formulation (docs/FWL_BLOCK_FORMULATION.md):
per model, a k × k Cholesky solve on a gathered submatrix of the
precomputed residualized Gram (p × p), followed by scalar log-score
arithmetic. Constraints: one GPU, float64 only, bounded host and device
memory, exact once-only counting, deterministic checkpoint/resume, future
multi-GPU sharding, streaming top-K and posterior-moment aggregation.

## Candidates

**A. Streamed direct batching.** Enumerate models size-by-size. For model
size k, walk the C(p,k) combinations in colexicographic rank order in
chunks of B; unrank ranks → (B, k) index tensors; gather (B, k, k) Gram
submatrices; batched `cholesky` + `solve_triangular`; accumulate reductions
on device. Every chunk is stateless: it is fully determined by
(k, first_rank, B).

**B. Segmented Gray-code sweep (no factor reuse).** Order models so
neighbours differ by one predictor, hoping for gather locality; each model
is still factorized from scratch. This is candidate A with a different —
and harder to unrank — order: the gather is a random-access indexed load
either way, and measured gather time is only 6–14% of the direct pipeline
(table below), so there is at most a ~10% theoretical win for materially
worse chunk bookkeeping (Gray order mixes model sizes, forcing p-padded
batches or per-chunk regrouping). Rejected without further measurement.

**C. Gray-code with Cholesky update/downdate.** Maintain per-lane factors;
one Gray step per lane = rank-1 update or downdate O(k²) instead of a fresh
O(k³/3) factorization. Theoretical flop advantage ~k/3.

## Measured evidence (float64, warm medians of 5, synchronized)

Direct batching — full pipeline (gather + Cholesky + solve + score):

| k | batch | models/s (Measured) | gather | cholesky | solve |
|---|---|---|---|---|---|
| 4 | 65,536 | 21,023,321 | 6.4% | 77.7% | 12.0% |
| 8 | 65,536 | 8,752,488 | 7.9% | 62.8% | 22.7% |
| 12 | 65,536 | 5,836,941 | 11.0% | 64.8% | 21.4% |
| 15 | 65,536 | 4,682,748 | 13.5% | 63.3% | 21.3% |
| 18 | 32,768 | 2,312,948 | 9.7% | 73.5% | 12.7% |
| 22 | 32,768 | 1,311,838 | 8.3% | 46.1% | 45.1% |
| 26 | 16,384 | 1,087,778 | 9.4% | 43.6% | 46.0% |
| 30 | 16,384 | 946,893 | 10.8% | 47.2% | 46.1% |

Gray-code rank-1 step (batched textbook update + triangular solve, one
model per lane per step) — the FAVOURABLE case (append/remove the last
column only):

| k | lanes | models/s (Measured) | ms/step |
|---|---|---|---|
| 8 | 65,536 | 13,692,414 | 4.79 |
| 15 | 65,536 | 5,472,278 | 11.98 |
| 22 | 32,768 | 1,382,552 | 23.70 |
| 30 | 16,384 | 768,336 | 21.32 |

Peak GPU memory during the k = 30 direct batch (B = 16,384): ~0.5 GiB
transient buffers (recorded in the JSON) — comfortably bounded.

**Reading:** the theoretical k/3 advantage of rank-1 updates does not
materialize. The update's outer loop over k rows is inherently sequential
(~6 small kernels per row → ~180 launches per step at k = 30), so the step
is launch-bound; batched Cholesky is a single vendor-optimized kernel. At
k = 30 Gray is 19% SLOWER than direct (768k vs 947k models/s); at k = 22
they are within noise; only at small k — where direct is already fastest in
absolute terms — does Gray lead, and small k contributes negligibly to the
total (C(p,k) mass sits at mid k). Moreover the measured Gray number
overstates reality: a true enumerator must remove ARBITRARY columns
(a Givens-chain column deletion, more work than rank-1) and pad ragged
per-lane sizes.

## Comparison against the required criteria

| criterion | A. direct batching | C. Gray update/downdate |
|---|---|---|
| float64 stability | each model factorized fresh from the exact Gram; no cross-model error accumulation | downdates lose positive-definiteness in the worst case; error accumulates along each 2^m-step sweep; needs periodic refactorization + monitoring |
| bounded memory | transient (B,k,k) buffers, freed per chunk | persistent lanes × p × p factors + per-lane state |
| implementation complexity | LOW — generalizes the validated Phase-1 scorer | HIGH — ragged active sets, arbitrary-column downdates, refactorization policy |
| checkpoint/resume | trivial: (k, next_rank) pair + reduction state | must serialize every lane factor or re-derive from scratch at segment boundaries |
| exact counting | ranks partition [0, C(p,k)) exactly; Σ_k C(p,k) = 2^p asserted | per-lane step counters; auditable but more moving parts |
| multi-GPU sharding (future) | split rank ranges — embarrassingly parallel, no state | partition subcubes; lane state per device |
| top-K / moment aggregation | identical streaming reductions per chunk | identical, plus factor state interleaved |
| measured speed (this GPU) | 0.95–21M models/s (k = 30…4) | not faster anywhere that matters; 19% slower at k = 30, favourable-case measurement |

## Decision

Implement **streamed direct batching**:

1. For k = 0…p, enumerate C(p,k) combination ranks in chunks of B.
2. Unrank on device via the combinadic (combinatorial number system) with a
   precomputed C(p,k) table in float64-exact int64 — deterministic,
   stateless, shardable by rank range.
3. Gather (B,k,k)/(B,k) sufficient-statistic submatrices; batched
   Cholesky; triangular solve; joint-shrink log score (FWL formulation).
4. Streaming on-device reductions per chunk: running log-sum-exp
   normalizer; PIP numerators; moment numerators; size distribution;
   top-K via per-chunk top-K merge. All in float64.
5. Checkpoint = (k, next_rank, reduction state) written atomically to disk;
   resume re-verifies dataset checksums and chunk arithmetic.

Gray-code (B and C) is rejected on measured evidence and implementation
risk, per the project rule that theoretical elegance alone must not decide.
Revisit only if a future device shows batched-Cholesky throughput far below
its rank-1-update throughput at mid k, and then only with a
positive-definiteness monitoring plan.

## Consequences

- The enumerator reuses the validated scoring formula unchanged (CLAUDE.md
  rule 8: no optimization of unvalidated formulas — the formula is the one
  already proven against Stata).
- Whole-space timing at p = 30 is NOT claimed here. Component rates above
  are measured at fixed k; end-to-end measurements happen on the
  progressive ladder (p = 12 → 24) before any p = 30 authorization.
- FP64 policy: pure float64 (GeForce 1/64 penalty accepted). Measured mid-k
  rates (~4.7M models/s at k = 15) already imply acceptable ladder times;
  no mixed-precision escape hatch is added.
