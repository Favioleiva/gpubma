# Canonical p30 replication report

The unchanged public `data/synthetic/panel_30_center15.parquet` was searched with BUDGET=5,000, seed=12345, beam_width=5, float64, DEVICE=auto selecting NVIDIA GeForce RTX 3060. Candidate order x1–x30, outcome y, w1/w2 always in, all 2,000 rows, no FE or sample transformation. g=2,000 and beta-binomial(1,1) match the reference convention.

## Archived reference and identity

The reference is the accepted `reports/artifacts/panel_30_center15_exact_results.zip`, whose selected exact-rank/true-model files were verified against its run manifest. The public compact transcription records source and member SHA-256 hashes in `src/gpubma/benchmarks/canonical_p30_reference.json`; no new enumeration was performed. The reference run records commit ef9d68af462564b5cbdce268be1ff2b43317dd24 and A100-SXM4-80GB. Current runtime is not compared as an identical hardware measurement.

Dataset SHA-256: `7b468ca0c09249a83b05638b53c884bc444fdd535ba5f698b8fa92c24f7dd6e0`. Local bytes and the public raw GitHub file matched.

| Target | Reference | Current BFG result | Classification |
|---|---|---|---|
| MAP identity | 536887295 (x1–x14, x30) | Same mask, found | EXACTLY REPLICATED |
| MAP log numerator | 1118.521780616737 | 1118.521780616737; absolute difference 0 | NUMERICALLY REPLICATED WITHIN TOLERANCE |
| Exact top-10 models discovered | 10 | 10/10 | EXACTLY REPLICATED |
| Top-10 rank-list overlap | 10 | 10/10 | EXACTLY REPLICATED |
| Historical milestone top-10 recall at 5,000 | 0.90 | 1.00 | BEHAVIORALLY REPLICATED |
| True model explicitly discovered | 32767 (x1–x15) | FOUND | BEHAVIORALLY REPLICATED |
| N_MAP | Verified historical first-hit target unavailable | 1,203 unique evaluations | REFERENCE NOT AVAILABLE |
| N_TRUE | Verified historical first-hit target unavailable | 2,742 unique evaluations | REFERENCE NOT AVAILABLE |
| Deterministic seed replay | Same current configuration/backend | Scores, masks, order, initial provenance and parents exactly match | EXACTLY REPLICATED |

The absolute log-score tolerance remains the existing 1e-9 from `tests/test_panel_30_center15_gpubma.py`; no tolerance was relaxed. Historical 90% recall belongs to its own retained milestone configuration; achieving 100% here is behavioral corroboration, not retroactive replacement of that result.

## True model versus MAP

M*=32767 has candidate size15, archived exact/global rank8 and PMP 0.01881348830750715. The exact MAP has candidate size15, rank1 and PMP 0.4715187997893168. Hamming distance is2: x15 is replaced by x30. M* is not the MAP. These exact/global quantities come only from the archived benchmark, separately from discovered-set weights. The main regression columns remain BFG ranks1–5; M* is not forced into them.

## Budget and runtime

Requested BUDGET=5,000; actual unique evaluations=5,000; universe=1,073,741,824. Fraction=4.656612873077393e-6, or 0.0004656612873077393%. All stages share the same scorer cap.

Primary standalone search: 0.448165s algorithm / 0.493966s fit-call wall time. Deterministic repeat: 0.252561s / 0.278576s. These are this machine's measurements, not portable performance guarantees.

Final notebook Run All: 29.894s; algorithm 0.629454s; fit-call 0.669746s; post-search reporting 16.387s. Stages do not retain separate runtimes, so stage runtimes are REFERENCE NOT AVAILABLE. Report/OLS-refit cost is not counted as BFG search time.

## Scope and limitations

This is one canonical public-data software/replication check at a predeclared 5,000 budget, not a new scaling campaign. The reference comparison occurs after discovery. No HOLDOUT, new exact universe computation, W-PCS continuation or additional dataset experiment was performed. No live Google account/runtime was operated. Fresh local kernels exercised the standalone notebook package on CUDA and CPU; live Colab remains a human environment check. The public main branch predates these helpers, so notebooks transparently bundle the reviewed release-candidate wheel until human publication.
