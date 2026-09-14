# Benchmark summary

These are transcribed historical DEVELOPMENT measurements, not new release-candidate runs. Milestone tables retain the closure's PARTIALLY SUPPORTED evidence qualification; independently audited scaling discovery is a separate source.

| p | Algorithm | Budget | Median fraction explored | MAP success minimum | Median top10 | Median top100 | Median mass | Median wall seconds |
|---|---|---|---|---|---|---|---|---|
| 12 | BFG | 800 | 0.1953125 | 1 | 1 | 0.58 | 0.997763886551 | 0.251543 |
| 12 | Baseline_Greedy | 800 | 0.1953125 | 1 | 0.9 | 0.365 | 0.962725852802 | 0.2472176 |
| 12 | Baseline_Random | 800 | 0.1953125 | 0 | 0.25 | 0.225 | 0.0862751906901 | 0.2379236 |
| 18 | BFG | 10000 | 0.0381469726562 | 1 | 1 | 0.78 | 0.987329626233 | 2.22592135 |
| 18 | Baseline_Greedy | 10000 | 0.0381469726562 | 1 | 1 | 0.295 | 0.945786468246 | 3.47251685 |
| 18 | Baseline_Random | 10000 | 0.0381469726562 | 0 | 0 | 0.04 | 0.0117435042472 | 1.98928045 |
| 30 | BFG | 5000 | 4.65661287308e-06 | 1 | 0.9 | 0.89 | 0.918023432498 | 1.0929116 |
| 30 | Baseline_Greedy | 5000 | 4.65661287308e-06 | 0 | 0.8 | 0.31 | 0.803736084202 | 1.0851873 |
| 30 | Baseline_Random | 5000 | 4.65661287308e-06 | 0 | 0 | 0 | 2.28762556242e-17 | 0.99532395 |

MAP success minimum=1 means all ten recorded seeds found MAP; minimum=0 means at least one did not, not that every seed failed. Inspect seed_results.csv for individual flags and N_MAP. Recorded timings depend on the historical GPU/configuration and include different phases from the release API. They do not promise a speedup on another machine. Only original measured comparisons support runtime-advantage claims; no new enumeration baseline was run.

The independent scaling audit corroborated 30/30 MAP and 29/30 true-model discoveries (five realizations at each p=12,15,18,21,24,30). Median saved-order N_MAP: 192,294,431,585,991,1216. The exact dataset rows include N_TRUE and right-censoring. The missing true model, p21/r01, was rank 2 and remained absent after 240,831 evaluations.

Canonical p30 BFG found MAP in 10/10 seeds at 5,000 evaluations versus greedy 8/10. It did not recover all top10 models or reach 95% mass, and its mass plateaued through 50,000. Scaling p30 terminal mass was much lower: 34.9221?63.1416%. Uniform random was substantially inferior in these retained configurations; this does not establish dominance over all local-search algorithms.

Historical exact-oracle PIP error columns are reference diagnostics, not evidence of a globally calibrated public BFG PIP estimator. True-model endpoints in the separate scaling table must not be assigned to milestone inputs. Ordinary API results do not contain exact oracle ranks or causal labels.

Software-only regression recipes include the existing p5 and p12 fixtures, CPU/RTX3060 resume and determinism, mask-width tests, and 16-evaluation input smokes for p12/p18/p30. The scientific benchmark campaign was not repeated. See docs/software_qa.md after candidate validation.
