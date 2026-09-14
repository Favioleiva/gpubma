# DEVELOPMENT benchmark evidence and software fixtures

`python benchmark/report.py` reads previously recorded random/greedy/BFG metrics. `python benchmark/report.py --ranks` reports best-so-far from already recorded exact ranks; unknown ranks are not replaced by fabricated values. No model score is computed by either report command.

The small p12 (4,096 models), medium p18 (262,144), and large p30 (1,073,741,824) fixtures are compact copies of existing synthetic X/y/control arrays. Their inventory binds each source and export hash. They contain no scientific oracle and no personal observations. `tests/test_benchmark_fixtures.py` runs only 16-evaluation software smokes, not scientific reproductions. Do not infer recovery rates from these tiny tests.

`recorded/benchmark_metrics.csv` and `seed_results.csv` retain all recorded budgets, unique counts, fractions, runtime, MAP flags/N_MAP, top10/top100 overlap, comparator and seed outcomes. `recorded_rank_observations.csv.gz` retains exact rank observations for seed20260715 at each benchmark's largest recorded budget, supporting trajectory views. `scaling_discovery.csv` supplies separate 30-dataset true/MAP identities, N_TRUE, censoring, posterior reference ranks and N_MAP. Missing true-model/comparator fields across these distinct experiments are not imputed or conflated. Negative map_eval_count sentinel -1 means not discovered at that run's budget, not N_MAP=-1.

The milestone search implementation is a historical benchmark-specific engine, distinct from the current package API. These results therefore support historical capabilities; they are not a promise that changing engine/defaults yields identical numbers. Random comparison is uniform random search; greedy is the milestone's greedy-from-sample comparator, not every greedy algorithm. The separate canonical genealogy experiment used another setup.

Exact reference quantities are benchmark-only historical diagnostics. Full p18/p30 oracle arrays and expensive enumeration runners are deliberately excluded. Complete historical oracle regeneration requires the separately archived research environment; this lightweight candidate does not claim standalone scientific reproduction of that campaign. Tests validate software and table integrity, not historical acceptance or full oracle truth.

NPZ arrays: `X`, `y`, `always_in` (w1,w2). Historical scorers included an intercept, shrink controls, g=n, Beta-Binomial(1,1), float64. No input is retrieved from the network. Optional future experiments require a separately defined execution scope.
