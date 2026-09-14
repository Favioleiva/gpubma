# Public BFG discovery API

```python
from gpubma import fit_bfg, BFGConfig, BFGResult
result = fit_bfg(y, X, candidate_names=None, always_in=None,
                 config=BFGConfig(budget_models=1000, beam_width=5, device="cpu"))
```

Use either `config` or keyword search options. Mixing them raises instead of silently ignoring options. `BFGConfig()` preserves the historical dataclass beam width 15; `fit_bfg()` without a config preserves functional width 5. Set width explicitly for reproducibility. Other retained defaults: seed 20260715, float64, shrink treatment of controls, benchmark g, Beta-Binomial(1,1) model prior, batch size 16384, wing_max_size 4096, elite_quantile .05, elite_calibration_size 500, reconnaissance allocation target 2500, adaptive allocation. Hard budget is the only accepted budget semantics.

`y` and `X` must be finite, with matching rows; constant candidate columns and duplicate names fail validation. Intercept is included automatically. `always_in` supplies additional numeric controls; do not add a duplicate intercept. Candidate variable names are labels, not causal annotations.

| Surface | Meaning |
|---|---|
| `best_model`, `best_log_score` | Best scored model record and score; no global MAP certificate |
| `top_models(k=20)` | Discovered ranks, ties in evaluation order |
| `discovery_table()` | Every scored ID, size, score, one-based unique-evaluation order, variables, first stage, source tags and available first traversal source |
| `genealogy()`, `parent_child_edges()` | Recorded scored-endpoint transitions; backward source can be the larger model |
| `champion_path()` | Strict improvements in best discovered score |
| `model_families()` | Descriptive connected components of retained edges, including isolated records; not posterior-mode classification |
| `search_diagnostics()` | Counts, shared budget, fraction, elapsed time, stage registrations, cache hits |
| `hardware`, `reproducibility` | Backend, device, precision, software versions, data/code/config fingerprint |
| `log_Z_seen` | Log sum of observed model scores only |
| `discovered_set_model_weights()` | Y_M/Z_seen; NOT global PMP |
| `discovered_set_pips()` | Inclusion under discovered-set weights; NOT global PIP |
| `summary()` | Human-readable discovery summary |
| `save_json(path)`, `BFGResult.load_json(path)` | Exclusive result creation and round-trip loading; decimal-string JSON model masks |

Do not infer physical within-GPU-batch timing from serialized evaluation order. `initial_provenance` is first registration stage; `source_tags` records later registration sources. Only one original traversal source is retained, so absent edges cannot establish absent model relationships. Oracle-only fields (exact rank, exact PMP, true model, N_MAP/N_TRUE) belong to benchmark records, never ordinary result objects.

## Resume

```python
cfg = BFGConfig(budget_models=1000, beam_width=5, device="cpu", checkpoint_dir="checkpoints")
result = fit_bfg(y, X, config=cfg)
cfg.resume = True
resumed = fit_bfg(y, X, config=cfg)
```

Intermediate saves occur at completed phase/shell boundaries. Optional `checkpoints=[100,500]` requests a save at the next such boundary; it does not interrupt a batch or modify search selection. A complete final checkpoint is always saved. The latest pointer is relative, atomically updated after payload hashes exist. Missing/corrupt state and changed input/config/code fail closed. These checks detect corruption; they are not an adversarial external-custody protocol. Legacy posterior-result checkpoints are incompatible and are not silently migrated. Result JSON is not a resumable execution checkpoint. Elapsed time after resume includes retained prior elapsed time, excluding downtime.

## Migration

Remove BFG `.pips`, `.map_pmp`, `.log_Z`, `.posterior_mean`, `.posterior_sd`, `.posterior_moments` assumptions. Explicit discovered-set methods have different semantics and are not numerical drop-in replacements. No mass estimator is called. ACESM settings, unused wall-clock/progress/elite-budget knobs, and `sampling_only` budget mode are rejected. The old internal reconstruction sampling allocation is retained as final reconnaissance solely to preserve ordered evaluations. The separate exhaustive APIs keep their distinct `BMAResult` type.

## Experimental Post-BFG Shell Recovery API

```python
from gpubma.bfg import shell_recovery as sr
```

A clean, separate experimental/benchmark namespace for representative reticular score-distribution estimation without modifying `fit_bfg()`:

| Function / Class | Purpose |
|---|---|
| `shell_population_size(p, k)` | Returns shell combination count $\binom{p}{k}$ |
| `sample_size(p, k, discovered=0, cap=100_000)` | Computes 1% capped sampling quota bounded by undiscovered remainder |
| `rank_indices(indices)`, `rank_mask(mask, p)` | Colexicographical rank mapping from indices or bitmask |
| `unrank_combination(rank, p, k)`, `mask_from_rank(rank, p, k)` | Exact unranking with arbitrary-precision integer support |
| `RemainderSampler(p, k, excluded_ranks, seed)` | Deterministic nested uniform without-replacement sampler for shell remainders |
| `RecoveryScorer(scorer, batch_size=65536)` | High-throughput GPU/CPU scorer that bypasses dictionary caching |
| `reconstruct_shell_mixture(discovered, sample, N_k)` | Computes representative mixture weights ($1/N_k$ vs $(N_k-D_k)/(N_k n_k)$) |
| `weighted_summary(values, weights)` | Weighted mean, variance, and quantiles |
| `weighted_distances(exact, values, weights)` | Exact discrete-CDF Kolmogorov-Smirnov and Wasserstein-1 distances |

