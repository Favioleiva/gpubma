# BFG Productionization Map: Research Prototype to Public Package Integration

This document maps all validated scientific algorithms, statistical estimators, and data structures developed in **Contracts 1–5** to their canonical production destinations within the `gpubma` package.

---

## 1. Architectural Component Mapping

| Research Component | Source Contract | Source Prototype File | Production Package Destination | Architectural Notes & Enhancements |
| :--- | :---: | :--- | :--- | :--- |
| **Exact Float64 GPU Bitmask Scorer** | Contract 1, 2, 3 | `contracts/Contract2/code/genealogical_engine.py`, `src/gpubma/gpu/batch_scorer.py` | `src/gpubma/bfg/scorer.py` | Float64 PyTorch GPU batch Cholesky scorer, FWL QR shrink formulation, Zellner $g$-prior, model prior, CPU fallback support. |
| **Extensible Model-ID Representation** | Contract 1, 2, 3 | `contracts/Contract2/code/genealogical_engine.py` | `src/gpubma/bfg/scorer.py` | Native integer bitmask for $K \le 64$; packed tuple/block representation for $K > 64$. |
| **Global Evaluation Cache** | Contract 2, 3, 4 | `contracts/Contract2/code/genealogical_engine.py` | `src/gpubma/bfg/scorer.py` | In-memory hash registry ensuring zero redundant model rescoring across all exploration phases. |
| **Exact Boundary Wings** | Contract 1, 4, 5 | `contracts/Contract4/code/bfg_denominator_engine.py` | `src/gpubma/bfg/sampling.py` | Exhaustive enumeration of extreme lattices where $\binom{K}{k} \le N_{\text{wing\_max}}$ (e.g. $k \le 3, k \ge K-3$). |
| **Uniform Random Reconnaissance** | Contract 1, 3, 4 | `contracts/Contract3/code/gpu_elite_search_engine.py` | `src/gpubma/bfg/sampling.py` | Zero-duplicate uniform $k$-combination sampling without replacement, deterministic RNG seeding. |
| **GPU Elite Search** | Contract 3 | `contracts/Contract3/code/gpu_elite_search_engine.py` | `src/gpubma/bfg/elite_search.py` | Calibration quantile estimation ($\tau_k$), sequential upper-tail discovery, finite-population prevalence $\widehat{H}_k$. |
| **Bidirectional Genealogical Search** | Contract 2 | `contracts/Contract2/code/genealogical_engine.py` | `src/gpubma/bfg/genealogy.py` | Forward greedy ($k \to K$), backward greedy ($k \to 0$), bidirectional collision, Hamming distance tracking. |
| **Genealogical Beam Search** | Contract 2 | `contracts/Contract2/code/genealogical_engine.py` | `src/gpubma/bfg/genealogy.py` | Multi-path beam search retaining top $B$ candidates per lattice generation, seeding from elite search hits. |
| **Multi-Path Elite Registry** | Contract 4, 5 | `contracts/Contract4/code/bfg_denominator_engine.py` | `src/gpubma/bfg/registry.py` | Provenance tracking (`RANDOM_BULK`, `RANDOM_TAIL`, `ELITE_HIT`, `FORWARD_GENEALOGY`, `BACKWARD_GENEALOGY`, `BEAM`, `EXACT_WING`, `MULTIPLE_SOURCE`), cumulative mass. |
| **ACESM Saturation Engine** | Contract 5 | `contracts/Contract5/code/acesm_engine.py` | `src/gpubma/bfg/acesm.py` | Anchored Weibull saturation ($\Phi(d) = 1 - \exp(-(\alpha d)^\beta)$) with validated locked default $\beta = 3.50$, L-BFGS-B in log space. |
| **Cumulative Evidence Builder** | Contract 5 | `contracts/Contract5/code/cumulative_curve_builder.py` | `src/gpubma/bfg/acesm.py` | Horvitz-Thompson inclusion-weighted cumulative curve $C_k(d)$ construction with boundary slope calculation. |
| **Adaptive Budget Allocator** | Contract 6 Spec | New design based on Contract 1–5 findings | `src/gpubma/bfg/allocation.py` | Dynamic allocation strategies: `uniform`, `posterior`, and `adaptive` ($A_k = \widehat{P}(k \mid y) \times U_k$). |
| **Global Reconstruction & Normalization** | Contract 4, 5 | `contracts/Contract5/code/global_reconstruction.py` | `src/gpubma/bfg/results.py` | Numerically stable LogSumExp evidence aggregation $\log \widehat{Z} = \text{LSE}_k(\log \widehat{Z}_k)$, $P(k \mid y)$ distribution. |
| **Posterior Inclusion Probabilities (PIPs)** | Contract 1, 4, 5 | `src/gpubma/cpu/enumeration.py` | `src/gpubma/bfg/results.py` | Statistically valid reconstructed variable inclusion: $\widehat{\text{PIP}}_j = \sum_k \widehat{P}(j \in M \mid k, y) \widehat{P}(k \mid y)$. |
| **Posterior Coefficients & Moments** | Contract 1 | `src/gpubma/cpu/enumeration.py` | `src/gpubma/bfg/results.py` | Aggregated BMA posterior means $E[\beta_j \mid y]$, standard deviations, and sign probabilities $P(\beta_j > 0 \mid y)$. |
| **Progressive Checkpoints & Resumption** | Contract 6 Spec | New design based on Contract 1–5 persistence | `src/gpubma/bfg/checkpoint.py` | Persistent state serialization (JSON / NPZ), interrupt/resume with zero redundant model evaluation. |
| **Configuration Dataclass** | Contract 6 Spec | New design | `src/gpubma/bfg/config.py` | Validated `BFGConfig` dataclass with serializable parameters and sensible defaults. |
| **Results & Diagnostics Interface** | Contract 6 Spec | `src/gpubma/result.py` | `src/gpubma/bfg/results.py`, `src/gpubma/bfg/diagnostics.py` | `BFGResult` and `LatticeResult` exposing clean APIs: `.summary()`, `.plot_convergence()`, `.plot_pips()`. |
| **Pipeline Coordinator & Functional API** | Contract 6 Spec | New design | `src/gpubma/bfg/engine.py`, `src/gpubma/__init__.py` | Main public entry point `fit_bfg()` orchestrating data validation, residualization, and all search phases. |
| **Command-Line Interface (CLI)** | Contract 6 Spec | New design | `src/gpubma/bfg/cli.py` | CLI entry point `gpubma-bfg` for standalone execution and cluster scheduling. |

---

## 2. Package Directory Layout

```text
src/gpubma/
├── __init__.py                # Exposes fit_bfg, BMAResult, bma_regress
├── api.py                     # Existing exact enumeration API
├── ...
└── bfg/
    ├── __init__.py            # BFG package exports (fit, fit_bfg, BFGConfig, BFGResult, etc.)
    ├── config.py              # BFGConfig dataclass and validation
    ├── engine.py              # BFGEngine and top-level fit_bfg function
    ├── scorer.py              # BFGScorer, PyTorch GPU batch Cholesky, Bitmask representation
    ├── sampling.py            # LatticeSampler, exact wings, no-replacement combinatorial draws
    ├── elite_search.py        # GPUEliteSearch, calibration, quantile estimation, prevalence
    ├── genealogy.py           # GenealogicalSearch, forward, backward, beam search
    ├── registry.py            # EliteRegistry, provenance tracking, cumulative mass
    ├── acesm.py               # ACESMReconstructor, Weibull saturation, curve builder
    ├── allocation.py          # BudgetAllocator (uniform, posterior, adaptive)
    ├── checkpoint.py          # CheckpointManager, state serialization, resume logic
    ├── results.py             # BFGResult, LatticeResult, PIPs, moments, MAP
    ├── diagnostics.py         # Summary tables, convergence tracking, diagnostics
    └── cli.py                 # CLI interface (gpubma-bfg)
```
