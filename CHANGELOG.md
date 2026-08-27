# Changelog

All notable changes to the `gpubma` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0.dev0] - 2026-08-27

### Added
- **BFG (Budgeted Fast GPU) Subpackage (`gpubma.bfg`)**:
  - High-level functional public API `fit_bfg()` and top-level export in `gpubma`.
  - Comprehensive `BFGConfig` dataclass supporting full parameterization and JSON serialization.
  - `BFGScorer`: Float64 PyTorch CUDA/CPU Cholesky batch likelihood evaluator with transparent zero-redundancy caching.
  - `LatticeSampler`: Zero-duplicate combinatorial uniform sampling without replacement supporting exclusion sets.
  - `ExactWingEnumerator`: Exhaustive boundary wing evaluation ($k \le 3$ and $k \ge p - 3$).
  - `EliteRegistry`: Multi-path model registry tracking complete genealogical, beam, and upper-tail exceedance provenance.
  - `GPUEliteSearch`: Two-stage finite-population calibration and sequential GPU discovery with conservative tail prevalence bounds.
  - `GenealogicalSearch`: Bidirectional greedy and forward/backward multi-beam search.
  - `ACESMReconstructor`: Anchored Cumulative Evidence Saturation Model with locked Weibull shape ($\beta = 3.50$) for unbiased denominator reconstruction ($|\Delta \log Z| < 0.010$, MAP PMP error $< 0.005$).
  - `BudgetAllocator`: Uniform, posterior-weighted, and adaptive lattice budget allocation strategies.
  - `CheckpointManager`: Progressive JSON/NPZ state persistence and lossless resumption.
  - `BFGResult` and `LatticeResult`: High-level structured result containers with `.summary()`, `.coefficients()`, `.top_models()`, and matplotlib diagnostic plotting methods (`.plot_convergence()`, `.plot_size_distribution()`, `.plot_pips()`).
  - Standalone Command-Line Interface `gpubma-bfg` (`gpubma.bfg.cli:main`).
- **Comprehensive Test Suite (`tests/bfg/`)**:
  - 19 new unit and integration tests verifying scorer parity, bitmask utilities, sampling, registry, genealogy, elite search, ACESM, allocation, checkpointing, small-$K$ exact parity, $K=30$ canonical benchmark regression, air-gapped zero data leakage, and deterministic seed reproducibility.

### Changed
- Top-level `gpubma.__init__.py` updated to export `fit_bfg`, `BFGResult`, `BFGConfig`.
- BMA test suite expanded from 155 to 174 tests with 100% pass rate.

---

## [0.1.0.dev0] - 2026-07-15

### Added
- Phase 1 exact float64 CPU reference implementation verified against Stata `bmaregress` oracle (~1e-12 tolerance).
- Phase 2 bounded-memory single-GPU exact enumerator with two-pass streaming reduction and chunked matrix generation.
- Full exact enumeration of $2^{30} = 1,073,741,824$ models on `panel_30_center15` benchmark on NVIDIA A100.
- Checkpointing, diagnostics (`python -m gpubma.doctor`), and publication figure generation.
