# GPUBMA

Exhaustive **Bayesian Model Averaging** for Gaussian linear regression, built
toward single-GPU enumeration of 2^30 = 1,073,741,824 candidate models.
Open source, BSD-3-Clause, Python-first. Stata is used only as an external
validation oracle on small datasets — never as a dependency.

**Status: Phase 1** — reproducible data, diagnostics, and an exact float64
CPU reference implementation. The production CUDA enumerator is *not*
implemented yet (see `STATUS.md` and `docs/PROJECT_SCOPE.md`).

## Install (development)

```bash
pip install -e .[dev]
```

Requires Python ≥ 3.10, NumPy, SciPy, pandas (PyArrow for Parquet, PyTorch
with CUDA for the GPU feasibility layer).

## Quick start

```python
import pandas as pd
from gpubma import bma_regress

df = pd.read_parquet("data/synthetic/panel_8.parquet")

result = bma_regress(
    data=df,
    outcome="y",
    predictors=[f"x{j}" for j in range(1, 9)],   # 2^8 = 256 candidate models
    controls=["w1", "w2"],                        # always included
    fixed_effects=["individual", "time"],         # always included
    entity_col="individual_id",
    time_col="period",
    backend="cpu",
    method="enumeration",
    precision="float64",
)
print(result.summary())
result.coefficients()             # PIP, posterior mean, posterior sd
result.inclusion_probabilities()
result.top_models()
result.model_size_distribution()
```

Estimator style:

```python
from gpubma import GPUBMARegressor
est = GPUBMARegressor(predictors=[f"x{j}" for j in range(1, 9)]).fit(df, outcome="y")
```

## Tools

```bash
python -m gpubma.doctor            # GPU/CUDA/environment diagnostics (real float64 test)
python -m gpubma.benchmark --max-predictors 15
python -m gpubma.gpu.feasibility   # GPU feasibility report
python scripts/generate_synthetic_panels.py   # regenerate frozen data
python scripts/download_grunfeld.py           # regenerate Grunfeld snapshot
python scripts/compare_fixed_effects.py       # dummies vs absorption report
python scripts/compare_stata_python.py        # deterministic comparisons
python -m pytest                              # 65 tests (GPU tests skip cleanly without CUDA)
```

## Honesty rules

The non-negotiable project rules live in `CLAUDE.md`: exhaustive enumeration
(no silent MC3), explicit float64, measured-vs-projected labelling, detected
(never assumed) hardware, reproducible seeds and checksums, and honest
recording of unresolved statistical questions. The default
`always_prior="shrink"` parameterization is **verified against executed
Stata `bmaregress` output** (StataNow/SE 19.5, six designs, worst absolute
difference 1.8e-12 — see `reports/comparison_report.md`); the alternative
`always_prior="flat"` convention is gpubma-specific and not Stata's.
