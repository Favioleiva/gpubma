# GPUBMA

Exhaustive **Bayesian Model Averaging** for Gaussian linear regression, built
toward single-GPU enumeration of 2^30 = 1,073,741,824 candidate models.
Open source, BSD-3-Clause, Python-first. Stata is used only as an external
validation oracle on small datasets — never as a dependency.

**Status: Phase 2 complete** — exact float64 CPU reference (Stata-verified
to ~1e-12 on seven designs), plus a bounded-memory exhaustive single-GPU
enumerator validated at p = 12…24 and executed at **p = 30
(1,073,741,824 models, ~95 s measured on an A100-SXM4-40GB)**. See
`STATUS.md`, `docs/ADR_0001_GPU_ENUMERATOR.md`, and
`docs/FWL_BLOCK_FORMULATION.md`.

## Install

```bash
pip install -e .[dev]                 # from a clone (development)
pip install "gpubma[gpu] @ git+https://github.com/Favioleiva/gpubma"
```

Requires Python ≥ 3.10, NumPy, SciPy, pandas (PyArrow for Parquet, PyTorch
with CUDA for the GPU enumerator).

## Run the exact 2^30 enumeration on Google Colab (`panel_30_center15`)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpubma/blob/main/notebooks/GPUBMA_A100_p30.ipynb)

The canonical notebook is **`notebooks/GPUBMA_A100_p30.ipynb`**
(<https://github.com/Favioleiva/gpubma/blob/main/notebooks/GPUBMA_A100_p30.ipynb>).
It targets the frozen **`panel_30_center15`** benchmark, published in this
repository at `data/synthetic/panel_30_center15.parquet` (+ metadata at
`data/synthetic/panel_30_center15_metadata.json`) — a fresh Colab clone
loads it directly, no external downloads or private access.

Requirements and workflow (**select an A100 GPU runtime** — the A100 40 GB
runtime is the validated target; do not run on CPU):

1. open the notebook in Colab (badge above) and pick the A100 runtime;
2. run bootstrap (clones this repo at a pinned public commit) and the
   canonical input-validation cell;
3. run the **mandatory CPU/GPU smoke test** (the expensive cell refuses to
   start without a PASS);
4. optionally mount Google Drive for checkpoint persistence;
5. set `RUN_FULL_EXACT_P30 = True` (it defaults to **False**) and run the
   clearly labelled expensive cell — the exact enumeration of all
   2^30 = 1,073,741,824 models;
6. after a disconnect, reopen and Run all: SHA-256-gated checkpoints on
   Drive resume the run with at most ~1 minute lost;
7. validate the completed posterior and download the compact results ZIP.

**Compute cost:** the full run consumes Colab compute units.
`panel_30_center15` (n = 2,000, correlated true regressors x1–x15 plus 15
structural-zero proxies x16–x30) is a harder design than the original
sparse benchmark, and **no runtime claim exists for it until the A100 run
is actually completed**. *Historical note:* the ~95 s / ~10–12 min PASS2
figures measured earlier on an A100-SXM4-40GB belong to the OLD
`panel_30` sparse benchmark (n = 1,000) and must not be attributed to
`panel_30_center15`.

**Scientific expectation:** because proxies correlate 0.79–0.89 with their
true sources, posterior mass may legitimately spread across
observationally similar models — **the exact true model need not have the
highest PMP**; family-level (true+proxy) recovery is the meaningful
outcome. See `reports/panel_30_center15_dgp_validation.md`.

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
python scripts/run_enumeration_ladder.py      # GPU validation ladder p=12..24
python -m pytest                              # 97 tests (GPU tests skip cleanly without CUDA)
```

## Honesty rules

The non-negotiable project rules: exhaustive enumeration
(no silent MC3), explicit float64, measured-vs-projected labelling, detected
(never assumed) hardware, reproducible seeds and checksums, and honest
recording of unresolved statistical questions. The default
`always_prior="shrink"` parameterization is **verified against executed
Stata `bmaregress` output** (StataNow/SE 19.5, six designs, worst absolute
difference 1.8e-12 — see `reports/comparison_report.md`); the alternative
`always_prior="flat"` convention is gpubma-specific and not Stata's.
Phase 2 extended the oracle to a per-model comparison of all 4,096
panel_12 models (worst |diff| 3.4e-12) and validated the GPU enumerator
bit-for-bit on reproducibility and checkpoint/resume
(`reports/enumeration_ladder.md`).
