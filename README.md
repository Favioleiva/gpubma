# GPUBMA

Exhaustive **Bayesian Model Averaging** for Gaussian linear regression, built
toward single-GPU enumeration of 2^30 = 1,073,741,824 candidate models.
Open source, BSD-3-Clause, Python-first. Stata is used only as an external
validation oracle on small datasets — never as a dependency.

**Status: canonical Phase 2 experiment complete** — exact float64 CPU
reference (Stata-verified to ~1e-12 on seven designs), a bounded-memory
single-GPU enumerator validated progressively at p = 12…24, and the exact
`panel_30_center15` run at **p = 30 (1,073,741,824 models)**. On an
NVIDIA A100-SXM4-80GB, measured PASS1 was 115.7 s (9.28M models/s), exact
PASS2 was 896 s, and peak GPU memory was 1.93 GiB. See `STATUS.md` and
`reports/CANONICAL_P30_RESULTS.md`.

## Install

```bash
pip install -e .[dev]                 # from a clone (development)
pip install "gpubma[gpu] @ git+https://github.com/Favioleiva/gpubma"
```

Requires Python ≥ 3.10, NumPy, SciPy, pandas (PyArrow for Parquet, PyTorch
with CUDA for the GPU enumerator).

## Run the exact 2^30 enumeration on Google Colab (`panel_30_center15`)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpubma/blob/main/notebooks/GPUBMA_A100_p30.ipynb)

The clean runnable notebook is **`notebooks/GPUBMA_A100_p30.ipynb`**
(<https://github.com/Favioleiva/gpubma/blob/main/notebooks/GPUBMA_A100_p30.ipynb>).
It targets the frozen **`panel_30_center15`** benchmark, published in this
repository at `data/synthetic/panel_30_center15.parquet` (+ metadata at
`data/synthetic/panel_30_center15_metadata.json`) — a fresh Colab clone
loads it directly, no external downloads or private access.

Requirements and workflow (**select an A100 GPU runtime** — the completed
canonical measurement used an A100-SXM4-80GB; do not run on CPU):

1. open the notebook in Colab (badge above) and pick the A100 runtime;
2. run bootstrap (clones this repo at a pinned public commit) and the
   canonical input-validation cell;
3. run the **mandatory CPU/GPU smoke test** (the expensive cell refuses to
   start without a PASS);
4. set `RUN_FULL_EXACT_P30 = True` (it defaults to **False**) and run the
   clearly labelled expensive cell — the exact enumeration of all
   2^30 = 1,073,741,824 models;
5. protect progress at any time with `download_checkpoint_bundle()` (a
   compact ZIP to your computer) and, after a destroyed runtime, restore
   it through the upload cell;
6. validate the completed posterior and download the compact results ZIP.

**Storage modes — Google Drive is optional and OFF by default
(`USE_GOOGLE_DRIVE = False`); no paid Drive plan is required.**

- *Default, free mode:* GitHub inputs → Colab local runtime (`/content`)
  → manual checkpoint/result downloads. The dataset is hosted publicly in
  this repository and read from the cloned copy — no uploads, credentials,
  or tokens. Colab local storage is **ephemeral**: checkpoints survive
  cell reruns in the same active runtime but are lost if the runtime is
  destroyed; downloading checkpoint bundles protects progress without
  Drive.
- *Optional persistence mode* (`USE_GOOGLE_DRIVE = True`): checkpoints and
  results live on your Drive, and a destroyed session resumes
  automatically after Run all. Scientific and numerical behavior is
  identical in both modes.

**Completed canonical evidence:** the executed notebook with retained outputs
and 23 final figures is
`notebooks/GPUBMA_A100_p30_middle15_stata_figures.ipynb`. Its compact,
versioned export is `reports/artifacts/panel_30_center15_exact_results.zip`.
The earlier ~95 s figure belongs to the old sparse `panel_30` benchmark and
must not be attributed to `panel_30_center15`.

**Scientific result:** x1–x14 are essentially recovered, while strong
substitution remains within the deliberately correlated x15/x30 family. The
MAP model uses x30 instead of x15 (PMP 47.15%); the exact generating model
ranks 8th (PMP 1.88%). Family-level recovery and posterior uncertainty are
therefore the meaningful criteria—not whether the generating model ranks
first. See `reports/CANONICAL_P30_RESULTS.md`.

### Regenerate the canonical figures without rerunning BMA

Install the plotting extra and point the generator at an existing results ZIP
or its extracted directory:

```bash
pip install -e .[plots]
```

```python
from gpubma import generate_canonical_figures

variable_names = [f"Variable {i}" for i in range(1, 31)]
variable_names[:5] = [
    "Initial income",
    "Mining production",
    "W × Mining production",
    "Capital stock",
    "W × Capital stock",
]

manifest = generate_canonical_figures(
    "reports/artifacts/panel_30_center15_exact_results.zip",
    "canonical_figures",
    variable_names=variable_names,
)
```

This writes the complete set of 23 PNGs followed by
`panel_30_center15_figure_manifest.json`. Generation fails if the manifest and
actual PNG filenames differ; every manifest record includes byte size and
SHA-256. Labels are positional, accept spaces and Unicode, and must contain
exactly one unique non-empty string per predictor. If `variable_names` is
omitted, names stored with the results are used; the resolver's final fallback
is `x1, ..., xp`. `BMAResult.predictor_names` already preserves the predictor
column names supplied to `bma_regress`. No enumerator or posterior calculation
is called.

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
python -m pytest                              # 155 collected; expensive CPU checks skip explicitly when documented
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
