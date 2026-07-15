# GPUBMA — Phase 1: Reproducible Data, GPU Diagnostics, and CPU Reference

You are Claude Fable 5 working inside Claude Code as the lead engineer for a new open-source Python project named `gpubma`.

This is the first, deliberately limited phase of the project.

Do not attempt to build the complete billion-model CUDA engine yet.

The purpose of this phase is to create the reproducible statistical and computational foundation required before GPU optimization begins.

## Project identity

Use the following names consistently:

```text
Project name: GPUBMA
Repository name: gpubma
Python package name: gpubma
Primary estimator class: GPUBMARegressor
Primary functional API: bma_regress
License: BSD-3-Clause
```

The package must be open source and usable directly from Python.

Stata is not the target platform and must not be a runtime dependency.

Stata will be used only as an external validation oracle on small datasets.

## Long-term objective

The eventual objective is an open-source Python package that performs exhaustive Bayesian Model Averaging for linear regression using one NVIDIA GPU.

The ultimate production experiment will contain:

```text
30 optional predictors
2^30 = 1,073,741,824 models
One local NVIDIA GPU or one Google Colab A100
Exhaustive enumeration
No MC3 substitution
```

Do not implement this production-scale enumerator during the current phase.

## Current phase objectives

Complete the following work:

1. Inspect the local computer and repository.
2. Detect the actual NVIDIA GPU and CUDA environment.
3. Produce a clear hardware and software diagnostic report.
4. obtain a small public panel dataset usable in both Python and Stata;
5. create deterministic synthetic panel datasets;
6. create a small exact CPU BMA reference implementation;
7. prepare Stata validation scripts;
8. establish support and validation plans for fixed effects;
9. create honest benchmark infrastructure;
10. document the results and unresolved statistical questions.

Do not optimize before the CPU reference is correct.

## 1. Repository inspection

Inspect the existing repository before creating or changing files.

Preserve unrelated existing work.

Determine:

* operating system;
* Python version;
* Git status;
* available package managers;
* C++ compiler availability;
* CUDA compiler availability;
* NVIDIA driver;
* installed Python scientific packages;
* whether Stata is installed and callable;
* whether a GPU-enabled Python stack is already installed.

Do not install large dependencies unless they are needed for this phase.

Record all findings in:

```text
reports/environment_report.md
reports/environment_report.json
```

## 2. GPU detection

Never assume the computer has an RTX 3060, RTX 3060 Ti, or any other specific card.

Detect the actual device.

Use available sources such as:

```bash
nvidia-smi
```

and a Python CUDA library when available.

Record:

* exact GPU model;
* GPU UUID when available;
* total VRAM;
* currently free VRAM;
* driver version;
* reported CUDA version;
* compute capability;
* multiprocessor count;
* maximum threads per block;
* shared memory per block;
* theoretical or documented FP32/FP64 limitations when reliably available;
* whether float64 kernels can execute;
* whether CUDA is actually accessible from Python;
* whether `nvcc` is installed;
* operating system and execution environment.

Create a user-facing command:

```bash
python -m gpubma.doctor
```

or an installed CLI equivalent:

```bash
gpubma doctor
```

The command must print a concise report and optionally save JSON.

Do not silently report CUDA availability merely because an NVIDIA driver exists. Test an actual small GPU calculation.

## 3. Public panel benchmark dataset

Use the classic Grunfeld panel dataset as the initial public benchmark when it can be obtained from a reliable source.

The same observations must be available to Python and Stata.

Prefer a reproducible download or a source bundled with a reputable statistical package.

Do not manually recreate the observations.

Save a local, versioned snapshot with provenance:

```text
data/public/grunfeld.csv
data/public/grunfeld.parquet
data/public/grunfeld.dta
data/public/grunfeld_provenance.json
```

The provenance record must include:

* source;
* retrieval date;
* source checksum when possible;
* local file checksums;
* row count;
* column names;
* panel identifier;
* time identifier;
* duplicate-key checks;
* missing-value checks.

Create a Python script that can reproduce the download and validation.

If direct download is unavailable, document the failure and use a reputable packaged copy rather than inventing the observations.

## 4. Deterministic synthetic panel generator

Create one parameterized generator rather than separate unrelated scripts.

The data-generating process must be explicit and documented.

Use a fixed default seed:

```text
20260715
```

Generate a balanced panel following a structure such as:

[
y_{it}
======

\alpha_i
+
\lambda_t
+
X_{it}\beta
+
\varepsilon_{it}.
]

The generator must preserve the same first predictors across dataset sizes.

For example, the first eight predictors in the 30-predictor dataset must be identical to those in the eight-predictor dataset when the panel dimensions and seed are the same.

Support:

```python
generate_panel(
    n_individuals=...,
    n_periods=...,
    n_predictors=...,
    seed=20260715,
)
```

Create at least these frozen datasets:

### Small validation dataset

```text
100 individuals
10 periods
1,000 observations
8 optional predictors
2^8 = 256 candidate models
```

### Medium validation dataset

```text
Same panel structure
12 optional predictors
2^12 = 4,096 candidate models
```

### Production-schema dataset

```text
Same deterministic design
30 optional predictors
2^30 = 1,073,741,824 candidate models
```

The production-schema dataset is only a data artifact in this phase. Do not attempt exhaustive evaluation of its model space yet.

Create:

```text
data/synthetic/panel_8.csv
data/synthetic/panel_8.parquet
data/synthetic/panel_8.dta

data/synthetic/panel_12.csv
data/synthetic/panel_12.parquet
data/synthetic/panel_12.dta

data/synthetic/panel_30.csv
data/synthetic/panel_30.parquet
data/synthetic/panel_30.dta
```

Create corresponding metadata files containing:

* seed;
* data-generating equation;
* true coefficients;
* individual effects;
* time effects;
* covariance structure;
* error variance;
* dimensions;
* checksums;
* expected number of models;
* generation timestamp;
* package versions.

Ensure that CSV, Parquet, and Stata files represent the same numerical dataset within documented serialization tolerances.

## 5. Candidate predictors and controls

Clearly separate:

```text
Outcome
Optional BMA predictors
Always-included continuous controls
Individual fixed effect
Time fixed effect
Identifiers
```

Optional predictors determine the number of candidate models.

Always-included controls and fixed effects must not increase the model count.

For (p) optional predictors:

[
N_{\text{models}}=2^p.
]

Validate this explicitly.

## 6. Fixed-effects design

The package must eventually support:

```python
fixed_effects=["individual"]
fixed_effects=["time"]
fixed_effects=["individual", "time"]
fixed_effects=["region", "industry", "year"]
```

For this phase, implement and compare two small-sample approaches.

### Reference approach: explicit dummy variables

Create a deterministic design matrix with:

* an explicit and documented base category;
* a consistent intercept convention;
* no dummy-variable trap;
* fixed effects included in every candidate model.

### Candidate production approach: absorption or residualization

Implement a transparent fixed-effects residualization method for:

* one-way individual fixed effects;
* one-way time fixed effects;
* two-way individual and time fixed effects.

Validate the residualized data against explicit dummies for ordinary least squares quantities.

Do not assume that equality of OLS slopes automatically establishes equality of Bayesian model scores.

Document the statistical issue:

> Under a g-prior, absorbing fixed effects may affect prior parameterization, effective sample size, degrees of freedom, marginal likelihood constants, and model comparison unless the always-included block is treated consistently.

Create tests and documentation before claiming BMA equivalence.

## 7. Stata validation files

Create Stata `.do` files for the small datasets.

The Stata scripts must load the exact `.dta` files generated by Python.

Create at least:

```text
validation/stata/small_no_fixed_effects.do
validation/stata/small_individual_fixed_effects.do
validation/stata/small_time_fixed_effects.do
validation/stata/small_two_way_fixed_effects.do
validation/stata/grunfeld_validation.do
```

Each script should:

1. load the frozen data;
2. validate row count and variable names;
3. run the corresponding exhaustive BMA command when supported;
4. explicitly request enumeration rather than sampling;
5. use documented fixed priors;
6. record the exact Stata version;
7. export available result matrices, scalars, coefficients, posterior inclusion probabilities, and model summaries;
8. save a plain-text log;
9. avoid manual copying of numerical output.

When Stata cannot export an item directly, document the limitation rather than fabricating it.

Do not require Stata for normal package installation or use.

## 8. CPU reference implementation

Implement a transparent exact CPU reference for Gaussian linear regression BMA.

This implementation is for correctness, not speed.

It must initially support:

* intercept;
* optional continuous predictors;
* always-included controls;
* fixed benchmark g-prior or another explicitly documented provisional fixed-g formulation;
* documented beta-binomial model prior;
* complete enumeration;
* float64 calculations;
* direct reconstruction of each model;
* stable normalization;
* top-model reporting.

The first API may be:

```python
from gpubma import bma_regress

result = bma_regress(
    data=df,
    outcome="y",
    predictors=[f"x{j}" for j in range(1, 9)],
    fixed_effects=None,
    backend="cpu",
    method="enumeration",
    precision="float64",
    deterministic=True,
)
```

Return at least:

* number of expected models;
* number of evaluated models;
* optional model masks for small runs;
* model log scores;
* normalized posterior model probabilities;
* posterior inclusion probabilities;
* posterior mean model size;
* model-size distribution;
* BMA coefficient means;
* BMA coefficient standard deviations when the formula has been validated;
* top-(K) models;
* runtime report.

Do not silently guess the exact Stata formula.

When the Stata parameterization is unresolved:

* isolate the formula behind a dedicated interface;
* label it provisional;
* cite the source used;
* add an unresolved item to `STATUS.md`;
* avoid describing the output as Stata-compatible.

## 9. Deterministic comparisons

Create comparison utilities that can compare:

```text
Python CPU vs Stata
Explicit dummies vs absorbed fixed effects
Repeated Python runs
CSV vs Parquet vs DTA inputs
```

The comparison report must include:

* quantity;
* reference value;
* candidate value;
* absolute difference;
* relative difference;
* tolerance;
* pass or fail.

Never round values before comparison.

Use rounding only for display.

## 10. Initial GPU feasibility work

Do not build the final exhaustive CUDA engine yet.

Create only a small feasibility layer that:

1. detects the GPU;
2. transfers sufficient statistics to the GPU;
3. executes a genuine float64 operation;
4. synchronizes correctly;
5. measures cold and warm execution separately;
6. compares the result with CPU;
7. reports whether the current machine is suitable for later kernel development.

Optionally create a simple GPU baseline for scoring a modest batch of models if this can be done cleanly without compromising the CPU work.

Do not report projected billion-model performance from a trivial matrix operation.

## 11. Benchmark ladder and runtime projection

Prepare a benchmark command such as:

```bash
gpubma benchmark --max-predictors 20
```

or:

```bash
python -m gpubma.benchmark
```

The benchmark ladder should eventually cover:

```text
8 predictors      256 models
10 predictors     1,024 models
12 predictors     4,096 models
15 predictors     32,768 models
18 predictors     262,144 models
20 predictors     1,048,576 models
24 predictors     16,777,216 models
26 predictors     67,108,864 models
28 predictors     268,435,456 models
30 predictors     1,073,741,824 models
```

During this phase, run only sizes that finish safely and quickly.

The report must distinguish:

```text
Measured
Projected
Not evaluated
```

For every measured run, report:

* exact GPU;
* backend;
* precision;
* number of observations;
* optional predictors;
* model count;
* compilation or initialization time;
* sufficient-statistics time;
* enumeration time;
* posterior reduction time;
* total time;
* models per second;
* peak GPU memory when measurable;
* repeat count;
* median runtime;
* minimum and maximum runtime.

Only project larger runs after a real model-evaluation benchmark exists.

A projected runtime must state:

* source benchmark;
* scaling assumption;
* whether throughput remained stable;
* confidence or uncertainty range;
* reasons the projection may fail.

## 12. Package API direction

Use a clean Python API.

Provide both functional and estimator-style interfaces eventually:

```python
from gpubma import bma_regress
```

and:

```python
from gpubma import GPUBMARegressor
```

Do not make the main function itself named `gpubma`.

Suggested result methods:

```python
result.summary()
result.coefficients()
result.inclusion_probabilities()
result.top_models()
result.model_size_distribution()
result.fixed_effects_report()
result.hardware_report()
result.runtime_report()
```

Support pandas and NumPy first.

Parquet input may be supported through pandas or PyArrow.

## 13. Suggested repository structure

Create or adapt:

```text
gpubma/
├── CLAUDE.md
├── README.md
├── LICENSE
├── pyproject.toml
├── STATUS.md
├── src/
│   └── gpubma/
│       ├── __init__.py
│       ├── api.py
│       ├── estimator.py
│       ├── result.py
│       ├── diagnostics/
│       ├── datasets/
│       ├── fixed_effects/
│       ├── priors/
│       ├── cpu/
│       ├── gpu/
│       ├── benchmarks/
│       └── validation/
├── scripts/
│   ├── download_grunfeld.py
│   ├── generate_synthetic_panels.py
│   └── compare_stata_python.py
├── data/
│   ├── public/
│   └── synthetic/
├── validation/
│   └── stata/
├── tests/
│   ├── unit/
│   ├── datasets/
│   ├── enumeration/
│   ├── fixed_effects/
│   ├── stata_parity/
│   └── gpu/
├── reports/
├── notebooks/
└── docs/
```

Do not commit very large generated artifacts without first considering repository size.

Small frozen validation datasets may be committed.

Document how larger artifacts should be regenerated.

## 14. Required documentation

Create:

```text
docs/PROJECT_SCOPE.md
docs/DATA_GENERATING_PROCESS.md
docs/FIXED_EFFECTS_DESIGN.md
docs/STATA_VALIDATION_PLAN.md
docs/GPU_DIAGNOSTICS.md
docs/BENCHMARK_PROTOCOL.md
docs/STATISTICAL_SPECIFICATION.md
```

Create a `CLAUDE.md` containing non-negotiable rules:

1. `gpubma` is a Python open-source package.
2. Stata is validation-only.
3. Exhaustive enumeration is the long-term target.
4. Never replace enumeration with MC3 without explicit authorization.
5. Never assume the GPU model.
6. Never silently use float32.
7. Never report projections as measurements.
8. Never optimize an unvalidated formula.
9. Never loosen tolerances merely to pass tests.
10. Keep generated data reproducible from scripts and fixed seeds.
11. Preserve checksums and provenance.
12. Record unresolved statistical questions honestly.

## 15. Test requirements

Add tests for:

* deterministic regeneration;
* identical row ordering;
* unique individual-time keys;
* balanced-panel structure;
* expected number of predictors;
* expected model count;
* equality of the first eight predictors across the 8-, 12-, and 30-predictor datasets;
* serialization round trips;
* explicit fixed-effect dummy rank;
* one-way residualization;
* two-way residualization;
* CPU model enumeration;
* posterior probabilities summing to one;
* PIP bounds;
* model-size probabilities summing to one;
* repeatable results;
* actual GPU float64 execution when CUDA is available;
* correct and explicit skipping when CUDA is unavailable.

## 16. Acceptance criteria for Phase 1

Phase 1 is complete only when:

1. The actual local GPU has been identified.
2. A reproducible hardware report exists.
3. The GPU has executed and validated a real float64 operation.
4. Grunfeld has been downloaded or obtained reproducibly with provenance.
5. The 8-, 12-, and 30-predictor synthetic datasets have been generated.
6. All formats have checksums and validation reports.
7. The small CPU exhaustive BMA runs successfully.
8. Exactly 256 models are evaluated for the 8-predictor dataset.
9. Posterior probabilities and PIPs pass internal consistency checks.
10. Fixed-effect explicit-dummy and absorption comparisons exist.
11. Stata validation `.do` files exist.
12. Actual Stata outputs are clearly distinguished from scripts merely prepared for later execution.
13. A benchmark report contains only measured values or clearly labeled projections.
14. All executed tests and their exact results are recorded.
15. `STATUS.md` describes the next phase without claiming the CUDA enumerator is already implemented.

## 17. Work execution

Begin working now rather than returning only a plan.

Use Fable 5’s agentic capabilities to inspect, implement, test, and document the first phase.

Keep the scope controlled.

Do not start the (2^{30}) production run.

Do not begin a complex custom CUDA extension until Phase 1 acceptance criteria pass.

At the end, report in Spanish:

1. repository state found;
2. exact GPU detected;
3. files created and modified;
4. datasets created;
5. checksums and dimensions;
6. CPU BMA functionality implemented;
7. fixed-effects results;
8. Stata scripts prepared or executed;
9. tests run and exact outcomes;
10. GPU calculations actually executed;
11. measured benchmark results;
12. unresolved statistical issues;
13. next concrete milestone.
