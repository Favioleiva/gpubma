# gpubma

GPU and CPU tools for Bayesian linear-model spaces. **BFG is a budgeted high-evidence model-space discovery algorithm.** The separate exact-reference notebook reproduces figures from the complete canonical p=30 universe.

## Install

This is a candidate prepared for human review. From the public repository checkout:

```sh
python -m pip install ".[notebooks]"
```

Python 3.10+ and PyTorch are required. Install the appropriate PyTorch CPU/CUDA build for your environment. Rendering the exact reference requires neither CUDA nor model enumeration.

## Examples

### Exact p=30 BMA reference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpubma/blob/main/examples/Exact_BMA_Canonical_p30_Figures.ipynb)

[Exact_BMA_Canonical_p30_Figures.ipynb](examples/Exact_BMA_Canonical_p30_Figures.ipynb) reproduces nine publication figures and the regression comparison table from the **complete 1,073,741,824-model universe**. Default `REGENERATE_FULL_ENUMERATION = False`. Compact hash-checked exact artifacts are committed under [benchmark/exact_p30/reference](benchmark/exact_p30/reference). No raw-score download or six-minute GPU run is needed. See [artifact provenance and workflow](docs/exact_p30_reference.md).

### BFG canonical p=30 example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpubma/blob/main/examples/BFG_Canonical_p30_Example.ipynb)

[BFG_Canonical_p30_Example.ipynb](examples/BFG_Canonical_p30_Example.ipynb) runs the canonical synthetic benchmark with a user-controlled evaluation budget. The preserved B=5,000 profile found MAP, truth, 10/10 exact top models, and 98/100 exact top models. These are benchmark observations, not guarantees on new data.

### BFG + shell recovery

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpubma/blob/main/examples/BFG_Canonical_p30_Shell_Recovery.ipynb)

[BFG_Canonical_p30_Shell_Recovery.ipynb](examples/BFG_Canonical_p30_Shell_Recovery.ipynb) pairs targeted BFG discovery (B=5,000) with representative uniform random shell recovery (recommended default cap 100,000 per shell) to reconstruct reticular score distributions across all 31 shells without equal-pooling bias. See [shell recovery documentation](docs/random_shell_recovery.md).

### Use BFG on your own data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpubma/blob/main/examples/BFG_User_Dataset_Example.ipynb)

[BFG_User_Dataset_Example.ipynb](examples/BFG_User_Dataset_Example.ipynb): change `DATA_PATH`, `TARGET`, `EXPLANATORY_VARIABLES`, and `BUDGET`. CSV, Parquet and Stata are supported; always-in controls are configured separately.

### Stata compatibility

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Favioleiva/gpubma/blob/main/examples/BFG_Public_Stata_Example.ipynb)

[BFG_Public_Stata_Example.ipynb](examples/BFG_Public_Stata_Example.ipynb) is a Grunfeld `.dta` compatibility / econometric parity example. Its two candidates give only four models; it is not a large model-space BFG demonstration.

## Hard global evaluation budget

`BUDGET` maps directly to **`BFGConfig.budget_models`**: the maximum number of **unique candidate models** BFG may score across all stages.

```python
from gpubma.bfg.example_workflow import ensure_public_data, load_example_data, fit_example

BUDGET = 5000
path = ensure_public_data("canonical_p30")
y, X, controls = load_example_data(path, "y", [f"x{i}" for i in range(1, 31)], ["w1", "w2"])
result, metrics = fit_example(y, X, controls, budget=BUDGET, seed=12345, device="auto")
assert result.n_models_evaluated <= BUDGET
print(result.best_model)
print(result.top_models(k=20))
```

The invariant is `N_unique_scored <= BUDGET`. A fully exhausted small universe can stop below the budget. See [public API](PUBLIC_API.md) and [budget semantics](docs/example_budget_semantics.md).

```sh
gpubma doctor
gpubma-bfg --data input.csv --outcome y --candidates x1,x2,x3 --budget-models 100 --device cpu --out discovery.json
```

## Scientific scope

BFG supports MAP/champion search, top-model recovery, genealogy/model-family discovery, controlled synthetic true-model recovery, deterministic reproducible search, and model-space compression. It provides the best models **found**, not a general exact-oracle certificate.

Discovered-set weights normalize over the evaluated set. Search alone does **not** establish exact global Z, global PMP, global PIP, or complete posterior integration. Exact global quantities in the canonical reference use full enumeration. Always-in controls are not candidate PIPs. W-PCS is not a validated production estimator and is not installed in this package.

The canonical exact benchmark uses n=2,000, candidates x1–x30, controls w1/w2, g=2,000, beta-binomial(1,1), and float64. Frozen RTX 3060 enumeration took 379.812 s (2,827,037 models/s). MAP is x1–x14+x30 (mask 536887295, PMP 0.4715187997893168). The generating model x1–x15 has rank 8 and PMP 0.01881348830750715. One variable substitution has binary inclusion Hamming distance 2.

[Experimental post-BFG shell recovery](docs/random_shell_recovery.md) remains benchmark functionality, separate from discovery and outside its budget; it is not the default inference API. [Small exact wings](docs/random_shell_recovery.md#future-small-shell-hybrid) are documented as future work only.

See [scientific scope](SCIENTIFIC_SCOPE.md), [limitations](LIMITATIONS.md), [benchmark inventory](benchmark/README.md), [test report](TEST_REPORT.md), [release notes](RELEASE_NOTES.md), and [public exclusions](PUBLIC_EXCLUSIONS.md). Cross-device bitwise equality is not promised. Benchmark results do not establish causal identification or performance on all datasets.

## Citation and license

[CITATION.cff](CITATION.cff) provides software citation metadata; cite the approved version/commit when published. No paper DOI is assigned. [BSD-3-Clause](LICENSE).
