# Model-space search selection

The selectable dimension alone chooses the strategy. `n` still enters the
existing likelihood and benchmark g convention; it never selects the search.

| Request | p <= 32 | p > 32 |
|---|---|---|
| `search="auto"` | exhaustive GPU | existing BFG |
| `search="exact"` | exhaustive GPU | informative ValueError |
| `search="bfg"` | existing BFG | existing BFG |

`MAX_EXACT_P = 32` lives in `src/gpubma/search.py`. There is no project-specific
enumerator. All selectable columns are freely encoded by their original bit
positions. Controls, intercepts and FE dummies remain always-in and do not
consume selectable bits. BFG retains its existing dimension limit (128) and
its native wide-mask dispatch; auto routing is not a promise of unlimited p.

## Functional and estimator APIs

```python
from gpubma import bma_regress, GPUBMARegressor, resolve_search, MAX_EXACT_P

print(resolve_search(len(predictors)))  # safe inspection, no enumeration
result = bma_regress(data, "y", predictors, controls=controls, search="auto")

# Choose only the option dictionary appropriate to the resolved strategy.
exact = bma_regress(data, "y", predictors, search="exact",
                    exact_options={"checkpoint_path": "exact.npz",
                                   "resume": False, "max_chunk": 65536})
bfg = bma_regress(data, "y", predictors, search="bfg",
                  bfg_options={"budget_models": 100000, "seed": 20260715})
estimator = GPUBMARegressor(predictors, search="auto").fit(data, "y")
```

These calls execute their selected strategy: the exact example must only be
run when full enumeration is intended. For bounded tests, the existing
low-level `enumerate_models_gpu(..., stop_after_chunks=...)` remains available.

`method="auto"`, `method="exact"`, and `method="bfg"` are aliases on the
existing method parameter; conflicting method/search requests fail. Omitted
search with the historical `method="enumeration"` retains the CPU reference
default, p<=20 all-scores storage guard and existing explicit backend behavior.
The legacy GPU batch scorer's separate p<=16 storage guard also remains.
These resource guards are not automatic search thresholds.

Explicit search defaults to GPU (`backend=None` resolves to `"gpu"`). Exact
requires CUDA, including when selected by auto; it never falls back to CPU.
BFG accepts `backend="cpu"` as before and retains its own device behavior.

The exact route returns `BMAResult`. Streaming avoids arrays of all model masks
and probabilities (`masks` and `pmp` are None). Top-model probabilities use the
full normalizer. `keep_scores=True` is optional for small validation universes;
it is rejected above the reference storage cap. Exact options are
`max_chunk`, `vram_budget_bytes`, `checkpoint_path`, `checkpoint_every_s`,
`resume`, `progress_every_s`, `progress`, and `keep_scores`.

The BFG route returns the existing `fit_bfg` result unchanged. Its posterior
reconstruction is approximate unless its own diagnostics establish complete
enumeration; it is never relabeled an exact result. `bfg_options` forwards
existing search/runtime options. Specify g, model_prior and always_prior at
the top level; options cannot override those or replace the data/configuration.
`top_k` and `compute_coefficients` belong to the exact result; BFG retains its
own summaries. Direct `fit_bfg` remains available with its native API.

## Statistical and checkpoint compatibility

The exact score, g resolution, model prior, coefficient moments, shrink/flat
always-in conventions and float64 operations are unchanged. The high-level
exact route uses the same FWL preparation as the legacy reference. BFG receives
original y/X and explicit always-in controls/dummies, excluding the intercept
it already adds. Absorbed FE are rejected for the BFG adapter because the
current BFG interface cannot accept their rank; use explicit dummies.

Masks, binomial tables, ranks and serialized counters are int64; totals are
Python integers. Both 2**32 and 2**32-1 fit without overflow. Enumeration remains
colexicographic within each model size. Checkpoint version 1 and its statistical
digest are unchanged; additional cursor consistency checks reject corrupt
counts. A reduced-universe old/new compatibility audit covers checkpoints in
both directions. No full p=30 or p=32 benchmark is required for this upgrade.

## BFG compatibility audit

Base revision: `a31e63a34227135d8ac3db0289d78e7732c389e4`.
The discovery release at `86c4466` exported discovery-only results. Later
commits `ebd13a2` (native BFG128) and `1a7d10c` (package completeness) replaced
the exported config/engine/results with the reconstruction implementation.
The historical discovery documentation and tests remained in the tree.

This is not just a result-class alias: discovery tests require successful
termination at budgets as small as one model, a `final_reconnaissance` stage,
and absence of global posterior attributes. The current reconstruction engine
requires fresh samples in nonexhausted model-size shells and can raise
`Insufficient reconstruction budget for lattice ...`; it returns posterior
attributes and uses a `reconstruction` checkpoint stage. Relabeling those
objects or bypassing that stage would change behavior or statistical meaning.

The strategy upgrade does not alter BFG search, scoring, priors, posterior
calculations, checkpoint stages or result semantics. Historical discovery
regressions must be reported separately from current-engine/selector tests;
they must not be hidden by changing numerical expectations.

The minimal interface repair rejects `fit_bfg(config=..., budget_models=...)`
(or any other explicit search option) before engine execution. Previously the
config branch silently ignored the supplied option. A small `functools.wraps`
guard detects explicitly supplied options, including values equal to defaults,
without changing the inspectable function signature or default values. Data,
candidate names, always-in columns and outcome labels remain accepted with
config. Valid calls return the same objects from the same engine. No scorer,
prior, search stage or posterior function was modified.

The new regression test was run before the guard (FAIL: did not raise) and
after (PASS). Safety snapshots cover existing p=10 CPU/CUDA and p=67 native
wide-engine test recipes plus bounded p=33 BFG. `examples/audit_bfg_strategy_compatibility.py`
records deterministic numerical fields and SHA256 without timing fields.
The pre-change revision above is the immediate rollback point; existing tests
and frozen numerical reference files are retained unchanged.

## Executed validation and remaining failures

On an NVIDIA GeForce RTX 3060, PyTorch 2.5.1+cu121:

- Relevant suite: **116 passed**, exit status 0. This includes every previously
  passing BFG test, the repaired existing regression, unchanged exact reference
  tests, and 41 new parameterized selector/exact/interface checks.
- Entire repository suite before changes: 74 passed, 43 failed, 8 errors.
- Entire repository suite after changes: 116 passed, 42 failed, 8 errors.
- No previously passing test regressed. The remaining failures/errors already
  existed, predominantly in the historical discovery/reporting contract.
- Deterministic BFG output SHA256, identical before and after:
  `09d5d98fb20336c25fb6bfeb3a5216cbc2cd61a4b1b4d139c88e65e2a65e3975`.
- Original versus adapted exact engines: bitwise identical scores, PIPs,
  coefficient means/SDs, normalization and top-model IDs for p=1,5,8 under
  shrink and flat always-in priors. Old/new checkpoints resumed in both directions.
- p=32 CUDA execution was bounded to the null and 32 singleton models. All
  model sizes, extreme/middle combination ranks, high bits and wide counts were
  checked separately. No full p=30/p=32 run or project inference was executed.

The full repository suite is **not green**. This release does not claim that
the historical discovery API discrepancy has been repaired. Correcting that
broader discrepancy requires a separate decision about canonical search and
posterior behavior. Nothing is skipped, xfailed or rewritten to conceal it.
[Machine-readable validation](search_strategy_validation.json) lists every
existing BFG test's before/after status, source hashes and the exact relevant
test selection. This permits a reproducible scoped validation without claiming
full-suite success.

## Downstream project migration

The published package replaces the need for a project-specific p=32 exact
enumerator. Replacement is not a file-for-file copy: project callers must drop
their vendor-path injection, pin this package revision and record installed
source hashes. The old project-only `allow_full_exact` argument is absent;
execution authorization belongs in the notebook/runtime before calling GPUBMA.
Retain that authorization gate during migration. Project-specific small-chunk
expectations also need review because this upgrade preserves the original
engine's chunk policy.

A slim vendored `__init__.py` can be replaced by normal package imports; importing
BFG symbols does not execute BFG. Plot overrides are separate: the project copy
has `Positive mean`/`Negative mean` colors and a grid-based interval label absent
from upstream. Removing it without adapting those labels would lose semantics;
no plotting changes are included in this search upgrade.

Source reconstruction, sample/registry/hash checks, the project A100 guard,
PASS1/PASS2 orchestration, NTL/mining conditional-shape analysis, and notebook
execution permissions remain in the thesis repository. They are not GPUBMA's
model-space strategy policy. No downstream project files were changed here.
