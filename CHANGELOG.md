# Changelog

## Reusable exact/BFG strategy selection - 2026-10-04

- Added opt-in `search="auto" | "exact" | "bfg"` to the functional and
  estimator APIs; preserved legacy calls without search.
- Centralized the selectable-dimension policy: exhaustive GPU through p=32,
  existing BFG above p=32; observations and always-in columns do not route search.
- Generalized the original streaming exact engine and validated wide counters,
  model masks, every model size and old/new checkpoint compatibility.
- Rejected ambiguous `fit_bfg(config=..., <search options>)` calls instead of
  silently discarding their options. Public signature/defaults and all BFG
  algorithm/scoring/result code remain unchanged.
- Documented the pre-existing discovery-versus-reconstruction API discrepancy;
  no restoration of historical search/posterior behavior is included.

## Final public packaging — 2026-09-14

- Added compact exact p30 reference artifacts, canonical figure notebook, hash verification and presentation tests.
- Preserved BFG implementation, budget behavior and bundled notebook wheel.
- Added explicit example inventory, artifact availability and experimental shell-recovery documentation.
- No frozen result or production inference default changed.


## 0.3.0rc1 (unpublished)

* Separate BFG discovery from experimental mass reconstruction.
* Add explicit discovered-set weights, genealogy views and reproducibility metadata.
* Enforce hard budgets in the public configuration; reject ignored/unsupported options.
* Seal portable resume payloads and preserve arbitrary-width masks.
* Add bounded CPU/CUDA, resume, serialization, archive/install and fixture checks.
* Align current public documentation with the final scientific closure.

Older research changelogs remain unchanged in the source archive; this candidate does not silently rewrite them.
