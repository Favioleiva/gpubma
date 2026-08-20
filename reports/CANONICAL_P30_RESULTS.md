# Canonical `panel_30_center15` exact results

This document closes the canonical p = 30 experiment as a reproducible
repository artifact. The scientific computation was not rerun during this
closure.

## Computational provenance

- Dataset: `data/synthetic/panel_30_center15.parquet`
- Dataset SHA-256:
  `7b468ca0c09249a83b05638b53c884bc444fdd535ba5f698b8fa92c24f7dd6e0`
- Code commit used by the A100 run:
  `ef9d68af462564b5cbdce268be1ff2b43317dd24`
- Hardware: NVIDIA A100-SXM4-80GB
- Environment: Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8,
  driver 580.82.07
- Method: exhaustive enumeration of exactly 1,073,741,824 models, float64,
  beta-binomial(1,1) model prior, `g = 2000`, no sampling, pruning,
  truncation or approximation
- PASS1: 115.7 s cumulative at 9,277,815 models/s
- Exact PASS2: 896 s
- Peak GPU memory: 1.93 GiB

The mandatory p = 15 smoke test passed before the full run. Its largest score
difference was 9.095e-13; the completed archive reports normalized posterior
mass, finite scores, zero rejected models and zero Cholesky failures.

## Scientific interpretation

The design intentionally contains strong imperfect substitutes. Recovery must
therefore be assessed through signal and family-level uncertainty, not by
requiring the single generating model to rank first.

- x1–x13 have PIP approximately 1 and x14 has PIP 0.9235.
- The material ambiguity is between x15 (PIP 0.0905) and its proxy x30
  (PIP 0.9638).
- Their family has probability 0.99998 of including at least one member.
- The MAP model substitutes x30 for x15 and has PMP 0.4715.
- The exact generating model ranks 8th and has PMP 0.0188.
- The top 10 and top 100 models contain 0.7267 and 0.9370 posterior mass.
- Posterior model size has mean 15.637, mode 15 and central 95% interval
  [15, 18].

The MAP substitution is scientifically reasonable BMA behavior under strong
collinearity: nearly all signal is recovered while posterior mass represents
uncertainty between observationally similar variables.

## Versioned artifacts and hashes

- Executed notebook:
  `notebooks/GPUBMA_A100_p30_middle15_stata_figures.ipynb`
  - size: 2,718,906 bytes
  - SHA-256:
    `970ffca537a31416dcd00c3e97b3d4d1d1a14e09d6538d4a97b044ec51685207`
- Final compact archive:
  `reports/artifacts/panel_30_center15_exact_results.zip`
  - size: 3,731,062 bytes
  - SHA-256:
    `bfabc5769a265e526e20b7be59a8a57547a26f237dccde73b3d30847aab8368f`

The previously cited archive hash
`b896824dd14a345c17ee370a6d8e33b78039e3c2e5156c4c0e70fc303ce4e46d`
was found and verified. It belonged to the 3,730,815-byte export whose figure
manifest listed 19 PNGs while the ZIP contained 23. The final archive differs
from it in exactly one member:
`results/panel_30_center15_figure_manifest.json`. Every scientific table,
array, report and figure is byte-for-byte unchanged.

## Final archive inventory

The ZIP contains 47 unique entries and no directories, path traversal,
checkpoints, caches, credentials, dataset copies or repository metadata:

| Type | Count |
|---|---:|
| JSON | 5 |
| Parquet | 17 |
| Markdown | 1 |
| NPZ | 1 |
| PNG | 23 |

The 23 manifest-matched figures are:

1. `coefdensity_slide_01_x1.png`
2. `coefdensity_slide_02_x2.png`
3. `coefdensity_slide_03_x3.png`
4. `coefdensity_slide_04_x4.png`
5. `coefdensity_slide_05_x5.png`
6. `coefdensity_slide_06_x8.png`
7. `coefdensity_slide_07_x9.png`
8. `coefdensity_slide_08_x6.png`
9. `coefdensity_slide_09_x7.png`
10. `coefdensity_slide_10_x11.png`
11. `coefdensity_top10.png`
12. `coefridgeline_ordered.png`
13. `coefsummary_ordered.png`
14. `coefsummary_ordered_exact_quantiles.png`
15. `corrmap.png`
16. `msize.png`
17. `pip.png`
18. `pmp_top100.png`
19. `pmp_top20.png`
20. `pmp_top200.png`
21. `pmp_top50.png`
22. `varmap_equal_width.png`
23. `varmap_proportional.png`

## Resolution of the 19-versus-23 discrepancy

The original figure-manifest cell was executed before four intentional final
figures were present: the two varmaps, the exact-quantile coefficient summary
and the coefficient ridgeline. `coefsummary_ordered.png` was also overwritten
after that manifest was produced, leaving its recorded size and hash stale.
The extra PNGs are not duplicates or auxiliary files; they are deliberate
outputs of the notebook.

The notebook now exposes `refresh_figure_manifest()` and invokes it again at
the start of export. The export then checks every PNG name, byte size and
SHA-256 against the refreshed manifest before offering the ZIP for download.
The final archive manifest contains all 23 figures and matches them exactly.
