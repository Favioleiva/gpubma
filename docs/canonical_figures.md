# Canonical discovered-set figures

The plotting port follows the researcher's historical **BFG model-space cloud and exhaustive evidence-frontier figures** and **Reserve Four-FE Publication Update** figure generators. It does not infer visual rules from images alone. The private review records the exact source-to-function mapping and source hashes; public runtime code has no dependency on those directories or datasets.

The cloud/ridges preserve serif labels, blue/orange translucent shapes (alpha .14), 1.15-point contours, dashed blue minimum frontiers, orange upper frontiers and red champion markers. The structural panels preserve the publication proportional model-width map, 5.2:1.4 inclusion-sidebar geometry, blue/red coefficient sign colors and white excluded cells. The size distribution uses the publication 11.8×4.4-inch blue/gray marker-line design. Coefficient ridges preserve its 5.6:1.4 geometry, .78 amplitude, .32 fill and sign/low-inclusion color language.

Public adaptations are explicit:

- Every conditional weight uses **all evaluated models E** in its denominator. A hatched remainder is other **evaluated** models, never unknown posterior mass.
- k counts candidate variables only. Always-in controls are labeled separately; a dark control row is not an inclusion discovery.
- Ridge bounds are actual minima/maxima. Silverman KDE requires eight observations and three distinct scores. No Gaussian fallback or support expansion is allowed; sparse/constant shells use empirical spikes. KDE endpoints touch the support baseline. Heights are unit peak, not mass estimates. Unobserved shells break frontier lines.
- The prior size curve comes only from the recorded model prior, multiplied by the exact combinatorial number of candidate models of that size. No illustrative prior is invented.
- Coefficient displays are **weighted empirical distributions of OLS point estimates across discovered models**, conditional on inclusion and available full-rank fits. They are neither posterior draws nor credible/confidence densities. The CSV reports available-fit weight. Exclusion mass is shown separately; 0.50 is a visual grouping only, not a validity threshold. Blue/red means coefficient sign, not significance. Always-in estimates use dark neutral ink. Singular model estimates are unavailable.
- The top-five table reports descriptive OLS refits and conventional homoskedastic SEs, not posterior SDs. They assume iid errors and do not adjust for model selection or clustering. AIC/BIC/HQIC count regression design rank plus the variance parameter. Residual standard error divides SSE by residual df; RMSE divides by N. Perfect-fit or undefined diagnostics are NA. Rank-deficient coefficients are NA, not arbitrary pseudoinverse coefficients. BFG numerator/prior/relative marginal-likelihood entries are separate from Gaussian OLS likelihood.
- Optional `FE_GROUPS` maps FE labels to **already supplied** always-in 0/1 dummy columns. Supply reference coding yourself. These dummies are suppressed from plots/table coefficient rows and represented by FE summary rows. Names are never guessed to be fixed effects. No FE extraction, demeaning or scoring rule is changed.
- An early champion gets an improvement-detail panel plus a compact full-budget panel. The complete plateau remains visible. Genealogy uses retained predecessor edges, with a separate CSV rather than an in-figure table. No universal dynasty is assumed.

```python
from gpubma.bfg.figures import canonical_bfg_figures, save_canonical_figures
from gpubma.bfg.model_report import refit_discovered_models, write_top5

report = refit_discovered_models(result, y, X, always_in=controls)
figures, tables = canonical_bfg_figures(result, report=report)
save_canonical_figures(figures, tables, 'new_run')  # exclusive new directory
write_top5(report, 'new_run', outcome='y')
```

No coefficient panel is fabricated when a result is supplied without original regression data. The notebook supplies the exact same in-memory rows used for discovery. Reporting refits do not change the search result, its scores, prior or ranking. Report runtime is distinct from search runtime and grows with the number of evaluated models.

Only 300-dpi PNGs are exported. The eleven-figure suite includes a separate observed-frontier plot. The LaTeX table requires `booktabs` and, for long tables, `longtable`; compile twice to align repeated headers. CSV/HTML previews need no LaTeX.

Three standalone notebooks share these public helpers. The canonical source is the existing panel_30_center15.parquet; the applied source is the existing public Grunfeld .dta. Official data bytes are hash-checked and fetched from the public repository if absent. The generic notebook requires the user's configured dataset. No embedded toy CSV or private path is used. The checked bundled wheel supports validation before public incorporation; the alternate installation branch uses the public GitHub pip command after publication. Missing normal dependencies can be installed with pip.

Synthetic reference panels use only the transcribed archived exact evidence. They report true/global MAP identities and exact ranks/PMPs separately from discovered weights. Main columns remain BFG ranks1–5, with T/M only on actual matching columns. The known true model is never inserted into the top-five comparison.
