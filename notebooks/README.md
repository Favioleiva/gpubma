# Notebooks

Two notebooks document the canonical `panel_30_center15` workflow:

- `GPUBMA_A100_p30.ipynb` is the clean, reproducible Colab workflow. Its
  expensive switch defaults to `False`, its outputs are cleared, and it is the
  preferred starting point for a fresh run.
- `GPUBMA_A100_p30_middle15_stata_figures.ipynb` is the executed scientific
  record. It retains the completed A100 outputs and the canonical BMA figures
  plus additional diagnostics. Do not use “Run all” casually: its recorded
  configuration intentionally has the full-run switch enabled.

The completed run enumerated exactly 1,073,741,824 models in float64. Its
compact export is versioned at
`reports/artifacts/panel_30_center15_exact_results.zip`; see
`reports/CANONICAL_P30_RESULTS.md` for provenance, inventory and SHA-256.

The notebooks are not required for Phase 1 reproduction; those results still
regenerate from `scripts/` and `python -m pytest`.
