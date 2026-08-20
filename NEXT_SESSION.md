# NEXT_SESSION — handoff after the canonical exact p = 30 milestone

The central `panel_30_center15` experiment is complete and versioned. Do not
repeat the 2^30 enumeration merely to reproduce documentation or figures: use
the retained notebook and compact results archive.

## Verified state (do not re-derive)

- Exact exhaustive float64 run: 1,073,741,824 models on an
  NVIDIA A100-SXM4-80GB.
- PASS1: 115.7 s at 9,277,815 models/s; exact PASS2: 896 s; peak GPU memory:
  1.93 GiB.
- Executed evidence notebook:
  `notebooks/GPUBMA_A100_p30_middle15_stata_figures.ipynb`.
- Final compact archive:
  `reports/artifacts/panel_30_center15_exact_results.zip`.
- Detailed archive inventory, SHA-256 values and scientific interpretation:
  `reports/CANONICAL_P30_RESULTS.md`.
- The runnable, output-cleared and safety-gated Colab workflow remains
  `notebooks/GPUBMA_A100_p30.ipynb`.
- Current tests: 154 passed, 1 intentionally skipped, 0 failed.

## Scientific conclusion

BMA recovers x1–x14 essentially, while the deliberately correlated x15/x30
family remains ambiguous. The MAP model substitutes x30 for x15; the true
model ranks 8th. This is evidence of honest posterior uncertainty between
substitutes, not failure to recover the data-generating signal.

## One next task

Integrate the validated standalone exact enumerator into the public
`bma_regress` GPU backend, preserving all current defaults and adding parity
tests against the existing standalone path. This is a new task and was
explicitly not started during the documentary closure.

## Hard limits

No MC3/MCMC substitution, silent float32, mixed precision, Tournament GPUBMA
or multi-GPU work without a separately authorized scope. Keep measured and
projected quantities explicitly distinct.
