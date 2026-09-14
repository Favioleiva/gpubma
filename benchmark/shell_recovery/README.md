# Compact post-BFG stratified shell recovery benchmark data

This directory contains compact summary artifacts and reconstructed reticular score distributions for the canonical p=30 synthetic benchmark (n=2,000, p=30, w1/w2 always in, g=2,000, beta-binomial(1,1)).

## Inventory

- `RANDOM_SHELL_CAP_SENSITIVITY.csv`: Summary metrics across precision caps (10k, 100k, 1M) including runtime, evaluations, KS distances, and Wasserstein-1 distances.
- `RANDOM_SHELL_CAP_SENSITIVITY_BY_SHELL.csv`: Shell-by-shell error metrics (KS, W1, quantiles) for each cap ladder step.
- `POST_BFG_RANDOM_SHELL_COUNTS.csv`: Discovered counts D_k, shell sizes N_k, sample sizes n_k, and sample fractions across all 31 shells.
- `BFG_RANDOM_RECOVERY_TIMING_COMPARISON.csv`: Comparative timing and evaluation accounting across full enumeration, BFG discovery, and shell recovery.
- `RECONSTRUCTED_HISTOGRAMS.npz`: Reconstructed reticular score distributions (2,000 bins) for each cap, weighting known discoveries by 1/N_k and random remainder draws by (N_k - D_k)/(N_k * n_k).

These artifacts enable reproducing publication-quality figures, error comparisons, and timing tables in sub-second time without requiring GPU execution or massive raw-sample storage.
