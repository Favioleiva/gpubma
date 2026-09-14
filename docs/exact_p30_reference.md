# Canonical exact p30 reference

The default notebook loads compact artifacts committed with this repository. All 2^30 models contribute to the 31 exact shell histograms (2,000 common score bins). Exact supports and shell quantiles come from the complete float64 shell scores, not histogram estimates. Cloud rendering uses deterministic exact bin counts; there is no random thinning. Ridge smoothing is display-only and stops at exact support.

The reference preserves n=2,000; x1–x30; always-in w1/w2; g=2,000; beta-binomial(1,1); float64. The score uses the frozen FWL convention, df=n−1 and centered-y TSS; a common model-independent log-marginal-likelihood constant is omitted. Metadata retains the precise scoring convention and timing definitions.

The RTX 3060 reference core runtime is 379.811713000061 s, rounded to 379.812 s / 2,827,037 models/s. It excludes imports, data loading, warm-up, initial allocation and final file flush. Historical A100 timings are retained as separate provenance, not substituted for this runtime. The coefficient curves reuse the verified full-universe A100 mixture grids (257 points per variable), with their original numerical values and SHA-256; they are not mixtures over only the leading models.

`benchmark/exact_p30/reference/REFERENCE_MANIFEST.json` records current SHA-256 and original frozen hashes. Only workstation file pointers were removed from the public metadata. The rendering helper has Python 3.10 quoting and NumPy 1.26-compatible trapezoid syntax; numerical rendering is unchanged. The table exporter adds a textual x15 → x30 substitution note without changing coefficients, diagnostics, probabilities or ranks. No frozen source file is rewritten.

The notebook checks hashes before rendering. Exports include nine title/caption-free PNGs, exact reticular and posterior summaries, and `exact_top5_regression_comparison.tex`. The LaTeX table uses booktabs/longtable and reports conventional model-specific OLS coefficients/SEs, sample size, sizes/rank, R², adjusted R², RMSE, residual standard error, log likelihood, AIC/BIC/HQIC, log marginal likelihood, log model prior, log posterior numerator, exact PMP and global rank. The dedicated synthetic reference panel retains M* at rank 8 and both true/MAP probabilities at full stored precision. It is not forced into the top five.

## Raw arrays and regeneration

The multi-gigabyte complete score and cumulative arrays are not required to reproduce these figures and are not included. Their original filenames and SHA-256 are recorded in the reference manifest. **No public full-array download is currently promised or configured.** Publishing an optional large release asset would require a separate human-approved upload and stable URL. The default workflow is complete with committed compact artifacts (option A), independent of that optional asset.

`REGENERATE_FULL_ENUMERATION = False` is the default and only executable path in this figure notebook. Setting it to True fails closed with the explicit warning: “This evaluates all 1,073,741,824 models.” Advanced full regeneration requires a separately provisioned canonical CUDA enumerator, matching configuration and fresh output location; this package does not claim a one-click reproduction of every raw score. No large enumeration runs during notebook or package tests.

## Publication and Colab

The public repository is `https://github.com/Favioleiva/gpubma`. After human publication of this tree, upload/open the exact notebook in Colab and Run All. The standalone bootstrap uses that repository's raw-content endpoint and verifies embedded hashes for helpers, manifest, and every reference file. It fails on missing or changed bytes; it cannot silently load an older scientific reference. A local checkout works immediately without public artifact hosting. The current candidate does not publish or update remote content.
