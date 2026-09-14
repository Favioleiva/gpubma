# Final candidate test report

Validation date: 2026-09-14. Python 3.12.6; PyTorch 2.5.1+cu121; NumPy 2.4.4; SciPy 1.16.3; NVIDIA GeForce RTX 3060. Tests are bounded synthetic/DEVELOPMENT checks. No HOLDOUT access or p30 full enumeration occurred.

| Check | Result |
|---|---|
| Existing package source vs canonical BFG notebook wheel | Every Python file identical (52 original sources byte-for-byte preserved) |
| Post-BFG stratified shell recovery module | PASS; `gpubma.bfg.shell_recovery` verified for arbitrary-precision unranking, sampling law, mixture weights |
| Build and isolated installed-wheel import | PASS; version 0.3.0rc1, imported from isolated site-packages |
| CPU and CUDA API / hard-budget smoke | PASS; nine-model cap on each backend |
| CLI help, hardware doctor, bounded CSV search | Four commands PASS |
| Public CPU/CUDA test suite against installed wheel | 91 tests PASS in 46.97 s, with source-tree pythonpath disabled (84 original + 7 shell recovery) |
| Exact artifact checksums / complete histogram coverage | PASS; counts sum to 1,073,741,824 across all 31 shells |
| Scientific figure artists / title-free PNG checks | All nine exact PASS; 5 shell recovery figures PASS; exact supports, probabilities, coefficient ordering preserved |
| Exact notebook, fresh kernel, unchanged defaults | PASS; 9.10 s total notebook execution, zero models evaluated |
| PNG comparison with frozen canonical suite | 9/9 byte-for-byte identical |
| BFG canonical notebook, fresh kernel | PASS; 26.84 s including figures and exports |
| BFG canonical shell recovery notebook, fresh kernel | PASS; 8.01 s including figures and exports |
| Grunfeld Stata notebook, fresh kernel | PASS; 11.49 s including figures and exports |
| User-data notebook, fresh kernel | PASS; canonical dataset explicitly configured, 27.32 s including figures and exports |
| Standalone exact-notebook bootstrap | PASS using a local fixture for unpublished HTTP content; nine identical PNGs, zero model evaluations |
| LaTeX compilation and visual check | PASS; two passes, two pages; full diagnostics and true/MAP reference panel |
| Python 3.10 syntax parse | PASS; runtime tests performed on Python 3.12 only |
| Relative paths / secrets / private artifacts | Final scan and ZIP inventory recorded in release manifest |

Notebook elapsed times include bootstrap, loading, rendering and output writing; they are not scorer benchmarks. The preserved discovery runtime and frozen enumeration runtime are separate measurements. The three BFG notebook sources and their embedded canonical wheel are unchanged. The figure notebook's optional regeneration switch fails closed; tests use False.

The wheel test environment shares installed scientific dependencies through a virtual environment but installs the actual candidate wheel with --no-deps. It is not a clean network dependency installation. Notebook kernels were fresh. Nonfatal IPython history-write warnings arose from the sandbox's read-only user profile; all cells completed successfully. MiKTeX required normal access to its installed user configuration; output remained in the review folder.

The initial new loader referred to `mask`; the archived schema uses `model_id`. This packaging-only bug was corrected and its affected tests/notebook rerun. No scientific input or package implementation changed. The standard build frontend produced a Windows temporary-directory permission issue on its output; the same setuptools wheel backend built the installed artifact directly.

Local reproduction is verified. Live Google Colab UI execution and fetching these new files from the public remote await human publication; this task does not update remote content. The optional multi-gigabyte raw arrays have no published download URL and are unnecessary for the default figure workflow.

Implementation and validation evidence are ready for human review. This report does not grant scientific acceptance, independent-audit status, or publication authorization.
