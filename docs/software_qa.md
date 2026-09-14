# Release-candidate software QA

The exported source archive passed **55 tests** on Python 3.12.6. CPU and NVIDIA RTX 3060 CUDA execution were available and exercised; CUDA cases were not reported as skips.

| Check | Result |
|---|---|
| Imports, configuration and API | Pass |
| Installed wheel API and CLI, including actual CSV-to-result execution | Pass |
| Legacy versus discovery search reference | Identical ordered IDs and float64 scores for 1,600 evaluations on CPU and CUDA; legacy mass fitting was never entered |
| Seed determinism, unique cache and hard budget | Pass, including budgets 1 and 2 |
| CPU scorer versus independent small exhaustive reference | Pass under inherited 1e-12 criterion |
| Interrupted resume at wings, genealogy, sampling, final reconnaissance | Pass on CPU and CUDA; scored records/weights exactly match uninterrupted runs |
| Missing/corrupt/changed-input checkpoints | Rejected |
| Result serialization and exclusive output creation | Pass; arbitrary-width decimal-string JSON masks |
| GPU masks at p64 and p90 | Small-mask software tests pass; not empirical discovery scaling evidence |
| Existing p12/p18/p30 public fixtures | Hash verification and 16-evaluation input smokes pass |
| Historical benchmark table integrity and read-only reporter | Pass |
| Wheel content | No benchmark data, raw oracles, tests or experimental mass estimators installed |

Validation used a clean source-archive extraction and a temporary wheel-install environment. Existing dependency site packages were shared; the `gpubma` import path was asserted inside the temporary environment. Dependencies were not downloaded. This is an isolated package/source check, not a from-scratch offline dependency-wheelhouse test.

Initial failures are retained in the private preparation evidence: an exhausted-beam empty history and conditional JSON tuple/list mismatch were corrected; seven assertions were corrected to the scorer's existing `gpu` backend label. An initial wheel harness incorrectly hid user-site dependencies with Python isolated mode; the corrected harness retained dependency access while verifying the package origin. Neither scientific evidence nor numerical tolerances were changed.

The QA scope is this candidate's complete bounded software suite. The historical research-wide suite, scientific scaling campaign, W-PCS ladder and expensive oracle scoring were not rerun. The original workspace and its prior evidence remain unchanged. No cross-platform scientific reproducibility guarantee follows from one host's tests.
