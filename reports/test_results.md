# Phase 1 test results

- Date: 2026-07-15 (local), machine: Windows 11 Pro, RTX 3060, Python 3.12.6
- Command: `python -X utf8 -m pytest -v -rs`
- Raw verbose log: `reports/test_results_raw.txt`

## Outcome

**65 passed, 1 skipped, 0 failed** (3.08 s).

The single skip is explicit and intentional:
`tests/stata_parity/test_stata_artifacts.py::test_python_vs_stata_pip_parity`
— "Stata outputs not available: .do scripts are prepared but were never
executed (no callable Stata on this machine). Parity comparison pending."

GPU tests were NOT skipped: CUDA float64 execution ran on the detected
NVIDIA GeForce RTX 3060 and matched the CPU reference.

## Coverage against the task's §15 requirements

| requirement | test(s) | result |
|---|---|---|
| deterministic regeneration | test_deterministic_regeneration, test_frozen_files_match_generator | pass |
| identical row ordering | test_identical_row_ordering | pass |
| unique individual-time keys | test_unique_individual_time_keys | pass |
| balanced-panel structure | test_balanced_panel_structure | pass |
| expected number of predictors | test_expected_number_of_predictors[8/12/30] | pass |
| expected model count | test_expected_model_count[8/12/30], test_exactly_256_models_evaluated | pass |
| first 8 predictors equal across 8/12/30 | test_first_eight_predictors_nested_across_sizes | pass |
| serialization round trips | test_round_trips_exact_across_formats[8/12/30], test_float_columns_bit_identical | pass |
| explicit FE dummy rank | test_explicit_dummy_rank_no_trap | pass |
| one-way residualization | test_one_way_residualization_matches_dummies[individual/time] | pass |
| two-way residualization | test_two_way_residualization_matches_dummies, test_two_way_within_rejects_unbalanced | pass |
| CPU model enumeration | test_matches_independent_formula[1/2/5], test_single_predictor_posterior_by_hand | pass |
| posterior probs sum to 1 | test_posterior_probabilities_sum_to_one | pass |
| PIP bounds | test_pip_bounds | pass |
| model-size probs sum to 1 | test_model_size_distribution_sums_to_one | pass |
| repeatable results | test_repeatable_results | pass |
| actual GPU float64 when CUDA available | test_actual_float64_gpu_execution, test_gpu_scores_match_cpu_reference, test_gpu_scorer_uses_float64_not_float32 | pass (executed on RTX 3060) |
| explicit skip without CUDA | requires_cuda marker with reasoned message; test_skip_reason_is_explicit_when_cuda_unavailable | pass |

Additional guards: MC3 substitution rejected, silent float32 rejected,
model/beta-binomial priors sum to 1 over the model space, comparison utility
never rounds before comparing.
