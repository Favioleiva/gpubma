# Deterministic comparison report

# Input format equivalence (panel_8)

- Reference: parquet
- Candidate: csv & dta
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| log scores parquet vs csv [max@i=0] | -2.19722457734 | -2.19722457734 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| PIP parquet vs csv [max@i=0] | 1 | 1 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| coef means parquet vs csv [max@i=0] | 1.56535421223 | 1.56535421223 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| log scores parquet vs dta [max@i=0] | -2.19722457734 | -2.19722457734 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| PIP parquet vs dta [max@i=0] | 1 | 1 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| coef means parquet vs dta [max@i=0] | 1.56535421223 | 1.56535421223 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |

# Repeated Python runs (determinism)

- Reference: run 1
- Candidate: run 2
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| log scores [max@i=0] | -2.19722457734 | -2.19722457734 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| PIP [max@i=0] | 1 | 1 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| coef means [max@i=0] | 1.56535421223 | 1.56535421223 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
