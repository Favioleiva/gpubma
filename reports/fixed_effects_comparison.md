# Fixed-effects comparison: explicit dummies vs absorption

Same fixed g and effective degrees of freedom were used on both sides; under that alignment the Bayesian model scores coincide by construction (Frisch-Waugh-Lovell). See docs/FIXED_EFFECTS_DESIGN.md for why this alignment is a statistical CHOICE that Stata may not share; BMA equivalence with Stata remains unverified.

# Fixed effects: explicit dummies vs within residualization (individual)

- Reference: explicit dummies (reference approach)
- Candidate: within transform (candidate production approach)
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| effective always-block rank | 102 | 102 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| OLS slopes (8 predictors) [max@i=4] | 0.268501134679 | 0.268501134679 | 9.992e-16 | 3.721e-15 | 1.0e-09 (abs) | PASS |
| BMA effective df | 898 | 898 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| BMA log scores (256 models) [max@i=207] | 627.473477808 | 627.473477808 | 1.364e-12 | 2.174e-15 | 1.0e-08 (abs) | PASS |
| BMA posterior model probs [max@i=95] | 0.091009824382 | 0.091009824382 | 4.138e-14 | 4.547e-13 | 1.0e-12 (abs) | PASS |
| BMA PIPs [max@i=0] | 1 | 1 | 8.571e-14 | 8.571e-14 | 1.0e-12 (abs) | PASS |
| BMA coefficient means [max@i=0] | 1.51332565554 | 1.51332565554 | 1.295e-13 | 8.554e-14 | 1.0e-10 (abs) | PASS |
| BMA coefficient sds [max@i=0] | 0.0358261014616 | 0.0358261014589 | 2.752e-12 | 7.681e-11 | 1.0e-10 (abs) | PASS |
| BMA mean model size | 5.41524976881 | 5.41524976881 | 5.258e-13 | 9.710e-14 | 1.0e-12 (abs) | PASS |

# Fixed effects: explicit dummies vs within residualization (time)

- Reference: explicit dummies (reference approach)
- Candidate: within transform (candidate production approach)
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| effective always-block rank | 12 | 12 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| OLS slopes (8 predictors) [max@i=0] | 1.57806137047 | 1.57806137047 | 1.998e-15 | 1.266e-15 | 1.0e-09 (abs) | PASS |
| BMA effective df | 988 | 988 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| BMA log scores (256 models) [max@i=7] | 499.688890809 | 499.688890809 | 4.547e-13 | 9.101e-16 | 1.0e-08 (abs) | PASS |
| BMA posterior model probs [max@i=31] | 0.785716714627 | 0.785716714628 | 8.937e-14 | 1.137e-13 | 1.0e-12 (abs) | PASS |
| BMA PIPs [max@i=0] | 1 | 1 | 6.661e-14 | 6.661e-14 | 1.0e-12 (abs) | PASS |
| BMA coefficient means [max@i=0] | 1.57832390552 | 1.57832390552 | 1.048e-13 | 6.640e-14 | 1.0e-10 (abs) | PASS |
| BMA coefficient sds [max@i=0] | 0.04347916261 | 0.0434791626081 | 1.890e-12 | 4.346e-11 | 1.0e-10 (abs) | PASS |
| BMA mean model size | 5.24494027313 | 5.24494027313 | 3.020e-13 | 5.758e-14 | 1.0e-12 (abs) | PASS |

# Fixed effects: explicit dummies vs within residualization (individual+time)

- Reference: explicit dummies (reference approach)
- Candidate: within transform (candidate production approach)
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| effective always-block rank | 111 | 111 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| OLS slopes (8 predictors) [max@i=6] | 0.0277253045209 | 0.0277253045209 | 1.131e-15 | 4.079e-14 | 1.0e-09 (abs) | PASS |
| BMA effective df | 889 | 889 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| BMA log scores (256 models) [max@i=31] | 709.867977348 | 709.867977348 | 1.364e-12 | 1.922e-15 | 1.0e-08 (abs) | PASS |
| BMA posterior model probs [max@i=31] | 0.742717183472 | 0.742717183472 | 8.449e-14 | 1.138e-13 | 1.0e-12 (abs) | PASS |
| BMA PIPs [max@i=5] | 0.111679148386 | 0.111679148386 | 4.580e-14 | 4.101e-13 | 1.0e-12 (abs) | PASS |
| BMA coefficient means [max@i=0] | 1.52765295772 | 1.52765295772 | 1.621e-14 | 1.061e-14 | 1.0e-10 (abs) | PASS |
| BMA coefficient sds [max@i=0] | 0.0330848304182 | 0.0330848304186 | 3.758e-13 | 1.136e-11 | 1.0e-10 (abs) | PASS |
| BMA mean model size | 5.30495205138 | 5.30495205138 | 1.776e-13 | 3.348e-14 | 1.0e-12 (abs) | PASS |
