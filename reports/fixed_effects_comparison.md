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
| BMA log scores (256 models) [max@i=71] | 555.031311495 | 555.031311495 | 1.364e-12 | 2.458e-15 | 1.0e-08 (abs) | PASS |
| BMA posterior model probs [max@i=95] | 0.091009824382 | 0.091009824382 | 4.138e-14 | 4.547e-13 | 1.0e-12 (abs) | PASS |
| BMA PIPs [max@i=4] | 0.999999999721 | 0.999999999721 | 7.438e-14 | 7.438e-14 | 1.0e-12 (abs) | PASS |
| BMA coefficient means [max@i=0] | 1.51332565554 | 1.51332565554 | 1.124e-13 | 7.424e-14 | 1.0e-10 (abs) | PASS |
| BMA coefficient sds [max@i=0] | 0.0358261014616 | 0.0358261014592 | 2.386e-12 | 6.660e-11 | 1.0e-10 (abs) | PASS |
| BMA mean model size | 5.41524976881 | 5.41524976881 | 4.468e-13 | 8.250e-14 | 1.0e-12 (abs) | PASS |

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
| BMA PIPs [max@i=0] | 1 | 1 | 4.152e-14 | 4.152e-14 | 1.0e-12 (abs) | PASS |
| BMA coefficient means [max@i=0] | 1.57832390552 | 1.57832390552 | 6.573e-14 | 4.164e-14 | 1.0e-10 (abs) | PASS |
| BMA coefficient sds [max@i=0] | 0.04347916261 | 0.0434791626088 | 1.185e-12 | 2.725e-11 | 1.0e-10 (abs) | PASS |
| BMA mean model size | 5.24494027313 | 5.24494027313 | 1.501e-13 | 2.862e-14 | 1.0e-12 (abs) | PASS |

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
| BMA PIPs [max@i=7] | 0.100993697078 | 0.100993697078 | 6.134e-14 | 6.074e-13 | 1.0e-12 (abs) | PASS |
| BMA coefficient means [max@i=0] | 1.52765295772 | 1.52765295772 | 5.373e-14 | 3.517e-14 | 1.0e-10 (abs) | PASS |
| BMA coefficient sds [max@i=0] | 0.0330848304182 | 0.0330848304195 | 1.235e-12 | 3.733e-11 | 1.0e-10 (abs) | PASS |
| BMA mean model size | 5.30495205138 | 5.30495205138 | 3.153e-13 | 5.944e-14 | 1.0e-12 (abs) | PASS |
