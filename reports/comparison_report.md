# Deterministic comparison report

Stata designs compared against REAL executed bmaregress output: 6/6.
Values are never rounded before comparison; rounding is display-only.

# Input format equivalence (panel_8)

- Reference: parquet
- Candidate: csv & dta
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| log scores parquet vs csv [max@i=0] | 73.2236395886 | 73.2236395886 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| PIP parquet vs csv [max@i=0] | 1 | 1 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| coef means parquet vs csv [max@i=0] | 1.56535409216 | 1.56535409216 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| log scores parquet vs dta [max@i=0] | 73.2236395886 | 73.2236395886 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| PIP parquet vs dta [max@i=0] | 1 | 1 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| coef means parquet vs dta [max@i=0] | 1.56535409216 | 1.56535409216 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |

# Repeated Python runs (determinism)

- Reference: run 1
- Candidate: run 2
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| log scores [max@i=0] | 73.2236395886 | 73.2236395886 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| PIP [max@i=0] | 1 | 1 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| coef means [max@i=0] | 1.56535409216 | 1.56535409216 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |

# Python vs Stata bmaregress — small_no_fe

- Reference: Stata bmaregress (StataNow/SE 19.5, batch export)
- Candidate: gpubma CPU reference (float64 enumeration)
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| models evaluated | 256 | 256 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| mean model size (Stata minus p_always) | 5.27940901628 | 5.27940901628 | 4.441e-15 | 8.412e-16 | 1.0e-09 (abs) | PASS |
| PIP[x1] | 1 | 1 | 2.764e-14 | 2.764e-14 | 1.0e-09 (abs) | PASS |
| coef mean[x1] | 1.56535409216 | 1.56535409216 | 4.619e-14 | 2.950e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x1] | 0.0449300070762 | 0.044930007077 | 7.606e-13 | 1.693e-11 | 1.0e-09 (abs) | PASS |
| PIP[x2] | 1 | 1 | 2.764e-14 | 2.764e-14 | 1.0e-09 (abs) | PASS |
| coef mean[x2] | -1.08302423498 | -1.08302423498 | 3.064e-14 | 2.829e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x2] | 0.0488432637206 | 0.048843263721 | 3.203e-13 | 6.558e-12 | 1.0e-09 (abs) | PASS |
| PIP[x3] | 1 | 1 | 2.764e-14 | 2.764e-14 | 1.0e-09 (abs) | PASS |
| coef mean[x3] | 0.721228837928 | 0.721228837928 | 2.021e-14 | 2.802e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x3] | 0.0469792332021 | 0.0469792332023 | 1.569e-13 | 3.341e-12 | 1.0e-09 (abs) | PASS |
| PIP[x4] | 1 | 1 | 2.764e-14 | 2.764e-14 | 1.0e-09 (abs) | PASS |
| coef mean[x4] | 0.468951005932 | 0.468951005932 | 1.321e-14 | 2.817e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x4] | 0.0477396847114 | 0.0477396847115 | 6.387e-14 | 1.338e-12 | 1.0e-09 (abs) | PASS |
| PIP[x5] | 0.999999917958 | 0.999999917958 | 2.776e-14 | 2.776e-14 | 1.0e-09 (abs) | PASS |
| coef mean[x5] | 0.296587737513 | 0.296587737513 | 8.382e-15 | 2.826e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x5] | 0.0471630680143 | 0.0471630680143 | 2.646e-14 | 5.611e-13 | 1.0e-09 (abs) | PASS |
| PIP[x6] | 0.105319308694 | 0.105319308694 | 8.696e-14 | 8.257e-13 | 1.0e-09 (abs) | PASS |
| coef mean[x6] | -0.00489548040308 | -0.00489548040308 | 3.986e-15 | 8.143e-13 | 1.0e-09 (abs) | PASS |
| coef sd[x6] | 0.0208333538713 | 0.0208333538713 | 7.994e-15 | 3.837e-13 | 1.0e-09 (abs) | PASS |
| PIP[x7] | 0.0926998426908 | 0.0926998426909 | 4.413e-14 | 4.761e-13 | 1.0e-09 (abs) | PASS |
| coef mean[x7] | 0.00358071327089 | 0.00358071327089 | 1.720e-15 | 4.805e-13 | 1.0e-09 (abs) | PASS |
| coef sd[x7] | 0.01811705956 | 0.01811705956 | 3.057e-15 | 1.687e-13 | 1.0e-09 (abs) | PASS |
| PIP[x8] | 0.0813899469414 | 0.0813899469415 | 1.125e-14 | 1.383e-13 | 1.0e-09 (abs) | PASS |
| coef mean[x8] | 0.00237702344216 | 0.00237702344216 | 3.812e-16 | 1.604e-13 | 1.0e-09 (abs) | PASS |
| coef sd[x8] | 0.0157106029758 | 0.0157106029758 | 7.355e-16 | 4.682e-14 | 1.0e-09 (abs) | PASS |

# Python vs Stata bmaregress — small_individual_fe

- Reference: Stata bmaregress (StataNow/SE 19.5, batch export)
- Candidate: gpubma CPU reference (float64 enumeration)
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| models evaluated | 256 | 256 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| mean model size (Stata minus p_always) | 5.45193405021 | 5.45193405021 | 1.075e-13 | 1.971e-14 | 1.0e-09 (abs) | PASS |
| PIP[x1] | 1 | 1 | 0.000e+00 | 0.000e+00 | 1.0e-09 (abs) | PASS |
| coef mean[x1] | 1.51328117125 | 1.51328117125 | 8.060e-14 | 5.326e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x1] | 0.0340104136625 | 0.0340104136607 | 1.802e-12 | 5.298e-11 | 1.0e-09 (abs) | PASS |
| PIP[x2] | 1 | 1 | 0.000e+00 | 0.000e+00 | 1.0e-09 (abs) | PASS |
| coef mean[x2] | -1.02404619167 | -1.02404619167 | 5.507e-14 | 5.377e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x2] | 0.0367867172349 | 0.0367867172342 | 7.578e-13 | 2.060e-11 | 1.0e-09 (abs) | PASS |
| PIP[x3] | 1 | 1 | 0.000e+00 | 0.000e+00 | 1.0e-09 (abs) | PASS |
| coef mean[x3] | 0.719843793996 | 0.719843793996 | 3.808e-14 | 5.290e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x3] | 0.0350899947978 | 0.0350899947974 | 3.999e-13 | 1.140e-11 | 1.0e-09 (abs) | PASS |
| PIP[x4] | 1 | 1 | 0.000e+00 | 0.000e+00 | 1.0e-09 (abs) | PASS |
| coef mean[x4] | 0.493933524377 | 0.493933524377 | 2.620e-14 | 5.305e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x4] | 0.0355307879103 | 0.0355307879101 | 1.808e-13 | 5.088e-12 | 1.0e-09 (abs) | PASS |
| PIP[x5] | 0.999999999982 | 0.999999999982 | 5.307e-14 | 5.307e-14 | 1.0e-09 (abs) | PASS |
| coef mean[x5] | 0.267101799036 | 0.267101799036 | 1.399e-14 | 5.237e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x5] | 0.0353653686544 | 0.0353653686543 | 5.327e-14 | 1.506e-12 | 1.0e-09 (abs) | PASS |
| PIP[x6] | 0.129880953284 | 0.129880953284 | 7.364e-14 | 5.669e-13 | 1.0e-09 (abs) | PASS |
| coef mean[x6] | -0.00515219640117 | -0.00515219640116 | 2.841e-15 | 5.513e-13 | 1.0e-09 (abs) | PASS |
| coef sd[x6] | 0.0185065941683 | 0.0185065941683 | 5.534e-15 | 2.990e-13 | 1.0e-09 (abs) | PASS |
| PIP[x7] | 0.156908285698 | 0.156908285698 | 5.246e-15 | 3.343e-14 | 1.0e-09 (abs) | PASS |
| coef mean[x7] | 0.00723308385114 | 0.00723308385114 | 3.227e-16 | 4.461e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x7] | 0.021760056136 | 0.021760056136 | 1.353e-15 | 6.218e-14 | 1.0e-09 (abs) | PASS |
| PIP[x8] | 0.165144811243 | 0.165144811243 | 6.092e-14 | 3.689e-13 | 1.0e-09 (abs) | PASS |
| coef mean[x8] | 0.00814891556722 | 0.00814891556721 | 3.027e-15 | 3.715e-13 | 1.0e-09 (abs) | PASS |
| coef sd[x8] | 0.0234411417338 | 0.0234411417338 | 3.060e-15 | 1.305e-13 | 1.0e-09 (abs) | PASS |

# Python vs Stata bmaregress — small_time_fe

- Reference: Stata bmaregress (StataNow/SE 19.5, batch export)
- Candidate: gpubma CPU reference (float64 enumeration)
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| models evaluated | 256 | 256 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| mean model size (Stata minus p_always) | 5.24555025341 | 5.24555025341 | 9.592e-14 | 1.829e-14 | 1.0e-09 (abs) | PASS |
| PIP[x1] | 1 | 1 | 0.000e+00 | 0.000e+00 | 1.0e-09 (abs) | PASS |
| coef mean[x1] | 1.57832337326 | 1.57832337326 | 1.887e-14 | 1.196e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x1] | 0.0432538826789 | 0.0432538826786 | 3.234e-13 | 7.477e-12 | 1.0e-09 (abs) | PASS |
| PIP[x2] | 1 | 1 | 0.000e+00 | 0.000e+00 | 1.0e-09 (abs) | PASS |
| coef mean[x2] | -1.06705304844 | -1.06705304844 | 1.088e-14 | 1.020e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x2] | 0.047128020031 | 0.0471280200309 | 1.366e-13 | 2.899e-12 | 1.0e-09 (abs) | PASS |
| PIP[x3] | 1 | 1 | 0.000e+00 | 0.000e+00 | 1.0e-09 (abs) | PASS |
| coef mean[x3] | 0.723980302648 | 0.723980302648 | 8.105e-15 | 1.119e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x3] | 0.0451384217541 | 0.045138421754 | 6.278e-14 | 1.391e-12 | 1.0e-09 (abs) | PASS |
| PIP[x4] | 1 | 1 | 0.000e+00 | 0.000e+00 | 1.0e-09 (abs) | PASS |
| coef mean[x4] | 0.481313713337 | 0.481313713337 | 5.607e-15 | 1.165e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x4] | 0.0459698588721 | 0.045969858872 | 2.707e-14 | 5.888e-13 | 1.0e-09 (abs) | PASS |
| PIP[x5] | 0.999999526922 | 0.999999526922 | 1.132e-14 | 1.132e-14 | 1.0e-09 (abs) | PASS |
| coef mean[x5] | 0.273984160069 | 0.273984160069 | 3.109e-15 | 1.135e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x5] | 0.045696458373 | 0.045696458373 | 8.549e-15 | 1.871e-13 | 1.0e-09 (abs) | PASS |
| PIP[x6] | 0.100770714481 | 0.100770714481 | 3.016e-14 | 2.993e-13 | 1.0e-09 (abs) | PASS |
| coef mean[x6] | -0.00436900698145 | -0.00436900698145 | 1.280e-15 | 2.930e-13 | 1.0e-09 (abs) | PASS |
| coef sd[x6] | 0.0193246711066 | 0.0193246711066 | 3.369e-15 | 1.743e-13 | 1.0e-09 (abs) | PASS |
| PIP[x7] | 0.0742448122197 | 0.0742448122198 | 1.764e-14 | 2.376e-13 | 1.0e-09 (abs) | PASS |
| coef mean[x7] | 0.00156367726952 | 0.00156367726952 | 4.181e-16 | 2.674e-13 | 1.0e-09 (abs) | PASS |
| coef sd[x7] | 0.013439746849 | 0.013439746849 | 8.153e-16 | 6.066e-14 | 1.0e-09 (abs) | PASS |
| PIP[x8] | 0.0705351997855 | 0.0705351997855 | 7.355e-15 | 1.043e-13 | 1.0e-09 (abs) | PASS |
| coef mean[x8] | 0.00103410777038 | 0.00103410777038 | 1.427e-16 | 1.380e-13 | 1.0e-09 (abs) | PASS |
| coef sd[x8] | 0.0127031631361 | 0.0127031631361 | 1.003e-15 | 7.893e-14 | 1.0e-09 (abs) | PASS |

# Python vs Stata bmaregress — small_two_way_fe

- Reference: Stata bmaregress (StataNow/SE 19.5, batch export)
- Candidate: gpubma CPU reference (float64 enumeration)
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| models evaluated | 256 | 256 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| mean model size (Stata minus p_always) | 5.32203923123 | 5.32203923123 | 8.882e-15 | 1.669e-15 | 1.0e-09 (abs) | PASS |
| PIP[x1] | 1 | 1 | 2.220e-16 | 2.220e-16 | 1.0e-09 (abs) | PASS |
| coef mean[x1] | 1.52763744308 | 1.52763744308 | 6.661e-16 | 4.361e-16 | 1.0e-09 (abs) | PASS |
| coef sd[x1] | 0.0312607439689 | 0.0312607439688 | 4.968e-14 | 1.589e-12 | 1.0e-09 (abs) | PASS |
| PIP[x2] | 1 | 1 | 2.220e-16 | 2.220e-16 | 1.0e-09 (abs) | PASS |
| coef mean[x2] | -1.00489603595 | -1.00489603595 | 2.887e-15 | 2.873e-15 | 1.0e-09 (abs) | PASS |
| coef sd[x2] | 0.033900228966 | 0.033900228966 | 3.005e-14 | 8.863e-13 | 1.0e-09 (abs) | PASS |
| PIP[x3] | 1 | 1 | 2.220e-16 | 2.220e-16 | 1.0e-09 (abs) | PASS |
| coef mean[x3] | 0.724282477737 | 0.724282477737 | 1.221e-15 | 1.686e-15 | 1.0e-09 (abs) | PASS |
| coef sd[x3] | 0.0321896728096 | 0.0321896728096 | 9.916e-15 | 3.080e-13 | 1.0e-09 (abs) | PASS |
| PIP[x4] | 1 | 1 | 2.220e-16 | 2.220e-16 | 1.0e-09 (abs) | PASS |
| coef mean[x4] | 0.508200566339 | 0.508200566339 | 7.772e-16 | 1.529e-15 | 1.0e-09 (abs) | PASS |
| coef sd[x4] | 0.0326655585413 | 0.0326655585413 | 4.378e-15 | 1.340e-13 | 1.0e-09 (abs) | PASS |
| PIP[x5] | 0.999999999935 | 0.999999999935 | 1.665e-15 | 1.665e-15 | 1.0e-09 (abs) | PASS |
| coef mean[x5] | 0.241360997903 | 0.241360997903 | 2.498e-16 | 1.035e-15 | 1.0e-09 (abs) | PASS |
| coef sd[x5] | 0.0327158573843 | 0.0327158573843 | 1.658e-15 | 5.069e-14 | 1.0e-09 (abs) | PASS |
| PIP[x6] | 0.119087916225 | 0.119087916225 | 1.935e-14 | 1.624e-13 | 1.0e-09 (abs) | PASS |
| coef mean[x6] | -0.00429312714738 | -0.00429312714738 | 6.184e-16 | 1.441e-13 | 1.0e-09 (abs) | PASS |
| coef sd[x6] | 0.0162126883765 | 0.0162126883765 | 2.671e-16 | 1.648e-14 | 1.0e-09 (abs) | PASS |
| PIP[x7] | 0.0963546652832 | 0.0963546652832 | 6.037e-15 | 6.265e-14 | 1.0e-09 (abs) | PASS |
| coef mean[x7] | 0.00260317134073 | 0.00260317134073 | 2.138e-16 | 8.213e-14 | 1.0e-09 (abs) | PASS |
| coef sd[x7] | 0.0127700445773 | 0.0127700445773 | 1.690e-15 | 1.323e-13 | 1.0e-09 (abs) | PASS |
| PIP[x8] | 0.106596649786 | 0.106596649786 | 2.046e-14 | 1.919e-13 | 1.0e-09 (abs) | PASS |
| coef mean[x8] | 0.00344858290246 | 0.00344858290247 | 6.444e-16 | 1.869e-13 | 1.0e-09 (abs) | PASS |
| coef sd[x8] | 0.0147143424501 | 0.0147143424501 | 2.692e-15 | 1.830e-13 | 1.0e-09 (abs) | PASS |

# Python vs Stata bmaregress — grunfeld_no_fe

- Reference: Stata bmaregress (StataNow/SE 19.5, batch export)
- Candidate: gpubma CPU reference (float64 enumeration)
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| models evaluated | 4 | 4 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| mean model size (Stata minus p_always) | 2 | 2 | 2.776e-14 | 1.388e-14 | 1.0e-09 (abs) | PASS |
| PIP[mvalue] | 1 | 1 | 0.000e+00 | 0.000e+00 | 1.0e-09 (abs) | PASS |
| coef mean[mvalue] | 0.114987219966 | 0.114987219966 | 1.457e-15 | 1.267e-14 | 1.0e-09 (abs) | PASS |
| coef sd[mvalue] | 0.00588355157198 | 0.00588355157196 | 1.366e-14 | 2.322e-12 | 1.0e-09 (abs) | PASS |
| PIP[kstock] | 1 | 1 | 1.399e-14 | 1.399e-14 | 1.0e-09 (abs) | PASS |
| coef mean[kstock] | 0.229530833383 | 0.229530833383 | 3.608e-15 | 1.572e-14 | 1.0e-09 (abs) | PASS |
| coef sd[kstock] | 0.0256846558165 | 0.0256846558165 | 1.375e-14 | 5.353e-13 | 1.0e-09 (abs) | PASS |

# Python vs Stata bmaregress — grunfeld_company_fe

- Reference: Stata bmaregress (StataNow/SE 19.5, batch export)
- Candidate: gpubma CPU reference (float64 enumeration)
- Overall: PASS

| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |
|---|---|---|---|---|---|---|
| models evaluated | 4 | 4 | 0.000e+00 | 0.000e+00 | 0.0e+00 (abs) | PASS |
| mean model size (Stata minus p_always) | 2 | 2 | 1.510e-14 | 7.550e-15 | 1.0e-09 (abs) | PASS |
| PIP[mvalue] | 1 | 1 | 4.663e-15 | 4.663e-15 | 1.0e-09 (abs) | PASS |
| coef mean[mvalue] | 0.109575917082 | 0.109575917082 | 1.554e-15 | 1.418e-14 | 1.0e-09 (abs) | PASS |
| coef sd[mvalue] | 0.0120292161627 | 0.0120292161627 | 3.242e-15 | 2.695e-13 | 1.0e-09 (abs) | PASS |
| PIP[kstock] | 1 | 1 | 1.110e-16 | 1.110e-16 | 1.0e-09 (abs) | PASS |
| coef mean[kstock] | 0.308522727417 | 0.308522727417 | 1.110e-15 | 3.599e-15 | 1.0e-09 (abs) | PASS |
| coef sd[kstock] | 0.0176070213119 | 0.0176070213118 | 1.125e-14 | 6.390e-13 | 1.0e-09 (abs) | PASS |
