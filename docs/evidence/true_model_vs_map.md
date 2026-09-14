# Known causal model versus posterior MAP

M* denotes the known data-generating causal model throughout this closure. M_MAP denotes the posterior champion. Some historical reports used M* for MAP; their notation is not carried forward. Differences between these models are not automatically a search failure or a proxy swap. [E03,E20,E23,E24]

The audited scaling experiment found M_MAP in 30/30 and explicitly evaluated M* in 29/30. All true models have recorded exact posterior ranks between 1 and 27; therefore the 29 discovered true models are among the reference high-evidence top27, but the rank-2 p21/r01 true model is absent. Posterior membership/rank is an exact-reference diagnostic; discovery means evaluated, not that BFG uniquely selected causal truth. Source values below are a keyed transcription from existing E21–E24, not new scoring.

| Dataset | N_MAP | N_TRUE | True rank | True PMP | MAP PMP | Hamming |
| --- | --- | --- | --- | --- | --- | --- |
| BFGSF1_p12_r01 | 191 | 191.0 | 1 | 0.5161068837375647 | 0.5161068837375647 | 0 |
| BFGSF1_p12_r02 | 192 | 192.0 | 1 | 0.5751192174858364 | 0.5751192174858364 | 0 |
| BFGSF1_p12_r03 | 187 | 187.0 | 1 | 0.38627229565061305 | 0.38627229565061305 | 0 |
| BFGSF1_p12_r04 | 196 | 188.0 | 2 | 0.10622814437191933 | 0.3395144171519738 | 1 |
| BFGSF1_p12_r05 | 192 | 192.0 | 1 | 0.33189174449888653 | 0.33189174449888653 | 0 |
| BFGSF1_p15_r01 | 294 | 294.0 | 1 | 0.43538209568788194 | 0.43538209568788194 | 0 |
| BFGSF1_p15_r02 | 289 | 289.0 | 1 | 0.4685504537908641 | 0.4685504537908641 | 0 |
| BFGSF1_p15_r03 | 293 | 293.0 | 1 | 0.49398974955979086 | 0.49398974955984704 | 0 |
| BFGSF1_p15_r04 | 296 | 296.0 | 1 | 0.2952047957900046 | 0.2952047957900046 | 0 |
| BFGSF1_p15_r05 | 296 | 296.0 | 1 | 0.5239893700268058 | 0.5239893700268058 | 0 |
| BFGSF1_p18_r01 | 431 | 431.0 | 1 | 0.203243976076535 | 0.203243976076535 | 0 |
| BFGSF1_p18_r02 | 432 | 432.0 | 1 | 0.2853984491732425 | 0.2853984491732425 | 0 |
| BFGSF1_p18_r03 | 428 | 428.0 | 1 | 0.27620558937411505 | 0.27620558937411505 | 0 |
| BFGSF1_p18_r04 | 428 | 428.0 | 1 | 0.3196154071212516 | 0.3196154071212516 | 0 |
| BFGSF1_p18_r05 | 434 | 434.0 | 1 | 0.2512037896643411 | 0.2512037896643411 | 0 |
| BFGSF1_p21_r01 | 577 | >240831 | 2 | 0.07049570081109238 | 0.2628718689724119 | 2 |
| BFGSF1_p21_r02 | 736 | 588.0 | 27 | 0.005864658170070019 | 0.09643492151957002 | 4 |
| BFGSF1_p21_r03 | 586 | 586.0 | 1 | 0.20708974223265844 | 0.20708974223265844 | 0 |
| BFGSF1_p21_r04 | 585 | 585.0 | 1 | 0.2631620073403644 | 0.2631620073403644 | 0 |
| BFGSF1_p21_r05 | 578 | 578.0 | 1 | 0.2791644804290306 | 0.2791644804290306 | 0 |
| BFGSF1_p24_r01 | 991 | 1011.0 | 17 | 0.006693483866446913 | 0.2388951300041118 | 1 |
| BFGSF1_p24_r02 | 769 | 769.0 | 1 | 0.19559764445474653 | 0.19559764445474653 | 0 |
| BFGSF1_p24_r03 | 995 | 995.0 | 1 | 0.06986855761150257 | 0.06986855761150257 | 0 |
| BFGSF1_p24_r04 | 1003 | 772.0 | 2 | 0.14154675696270289 | 0.14492799067711204 | 2 |
| BFGSF1_p24_r05 | 771 | 769.0 | 2 | 0.06834808521275289 | 0.2198279473656639 | 2 |
| BFGSF1_p30_r01 | 1216 | 1216.0 | 1 | 0.20151408001741222 | 0.20151408001741222 | 0 |
| BFGSF1_p30_r02 | 1607 | 1591.0 | 8 | 0.010402282087595372 | 0.02327648271111051 | 1 |
| BFGSF1_p30_r03 | 1583 | 1606.0 | 2 | 0.03533739004801557 | 0.07535215802667336 | 1 |
| BFGSF1_p30_r04 | 1207 | 1207.0 | 1 | 0.08691007659647047 | 0.08691007659647047 | 0 |
| BFGSF1_p30_r05 | 1207 | 1207.0 | 1 | 0.10515509250607812 | 0.10515509250607812 | 0 |

## Interpreting divergence

Eight datasets have distinct true and MAP models. Registered proxy substitutions occur in p21/r01, p21/r02, p24/r04 and p24/r05; only three are pure one-for-one swaps. The p21/r02 case additionally includes two noise additions. p12/r04, p24/r01 and p30/r03 add a variable without a registered causal-for-proxy replacement; p30/r02 omits one causal variable. Complete identities, removed/added predictors, substitution records, budgets and terminal discovered mass appear in [scaling_discovery.csv](../../benchmark/recorded/scaling_discovery.csv). [E20,E24]

For the separate canonical panel_30_center15 experiment, M* is 32767 (x1–x15), and M_MAP is 536887295 (x1–x14,x30), a Hamming-2 replacement. The retained branch table records true log score 1115.3003957189062 and MAP log score 1118.521780616737. The approved report identifies true posterior rank 8 and approximately 1.88% PMP; the canonical exact MAP PMP is 0.4715187997893168. The older Contract 1 narrative's reported true-score 1115.3040 should not override the retained Contract 2 table. Beam width >= 5 is reported to recover both branches. The per-run table does not supply a separately verified N_TRUE for that experiment, so none is invented. Its complete-search evaluated counts are not first-hit counts. [E02–E06,E09,E46]

The defensible statement is: “BFG did not merely locate posterior champions; in controlled contaminated-DGP experiments it also recovered the known causal generating model among evaluated candidates in 29 of 30 audited scaling datasets, allowing causal truth and posterior optimality to be studied separately.” Retain the explicit exception and the single-benchmark beam evidence; do not claim universal causal recovery or causal truth of the MAP.

Exact block inclusion is Pr(causal OR proxy), not the sum of marginal PIPs. The audited union support is high in the four actual substitution cases, but the p30/r02 block02 counterexample has causal PIP 0.524265, proxy PIP 0.250961 and union 0.572232. BMA does not guarantee causal-signal preservation merely by averaging models. [E20,E49]
