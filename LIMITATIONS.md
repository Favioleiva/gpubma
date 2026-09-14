# Limitations

| Limitation | Disposition |
|---|---|
| Best found model may not be global MAP | Benchmark matches do not certify new datasets |
| True model not invariably discovered | 29/30 scaling; p21/r01 rank-2 miss after 240,831 evaluations |
| Incomplete top-K/mass coverage | Canonical p30 top10=.9, top100=.89; harder scaling p30 terminal mass .349221–.631416 |
| No general superiority to every greedy method | Comparators/configurations differ; p12 budget200 greedy mass slightly exceeded BFG |
| Search samples are intentionally nonrepresentative | High-evidence targeting is not posterior sampling |
| Global Z and PMP remain uncalibrated | Positive missing mass means Y/Z_seen > Y/Z |
| Global PIP additionally needs missing composition | Denominator calibration alone is insufficient; bias can have either sign |
| W-PCS practical reconstruction rejected | Predictive validity not established over executed frontier; p18 N5000 ≈20.6min; p30 incomplete |
| No p≈90 empirical search claim | Integer-mask software regression is not scaling validation |
| No MCMC calibration/mixing result | Future research only |
| Genealogy incomplete | First registered edges only; components are descriptive, not posterior modes |
| Fixed wall time not enforced | Only unique-evaluation budget is a hard stopping resource |
| Reproducibility scoped | Same input/config/software/backend tested; cross-platform bitwise identity not promised |
| Historical benchmark acceptance uneven | Milestone tables retained with closure qualifications; scaling discovery audit corroborates stated counts |

For observed models, let q=Z_seen/Z. Then PMP(M)=q·weight_seen(M). For predictors, PIP_j=q·PIP_j_seen+(1-q)·PIP_j_missing. Neither q nor the missing inclusion composition is supplied by the discovery API. Even a perfect estimate of Z cannot supply an unknown PIP numerator. Search success does not prove causal truth, complete posterior integration or calibrated global moments.
