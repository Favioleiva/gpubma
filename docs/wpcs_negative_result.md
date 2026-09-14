# W-PCS: practical posterior-normalizer reconstruction rejected

W-PCS explored whether observed BFG model scores could reconstruct enough unobserved mass to estimate global Z more cheaply than exhaustive enumeration. The corrected observed-only predictive-validity gate did not pass in the executed frontier:

* p12/r01: N=500,1000,2000.
* p18/r01: N=500,1000,2000,5000.

All listed checkpoints returned `PREDICTIVE_VALIDITY_NOT_ESTABLISHED`. The researcher reported approximately 20.6 minutes for p18 N=5000 alone. The p30 ladder was not completed within the dissertation budget. N_valid(p12)>2000 and N_valid(p18)>5000 are right-censored on the executed checkpoint grids; no outcomes are assigned to unexecuted checkpoints. N_valid means predictive-gate pass; support and estimate availability are separate.

These last live outcomes/timing are explicitly human-reported closure evidence; this release does not claim possession of canonical point files that were not located. Earlier blind-workstation N500 results remain historical evidence, not recreated artifacts.

Predictive validity was not established within the evaluated computational frontier. As cost increased substantially with N, W-PCS was rejected as a practically useful global-normalizer reconstruction mechanism for this dissertation. This is not proof that W-PCS can never estimate Z. It does not invalidate independently demonstrated BFG model-discovery results.

Experimental posterior-mass reconstruction work is retained for research reproducibility but is not part of the validated BFG discovery API. Historical implementation stays in the original research archive; the public install contains no W-PCS/ACESM estimator.
