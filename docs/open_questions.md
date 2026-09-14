# Open research questions

* How do N_MAP and N_TRUE scale beyond the observed dimensions? Can champion discovery remain useful near p≈90?
* Where does efficient mode discovery cease to approximate a useful fraction of global posterior mass?
* How do proxies, collinearity, prior choice and multimodality affect true-model rank, model families and predictor inclusion?
* Can BFG champions initialize distinct MC3/MCMC chains and improve mixing? At what dimension does ordinary MC3 mixing fail in these contaminated spaces?
* Can posterior sampling estimate q=Z_seen/Z from overlap with BFG, without explicit global enumeration? Is a per-overlap-model ratio of sampled PMP to discovered-set weight stable, accounting for autocorrelation, rare models and dependence?
* Could a defensible distribution over q induce Z=Z_seen/q? This is an unvalidated idea, not an implemented estimator.
* How can missing posterior inclusion composition be estimated even if q or Z is calibrated?
* Can GPU posterior sampling/calibration be practical near p≈90? How should independent error and mixing diagnostics be established?
* Which discovery summaries remain reliable without normalization, and how should conditional weights be reported to prevent confusion with global posterior quantities?

No sampler, hybrid calibration or new scaling campaign is part of this release.
