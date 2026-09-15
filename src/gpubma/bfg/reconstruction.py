"""Identity-aware summaries calibrated to an ACESM lattice denominator.

Known models retain z_M/Z. Any fitted unobserved mass is represented by a
score-weighted fresh probability sample for inclusion/moment integration.
Those integration weights are NOT individual sampled-model probabilities.
This is an approximate plug-in conditional-distribution reconstruction, not an
unbiased estimator or an exactness guarantee for a partially observed lattice.
"""
import math
import numpy as np
from scipy.special import logsumexp


def calibrated_lattice(log_Z_hat, known_scores, sample_ids, population_size):
    """Return a feasible denominator, integration weights, and missing mass.

    ``known_scores`` contains each evaluated identity exactly once, including
    fresh sampled identities. The caller must freeze discovery before sampling.
    """
    if not known_scores:
        raise ValueError('No evaluated models: lattice posterior is unidentified')
    ids = list(known_scores)
    scores = np.array([known_scores[m] for m in ids], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('Reconstruction requires finite evaluated model scores')
    log_known = float(logsumexp(scores))
    exact = len(ids) == population_size
    if len(ids) > population_size:
        raise ValueError('Evaluated identities exceed lattice population')
    if not exact and (not sample_ids or not set(sample_ids).issubset(known_scores)):
        raise ValueError('Unobserved lattice requires fresh probability-sample identities')
    if not exact and not math.isfinite(float(log_Z_hat)):
        raise ValueError('Unobserved lattice requires a finite ACESM log total')
    # A valid total cannot be smaller than the actually observed positive mass.
    logz = log_known if exact else max(float(log_Z_hat), log_known)
    weights = dict(zip(ids, np.exp(scores-logz)))
    missing = max(0., -math.expm1(log_known-logz))
    if missing:
        sample_scores = np.array([known_scores[m] for m in sample_ids])
        conditional = np.exp(sample_scores-logsumexp(sample_scores))
        for m, w in zip(sample_ids, conditional):
            weights[m] += missing * float(w)
    return logz, weights, missing
