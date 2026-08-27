"""Unit tests for genealogical forward, backward, and beam search."""

import numpy as np
import pytest

from gpubma.bfg.genealogy import GenealogicalSearch
from gpubma.bfg.registry import EliteRegistry
from gpubma.bfg.scorer import BFGScorer, count_set_bits
from gpubma.priors.model_priors import log_model_prior_function


def test_genealogical_searches():
    rng = np.random.default_rng(20260715)
    n, p = 80, 8
    X = rng.normal(size=(n, p))
    # Signal in x1, x2, x3
    y = X[:, 0] * 2.0 + X[:, 1] * 1.5 + X[:, 2] * 1.0 + rng.normal(size=n)
    log_prior_fn, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), n_predictors=p)

    scorer = BFGScorer(
        X_r=X, y_r=y, df_resid=n-1, g=float(n), log_model_prior=log_prior_fn, device="cpu"
    )
    registry = EliteRegistry(p=p)
    genealogy = GenealogicalSearch(scorer=scorer, registry=registry)

    # 1. Forward Greedy
    fwd_res = genealogy.forward_greedy(start_model_id=0)
    assert len(fwd_res.path) == p + 1
    assert fwd_res.path[0][0] == 0
    assert fwd_res.path[-1][0] == (1 << p) - 1

    # 2. Backward Greedy
    bwd_res = genealogy.backward_greedy(start_model_id=(1 << p) - 1)
    assert len(bwd_res.path) == p + 1
    assert bwd_res.path[0][0] == (1 << p) - 1
    assert bwd_res.path[-1][0] == 0

    # 3. Forward Beam
    beam_res = genealogy.forward_beam(start_seeds=[0], beam_width=3)
    assert beam_res.best_score > float("-inf")
    assert registry.total_registered() > 0
