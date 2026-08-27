import tempfile
from pathlib import Path
import numpy as np
import pytest

from gpubma.bfg.checkpoint import CheckpointManager
from gpubma.bfg.registry import EliteRegistry, ModelProvenance
from gpubma.bfg.scorer import BFGScorer
from gpubma.priors.model_priors import log_model_prior_function


def test_checkpoint_save_and_resume():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        p = 6
        log_prior_fn, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), n_predictors=p)
        X = np.random.randn(50, p)
        y = np.random.randn(50)

        scorer = BFGScorer(X_r=X, y_r=y, df_resid=49, g=50.0, log_model_prior=log_prior_fn, device="cpu")
        registry = EliteRegistry(p=p)

        # Score some models
        scorer.score_batch([1, 2, 3])
        registry.register(1, scorer.cache[1], ModelProvenance.FORWARD_GENEALOGY)
        registry.register(2, scorer.cache[2], ModelProvenance.BEAM)

        manager = CheckpointManager(checkpoint_dir=tmp_path)
        P_hat_k = np.full(p + 1, 1.0 / (p + 1))
        pips = np.full(p, 0.5)

        ckpt_path = manager.save_checkpoint(
            checkpoint_id=1,
            eval_count=3,
            elapsed_seconds=1.5,
            log_Z_hat=120.0,
            log_Z_by_k={k: 20.0 for k in range(p + 1)},
            P_hat_k=P_hat_k,
            pips=pips,
            map_model_id=2,
            map_log_score=scorer.cache[2],
            map_pmp=0.4,
            scorer=scorer,
            registry=registry,
        )
        assert ckpt_path.exists()
        assert (ckpt_path / "state.json").exists()
        assert (ckpt_path / "evaluation_cache.npz").exists()

        # Now simulate a fresh resume session
        new_scorer = BFGScorer(X_r=X, y_r=y, df_resid=49, g=50.0, log_model_prior=log_prior_fn, device="cpu")
        new_registry = EliteRegistry(p=p)
        assert new_scorer.n_unique_evaluated == 0

        state = manager.load_latest_checkpoint(scorer=new_scorer, registry=new_registry)
        assert state is not None
        assert state.checkpoint_id == 1
        assert state.log_Z_hat == 120.0
        assert new_scorer.n_unique_evaluated == 3
        assert new_registry.total_registered() == 2
