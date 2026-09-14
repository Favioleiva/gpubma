"""Core pipeline engine and functional API for BFG model discovery."""

from __future__ import annotations

import math
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from gpubma.bfg.allocation import BudgetAllocator
from gpubma.bfg.checkpoint import CheckpointManager
from gpubma.bfg.config import BFGConfig
from gpubma.bfg.elite_search import GPUEliteSearch
from gpubma.bfg.genealogy import GenealogicalSearch
from gpubma.bfg.registry import EliteRegistry, ModelProvenance
from gpubma.bfg.results import BFGResult
from gpubma.bfg.sampling import ExactWingEnumerator, LatticeSampler
from gpubma.bfg.scorer import (
    BFGScorer,
    count_set_bits,
    detect_hardware,
    model_id_to_indices,
    model_id_to_vars,
)
from gpubma.priors.gpriors import resolve_g
from gpubma.priors.model_priors import log_model_prior_function


def _to_numpy_2d(data: Any, name: str = "X") -> np.ndarray:
    """Convert input matrix to contiguous float64 2D numpy array."""
    if hasattr(data, "to_numpy"):
        arr = data.to_numpy(dtype=np.float64)
    elif isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.float64)
    elif hasattr(data, "cpu") and hasattr(data, "numpy"):  # PyTorch tensor
        arr = np.ascontiguousarray(data.detach().cpu().numpy(), dtype=np.float64)
    else:
        arr = np.ascontiguousarray(np.array(data, dtype=np.float64))

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {arr.shape}.")
    return arr


def _to_numpy_1d(data: Any, name: str = "y") -> np.ndarray:
    """Convert input vector to contiguous float64 1D numpy array."""
    if hasattr(data, "to_numpy"):
        arr = data.to_numpy(dtype=np.float64).ravel()
    elif isinstance(data, np.ndarray):
        arr = np.ascontiguousarray(data, dtype=np.float64).ravel()
    elif hasattr(data, "cpu") and hasattr(data, "numpy"):
        arr = np.ascontiguousarray(data.detach().cpu().numpy(), dtype=np.float64).ravel()
    else:
        arr = np.ascontiguousarray(np.array(data, dtype=np.float64)).ravel()

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape {arr.shape}.")
    return arr


def validate_inputs(
    y: Any,
    X: Any,
    candidate_names: Optional[Sequence[str]] = None,
    always_in: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[np.ndarray]]:
    """Validate data integrity, ranks, names, and absence of NaNs/Infs."""
    y_arr = _to_numpy_1d(y, "y")
    X_arr = _to_numpy_2d(X, "X")

    n_obs, n_predictors = X_arr.shape
    if len(y_arr) != n_obs:
        raise ValueError(f"Dimension mismatch: y has {len(y_arr)} rows, X has {n_obs} rows.")

    if n_obs < 3:
        raise ValueError(f"Insufficient observations: N={n_obs}. Must have N >= 3.")
    if n_predictors < 1:
        raise ValueError("At least 1 candidate predictor is required.")

    # Check for NaN / Inf
    if np.isnan(y_arr).any() or np.isinf(y_arr).any():
        raise ValueError("Outcome y contains NaN or Infinite values.")
    if np.isnan(X_arr).any() or np.isinf(X_arr).any():
        raise ValueError("Design matrix X contains NaN or Infinite values.")

    # Candidate names
    if candidate_names is not None:
        names = list(candidate_names)
        if len(names) != n_predictors:
            raise ValueError(
                f"Length of candidate_names ({len(names)}) does not match columns in X ({n_predictors})."
            )
    elif hasattr(X, "columns"):
        names = [str(col) for col in X.columns]
    else:
        names = [f"x{j + 1}" for j in range(n_predictors)]

    # Check for duplicate column names
    if len(names) != len(set(names)):
        raise ValueError("Duplicate candidate predictor names found.")

    # Check for constant columns
    col_stds = np.std(X_arr, axis=0)
    if (col_stds < 1e-12).any():
        bad_idx = np.where(col_stds < 1e-12)[0]
        raise ValueError(f"Candidate predictors {np.array(names)[bad_idx].tolist()} have zero variance.")

    # Always-in controls validation
    A_arr = None
    if always_in is not None:
        A_arr = _to_numpy_2d(always_in, "always_in")
        if len(A_arr) != n_obs:
            raise ValueError(f"always_in has {len(A_arr)} rows, expected {n_obs}.")
        if np.isnan(A_arr).any() or np.isinf(A_arr).any():
            raise ValueError("always_in contains NaN or Infinite values.")

    return y_arr, X_arr, names, A_arr


class BFGEngine:
    """Master pipeline orchestrator for BFG budgeted model-space discovery."""

    def __init__(self, config: Optional[BFGConfig] = None):
        self.config = config if config is not None else BFGConfig()
        self.config.__post_init__()

    @staticmethod
    def _allocation_information(scorer, p):
        """Known-score mass priorities and actual unscored lattice capacities.

        Directed-search mass is a lower bound, NOT a certified posterior/grid
        estimate. It is used only for allocating exploration, never inference.
        """
        grouped = [[] for _ in range(p + 1)]
        for model_id, score in scorer.cache.items():
            grouped[model_id.bit_count()].append(score)
        log_mass = np.array([logsumexp(s) if s else -np.inf for s in grouped])
        if np.isfinite(log_mass).any():
            priorities = np.exp(log_mass - logsumexp(log_mass))
        else:
            priorities = np.full(p + 1, 1.0 / (p + 1))
        remaining = {k: math.comb(p, k) - len(grouped[k]) for k in range(p + 1)}
        return priorities, remaining

    def fit(
        self,
        y: Any,
        X: Any,
        candidate_names: Optional[Sequence[str]] = None,
        always_in: Optional[Any] = None,
        outcome_name: str = "y",
    ) -> BFGResult:
        """Execute model discovery without global posterior estimation."""
        t0_total = time.perf_counter()

        # 1. Validation & Data Preparation
        y_raw, X_raw, names, A_raw = validate_inputs(
            y=y, X=X, candidate_names=candidate_names, always_in=always_in
        )
        n, p = X_raw.shape
        rng = np.random.default_rng(self.config.seed)

        # 2. FWL Residualization
        if A_raw is not None:
            A = np.column_stack([np.ones(n, dtype=np.float64), A_raw])
            k_always = A_raw.shape[1]
        else:
            A = np.ones((n, 1), dtype=np.float64)
            k_always = 0

        Q, _ = np.linalg.qr(A)
        y_r = y_raw - Q @ (Q.T @ y_raw)
        X_r = X_raw - Q @ (Q.T @ X_raw)
        y_c = y_raw - y_raw.mean()

        if self.config.always_prior == "shrink":
            tss_norm = float(y_c @ y_c)
            df_resid = n - 1
        else:
            tss_norm = float(y_r @ y_r)
            df_resid = n - A.shape[1]

        # Prior setup
        g_spec = resolve_g(self.config.g, n_obs=n, n_predictors=p)
        log_prior_fn, prior_desc = log_model_prior_function(self.config.model_prior, n_predictors=p)

        # Scorer & Registries
        scorer = BFGScorer(
            X_r=X_r,
            y_r=y_r,
            df_resid=df_resid,
            g=g_spec.g,
            log_model_prior=log_prior_fn,
            prior_description=prior_desc,
            tss_norm=tss_norm,
            k_always=k_always,
            device=self.config.device,
            max_eval_budget=self.config.budget_models if self.config.budget_semantics == "hard" else None,
        )
        registry = EliteRegistry(p=p)
        sampler = LatticeSampler(p=p)
        wing_enum = ExactWingEnumerator(scorer=scorer, p=p)
        genealogy = GenealogicalSearch(scorer=scorer, registry=registry)
        elite_searcher = GPUEliteSearch(scorer=scorer, registry=registry)
        checkpoint_mgr = CheckpointManager(checkpoint_dir=self.config.checkpoint_dir)

        fingerprint_config = self.config.to_dict()
        for key in ('resume', 'checkpoint_dir', 'checkpoints', 'verbose', 'progress_interval'):
            fingerprint_config.pop(key, None)
        identity = hashlib.sha256(json.dumps(fingerprint_config, sort_keys=True).encode())
        identity.update(json.dumps(names).encode())
        identity.update(b'bfg-discovery-resume-schema-1')
        for module in sorted(Path(__file__).parent.glob('*.py')):
            identity.update(module.name.encode())
            identity.update(module.read_bytes())
        for array in (y_raw, X_raw, A_raw):
            if array is not None:
                identity.update(str(array.shape).encode())
                identity.update(np.ascontiguousarray(array).tobytes())
        fingerprint = identity.hexdigest()
        execution = dict(next_wing=0, genealogy_done=False, next_lattice=0,
            exact_wings=[], wing_log_Z={}, allocation=None,
            allocation_priorities=None, allocation_capacities=None, remaining_budget=None,
            completed_thresholds=[], final_reconnaissance=None, final_reconnaissance_reserve=None)
        elapsed_before_resume = 0.0

        # Resume state if requested
        ckpt_history = []
        if self.config.resume and self.config.checkpoint_dir is not None:
            prev_state = checkpoint_mgr.load_latest_checkpoint(scorer=scorer, registry=registry,
                                                               expected_fingerprint=fingerprint)
            if prev_state is None:
                raise ValueError('resume=True requires a complete committed checkpoint')
            if prev_state is not None:
                if 'execution' not in prev_state.metadata or 'rng_state' not in prev_state.metadata:
                    raise ValueError('Checkpoint lacks resumable execution/RNG state')
                execution = prev_state.metadata['execution']
                rng.bit_generator.state = prev_state.metadata['rng_state']
                elapsed_before_resume = prev_state.elapsed_seconds
                ckpt_history.append(prev_state.to_dict())
                if self.config.verbose:
                    print(
                        f"[BFG Engine] Resumed from checkpoint {prev_state.checkpoint_id} "
                        f"({prev_state.evaluated_count:,} cached models)."
                    )

        # 3. Identify Exact Wings (under hard budget ceiling)
        exact_wings = set(execution['exact_wings'])
        wing_log_Z = {int(k): v for k, v in execution['wing_log_Z'].items()}

        def save_progress(stage):
            """Commit an entire resumable search boundary; no partially updated phase."""
            if self.config.checkpoint_dir is None:
                return
            thresholds = self.config.checkpoints
            crossed = [v for v in (thresholds or []) if v <= scorer.n_unique_evaluated
                       and v not in execution['completed_thresholds']]
            if thresholds and not crossed and stage != 'complete':
                return
            if scorer.n_unique_evaluated == 0:
                return
            execution['completed_thresholds'].extend(crossed)
            execution['exact_wings'] = sorted(exact_wings)
            execution['wing_log_Z'] = wing_log_Z
            # Checkpoint summaries are explicitly observed-mass diagnostics, not
            # global posterior estimates. The cache is the discovery ledger.
            masses, _ = self._allocation_information(scorer, p)
            ids = list(scorer.cache)
            scores = np.array([scorer.cache[m] for m in ids])
            logz = float(logsumexp(scores))
            pip = np.zeros(p)
            for m, weight in zip(ids, np.exp(scores-logz)):
                pip[model_id_to_indices(m, p)] += weight
            best = ids[int(np.argmax(scores))]
            path = checkpoint_mgr.save_checkpoint(
                checkpoint_id=(ckpt_history[-1]['checkpoint_id']+1 if ckpt_history else 1),
                eval_count=scorer.n_unique_evaluated,
                elapsed_seconds=elapsed_before_resume+time.perf_counter()-t0_total,
                log_Z_seen=logz, exact_wing_log_mass=wing_log_Z, discovered_size_weights=masses, discovered_set_pips=pip,
                best_model_id=best, best_log_score=scorer.cache[best],
                best_discovered_weight=float(np.exp(scorer.cache[best]-logz)), scorer=scorer, registry=registry,
                metadata=dict(fingerprint=fingerprint, rng_state=rng.bit_generator.state,
                    execution=execution, stage=stage, summary_semantics='evaluated-model conditional diagnostics'))
            ckpt_history.append(json.loads((path/'state.json').read_text(encoding='utf-8')))
        t_wings_start = time.perf_counter()

        max_wing_budget = self.config.budget_models if self.config.budget_semantics != "hard" else int(self.config.budget_models * 0.35)

        for k in range(execution['next_wing'], p + 1):
            n_k_comb = math.comb(p, k)
            if wing_enum.is_exact_wing(k, max_wing_size=self.config.wing_max_size):
                if scorer.n_unique_evaluated + n_k_comb <= max_wing_budget or n_k_comb <= 64:
                    models, scores, log_Z_exact = wing_enum.enumerate_lattice(k)
                    registry.register_batch(models, scores, ModelProvenance.EXACT_WING, source_tag=f"exact_wing_k{k}")
                    if all(m in scorer.cache for m in models) and len(models) == n_k_comb:
                        exact_wings.add(k)
                        wing_log_Z[k] = log_Z_exact
            execution['next_wing'] = k + 1
            save_progress('wings')

        t_wings = time.perf_counter() - t_wings_start

        # 4. Genealogical Multi-Path Search (Greedy + Forward/Backward Beam Search)
        t_genealogy_start = time.perf_counter()
        if not execution['genealogy_done']:
            fwd_res = genealogy.forward_greedy(start_model_id=0)
            bwd_res = genealogy.backward_greedy(start_model_id=(1 << p) - 1)

        # Run beam search if budget allows
        if not execution['genealogy_done'] and (self.config.budget_semantics != "hard" or scorer.n_unique_evaluated < self.config.budget_models):
            beam_fwd = genealogy.forward_beam(
                start_seeds=[0], beam_width=self.config.beam_width
            )
            beam_bwd = genealogy.backward_beam(
                start_seeds=[(1 << p) - 1], beam_width=self.config.beam_width
            )
        t_genealogy = time.perf_counter() - t_genealogy_start
        execution['genealogy_done'] = True
        save_progress('genealogy')

        # 5. Adaptive Budget Allocation for Non-Wing Lattices
        t_sampling_start = time.perf_counter()
        if self.config.budget_semantics == "hard":
            remaining_budget = max(0, self.config.budget_models - scorer.n_unique_evaluated)
        else:
            remaining_budget = self.config.budget_models

        allocation_priorities, allocation_capacities = self._allocation_information(scorer, p)
        # Preserve the historical final-sample reserve and RNG trajectory.
        # These evaluations now serve discovery only; no mass estimator runs.
        nonempty = sum(v > 0 for v in allocation_capacities.values())
        reserve = min(remaining_budget, max(nonempty, remaining_budget // 3))
        if execution['final_reconnaissance_reserve'] is not None:
            reserve = execution['final_reconnaissance_reserve']
        else:
            execution['final_reconnaissance_reserve'] = reserve
        budget_allocated = BudgetAllocator.allocate(
            total_budget=max(0, remaining_budget-reserve),
            p=p,
            exact_wings=exact_wings,
            strategy=self.config.allocation_strategy,
            min_per_lattice=min(self.config.recon_sample_per_lattice, max(50, remaining_budget // max(1, p + 1 - len(exact_wings)))),
            P_k_hat=allocation_priorities,
            remaining_by_k=allocation_capacities,
        )
        if execution['allocation'] is not None:
            budget_allocated = {int(k): v for k, v in execution['allocation'].items()}
            allocation_priorities = np.array(execution['allocation_priorities'])
            allocation_capacities = {int(k): v for k, v in execution['allocation_capacities'].items()}
            remaining_budget = execution['remaining_budget']
        else:
            execution.update(allocation=budget_allocated, allocation_priorities=allocation_priorities.tolist(),
                allocation_capacities=allocation_capacities, remaining_budget=remaining_budget)

        original_scorer_limit = scorer.max_eval_budget
        if original_scorer_limit is not None:
            scorer.max_eval_budget = max(scorer.n_unique_evaluated, original_scorer_limit-reserve)

        # 6. Random Reconnaissance & GPU Elite Search per Non-Wing Lattice
        curves_by_k: Dict[int, Any] = {}
        sample_models_by_k: Dict[int, List[int]] = {}

        for k in range(execution['next_lattice'], p + 1):
            if k in exact_wings:
                continue

            n_k_budget = budget_allocated.get(k, 0)
            if self.config.budget_semantics == "hard":
                curr_rem = max(0, scorer.max_eval_budget - scorer.n_unique_evaluated)
                n_k_budget = min(n_k_budget, curr_rem)

            if n_k_budget > 0:
                r_k = min(n_k_budget, self.config.elite_calibration_size, max(2, n_k_budget // 2))
                m_k = max(0, n_k_budget - r_k)

                # GPU Elite Search (Calibration + Sequential Exploration)
                elite_res = elite_searcher.run_lattice_elite_search(
                    k=k,
                    r_k=r_k,
                    m_k=m_k,
                    target_q=self.config.elite_quantile,
                    rng=rng,
                    batch_size=self.config.batch_size,
                )

                # Seed additional beam search from elite hits if found and budget allows
                if elite_res.retained_model_ids and (
                    self.config.budget_semantics != "hard" or scorer.n_unique_evaluated < self.config.budget_models
                ):
                    genealogy.forward_beam(
                        start_seeds=elite_res.retained_model_ids[: self.config.beam_width],
                        beam_width=self.config.beam_width,
                    )
            execution['next_lattice'] = k + 1
            save_progress('sampling')

        t_sampling = time.perf_counter() - t_sampling_start
        scorer.max_eval_budget = original_scorer_limit

        # Freeze discovery before drawing fresh final_reconnaissance samples. Calibration
        # scores from elite search are no longer reused as a probability sample.
        recon = execution['final_reconnaissance']
        if recon is None:
            priorities, capacities = self._allocation_information(scorer, p)
            available = (max(0, self.config.budget_models-scorer.n_unique_evaluated)
                         if self.config.budget_semantics == 'hard' else reserve)
            recon = dict(next_k=0, samples={}, discovery={
                str(k): [m for m in scorer.cache if m.bit_count() == k] for k in range(p+1)},
                allocation=BudgetAllocator.allocate(total_budget=available, p=p,
                    exact_wings={k for k,v in capacities.items() if v == 0},
                    strategy=self.config.allocation_strategy, P_k_hat=priorities,
                    min_per_lattice=min(self.config.recon_sample_per_lattice,
                        available//max(sum(v > 0 for v in capacities.values()), 1)),
                    remaining_by_k=capacities))
            execution['final_reconnaissance'] = recon
        recon['allocation'] = {int(k): v for k,v in recon['allocation'].items()}
        for k in range(recon['next_k'], p+1):
            discovered = recon['discovery'][str(k)]
            available_k = math.comb(p,k)-len(discovered)
            sample_ids = sampler.sample_combinations(k, recon['allocation'].get(k, 0), rng,
                                                     exclude_set=set(discovered))
            scores = scorer.score_batch(sample_ids, chunk_size=self.config.batch_size)
            if any(m not in scorer.cache for m in sample_ids):
                raise ValueError('Final reconnaissance sample exceeded its reserved evaluation budget')
            registry.register_batch(sample_ids, [scores[m] for m in sample_ids],
                                    ModelProvenance.RANDOM_BULK, source_tag=f'final_reconnaissance_k{k}')
            recon['samples'][str(k)] = sample_ids
            recon['next_k'] = k+1
            save_progress('final_reconnaissance')
        save_progress('complete')
        return BFGResult.from_search(
            scorer=scorer, registry=registry, names=names, n_obs=n,
            outcome=outcome_name, config=self.config,
            elapsed_seconds=elapsed_before_resume + time.perf_counter() - t0_total,
            fingerprint=fingerprint, checkpoints=ckpt_history,
            exact_wings=sorted(exact_wings))


def fit_bfg(y: Any, X: Any, candidate_names=None, *, always_in=None,
            config: Optional[BFGConfig] = None, outcome_name='y', **search_options) -> BFGResult:
    """Find high-evidence models under a hard unique-evaluation budget.

    Supply a BFGConfig or keyword search options, never both. No global
    normalizer, model probabilities, inclusion probabilities or moments are
    estimated. Without a config, the historical functional beam width is 5;
    BFGConfig() retains its historical width 15. Set it explicitly to compare runs.
    """
    if config is not None and search_options:
        raise TypeError('Supply config OR search options; options cannot be silently ignored')
    if config is None:
        search_options.setdefault('beam_width', 5)
        config = BFGConfig(**search_options)
    return BFGEngine(config).fit(y, X, candidate_names, always_in, outcome_name)
