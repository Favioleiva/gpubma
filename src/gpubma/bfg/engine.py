"""Core pipeline engine and functional API for BFG Bayesian Model Averaging."""

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

from gpubma.bfg.acesm import ACESMReconstructor, CumulativeCurveBuilder
from gpubma.bfg.allocation import BudgetAllocator
from gpubma.bfg.checkpoint import CheckpointManager
from gpubma.bfg.config import BFGConfig
from gpubma.bfg.elite_search import GPUEliteSearch
from gpubma.bfg.genealogy import GenealogicalSearch
from gpubma.bfg.registry import EliteRegistry, ModelProvenance
from gpubma.bfg.results import BFGResult, LatticeResult
from gpubma.bfg.reconstruction import calibrated_lattice
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
    """Master pipeline orchestrator for BFG model-space search and evidence reconstruction."""

    def __init__(self, config: Optional[BFGConfig] = None):
        self.config = config if config is not None else BFGConfig()

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
        """Execute full BFG search and Bayesian model averaging."""
        t0_total = time.perf_counter()

        # 1. Validation & Data Preparation
        y_raw, X_raw, names, A_raw = validate_inputs(
            y=y, X=X, candidate_names=candidate_names, always_in=always_in
        )
        n, p = X_raw.shape
        if p > 128:
            raise ValueError(
                f"Unsupported dimension p={p}. "
                f"GPU BMA supports p <= 60 (legacy BFG engine) and 61 <= p <= 128 (BFG128 engine)."
            )
        if p > 60:
            from gpubma.bfg.bfg128.engine import BFG128Engine
            engine128 = BFG128Engine(config=self.config)
            return engine128.fit(
                y=y_raw,
                X=X_raw,
                candidate_names=names,
                always_in=A_raw,
                outcome_name=outcome_name,
            )
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
        identity.update(b'contract6-resume-schema-1')
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
            completed_thresholds=[], reconstruction=None, reconstruction_reserve=None)
        elapsed_before_resume = 0.0

        # Resume state if requested
        ckpt_history = []
        if self.config.resume and self.config.checkpoint_dir is not None:
            prev_state = checkpoint_mgr.load_latest_checkpoint(scorer=scorer, registry=registry,
                                                               expected_fingerprint=fingerprint)
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
            if thresholds and not crossed:
                return
            if scorer.n_unique_evaluated == 0:
                return
            execution['completed_thresholds'].extend(crossed)
            execution['exact_wings'] = sorted(exact_wings)
            execution['wing_log_Z'] = wing_log_Z
            # Checkpoint summaries are explicitly observed-mass diagnostics, not
            # prematurely certified ACESM posteriors. Cache is the sufficient ledger.
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
                log_Z_hat=logz, log_Z_by_k=wing_log_Z, P_hat_k=masses, pips=pip,
                map_model_id=best, map_log_score=scorer.cache[best],
                map_pmp=float(np.exp(scorer.cache[best]-logz)), scorer=scorer, registry=registry,
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
        # Hold out a third of the available budget for fresh probability samples
        # after ALL directed search. Later discoveries cannot contaminate them.
        nonempty = sum(v > 0 for v in allocation_capacities.values())
        reserve = min(remaining_budget, max(nonempty, remaining_budget // 3))
        if execution['reconstruction_reserve'] is not None:
            reserve = execution['reconstruction_reserve']
        else:
            execution['reconstruction_reserve'] = reserve
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

        # Freeze discovery before drawing fresh reconstruction samples. Calibration
        # scores from elite search are no longer reused as a probability sample.
        recon = execution['reconstruction']
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
            execution['reconstruction'] = recon
        recon['allocation'] = {int(k): v for k,v in recon['allocation'].items()}
        for k in range(recon['next_k'], p+1):
            discovered = recon['discovery'][str(k)]
            available_k = math.comb(p,k)-len(discovered)
            sample_ids = sampler.sample_combinations(k, recon['allocation'].get(k, 0), rng,
                                                     exclude_set=set(discovered))
            if available_k and not sample_ids:
                raise ValueError(f'Insufficient reconstruction budget for lattice {k}; increase budget_models')
            scores = scorer.score_batch(sample_ids, chunk_size=self.config.batch_size)
            if any(m not in scorer.cache for m in sample_ids):
                raise ValueError('Reconstruction sample exceeded its reserved evaluation budget')
            registry.register_batch(sample_ids, [scores[m] for m in sample_ids],
                                    ModelProvenance.RANDOM_BULK, source_tag=f'reconstruction_k{k}')
            recon['samples'][str(k)] = sample_ids
            recon['next_k'] = k+1
            save_progress('reconstruction')
        for k in range(p+1):
            sample_models_by_k[k] = recon['samples'][str(k)]
            if sample_models_by_k[k]:
                curves_by_k[k] = CumulativeCurveBuilder.build_empirical_curve(k,
                    [scorer.cache[m] for m in recon['discovery'][str(k)]],
                    [scorer.cache[m] for m in sample_models_by_k[k]], math.comb(p,k))

        # 7. ACESM Denominator Saturation Fitting
        # A lattice exhausted by later search/sampling is exact too. Never fit
        # an extrapolated plateau when its entire model population is known.
        for k in range(p + 1):
            cached_scores = [score for model, score in scorer.cache.items() if model.bit_count() == k]
            if len(cached_scores) == math.comb(p, k):
                exact_wings.add(k)
                wing_log_Z[k] = float(logsumexp(cached_scores))
        t_acesm_start = time.perf_counter()
        log_Z_hat_by_k: Dict[int, float] = {}
        lattice_results: Dict[int, LatticeResult] = {}
        integration_by_k = {}
        unseen_mass_by_k = {}

        for k in range(p + 1):
            N_k = math.comb(p, k)
            champ_rec = registry.get_champion(k)
            best_id = champ_rec.model_id if champ_rec else 0
            best_score = champ_rec.log_score if champ_rec else float("-inf")
            n_eval_k = len(registry.by_k[k])

            if k in exact_wings:
                log_Z_hat_by_k[k] = wing_log_Z[k]
                lattice_results[k] = LatticeResult(
                    k=k,
                    N_k=N_k,
                    log_Z_hat=wing_log_Z[k],
                    best_model_id=best_id,
                    best_score=best_score,
                    evaluated_count=N_k,
                    elite_count=N_k,
                    acesm_parameters={"is_exact_wing": 1.0},
                    is_wing=True,
                    is_boundary_collapsed=False,
                    budget_spent=N_k,
                )
            else:
                curve = curves_by_k[k]
                acesm_res = ACESMReconstructor.fit_lattice(
                    curve=curve,
                    beta=self.config.acesm_beta,
                    lambda_momentum=self.config.acesm_lambda_momentum,
                )
                actual_scores = {m:s for m,s in scorer.cache.items() if m.bit_count() == k}
                feasible_logz, integration, missing = calibrated_lattice(
                    acesm_res.log_Z_hat, actual_scores, sample_models_by_k[k], N_k)
                integration_by_k[k] = integration
                unseen_mass_by_k[k] = missing
                log_Z_hat_by_k[k] = feasible_logz
                lattice_results[k] = LatticeResult(
                    k=k,
                    N_k=N_k,
                    log_Z_hat=feasible_logz,
                    best_model_id=best_id,
                    best_score=best_score,
                    evaluated_count=n_eval_k,
                    elite_count=curve.n_elites,
                    acesm_parameters={
                        "delta_Z_hat": acesm_res.delta_Z_hat,
                        "alpha_hat": acesm_res.alpha_hat,
                        "beta_hat": acesm_res.beta_hat,
                        "final_loss": acesm_res.final_loss,
                        "unconstrained_log_Z_hat": acesm_res.log_Z_hat,
                        "unseen_probability": missing,
                    },
                    is_wing=False,
                    is_boundary_collapsed=acesm_res.is_boundary_collapsed,
                    budget_spent=budget_allocated.get(k, self.config.recon_sample_per_lattice),
                )

        t_acesm = time.perf_counter() - t_acesm_start

        # 8. Global Normalization & Evidence Aggregation
        all_log_Z_k = [log_Z_hat_by_k[k] for k in range(p + 1)]
        global_log_Z = float(logsumexp(all_log_Z_k))
        P_hat_k = np.exp(np.array(all_log_Z_k, dtype=np.float64) - global_log_Z)
        P_hat_k = P_hat_k / np.sum(P_hat_k)

        # 9. One calibrated integration measure for PIPs and coefficient moments.
        pips_array = np.zeros(p, dtype=np.float64)
        coef_mean_array = np.zeros(p, dtype=np.float64)
        coef_m2_array = np.zeros(p, dtype=np.float64)
        sign_pos_array = np.zeros(p, dtype=np.float64)

        global_integration = {}
        for k in range(p + 1):
            if k in exact_wings:
                integration_by_k[k] = {m: float(np.exp(s-wing_log_Z[k]))
                    for m,s in scorer.cache.items() if m.bit_count() == k}
                unseen_mass_by_k[k] = 0.0
            for model, conditional_weight in integration_by_k[k].items():
                weight = float(P_hat_k[k] * conditional_weight)
                global_integration[model] = weight
                if model == 0 or weight == 0:
                    continue
                moments = scorer.compute_model_coefficients(model)
                idx = moments['indices']
                mean, sd = moments['coef_mean'], moments['coef_sd']
                pips_array[idx] += weight
                coef_mean_array[idx] += weight * mean
                coef_m2_array[idx] += weight * (sd**2+mean**2)
                # Retained historical point-sign definition is explicit below;
                # it must not be confused with integrated Pr(beta > 0).
                sign_pos_array[np.array(idx)[mean > 0]] += weight

        coef_sd_array = np.sqrt(np.maximum(coef_m2_array - coef_mean_array ** 2, 0.0))
        sign_pos_array = np.clip(sign_pos_array, 0.0, 1.0)

        # 10. Global MAP Model
        global_champ = registry.get_champion()
        map_model_id = global_champ.model_id if global_champ else 0
        map_log_score = global_champ.log_score if global_champ else float("-inf")
        map_pmp = float(math.exp(min(map_log_score - global_log_Z, 0.0)))
        map_vars = model_id_to_vars(map_model_id, names)

        t_total = elapsed_before_resume + time.perf_counter() - t0_total

        # 11. Checkpoint Final State
        if self.config.checkpoint_dir is not None:
            execution['exact_wings'] = sorted(exact_wings)
            execution['wing_log_Z'] = wing_log_Z
            final_path = checkpoint_mgr.save_checkpoint(
                checkpoint_id=(ckpt_history[-1]['checkpoint_id'] + 1 if ckpt_history else 1),
                eval_count=scorer.n_unique_evaluated,
                elapsed_seconds=t_total,
                log_Z_hat=global_log_Z,
                log_Z_by_k=log_Z_hat_by_k,
                P_hat_k=P_hat_k,
                pips=pips_array,
                map_model_id=map_model_id,
                map_log_score=map_log_score,
                map_pmp=map_pmp,
                scorer=scorer,
                registry=registry,
                metadata=dict(fingerprint=fingerprint, rng_state=rng.bit_generator.state,
                    execution=execution, stage='complete', summary_semantics='final reconstructed posterior'),
            )
            ckpt_history.append(json.loads((final_path/'state.json').read_text(encoding='utf-8')))

        # 12. Return Result
        return BFGResult(
            outcome=outcome_name,
            candidate_names=names,
            n_obs=n,
            n_predictors=p,
            total_universe_models=1 << p,
            n_models_evaluated=scorer.n_unique_evaluated,
            log_Z=global_log_Z,
            model_size_posterior=pd.Series(P_hat_k, index=pd.RangeIndex(p + 1, name="model_size"), name="posterior_probability"),
            pips=pd.Series(pips_array, index=names, name="pip"),
            posterior_mean=pd.Series(coef_mean_array, index=names, name="post_mean"),
            posterior_sd=pd.Series(coef_sd_array, index=names, name="post_sd"),
            sign_probability=pd.Series(sign_pos_array, index=names, name="p_pos"),
            map_model=map_vars,
            map_model_id=map_model_id,
            map_log_score=map_log_score,
            map_pmp=map_pmp,
            lattice_results=lattice_results,
            elite_registry=registry.to_dataframe(),
            checkpoints=ckpt_history,
            runtime={
                "total_seconds": t_total,
                "wings_seconds": t_wings,
                "genealogy_seconds": t_genealogy,
                "sampling_seconds": t_sampling,
                "acesm_seconds": t_acesm,
                "backend": scorer.backend,
                "throughput_models_per_sec": scorer.n_unique_evaluated / max(t_total, 1e-6),
            },
            hardware={
                "device_name": scorer.device_name,
                "backend": scorer.backend,
                "precision": "float64",
            },
            diagnostics={
                "cache_hits": scorer.n_cache_hits,
                "eval_calls": scorer.n_eval_calls,
                "exact_wings": sorted(list(exact_wings)),
                "allocation_basis": "normalized evaluated-score mass; exploratory lower-bound priorities, not certified posterior",
                "allocation_priorities": allocation_priorities.tolist(),
                "allocation_budget": budget_allocated,
                "allocation_remaining_budget": remaining_budget,
                "reconstruction_method": "ACESM total with fresh-sample calibrated conditional summaries; approximate",
                "reconstruction_sample_ids": sample_models_by_k,
                "integration_weights": global_integration,
                "unseen_probability": float(sum(P_hat_k[k]*unseen_mass_by_k[k] for k in range(p+1))),
                "posterior_certified": scorer.n_unique_evaluated == (1 << p),
                "sign_probability_semantics": "posterior integration weight of positive conditional point means; not integrated coefficient sign probability",
                "compression_factor": (1 << p) / max(scorer.n_unique_evaluated, 1),
            },
        )


def fit_bfg(
    y: Any,
    X: Any,
    candidate_names: Optional[Sequence[str]] = None,
    *,
    always_in: Optional[Any] = None,
    budget_models: int = 100_000,
    device: str = "cuda",
    seed: int = 20260715,
    g: Union[str, float] = "benchmark",
    model_prior: Tuple[str, float, float] = ("betabinomial", 1.0, 1.0),
    always_prior: str = "shrink",
    acesm_beta: float = 3.5,
    beam_width: int = 5,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    resume: bool = False,
    config: Optional[BFGConfig] = None,
    verbose: bool = True,
    outcome_name: str = "y",
    **kwargs,
) -> BFGResult:
    """Public high-level functional API for BFG Bayesian Model Averaging.

    Parameters
    ----------
    y : array-like, Series, or 1D Tensor
        Target outcome variable.
    X : array-like, DataFrame, or 2D Tensor
        Design matrix of candidate predictors.
    candidate_names : Optional[Sequence[str]], default=None
        Names of candidate predictors. If None and X is a DataFrame, column names are used.
    always_in : Optional[array-like or DataFrame], default=None
        Controls and fixed-effect dummies always included in the model.
    budget_models : int, default=100_000
        Total model evaluation budget.
    device : str, default="cuda"
        Compute device ("cuda", "cuda:0", "cpu").
    seed : int, default=20260715
        Deterministic random seed.
    g : Union[str, float], default="benchmark"
        Zellner g-prior specification.
    model_prior : Tuple[str, float, float], default=("betabinomial", 1.0, 1.0)
        Model size prior specification.
    always_prior : str, default="shrink"
        Always-included slope prior: "shrink" (Stata bmaregress compatible) or "flat".
    acesm_beta : float, default=3.5
        Locked ACESM Weibull shape parameter.
    beam_width : int, default=5
        Genealogical beam width.
    checkpoint_dir : Optional[Union[str, Path]], default=None
        Directory for progressive checkpoints and state resumption.
    resume : bool, default=False
        Whether to resume from existing checkpoints in checkpoint_dir.
    config : Optional[BFGConfig], default=None
        Custom configuration instance overriding individual parameters.
    verbose : bool, default=True
        Whether to print progress messages.
    outcome_name : str, default="y"
        Name of outcome variable for reporting.

    Returns
    -------
    BFGResult
        Structured BMA result containing reconstructed log Z, PIPs, MAP model,
        posterior model size distribution, and moments.

    Example
    -------
    >>> from gpubma import fit_bfg
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> y = X[:, 0] * 2.0 + X[:, 1] * 1.5 + np.random.randn(100)
    >>> result = fit_bfg(y, X, budget_models=5000, seed=20260715)
    >>> print(result.summary())
    """
    if config is None:
        cfg = BFGConfig(
            budget_models=budget_models,
            device=device,
            seed=seed,
            g=g,
            model_prior=model_prior,
            always_prior=always_prior,
            acesm_beta=acesm_beta,
            beam_width=beam_width,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            verbose=verbose,
            **kwargs,
        )
    else:
        cfg = config

    engine = BFGEngine(config=cfg)
    return engine.fit(
        y=y,
        X=X,
        candidate_names=candidate_names,
        always_in=always_in,
        outcome_name=outcome_name,
    )

