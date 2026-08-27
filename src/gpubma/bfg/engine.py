"""Core pipeline engine and functional API for BFG Bayesian Model Averaging."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from gpubma.bfg.acesm import ACESMFitResult, ACESMReconstructor, CumulativeCurveBuilder
from gpubma.bfg.allocation import BudgetAllocator
from gpubma.bfg.checkpoint import CheckpointManager
from gpubma.bfg.config import BFGConfig
from gpubma.bfg.elite_search import GPUEliteSearch
from gpubma.bfg.genealogy import GenealogicalSearch
from gpubma.bfg.registry import EliteRegistry, ModelProvenance
from gpubma.bfg.results import BFGResult, LatticeResult
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

        # Resume state if requested
        ckpt_history = []
        if self.config.resume and self.config.checkpoint_dir is not None:
            prev_state = checkpoint_mgr.load_latest_checkpoint(scorer=scorer, registry=registry)
            if prev_state is not None:
                ckpt_history.append(prev_state.to_dict())
                if self.config.verbose:
                    print(
                        f"[BFG Engine] Resumed from checkpoint {prev_state.checkpoint_id} "
                        f"({prev_state.evaluated_count:,} cached models)."
                    )

        # 3. Identify Exact Wings (under hard budget ceiling)
        exact_wings: Set[int] = set()
        wing_log_Z: Dict[int, float] = {}
        t_wings_start = time.perf_counter()

        max_wing_budget = self.config.budget_models if self.config.budget_semantics != "hard" else int(self.config.budget_models * 0.35)

        for k in range(p + 1):
            n_k_comb = math.comb(p, k)
            if wing_enum.is_exact_wing(k, max_wing_size=self.config.wing_max_size):
                if scorer.n_unique_evaluated + n_k_comb <= max_wing_budget or n_k_comb <= 64:
                    exact_wings.add(k)
                    models, scores, log_Z_exact = wing_enum.enumerate_lattice(k)
                    wing_log_Z[k] = log_Z_exact
                    registry.register_batch(models, scores, ModelProvenance.EXACT_WING, source_tag=f"exact_wing_k{k}")

        t_wings = time.perf_counter() - t_wings_start

        # 4. Genealogical Multi-Path Search (Greedy + Forward/Backward Beam Search)
        t_genealogy_start = time.perf_counter()
        fwd_res = genealogy.forward_greedy(start_model_id=0)
        bwd_res = genealogy.backward_greedy(start_model_id=(1 << p) - 1)

        # Run beam search if budget allows
        if self.config.budget_semantics != "hard" or scorer.n_unique_evaluated < self.config.budget_models:
            beam_fwd = genealogy.forward_beam(
                start_seeds=[0], beam_width=self.config.beam_width
            )
            beam_bwd = genealogy.backward_beam(
                start_seeds=[(1 << p) - 1], beam_width=self.config.beam_width
            )
        t_genealogy = time.perf_counter() - t_genealogy_start

        # 5. Adaptive Budget Allocation for Non-Wing Lattices
        t_sampling_start = time.perf_counter()
        if self.config.budget_semantics == "hard":
            remaining_budget = max(0, self.config.budget_models - scorer.n_unique_evaluated)
        else:
            remaining_budget = self.config.budget_models

        budget_allocated = BudgetAllocator.allocate(
            total_budget=remaining_budget,
            p=p,
            exact_wings=exact_wings,
            strategy=self.config.allocation_strategy,
            min_per_lattice=min(self.config.recon_sample_per_lattice, max(50, remaining_budget // max(1, p + 1 - len(exact_wings)))),
        )

        # 6. Random Reconnaissance & GPU Elite Search per Non-Wing Lattice
        curves_by_k: Dict[int, Any] = {}
        sample_scores_by_k: Dict[int, np.ndarray] = {}
        sample_models_by_k: Dict[int, List[int]] = {}

        for k in range(p + 1):
            if k in exact_wings:
                continue

            n_k_budget = budget_allocated.get(k, 0)
            if self.config.budget_semantics == "hard":
                curr_rem = max(0, self.config.budget_models - scorer.n_unique_evaluated)
                n_k_budget = min(n_k_budget, curr_rem)

            if n_k_budget > 0:
                r_k = min(self.config.elite_calibration_size, max(2, n_k_budget // 2))
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
                calib_scores = elite_res.calibration_scores
            else:
                calib_scores = np.array([], dtype=np.float64)

            sample_scores_by_k[k] = calib_scores

            # Retrieve discovered elite models vs random sample models for lattice k
            known_scores = registry.get_elite_scores(k)
            if not known_scores:
                champ_k = registry.get_champion(k)
                if champ_k is not None:
                    known_scores = [champ_k.log_score]

            # Build empirical cumulative evidence curve C_k^obs(d)
            curve = CumulativeCurveBuilder.build_empirical_curve(
                k=k,
                discovered_scores=known_scores,
                sampled_scores=calib_scores,
                N_k=math.comb(p, k),
                n_grid_points=80,
            )
            curves_by_k[k] = curve

        t_sampling = time.perf_counter() - t_sampling_start

        # 7. ACESM Denominator Saturation Fitting
        t_acesm_start = time.perf_counter()
        log_Z_hat_by_k: Dict[int, float] = {}
        lattice_results: Dict[int, LatticeResult] = {}

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
                log_Z_hat_by_k[k] = acesm_res.log_Z_hat
                lattice_results[k] = LatticeResult(
                    k=k,
                    N_k=N_k,
                    log_Z_hat=acesm_res.log_Z_hat,
                    best_model_id=best_id,
                    best_score=best_score,
                    evaluated_count=n_eval_k,
                    elite_count=curve.n_elites,
                    acesm_parameters={
                        "delta_Z_hat": acesm_res.delta_Z_hat,
                        "alpha_hat": acesm_res.alpha_hat,
                        "beta_hat": acesm_res.beta_hat,
                        "final_loss": acesm_res.final_loss,
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

        # 9. Statistically Valid PIP & Coefficient Reconstruction
        pips_array = np.zeros(p, dtype=np.float64)
        coef_mean_array = np.zeros(p, dtype=np.float64)
        coef_m2_array = np.zeros(p, dtype=np.float64)
        sign_pos_array = np.zeros(p, dtype=np.float64)

        # Accumulate lattice-conditional inclusion mass
        for k in range(p + 1):
            p_k_weight = P_hat_k[k]
            if k == 0 or p_k_weight <= 1e-15:
                continue

            if k in exact_wings:
                # Exact wing: sum over all models in lattice
                models_k = [m for m in registry.by_k[k].keys()]
                scores_k = np.array([registry.by_k[k][m].log_score for m in models_k], dtype=np.float64)
                pmp_k = np.exp(scores_k - wing_log_Z[k])
                for m, w in zip(models_k, pmp_k):
                    idx = model_id_to_indices(m, p)
                    pips_array[idx] += p_k_weight * w
            else:
                # ACESM lattice: known elites + weighted probability sample
                curve = curves_by_k[k]
                U_k = curve.U_obs
                known_recs = list(registry.by_k[k].values())

                if known_recs:
                    known_ids = [r.model_id for r in known_recs]
                    known_scores = [r.log_score for r in known_recs]
                    known_rel_mass = np.exp(np.array(known_scores, dtype=np.float64) - U_k)
                    z_known_rel = float(np.sum(known_rel_mass))
                else:
                    known_ids, known_rel_mass, z_known_rel = [], np.array([]), 0.0

                samp_scores = sample_scores_by_k.get(k, np.array([]))
                if len(samp_scores) > 0:
                    n_non_elite = max(math.comb(p, k) - len(known_ids), 1)
                    samp_wgt = float(n_non_elite) / float(len(samp_scores))
                    samp_rel_mass = np.exp(samp_scores - U_k) * samp_wgt
                    z_samp_rel = float(np.sum(samp_rel_mass))
                else:
                    samp_rel_mass, z_samp_rel = np.array([]), 0.0

                z_total_rel = max(z_known_rel + z_samp_rel, 1e-300)

                for m, m_rel in zip(known_ids, known_rel_mass):
                    prob_in_k = m_rel / z_total_rel
                    idx = model_id_to_indices(m, p)
                    pips_array[idx] += p_k_weight * prob_in_k

        pips_array = np.clip(pips_array, 0.0, 1.0)

        # Posterior coefficients computed over registered models holding posterior mass
        all_registered = list(registry.records.values())
        if all_registered:
            reg_scores = np.array([r.log_score for r in all_registered], dtype=np.float64)
            reg_weights = np.exp(np.clip(reg_scores - global_log_Z, -700.0, 0.0))
            sum_reg_weights = max(float(np.sum(reg_weights)), 1e-12)

            for r, w in zip(all_registered, reg_weights):
                if r.model_id == 0 or w < 1e-8:
                    continue
                norm_w = w / sum_reg_weights
                moments = scorer.compute_model_coefficients(r.model_id)
                idx = moments["indices"]
                b_mean = moments["coef_mean"]
                b_sd = moments["coef_sd"]

                coef_mean_array[idx] += norm_w * b_mean
                coef_m2_array[idx] += norm_w * (b_sd ** 2 + b_mean ** 2)
                pos_mask = b_mean > 0
                sign_pos_array[np.array(idx)[pos_mask]] += norm_w

        coef_sd_array = np.sqrt(np.maximum(coef_m2_array - coef_mean_array ** 2, 0.0))
        sign_pos_array = np.clip(sign_pos_array, 0.0, 1.0)

        # 10. Global MAP Model
        global_champ = registry.get_champion()
        map_model_id = global_champ.model_id if global_champ else 0
        map_log_score = global_champ.log_score if global_champ else float("-inf")
        map_pmp = float(math.exp(min(map_log_score - global_log_Z, 0.0)))
        map_vars = model_id_to_vars(map_model_id, names)

        t_total = time.perf_counter() - t0_total

        # 11. Checkpoint Final State
        if self.config.checkpoint_dir is not None:
            checkpoint_mgr.save_checkpoint(
                checkpoint_id=len(ckpt_history) + 1,
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
            )

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
