"""Native 128-bit search engine and pipeline for 61 <= p <= 128."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

from gpubma import __version__ as gpubma_version
from gpubma.bfg.config import BFGConfig
from gpubma.bfg.engine import validate_inputs
from gpubma.bfg.registry import ModelProvenance
from gpubma.bfg.scorer import BFGScorer, count_set_bits, model_id_to_vars
from gpubma.priors.gpriors import resolve_g
from gpubma.priors.model_priors import log_model_prior_function

from gpubma.bfg.bfg128.bitops import pack_indices, popcount, unpack_mask
from gpubma.bfg.bfg128.elite_search import WideEliteSearch
from gpubma.bfg.bfg128.genealogy import WideGenealogicalSearch
from gpubma.bfg.bfg128.recovery import ShellRecoveryResult128, WideShellRecovery
from gpubma.bfg.bfg128.registry import WideEliteRegistry, WideModelRecord
from gpubma.bfg.bfg128.results import BFG128Result


class BFG128Engine:
    """Master pipeline orchestrator for BFG128 (61 <= p <= 128)."""

    def __init__(self, config: Optional[BFGConfig] = None):
        self.config = config if config is not None else BFGConfig()

    def fit(
        self,
        y: Any,
        X: Any,
        candidate_names: Optional[Sequence[str]] = None,
        always_in: Optional[Any] = None,
        outcome_name: str = "y",
    ) -> BFG128Result:
        """Execute full BFG128 search and Bayesian model averaging."""
        t0_total = time.perf_counter()

        # 1. Validation & Data Preparation
        y_raw, X_raw, names, A_raw = validate_inputs(
            y=y, X=X, candidate_names=candidate_names, always_in=always_in
        )
        n, p = X_raw.shape
        if not (61 <= p <= 128):
            raise ValueError(
                f"BFG128Engine requires 61 <= p <= 128, got p={p}. "
                "Use legacy BFGEngine for p <= 60."
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
        registry = WideEliteRegistry(p=p)
        genealogy = WideGenealogicalSearch(scorer=scorer, registry=registry)
        elite_searcher = WideEliteSearch(scorer=scorer, registry=registry)
        recovery_engine = WideShellRecovery(
            scorer=scorer,
            registry=registry,
            census_max_combinations=self.config.wing_max_size,
        )

        # Resumption signature
        fingerprint_config = self.config.to_dict()
        for key in ('resume', 'checkpoint_dir', 'checkpoints', 'verbose', 'progress_interval'):
            fingerprint_config.pop(key, None)
        identity = hashlib.sha256(json.dumps(fingerprint_config, sort_keys=True).encode())
        identity.update(json.dumps(names).encode())
        identity.update(b'contract8-bfg128-resume-v1')
        for array in (y_raw, X_raw, A_raw):
            if array is not None:
                identity.update(str(array.shape).encode())
                identity.update(np.ascontiguousarray(array).tobytes())
        fingerprint = identity.hexdigest()

        # Checkpoint directory
        ckpt_dir = Path(self.config.checkpoint_dir) if self.config.checkpoint_dir is not None else None
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        execution_state = {
            "wings_done": False,
            "genealogy_done": False,
            "elite_search_done": False,
            "recovery_done": False,
            "next_recovery_k": 0,
            "shell_raw_indices": {k: 0 for k in range(p + 1)},
        }
        elapsed_before_resume = 0.0
        ckpt_history: List[Dict[str, Any]] = []
        shell_results: Dict[int, ShellRecoveryResult128] = {}

        # Load checkpoint if resuming
        if self.config.resume and ckpt_dir is not None:
            state_file = ckpt_dir / "bfg128_state.json"
            reg_file = ckpt_dir / "bfg128_registry.parquet"
            if state_file.is_file() and reg_file.is_file():
                meta = json.loads(state_file.read_text(encoding="utf-8"))
                if meta.get("backend") is not None and meta.get("backend") != "bfg128":
                    raise ValueError(
                        f"Incompatible checkpoint backend: {meta.get('backend')}, expected 'bfg128'."
                    )
                if meta.get("fingerprint") != fingerprint:
                    raise ValueError(
                        f"Checkpoint fingerprint mismatch in {ckpt_dir}. "
                        "Dataset or configuration changed; cannot safely resume."
                    )
                execution_state = meta["execution_state"]
                execution_state["shell_raw_indices"] = {
                    int(k): v for k, v in execution_state["shell_raw_indices"].items()
                }
                elapsed_before_resume = meta.get("elapsed_seconds", 0.0)

                # Reconstitute registry
                reg_df = pd.read_parquet(reg_file)
                for _, r in reg_df.iterrows():
                    parent_key = (int(r["parent_lo"]), int(r["parent_hi"])) if pd.notna(r["parent_lo"]) else None
                    registry.register(
                        mask_lo=int(r["mask_lo"]),
                        mask_hi=int(r["mask_hi"]),
                        log_score=float(r["log_score"]),
                        provenance=ModelProvenance(r["provenance"]),
                        parent_key=parent_key,
                        generation=int(r["generation"]),
                        source_tag=str(r["source_tags"]),
                        k=int(r["model_size"]),
                    )

                # Reconstitute shell_results if present
                if "shell_results" in meta:
                    for k_str, r_data in meta["shell_results"].items():
                        k_val = int(k_str)
                        champ_k = tuple(r_data["champion_key"]) if r_data.get("champion_key") else None
                        shell_results[k_val] = ShellRecoveryResult128(
                            k=r_data["k"],
                            N_k=r_data["N_k"],
                            D_k=r_data["D_k"],
                            remainder_k=r_data["remainder_k"],
                            n_recovery=r_data["n_recovery"],
                            binding_rule=r_data["binding_rule"],
                            log_Z_k=r_data["log_Z_k"],
                            pip_k=np.array(r_data["pip_k"], dtype=np.float64),
                            champion_key=champ_k,
                            champion_score=r_data["champion_score"],
                            next_raw_index=r_data["next_raw_index"],
                            runtime_seconds=r_data.get("runtime_seconds", 0.0),
                        )

                if self.config.verbose:
                    print(
                        f"[BFG128 Engine] Resumed from checkpoint: "
                        f"{registry.total_registered():,} models loaded."
                    )

        def save_checkpoint_state(stage: str):
            if ckpt_dir is None:
                return
            state_data = {
                "backend": "bfg128",
                "schema_version": "1.0",
                "package_version": gpubma_version,
                "fingerprint": fingerprint,
                "stage": stage,
                "n_predictors": p,
                "n_obs": n,
                "seed": self.config.seed,
                "candidate_names": names,
                "model_prior": self.config.model_prior,
                "g_prior": str(self.config.g),
                "budget_models": self.config.budget_models,
                "recon_sample_per_lattice": self.config.recon_sample_per_lattice,
                "n_models_evaluated": registry.total_registered(),
                "elapsed_seconds": elapsed_before_resume + (time.perf_counter() - t0_total),
                "execution_state": execution_state,
                "shell_results": {
                    str(k): {
                        "k": res.k,
                        "N_k": res.N_k,
                        "D_k": res.D_k,
                        "remainder_k": res.remainder_k,
                        "n_recovery": res.n_recovery,
                        "binding_rule": res.binding_rule,
                        "log_Z_k": res.log_Z_k,
                        "pip_k": res.pip_k.tolist(),
                        "champion_key": list(res.champion_key) if res.champion_key else None,
                        "champion_score": res.champion_score,
                        "next_raw_index": res.next_raw_index,
                        "runtime_seconds": res.runtime_seconds,
                    }
                    for k, res in shell_results.items()
                },
            }
            (ckpt_dir / "bfg128_state.json").write_text(json.dumps(state_data, indent=2), encoding="utf-8")
            registry.to_dataframe().to_parquet(ckpt_dir / "bfg128_registry.parquet", index=False)
            ckpt_history.append(state_data)

        # -------------------------------------------------------------
        # Phase 1: Exact Wing Enumeration
        # -------------------------------------------------------------
        t0_wings = time.perf_counter()
        exact_wings: Set[int] = set()
        wing_log_Z: Dict[int, float] = {}

        if not execution_state["wings_done"]:
            # Wings k=0 and k=1
            r0 = recovery_engine.enumerate_census_shell(0)
            exact_wings.add(0)
            wing_log_Z[0] = r0.log_Z_k

            r1 = recovery_engine.enumerate_census_shell(1)
            exact_wings.add(1)
            wing_log_Z[1] = r1.log_Z_k

            # Upper wings k=p and k=p-1
            rp = recovery_engine.enumerate_census_shell(p)
            exact_wings.add(p)
            wing_log_Z[p] = rp.log_Z_k

            rp1 = recovery_engine.enumerate_census_shell(p - 1)
            exact_wings.add(p - 1)
            wing_log_Z[p - 1] = rp1.log_Z_k

            # Check if k=2 or k=p-2 fit under wing_max_size
            if math.comb(p, 2) <= self.config.wing_max_size:
                r2 = recovery_engine.enumerate_census_shell(2)
                exact_wings.add(2)
                wing_log_Z[2] = r2.log_Z_k

                rp2 = recovery_engine.enumerate_census_shell(p - 2)
                exact_wings.add(p - 2)
                wing_log_Z[p - 2] = rp2.log_Z_k

            execution_state["wings_done"] = True
            save_checkpoint_state("wings")

        t_wings = time.perf_counter() - t0_wings

        # -------------------------------------------------------------
        # Phase 2: Multi-Path Genealogical Search
        # -------------------------------------------------------------
        t0_genealogy = time.perf_counter()
        if not execution_state["genealogy_done"]:
            max_budget = self.config.budget_models if self.config.budget_semantics == "hard" else None
            # Forward greedy from null model
            fwd_res = genealogy.forward_greedy(
                start_key=(0, 0),
                target_k=p,
                max_eval_budget=max_budget,
            )
            # Backward greedy from full model
            full_key = pack_indices(range(p), p)
            bwd_res = genealogy.backward_greedy(
                start_key=full_key,
                target_k=0,
                max_eval_budget=max_budget,
            )
            # Beam search from discovered path champions
            beam_seeds = [fwd_res.best_model_key, bwd_res.best_model_key]
            genealogy.forward_beam(
                start_seeds=beam_seeds,
                beam_width=self.config.beam_width,
                max_eval_budget=max_budget,
            )
            genealogy.backward_beam(
                start_seeds=beam_seeds,
                beam_width=self.config.beam_width,
                max_eval_budget=max_budget,
            )

            execution_state["genealogy_done"] = True
            save_checkpoint_state("genealogy")

        t_genealogy = time.perf_counter() - t0_genealogy

        # -------------------------------------------------------------
        # Phase 3: GPU Elite Search Across Candidate Shells
        # -------------------------------------------------------------
        t0_elite = time.perf_counter()
        if not execution_state["elite_search_done"]:
            # Allocate elite reconnaissance to active shells around greedy/beam champions
            active_k_shells = sorted({
                rec.model_size for rec in registry.records.values()
                if rec.model_size not in exact_wings
            })
            remaining_eval = max(0, self.config.budget_models - registry.total_registered())
            if active_k_shells and remaining_eval > 0:
                per_shell_budget = max(50, remaining_eval // (len(active_k_shells) * 2))
                for k in active_k_shells:
                    if self.config.budget_semantics == "hard" and registry.total_registered() >= self.config.budget_models:
                        break
                    r_k = min(per_shell_budget // 2, self.config.elite_calibration_size)
                    m_k = max(0, per_shell_budget - r_k)
                    elite_res = elite_searcher.run_lattice_elite_search(
                        k=k,
                        r_k=r_k,
                        m_k=m_k,
                        target_q=self.config.elite_quantile,
                        seed=self.config.seed + k,
                        batch_size=self.config.batch_size,
                    )
                    # Seed beam search if strong hits found
                    if elite_res.retained_model_keys and (
                        self.config.budget_semantics != "hard" or registry.total_registered() < self.config.budget_models
                    ):
                        genealogy.forward_beam(
                            start_seeds=elite_res.retained_model_keys[: self.config.beam_width],
                            beam_width=self.config.beam_width,
                            max_eval_budget=self.config.budget_models if self.config.budget_semantics == "hard" else None,
                        )

            execution_state["elite_search_done"] = True
            save_checkpoint_state("elite_search")

        t_elite = time.perf_counter() - t0_elite

        # -------------------------------------------------------------
        # Phase 4: Stratified Shell Recovery & Online Evidence Accumulation
        # -------------------------------------------------------------
        t0_recovery = time.perf_counter()
        if not execution_state["recovery_done"]:
            # Default recovery cap: e.g. 5,000 per shell for standard BFG run or config
            recovery_cap = max(self.config.recon_sample_per_lattice, 500)

            for k in range(p + 1):
                if k in exact_wings:
                    # Wing already computed
                    res = recovery_engine.enumerate_census_shell(k)
                else:
                    raw_idx = execution_state["shell_raw_indices"].get(k, 0)
                    res = recovery_engine.recover_shell(
                        k=k,
                        target_evaluations=recovery_cap,
                        seed=self.config.seed + k * 10007,
                        start_raw_index=raw_idx,
                        batch_size=self.config.batch_size,
                    )
                    execution_state["shell_raw_indices"][k] = res.next_raw_index

                shell_results[k] = res

            execution_state["recovery_done"] = True
            save_checkpoint_state("recovery")
        t_recovery = time.perf_counter() - t0_recovery

        # -------------------------------------------------------------
        # Phase 5: Global Normalization & Evidence Aggregation
        # -------------------------------------------------------------
        all_log_Z_k = [shell_results[k].log_Z_k for k in range(p + 1)]
        global_log_Z = float(logsumexp(all_log_Z_k))

        # Model size posterior P(k | y)
        P_hat_k = np.exp(np.array(all_log_Z_k, dtype=np.float64) - global_log_Z)
        P_hat_k = P_hat_k / np.sum(P_hat_k)

        # Global Posterior Inclusion Probabilities (PIPs)
        pips_array = np.zeros(p, dtype=np.float64)
        for k in range(p + 1):
            pips_array += P_hat_k[k] * shell_results[k].pip_k
        pips_array = np.clip(pips_array, 0.0, 1.0)

        # -------------------------------------------------------------
        # Best Discovered MAP Model & Moments
        # -------------------------------------------------------------
        global_champ = registry.get_champion()
        best_rec_score = max((r.champion_score for r in shell_results.values() if r.champion_score is not None), default=float("-inf"))

        if global_champ and global_champ.log_score >= best_rec_score:
            map_key = global_champ.key
            map_log_score = global_champ.log_score
        else:
            best_shell = max(shell_results.values(), key=lambda r: r.champion_score)
            map_key = best_shell.champion_key if best_shell.champion_key is not None else (0, 0)
            map_log_score = best_shell.champion_score

        map_model_id = (map_key[1] << 64) | map_key[0]
        map_pmp = float(math.exp(min(map_log_score - global_log_Z, 0.0)))
        map_vars = model_id_to_vars(map_model_id, names)

        # MAP moments
        map_moments = scorer.compute_model_coefficients(map_model_id)
        coef_mean_array = np.zeros(p, dtype=np.float64)
        coef_sd_array = np.zeros(p, dtype=np.float64)
        sign_pos_array = np.zeros(p, dtype=np.float64)

        if map_moments["indices"]:
            idx_list = map_moments["indices"]
            coef_mean_array[idx_list] = map_moments["coef_mean"]
            coef_sd_array[idx_list] = map_moments["coef_sd"]
            sign_pos_array[np.array(idx_list)[map_moments["coef_mean"] > 0]] = 1.0

        t_total = elapsed_before_resume + time.perf_counter() - t0_total

        # Save final state
        save_checkpoint_state("complete")

        return BFG128Result(
            outcome=outcome_name,
            candidate_names=names,
            n_obs=n,
            n_predictors=p,
            total_universe_models=1 << p,
            n_models_evaluated=registry.total_registered() + sum(r.n_recovery for r in shell_results.values()),
            log_Z=global_log_Z,
            model_size_posterior=pd.Series(
                P_hat_k, index=pd.RangeIndex(p + 1, name="model_size"), name="posterior_probability"
            ),
            pips=pd.Series(pips_array, index=names, name="pip"),
            posterior_mean=pd.Series(coef_mean_array, index=names, name="post_mean"),
            posterior_sd=pd.Series(coef_sd_array, index=names, name="post_sd"),
            sign_probability=pd.Series(sign_pos_array, index=names, name="p_pos"),
            map_model=map_vars,
            map_model_id=map_model_id,
            map_model_key=map_key,
            map_log_score=map_log_score,
            map_pmp=map_pmp,
            shell_results=shell_results,
            elite_registry=registry.to_dataframe(),
            checkpoints=ckpt_history,
            runtime={
                "total_seconds": t_total,
                "wings_seconds": t_wings,
                "genealogy_seconds": t_genealogy,
                "elite_seconds": t_elite,
                "recovery_seconds": t_recovery,
                "backend": scorer.backend,
            },
            hardware={
                "device_name": scorer.device_name,
                "backend": scorer.backend,
                "precision": "float64",
            },
            diagnostics={
                "exact_wings": sorted(list(exact_wings)),
                "reconstruction_method": "direct stratified recovery with online streaming accumulators",
                "compression_factor": float(1 << p) / max(registry.total_registered(), 1),
            },
        )
