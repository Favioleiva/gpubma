"""Result containers and posterior inference summaries for BFG."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from gpubma.bfg.scorer import count_set_bits, model_id_to_vars


@dataclass
class LatticeResult:
    """Detailed summary of reconstructed evidence and search state in lattice k."""
    k: int
    N_k: int
    log_Z_hat: float
    best_model_id: int
    best_score: float
    evaluated_count: int
    elite_count: int
    acesm_parameters: Dict[str, float]
    is_wing: bool
    is_boundary_collapsed: bool
    budget_spent: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "N_k": self.N_k,
            "log_Z_hat": self.log_Z_hat,
            "best_model_id": self.best_model_id,
            "best_score": self.best_score,
            "evaluated_count": self.evaluated_count,
            "elite_count": self.elite_count,
            "acesm_parameters": self.acesm_parameters,
            "is_wing": self.is_wing,
            "is_boundary_collapsed": self.is_boundary_collapsed,
            "budget_spent": self.budget_spent,
        }


@dataclass
class BFGResult:
    """Comprehensive Bayesian Model Averaging result produced by BFG.

    Attributes
    ----------
    outcome : str
        Name of dependent outcome variable.
    candidate_names : List[str]
        List of candidate predictor names.
    n_obs : int
        Number of observations.
    n_predictors : int
        Number of candidate predictors K.
    total_universe_models : int
        Total model space size 2^K.
    n_models_evaluated : int
        Unique models evaluated during the BFG search.
    log_Z : float
        Reconstructed global marginal log likelihood denominator log Z_hat.
    model_size_posterior : pd.Series
        Posterior probability distribution over model sizes P(k | y).
    pips : pd.Series
        Posterior Inclusion Probabilities (PIPs) for each candidate predictor.
    posterior_mean : pd.Series
        Estimated BMA posterior mean E[beta_j | y].
    posterior_sd : pd.Series
        Estimated BMA posterior standard deviation SD[beta_j | y].
    sign_probability : pd.Series
        Posterior probability that coefficient is positive P(beta_j > 0 | y).
    map_model : List[str]
        Predictor names included in the global MAP model.
    map_model_id : int
        Integer bitmask ID of the MAP model.
    map_log_score : float
        Canonical log score log z_M of the MAP model.
    map_pmp : float
        Estimated Posterior Model Probability of the MAP model.
    lattice_results : Dict[int, LatticeResult]
        Per-lattice reconstruction and diagnostic records.
    elite_registry : pd.DataFrame
        DataFrame of registered models and provenance tags.
    checkpoints : List[Dict[str, Any]]
        Progressive checkpoint states recorded during search.
    runtime : Dict[str, Any]
        Detailed execution timing breakdown.
    hardware : Dict[str, Any]
        Device and hardware metadata.
    diagnostics : Dict[str, Any]
        Convergence, plateau, and compression diagnostics.
    """

    outcome: str
    candidate_names: List[str]
    n_obs: int
    n_predictors: int
    total_universe_models: int
    n_models_evaluated: int
    log_Z: float
    model_size_posterior: pd.Series
    pips: pd.Series
    posterior_mean: pd.Series
    posterior_sd: pd.Series
    sign_probability: pd.Series
    map_model: List[str]
    map_model_id: int
    map_log_score: float
    map_pmp: float
    lattice_results: Dict[int, LatticeResult]
    elite_registry: pd.DataFrame
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    runtime: Dict[str, Any] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- API
    def coefficients(self) -> pd.DataFrame:
        """Return a structured summary DataFrame of posterior coefficients and inclusion."""
        return pd.DataFrame({
            "predictor": self.candidate_names,
            "pip": self.pips.values,
            "post_mean": self.posterior_mean.values,
            "post_sd": self.posterior_sd.values,
            "p_pos": self.sign_probability.values,
        })

    def inclusion_probabilities(self) -> pd.Series:
        """Return Series of Posterior Inclusion Probabilities (PIPs)."""
        return self.pips.copy()

    def top_models(self, n: int = 10) -> pd.DataFrame:
        """Return top n evaluated models by estimated Posterior Model Probability (PMP)."""
        if self.elite_registry.empty:
            return pd.DataFrame(columns=["rank", "model_id", "size", "predictors", "log_score", "pmp"])

        sorted_df = self.elite_registry.sort_values("log_score", ascending=False).head(n).copy()
        rows = []
        for rank_idx, (_, row) in enumerate(sorted_df.iterrows(), start=1):
            m_id = int(row["model_id"])
            log_s = float(row["log_score"])
            pmp_val = math.exp(min(log_s - self.log_Z, 0.0))
            var_names = model_id_to_vars(m_id, self.candidate_names)
            rows.append({
                "rank": rank_idx,
                "model_id": m_id,
                "size": count_set_bits(m_id),
                "predictors": " ".join(var_names) if var_names else "(null)",
                "log_score": log_s,
                "pmp": pmp_val,
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Generate a concise, publication-grade summary of BFG posterior inference."""
        comp_factor = float(self.total_universe_models) / max(self.n_models_evaluated, 1)
        lines = [
            "GPUBMA BFG Bayesian Model Averaging (Budgeted GPU Search)",
            "=" * 72,
            f"Outcome:                   {self.outcome}",
            f"Observations (N):          {self.n_obs}",
            f"Candidate Predictors (K):  {self.n_predictors:>5}   Universe Size: 2^{self.n_predictors} = {self.total_universe_models:,}",
            f"Unique Models Evaluated:   {self.n_models_evaluated:,} (Compression Factor: {comp_factor:,.1f}x)",
            f"Reconstructed Log Denom:   log Z = {self.log_Z:.6f}",
            f"Global MAP Model ID:       {self.map_model_id} (Size {len(self.map_model)})",
            f"Global MAP Predictors:     {' '.join(self.map_model) if self.map_model else '(null)'}",
            f"Global MAP Log Score:      {self.map_log_score:.6f}",
            f"Global MAP PMP:            {self.map_pmp:.4f}",
            f"Execution Device / Engine: {self.hardware.get('device_name', 'CPU')} ({self.runtime.get('backend', 'gpu')})",
            f"Total Search Time:         {self.runtime.get('total_seconds', 0.0):.2f} s",
            "",
            "Posterior Coefficients & Inclusion Probabilities:",
            self.coefficients().to_string(index=False, float_format=lambda v: f"{v: .6f}"),
            "",
            "Top Evaluated Models by Posterior Probability:",
            self.top_models(5).to_string(index=False, float_format=lambda v: f"{v: .6f}"),
        ]
        return "\n".join(lines)

    def plot_convergence(self, **kwargs):
        """Plot convergence of log Z and MAP score across progressive checkpoints."""
        from gpubma.bfg.diagnostics import plot_convergence
        return plot_convergence(self, **kwargs)

    def plot_model_size(self, **kwargs):
        """Plot posterior model size distribution P(k | y)."""
        from gpubma.bfg.diagnostics import plot_model_size
        return plot_model_size(self, **kwargs)

    def plot_pips(self, **kwargs):
        """Plot Posterior Inclusion Probabilities across candidate predictors."""
        from gpubma.bfg.diagnostics import plot_pips
        return plot_pips(self, **kwargs)

    def plot_allocation(self, **kwargs):
        """Plot per-lattice budget allocation and evaluations."""
        from gpubma.bfg.diagnostics import plot_allocation
        return plot_allocation(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<BFGResult outcome={self.outcome!r} K={self.n_predictors} "
            f"evaluated={self.n_models_evaluated:,} log_Z={self.log_Z:.4f}>"
        )
