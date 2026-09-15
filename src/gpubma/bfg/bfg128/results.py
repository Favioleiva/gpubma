"""Result container for native 128-bit BFG Bayesian Model Averaging."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from gpubma.bfg.scorer import model_id_to_vars
from gpubma.bfg.bfg128.recovery import ShellRecoveryResult128


@dataclass
class BFG128Result:
    """Comprehensive Bayesian Model Averaging result for 61 <= p <= 128."""

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
    map_model_key: Tuple[int, int]
    map_log_score: float
    map_pmp: float
    shell_results: Dict[int, ShellRecoveryResult128]
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

    @property
    def backend(self) -> str:
        """Name of execution backend engine ('bfg128' for 61 <= p <= 128)."""
        return "bfg128"

    def inclusion_probabilities(self) -> pd.Series:
        """Return Series of Posterior Inclusion Probabilities (PIPs)."""
        return self.pips.copy()

    def top_models(self, n: int = 10) -> pd.DataFrame:
        """Return top n evaluated models by estimated Posterior Model Probability (PMP)."""
        if self.elite_registry.empty:
            return pd.DataFrame(columns=["rank", "mask_lo", "mask_hi", "size", "predictors", "log_score", "pmp"])

        sorted_df = self.elite_registry.sort_values("log_score", ascending=False).head(n).copy()
        rows = []
        for rank_idx, (_, row) in enumerate(sorted_df.iterrows(), start=1):
            lo = int(row["mask_lo"])
            hi = int(row["mask_hi"])
            m_id = (hi << 64) | lo
            log_s = float(row["log_score"])
            pmp_val = math.exp(min(log_s - self.log_Z, 0.0))
            var_names = model_id_to_vars(m_id, self.candidate_names)
            rows.append({
                "rank": rank_idx,
                "mask_lo": hex(lo),
                "mask_hi": hex(hi),
                "size": int(row["model_size"]),
                "predictors": " ".join(var_names) if var_names else "(null)",
                "log_score": log_s,
                "pmp": pmp_val,
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Generate a concise, publication-grade summary of BFG128 posterior inference."""
        comp_factor = float(self.total_universe_models) / max(self.n_models_evaluated, 1)
        lines = [
            "GPUBMA BFG128 Bayesian Model Averaging (Native 128-bit Engine)",
            "=" * 72,
            f"Outcome:                   {self.outcome}",
            f"Observations (N):          {self.n_obs}",
            f"Candidate Predictors (K):  {self.n_predictors:>5}   Universe Size: 2^{self.n_predictors} = {self.total_universe_models:.2e}",
            f"Unique Models Evaluated:   {self.n_models_evaluated:,} (Compression Factor: {comp_factor:,.1e}x)",
            f"Reconstructed Log Denom:   log Z = {self.log_Z:.6f}",
            f"Best-found MAP Key:        (0x{self.map_model_key[0]:016x}, 0x{self.map_model_key[1]:016x}) (Size {len(self.map_model)})",
            f"Best Discovered MAP:       {' '.join(self.map_model) if self.map_model else '(null)'}",
            f"Best Discovered Score:     {self.map_log_score:.6f}",
            f"Best Discovered PMP:       {self.map_pmp:.4f}",
            f"Execution Device / Engine: {self.hardware.get('device_name', 'CPU')} ({self.runtime.get('backend', 'gpu')})",
            f"Total Execution Time:      {self.runtime.get('total_seconds', 0.0):.2f} s",
            "",
            "Posterior Coefficients & Inclusion Probabilities:",
            self.coefficients().to_string(index=False, float_format=lambda v: f"{v: .6f}"),
            "",
            "Top Evaluated Models by Posterior Probability:",
            self.top_models(5).to_string(index=False, float_format=lambda v: f"{v: .6f}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<BFG128Result outcome={self.outcome!r} K={self.n_predictors} "
            f"evaluated={self.n_models_evaluated:,} log_Z={self.log_Z:.4f}>"
        )
