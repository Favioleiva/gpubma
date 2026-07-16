"""Result object returned by :func:`gpubma.bma_regress`."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BMAResult:
    outcome: str
    predictor_names: list
    n_obs: int
    n_predictors: int
    n_models_expected: int
    n_models_evaluated: int
    df_resid: int
    g_spec: object
    model_prior_description: str
    fixed_effects_info: dict
    backend: str
    precision: str
    pip: np.ndarray
    pmp: np.ndarray = None
    masks: np.ndarray = None
    log_scores: np.ndarray = None
    coef_mean: np.ndarray = None
    coef_sd: np.ndarray = None
    mean_model_size: float = float("nan")
    size_distribution: np.ndarray = None
    _top_models: list = field(default_factory=list)
    runtime: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    # ----------------------------------------------------------------- API
    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "predictor": self.predictor_names,
                "pip": self.pip,
                "post_mean": self.coef_mean,
                "post_sd": self.coef_sd,
            }
        )

    def inclusion_probabilities(self) -> pd.Series:
        return pd.Series(self.pip, index=self.predictor_names, name="pip")

    def top_models(self, k: int = 10) -> pd.DataFrame:
        rows = []
        for m in self._top_models[:k]:
            rows.append(
                {
                    "mask": m["mask"],
                    "predictors": " ".join(self.predictor_names[j] for j in m["included"]) or "(null)",
                    "size": m["size"],
                    "pmp": m["pmp"],
                    "log_score": m["log_score"],
                }
            )
        return pd.DataFrame(rows)

    def model_size_distribution(self) -> pd.Series:
        return pd.Series(
            self.size_distribution,
            index=pd.RangeIndex(len(self.size_distribution), name="model_size"),
            name="posterior_probability",
        )

    def fixed_effects_report(self) -> dict:
        return dict(self.fixed_effects_info)

    def hardware_report(self) -> dict:
        return dict(self.hardware)

    def runtime_report(self) -> dict:
        return dict(self.runtime)

    def summary(self) -> str:
        lines = [
            "GPUBMA exhaustive Bayesian Model Averaging (linear regression)",
            "=" * 64,
            f"Outcome:            {self.outcome}",
            f"Observations:       {self.n_obs}",
            f"Optional predictors:{self.n_predictors:>5}   candidate models: {self.n_models_expected:,}",
            f"Models evaluated:   {self.n_models_evaluated:,} (exhaustive enumeration)",
            f"Effective df:       {self.df_resid}",
            f"g-prior:            {self.g_spec.describe()}",
            f"Model prior:        {self.model_prior_description}",
            f"Fixed effects:      {self.fixed_effects_info.get('fixed_effects') or 'none'}"
            + (
                f" (method: {self.fixed_effects_info.get('fe_method')})"
                if self.fixed_effects_info.get("fixed_effects")
                else ""
            ),
            f"Backend/precision:  {self.backend}/{self.precision}",
            f"Posterior mean model size: {self.mean_model_size:.4f}",
            "",
            "Coefficients (BMA posterior):",
            self.coefficients().to_string(
                index=False, float_format=lambda v: f"{v: .6f}"
            ),
            "",
            "Top models by posterior probability:",
            self.top_models(5).to_string(index=False, float_format=lambda v: f"{v: .6f}"),
        ]
        if self.notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in self.notes]
        return "\n".join(lines)

    def __repr__(self) -> str:  # keep console output short
        return (
            f"<BMAResult outcome={self.outcome!r} p={self.n_predictors} "
            f"models={self.n_models_evaluated:,} backend={self.backend}>"
        )
