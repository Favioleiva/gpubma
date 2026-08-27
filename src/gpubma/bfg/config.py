"""Configuration dataclass for the BFG (Budgeted Fast GPU) BMA Engine."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class BFGConfig:
    """Configuration container for BFG model-space search and denominator reconstruction.

    Parameters
    ----------
    budget_models : int, default=100_000
        Total model evaluation budget across all lattices.
    budget_seconds : Optional[float], default=None
        Maximum wall-clock execution time in seconds. If set, search halts
        when elapsed time exceeds this threshold.
    batch_size : int, default=16384
        GPU batch evaluation size for candidate scoring.
    seed : int, default=20260715
        Deterministic random seed for reproducibility.
    device : str, default="cuda"
        Target execution device ("cuda", "cuda:0", "cpu"). Automatically detects
        and falls back cleanly if requested device is unavailable for small tasks.
    precision : str, default="float64"
        Floating-point precision. Must be "float64" as per CLAUDE.md Rule 6.
    always_prior : str, default="shrink"
        Treatment of always-included controls: "shrink" (Stata bmaregress compatible)
        or "flat".
    g : Union[str, float], default="benchmark"
        Zellner g-prior specification ("benchmark", "uip", "ric", or float value).
    model_prior : Tuple[str, float, float], default=("betabinomial", 1.0, 1.0)
        Model size prior specification (family, param1, param2).
    wing_max_size : int, default=4096
        Maximum lattice size comb(p, k) to treat as an exact boundary wing.
        Lattices with comb(p, k) <= wing_max_size are enumerated exhaustively.
    recon_sample_per_lattice : int, default=2500
        Number of random reconnaissance models sampled per non-wing lattice.
    elite_quantile : float, default=0.05
        Target quantile for GPU Elite Search calibration (e.g. 0.05 for Top 5%).
    elite_calibration_size : int, default=500
        Number of random models evaluated to calibrate the elite threshold tau_k.
    elite_search_budget : int, default=2500
        Maximum models inspected during sequential elite search per lattice.
    beam_width : int, default=5
        Number of elite candidate models maintained per generation during genealogical beam search.
    acesm_beta : float, default=3.5
        Locked shape parameter for the Anchored Weibull Saturation Model.
    acesm_lambda_momentum : float, default=0.0
        Momentum/boundary-slope regularization penalty weight.
    allocation_strategy : str, default="adaptive"
        Budget allocation across lattices: "uniform", "posterior", or "adaptive".
    checkpoints : Optional[List[int]], default=None
        List of cumulative model evaluation counts at which progressive checkpoints are saved.
    checkpoint_dir : Optional[Union[str, Path]], default=None
        Directory where checkpoint states are saved and loaded.
    resume : bool, default=False
        If True, attempts to resume execution from the latest checkpoint in checkpoint_dir.
    progress_interval : float, default=30.0
        Interval in seconds between progress logging statements.
    verbose : bool, default=True
        Whether to print informative progress and diagnostic messages to console.
    """

    budget_models: int = 100_000
    budget_seconds: Optional[float] = None
    batch_size: int = 16384
    seed: int = 20260715
    device: str = "cuda"
    precision: str = "float64"
    always_prior: str = "shrink"
    g: Union[str, float] = "benchmark"
    model_prior: Tuple[str, float, float] = ("betabinomial", 1.0, 1.0)
    wing_max_size: int = 4096
    recon_sample_per_lattice: int = 2500
    elite_quantile: float = 0.05
    elite_calibration_size: int = 500
    elite_search_budget: int = 2500
    beam_width: int = 15
    acesm_beta: float = 3.5
    acesm_lambda_momentum: float = 0.0
    allocation_strategy: str = "adaptive"
    budget_semantics: str = "hard"
    checkpoints: Optional[List[int]] = None
    checkpoint_dir: Optional[Union[str, Path]] = None
    resume: bool = False
    progress_interval: float = 30.0
    verbose: bool = True

    def __post_init__(self):
        if self.precision != "float64":
            raise ValueError(
                f"Unsupported precision '{self.precision}'. BFG requires strict float64 arithmetic."
            )
        if self.budget_models <= 0:
            raise ValueError(f"budget_models must be positive, got {self.budget_models}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {self.beam_width}.")
        if self.always_prior not in ("shrink", "flat"):
            raise ValueError(f"always_prior must be 'shrink' or 'flat', got '{self.always_prior}'.")
        if self.allocation_strategy not in ("uniform", "posterior", "adaptive"):
            raise ValueError(
                f"allocation_strategy must be 'uniform', 'posterior', or 'adaptive', "
                f"got '{self.allocation_strategy}'."
            )
        if self.budget_semantics not in ("hard", "sampling_only"):
            raise ValueError(
                f"budget_semantics must be 'hard' or 'sampling_only', got '{self.budget_semantics}'."
            )
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.checkpoint_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a serializable dictionary."""
        d = asdict(self)
        if d["checkpoint_dir"] is not None:
            d["checkpoint_dir"] = str(d["checkpoint_dir"])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BFGConfig:
        """Construct BFGConfig from a dictionary."""
        return cls(**data)

    def save_json(self, path: Union[str, Path]) -> None:
        """Serialize configuration to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> BFGConfig:
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
