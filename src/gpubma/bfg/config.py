"""Configuration for reproducible, hard-budget BFG discovery."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

@dataclass
class BFGConfig:
    """Search settings. Every stage shares budget_models; float64 is required.

    recon_sample_per_lattice controls reconnaissance allocation, not posterior
    reconstruction. allocation_strategy='posterior' is a historical name for
    observed-score allocation priorities, not a global posterior estimate.
    Checkpoints save at completed phase/lattice boundaries, not exactly at N.
    CUDA falls back to CPU when unavailable; inspect result.hardware.
    """
    budget_models: int = 100_000
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
    beam_width: int = 15
    allocation_strategy: str = "adaptive"
    budget_semantics: str = "hard"
    checkpoints: Optional[List[int]] = None
    checkpoint_dir: Optional[Union[str, Path]] = None
    resume: bool = False
    verbose: bool = True

    def __post_init__(self):
        for key in ('budget_models', 'batch_size', 'beam_width', 'wing_max_size',
                    'recon_sample_per_lattice', 'elite_calibration_size'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if not 0 < self.elite_quantile < 1:
            raise ValueError('elite_quantile must lie strictly between 0 and 1')
        if self.resume and self.checkpoint_dir is None:
            raise ValueError('resume requires checkpoint_dir')
        if self.device != 'cpu' and self.device != 'cuda' and not (
                self.device.startswith('cuda:') and self.device[5:].isdigit()):
            raise ValueError('device must be cpu, cuda or cuda:<index>')
        if self.checkpoints is not None and any(
                isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.checkpoints):
            raise ValueError('checkpoint thresholds must be positive integers')
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
        if self.budget_semantics not in ("hard",):
            raise ValueError(
                f"budget_semantics must be 'hard' in the discovery API, got '{self.budget_semantics}'."
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
