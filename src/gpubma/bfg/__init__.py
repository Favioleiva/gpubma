"""BFG (Budgeted Fast GPU) Bayesian Model Averaging Subpackage."""

from gpubma.bfg.acesm import ACESMFitResult, ACESMReconstructor, CumulativeCurveBuilder, CumulativeEvidenceCurve
from gpubma.bfg.allocation import BudgetAllocator
from gpubma.bfg.checkpoint import CheckpointManager, CheckpointState
from gpubma.bfg.config import BFGConfig
from gpubma.bfg.elite_search import GPUEliteSearch, ThresholdEstimator
from gpubma.bfg.engine import BFGEngine, fit_bfg
from gpubma.bfg.genealogy import GenealogicalSearch
from gpubma.bfg.registry import EliteRegistry, ModelProvenance, ModelRecord
from gpubma.bfg.results import BFGResult, LatticeResult
from gpubma.bfg.sampling import ExactWingEnumerator, LatticeSampler
from gpubma.bfg.scorer import BFGScorer, count_set_bits, hamming_distance

# Alias `fit` to `fit_bfg`
fit = fit_bfg

__all__ = [
    "fit",
    "fit_bfg",
    "BFGConfig",
    "BFGResult",
    "LatticeResult",
    "BFGEngine",
    "BFGScorer",
    "EliteRegistry",
    "ModelProvenance",
    "ModelRecord",
    "LatticeSampler",
    "ExactWingEnumerator",
    "GPUEliteSearch",
    "ThresholdEstimator",
    "GenealogicalSearch",
    "ACESMReconstructor",
    "CumulativeCurveBuilder",
    "CumulativeEvidenceCurve",
    "ACESMFitResult",
    "BudgetAllocator",
    "CheckpointManager",
    "CheckpointState",
    "count_set_bits",
    "hamming_distance",
]
