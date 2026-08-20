"""GPUBMA: exhaustive Bayesian Model Averaging for linear regression.

Phase 1 provides an exact float64 CPU reference implementation, deterministic
datasets, diagnostics, and a GPU feasibility layer. The production CUDA
enumerator is intentionally not implemented yet.
"""

from gpubma.api import bma_regress
from gpubma.estimator import GPUBMARegressor
from gpubma.plots import generate_canonical_figures
from gpubma.result import BMAResult

__version__ = "0.1.0.dev0"

__all__ = [
    "bma_regress",
    "GPUBMARegressor",
    "BMAResult",
    "generate_canonical_figures",
    "__version__",
]
