"""BFG discovery and separate exhaustive BMA reference APIs."""
from gpubma.adapters import load_bma_run_as_inputs
from gpubma.api import bma_regress
from gpubma.bfg import fit_bfg, BFGResult, BFGConfig
from gpubma.estimator import GPUBMARegressor
from gpubma.fixed_effects.design import two_way_residualize, within_transform
from gpubma.gpu.enumerator import enumerate_models_gpu
from gpubma.gpu.structured import (
    enumerate_structured_models_gpu,
    get_translog_heredity_masks,
)
from gpubma.latex import (
    generate_fe_comparison_table,
    generate_latex_bundle,
    generate_master_manifest,
)
from gpubma.plots import (
    CANONICAL_8_FIGURE_FILENAMES,
    CANONICAL_FIGURE_FILENAMES,
    generate_canonical_figures,
    resolve_variable_names,
)
from gpubma.result import BMAResult

__version__ = "0.3.0rc1"

__all__ = [
    "bma_regress",
    "fit_bfg",
    "BFGResult",
    "BFGConfig",
    "GPUBMARegressor",
    "BMAResult",
    "enumerate_models_gpu",
    "enumerate_structured_models_gpu",
    "get_translog_heredity_masks",
    "two_way_residualize",
    "within_transform",
    "generate_canonical_figures",
    "generate_latex_bundle",
    "generate_fe_comparison_table",
    "generate_master_manifest",
    "load_bma_run_as_inputs",
    "resolve_variable_names",
    "CANONICAL_8_FIGURE_FILENAMES",
    "CANONICAL_FIGURE_FILENAMES",
    "__version__",
]
