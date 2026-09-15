"""BFG128 Native 128-bit Large Model Space Engine.

This package provides native (mask_lo, mask_hi) model representation, high-performance
scalar and vectorized bit operations, direct uniform shell samplers, wide elite registry,
multi-path genealogy, GPU elite search, and stratified shell recovery for large model
spaces (61 <= p <= 128).
"""

from gpubma.bfg.bfg128.types import (
    DTYPE_MODEL_128,
    LEGACY_MAX_P,
    MAX_P,
    MIN_WIDE_P,
    UINT64_BITS,
    UINT64_MAX,
    Mask128,
    mask_from_hex,
    mask_to_hex,
    validate_mask128,
)
from gpubma.bfg.bfg128.bitops import (
    add_bit,
    batch_get_children,
    batch_get_parents,
    contains,
    from_structured_array,
    get_children,
    get_parents,
    pack_indices,
    pack_indices_batch,
    popcount,
    popcount_batch,
    remove_bit,
    to_structured_array,
    unpack_mask,
    unpack_mask_batch,
)
from gpubma.bfg.bfg128.sampling import (
    DirectUniformShellSampler128,
    SampledBatch128,
    sample_shell_128,
)
from gpubma.bfg.bfg128.registry import (
    WideEliteRegistry,
    WideModelRecord,
)
from gpubma.bfg.bfg128.genealogy import (
    SearchTrajectoryResult128,
    WideGenealogicalSearch,
)
from gpubma.bfg.bfg128.elite_search import (
    WideEliteSearch,
    WideEliteSearchResult,
)
from gpubma.bfg.bfg128.recovery import (
    ShellAccumulator128,
    ShellRecoveryResult128,
    WideShellRecovery,
)
from gpubma.bfg.bfg128.results import (
    BFG128Result,
)
from gpubma.bfg.bfg128.engine import (
    BFG128Engine,
)

__all__ = [
    # Types and constants
    "Mask128",
    "DTYPE_MODEL_128",
    "UINT64_BITS",
    "UINT64_MAX",
    "MAX_P",
    "MIN_WIDE_P",
    "LEGACY_MAX_P",
    "validate_mask128",
    "mask_to_hex",
    "mask_from_hex",
    # Scalar bit operations
    "pack_indices",
    "unpack_mask",
    "popcount",
    "contains",
    "add_bit",
    "remove_bit",
    "get_parents",
    "get_children",
    # Batch bit operations
    "popcount_batch",
    "pack_indices_batch",
    "unpack_mask_batch",
    "batch_get_parents",
    "batch_get_children",
    "to_structured_array",
    "from_structured_array",
    # Sampler
    "DirectUniformShellSampler128",
    "SampledBatch128",
    "sample_shell_128",
    # Registry
    "WideEliteRegistry",
    "WideModelRecord",
    # Genealogy
    "WideGenealogicalSearch",
    "SearchTrajectoryResult128",
    # Elite Search
    "WideEliteSearch",
    "WideEliteSearchResult",
    # Recovery
    "WideShellRecovery",
    "ShellRecoveryResult128",
    "ShellAccumulator128",
    # Results
    "BFG128Result",
    # Engine
    "BFG128Engine",
]
