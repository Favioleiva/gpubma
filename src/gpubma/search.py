"""Model-space strategy policy, independent of sample size and always-in terms."""
from numbers import Integral

MAX_EXACT_P = 32
# Storage limits of the legacy all-scores reference, not strategy boundaries.
MAX_REFERENCE_P = 20


def validate_exact_p(p: int) -> None:
    """Validate selectable dimension; the low-level null-only universe is valid."""
    if isinstance(p, bool) or not isinstance(p, Integral) or p < 0:
        raise ValueError("p must be a nonnegative integer count of selectable regressors")
    if p > MAX_EXACT_P:
        raise ValueError(
            f"Exact enumeration supports at most {MAX_EXACT_P} selectable regressors. "
            f"Received p={p}. Use search='bfg' or search='auto'."
        )


def resolve_search(p: int, search: str = "auto") -> str:
    """Return 'exact' or 'bfg' using selectable p, never observation count n.

    Controls, intercepts and fixed effects are always-in terms, not selectable
    candidates. BFG retains its own input/backend limits and validation.
    """
    if isinstance(p, bool) or not isinstance(p, Integral) or p < 1:
        raise ValueError("p must be a positive integer count of selectable regressors")
    if search not in ("auto", "exact", "bfg"):
        raise ValueError("search must be 'auto', 'exact', or 'bfg'")
    if search == "auto":
        return "exact" if p <= MAX_EXACT_P else "bfg"
    if search == "exact":
        validate_exact_p(p)
    return search
