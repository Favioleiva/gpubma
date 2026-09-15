"""Canonical types, constants, and data structures for BFG128 representation.

This module defines the 128-bit model identity representation for BFG:
    (mask_lo, mask_hi)
where mask_lo and mask_hi are unsigned 64-bit integers:
    mask_lo : uint64, bits 0..63   (predictor indices 0..63)
    mask_hi : uint64, bits 64..127 (predictor indices 64..127)

No signed reinterpretation.
No string representations as internal keys.
No 128-bit big-integer objects for internal indexing.
"""

from typing import Tuple, Union, Optional
import numpy as np

# Canonical type alias for scalar 128-bit model mask in Python
Mask128 = Tuple[int, int]

# Bit width constants
UINT64_BITS: int = 64
UINT64_MAX: int = 0xFFFF_FFFF_FFFF_FFFF
MAX_P: int = 128
MIN_WIDE_P: int = 61
LEGACY_MAX_P: int = 60

# NumPy structured dtype for 128-bit model representation
# Allows contiguous tabular storage and fast vectorized operations
DTYPE_MODEL_128 = np.dtype([("lo", np.uint64), ("hi", np.uint64)])


def validate_mask128(mask_lo: int, mask_hi: int, p: Optional[int] = None) -> None:
    """Validate that (mask_lo, mask_hi) are valid uint64 words and respect dimension p.

    Parameters
    ----------
    mask_lo : int
        Low 64-bit word (variables 0..63).
    mask_hi : int
        High 64-bit word (variables 64..127).
    p : int, optional
        Number of candidate predictors (1 <= p <= 128). If provided, verifies that
        no active bits are at or above index p.

    Raises
    ------
    TypeError
        If mask_lo or mask_hi are not integer types.
    ValueError
        If words are outside [0, 2^64 - 1], or if bits >= p are active.
    """
    if not isinstance(mask_lo, (int, np.integer)):
        raise TypeError(f"mask_lo must be an integer, got {type(mask_lo).__name__}")
    if not isinstance(mask_hi, (int, np.integer)):
        raise TypeError(f"mask_hi must be an integer, got {type(mask_hi).__name__}")

    lo_val = int(mask_lo)
    hi_val = int(mask_hi)

    if lo_val < 0 or lo_val > UINT64_MAX:
        raise ValueError(f"mask_lo={lo_val} is out of unsigned 64-bit range [0, 0x{UINT64_MAX:x}]")
    if hi_val < 0 or hi_val > UINT64_MAX:
        raise ValueError(f"mask_hi={hi_val} is out of unsigned 64-bit range [0, 0x{UINT64_MAX:x}]")

    if p is not None:
        if p < 1 or p > MAX_P:
            raise ValueError(f"Predictor count p={p} must be in range [1, {MAX_P}]")
        if p < 64:
            if hi_val != 0:
                raise ValueError(
                    f"mask_hi must be 0 when p={p} < 64, but got 0x{hi_val:016x}"
                )
            if (lo_val >> p) != 0:
                raise ValueError(
                    f"mask_lo has active bits >= p={p}: 0x{lo_val:016x}"
                )
        elif p < 128:
            bits_in_hi = p - 64
            if (hi_val >> bits_in_hi) != 0:
                raise ValueError(
                    f"mask_hi has active bits >= p={p} (hi bit index >= {bits_in_hi}): 0x{hi_val:016x}"
                )


def mask_to_hex(mask_lo: int, mask_hi: int) -> str:
    """Format a 128-bit mask as a human-readable canonical hex string for logging/display.

    Note: This is strictly for reporting and debugging, never for internal dictionary keys.
    """
    return f"0x{int(mask_hi):016x}_{int(mask_lo):016x}"


def mask_from_hex(s: str) -> Tuple[int, int]:
    """Parse a canonical hex string formatted as '0x<hi>_<lo>' into (mask_lo, mask_hi)."""
    s_clean = s.strip()
    if s_clean.startswith("0x") or s_clean.startswith("0X"):
        s_clean = s_clean[2:]
    parts = s_clean.split("_")
    if len(parts) == 1:
        val = int(parts[0], 16)
        return (val & UINT64_MAX, (val >> 64) & UINT64_MAX)
    elif len(parts) == 2:
        hi = int(parts[0], 16)
        lo = int(parts[1], 16)
        return (lo & UINT64_MAX, hi & UINT64_MAX)
    else:
        raise ValueError(f"Cannot parse hex mask: {s}")
