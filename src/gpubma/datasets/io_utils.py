"""Serialization helpers: multi-format writing, checksums, round-trip checks.

Serialization tolerances (documented):
- CSV: written with float_format="%.17g" (17 significant digits), which is
  sufficient for exact float64 round-trip through decimal text (tolerance 0.0).
  pandas' default CSV float formatting is NOT exact and is not used.
- Parquet: float64 stored exactly (tolerance 0.0).
- Stata .dta (version 118): float64 stored as Stata double, exact
  (tolerance 0.0). Integer identifiers may come back as different integer
  widths; values compare exactly after casting.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_sha256(df: pd.DataFrame) -> str:
    """Order- and dtype-sensitive content hash of the numerical values."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    for col in df.columns:
        values = df[col].to_numpy()
        if values.dtype.kind in "iu":
            values = values.astype(np.int64)
        elif values.dtype.kind == "f":
            values = values.astype(np.float64)
        else:
            h.update("|".join(map(str, values)).encode())
            continue
        h.update(np.ascontiguousarray(values).tobytes())
    return h.hexdigest()


def write_all_formats(df: pd.DataFrame, base_path, stem: str) -> dict:
    """Write ``<stem>.csv/.parquet/.dta`` under base_path; return checksums."""
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": base / f"{stem}.csv",
        "parquet": base / f"{stem}.parquet",
        "dta": base / f"{stem}.dta",
    }
    # %.17g guarantees exact float64 round-trip through decimal text
    df.to_csv(paths["csv"], index=False, float_format="%.17g")
    df.to_parquet(paths["parquet"], index=False)
    df.to_stata(paths["dta"], write_index=False, version=118)
    return {fmt: {"path": str(p), "sha256": file_sha256(p)} for fmt, p in paths.items()}


def read_any(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        # round_trip parser: exact IEEE-754 decimal-to-double conversion
        return pd.read_csv(path, float_precision="round_trip")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".dta":
        return pd.read_stata(path)
    raise ValueError(f"unsupported format: {path.suffix}")


def compare_frames_exact(a: pd.DataFrame, b: pd.DataFrame, float_atol: float = 0.0) -> dict:
    """Compare two frames column by column; returns a small report dict."""
    report = {"equal_shape": a.shape == b.shape, "columns_match": list(a.columns) == list(b.columns),
              "max_abs_diff": {}, "pass": True}
    if not (report["equal_shape"] and report["columns_match"]):
        report["pass"] = False
        return report
    for col in a.columns:
        av, bv = a[col].to_numpy(), b[col].to_numpy()
        if av.dtype.kind == "f" or bv.dtype.kind == "f":
            diff = float(np.max(np.abs(av.astype(np.float64) - bv.astype(np.float64)))) if len(av) else 0.0
            report["max_abs_diff"][col] = diff
            if diff > float_atol:
                report["pass"] = False
        else:
            same = bool((av.astype(np.int64) == bv.astype(np.int64)).all()) if av.dtype.kind in "iu" else bool((av == bv).all())
            report["max_abs_diff"][col] = 0.0 if same else float("inf")
            report["pass"] = report["pass"] and same
    return report
