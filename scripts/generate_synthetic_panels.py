"""Regenerate the frozen synthetic panel datasets (8, 12, 30 predictors).

Deterministic: seed 20260715, 100 individuals x 10 periods. Writes CSV,
Parquet, and Stata .dta plus a metadata JSON per dataset, and validates
serialization round trips and cross-size predictor nesting.

Usage:  python scripts/generate_synthetic_panels.py
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpubma.datasets.io_utils import (  # noqa: E402
    compare_frames_exact, frame_sha256, read_any, write_all_formats,
)
from gpubma.datasets.synthetic import DEFAULT_SEED, generate_panel  # noqa: E402

N_INDIVIDUALS, N_PERIODS = 100, 10
SIZES = (8, 12, 30)
OUT = ROOT / "data" / "synthetic"


def main() -> int:
    frames = {}
    for p in SIZES:
        df, meta = generate_panel(N_INDIVIDUALS, N_PERIODS, p, seed=DEFAULT_SEED)
        frames[p] = df
        stem = f"panel_{p}"
        checks = write_all_formats(df, OUT, stem)

        # round-trip validation (documented tolerances: exact for all formats)
        round_trips = {}
        for fmt in ("csv", "parquet", "dta"):
            back = read_any(checks[fmt]["path"])
            rep = compare_frames_exact(df, back, float_atol=0.0)
            round_trips[fmt] = rep
            assert rep["pass"], f"{stem}.{fmt} round trip failed: {rep}"

        meta.update({
            "content_sha256": frame_sha256(df),
            "file_checksums": checks,
            "round_trip_exact": {k: v["pass"] for k, v in round_trips.items()},
            "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "package_versions": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "pyarrow")
            },
            "python_version": sys.version.split()[0],
        })
        (OUT / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"{stem}: {len(df)} rows, {p} predictors, 2^{p} = {2**p:,} models; "
              f"content sha256 {meta['content_sha256'][:16]}…")

    # cross-size nesting: first 8 predictors and y identical across datasets
    base = frames[8]
    for p in (12, 30):
        for col in ["individual_id", "period", "y", "w1", "w2"] + [f"x{j}" for j in range(1, 9)]:
            same = np.array_equal(base[col].to_numpy(), frames[p][col].to_numpy())
            assert same, f"column {col} differs between panel_8 and panel_{p}"
    print("nesting check: first 8 predictors, y, controls identical across 8/12/30 [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
