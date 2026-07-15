import numpy as np
import pytest

from gpubma.datasets.io_utils import compare_frames_exact, read_any


@pytest.mark.parametrize("p", [8, 12, 30])
def test_round_trips_exact_across_formats(frozen_paths, p):
    frames = {fmt: read_any(path) for fmt, path in frozen_paths[p].items()}
    ref = frames["parquet"]
    for fmt in ("csv", "dta"):
        rep = compare_frames_exact(ref, frames[fmt], float_atol=0.0)
        assert rep["pass"], f"panel_{p} parquet vs {fmt}: {rep}"


def test_float_columns_bit_identical(frozen_paths):
    a = read_any(frozen_paths[8]["csv"])
    b = read_any(frozen_paths[8]["parquet"])
    for col in ("y", "x1", "x8", "w2"):
        assert np.array_equal(a[col].to_numpy(), b[col].to_numpy()), col
