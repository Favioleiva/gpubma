"""Tests for BFG Command-Line Interface."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from gpubma.bfg.cli import main


def test_cli_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        rng = np.random.default_rng(20260715)
        n, p = 100, 6
        X = rng.normal(size=(n, p))
        y = X[:, 0] * 2.0 + rng.normal(size=n)
        cols = [f"x{j + 1}" for j in range(p)]

        df = pd.DataFrame(X, columns=cols)
        df["target"] = y

        data_file = tmp_path / "test_data.csv"
        out_file = tmp_path / "summary.txt"
        df.to_csv(data_file, index=False)

        exit_code = main([
            "--data", str(data_file),
            "--outcome", "target",
            "--budget", "500",
            "--out", str(out_file),
        ])

        assert exit_code == 0
        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "GPUBMA BFG Bayesian Model Averaging" in content
        assert "target" in content
