import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA_SYNTHETIC = ROOT / "data" / "synthetic"
DATA_PUBLIC = ROOT / "data" / "public"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def panel8() -> pd.DataFrame:
    return pd.read_parquet(DATA_SYNTHETIC / "panel_8.parquet")


@pytest.fixture(scope="session")
def frozen_paths() -> dict:
    return {
        p: {fmt: DATA_SYNTHETIC / f"panel_{p}.{fmt}" for fmt in ("csv", "parquet", "dta")}
        for p in (8, 12, 30)
    }
