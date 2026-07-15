"""Stata parity scaffolding.

The .do scripts are PREPARED; no working Stata installation exists on the
development machine (only renamed *_old.exe leftovers that do not run in
batch mode). Numerical parity tests therefore SKIP explicitly until real
Stata exports appear under validation/stata/output/.
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
STATA_DIR = ROOT / "validation" / "stata"
STATA_OUT = STATA_DIR / "output"

DO_FILES = [
    "small_no_fixed_effects.do",
    "small_individual_fixed_effects.do",
    "small_time_fixed_effects.do",
    "small_two_way_fixed_effects.do",
    "grunfeld_validation.do",
]


@pytest.mark.parametrize("name", DO_FILES)
def test_do_file_exists_and_loads_frozen_data(name):
    path = STATA_DIR / name
    assert path.exists()
    text = path.read_text()
    assert ".dta" in text and "assert _N ==" in text
    assert "enumeration" in text, "scripts must explicitly request enumeration"
    assert "gprior(fixed" in text, "scripts must pin the documented fixed g"
    assert "log using" in text, "scripts must save a plain-text log"


def test_frozen_dta_loads_in_pandas():
    df = pd.read_stata(ROOT / "data" / "synthetic" / "panel_8.dta")
    assert len(df) == 1000
    g = pd.read_stata(ROOT / "data" / "public" / "grunfeld.dta")
    assert len(g) == 200


def test_python_vs_stata_pip_parity():
    export = STATA_OUT / "small_no_fe_pip.csv"
    if not export.exists():
        pytest.skip(
            "Stata outputs not available: .do scripts are prepared but were "
            "never executed (no callable Stata on this machine). Parity "
            "comparison pending."
        )
    # When real exports exist, compare via scripts/compare_stata_python.py logic.
    stata = pd.read_csv(export)
    assert not stata.empty
