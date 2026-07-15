"""Stata artifact structure checks.

The .do scripts were EXECUTED in batch mode on StataNow/SE 19.5
(2026-07-15); real exports live under validation/stata/output/.
Numerical parity is covered by test_stata_parity.py.
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


def test_stata_exports_exist_and_are_license_free():
    """Exports exist for all six designs and contain no license information."""
    stems = ["small_no_fe", "small_individual_fe", "small_time_fe",
             "small_two_way_fe", "grunfeld_no_fe", "grunfeld_company_fe"]
    for stem in stems:
        for kind in ("b_bma", "pip", "v_diag", "scalars"):
            path = STATA_OUT / f"{stem}_{kind}.csv"
            assert path.exists(), f"missing Stata export {path.name}"
    for path in STATA_OUT.iterdir():
        text = path.read_text(errors="ignore").lower()
        assert "serial number" not in text and "licensed to" not in text, (
            f"license information leaked into {path.name}")
