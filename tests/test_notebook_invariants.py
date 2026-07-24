"""Static invariants of the canonical Colab notebook (no execution).

Guards the published A100 workflow: valid .ipynb JSON, every code cell
parses as pure Python (no shell magics), the active benchmark is
panel_30_center15 (no stale reference to the old sparse panel_30 artifact
in code cells), the expensive run and every dangerous override default to
disabled, the pinned commit is a real 40-hex SHA, no local Windows paths,
no credential patterns, no stored outputs, and checkpoint files stay
ignored by git.
"""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "GPUBMA_A100_p30.ipynb"


def _nb():
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _src(cell) -> str:
    return "".join(cell["source"])


def test_valid_ipynb_json_v4():
    nb = _nb()
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 10
    for c in nb["cells"]:
        assert c["cell_type"] in ("code", "markdown")


def test_every_code_cell_parses_as_pure_python():
    for i, c in enumerate(_code_cells(_nb())):
        src = _src(c)
        assert not re.search(r"^\s*[!%]", src, re.M), f"cell {i}: shell/magic line"
        ast.parse(src)  # raises SyntaxError on failure


def test_outputs_cleared_and_no_execution_counts():
    for c in _code_cells(_nb()):
        assert c.get("outputs") == [], "stored outputs must be cleared"
        assert c.get("execution_count") in (None, 0)


def test_active_cells_use_center15_not_old_panel_30():
    code = "\n".join(_src(c) for c in _code_cells(_nb()))
    assert "panel_30_center15.parquet" in code
    assert "panel_30_center15_metadata.json" in code
    # the OLD sparse benchmark artifacts must not be loaded by any code cell
    assert re.search(r"panel_30(?!_center15)\.parquet", code) is None
    assert re.search(r"panel_30(?!_center15)_metadata", code) is None
    assert "panel_8" not in code and "panel_12" not in code


def test_expensive_run_and_overrides_default_disabled():
    code = "\n".join(_src(c) for c in _code_cells(_nb()))
    assert re.search(r"^RUN_FULL_EXACT_P30\s*=\s*False", code, re.M)
    assert re.search(r"^EXPERT_OVERRIDE_SKIP_SMOKE_GATE\s*=\s*False", code, re.M)
    assert "RUN_FULL_EXACT_P30 = True" not in code.replace(
        "set RUN_FULL_EXACT_P30 = True", "")  # only mentioned in instructions


def test_smoke_gate_wired_to_expensive_cell():
    cells = [_src(c) for c in _code_cells(_nb())]
    expensive = [s for s in cells if "EXPENSIVE" in s or "RUN_FULL_EXACT_P30" in s]
    gate = [s for s in expensive if "SMOKE_TEST_PASSED" in s
            and "enumerate_models_gpu" in s]
    assert gate, "the expensive cell must check SMOKE_TEST_PASSED"


def test_pinned_commit_is_real_sha():
    code = "\n".join(_src(c) for c in _code_cells(_nb()))
    m = re.search(r"^PINNED_COMMIT\s*=\s*'([0-9a-f]{40})'", code, re.M)
    assert m, "PINNED_COMMIT must be a resolved 40-hex SHA, not a placeholder"


def test_no_local_windows_paths_or_credentials():
    raw = NB_PATH.read_text(encoding="utf-8")
    assert "X:\\\\" not in raw and "X:/Claude" not in raw
    assert not re.search(r"[A-Z]:\\\\Users", raw)
    for pat in ("ghp_", "github_pat_", "AKIA", "BEGIN PRIVATE KEY",
                "hf_", "api_key"):
        assert pat not in raw, f"credential-like pattern {pat!r}"


def test_float64_only_and_exact_enumeration_language():
    code = "\n".join(_src(c) for c in _code_cells(_nb()))
    assert "DTYPE = 'float64'" in code
    assert "1 << 30" in code or "2**30" in code or "1_073_741_824" in code
    # the only enumeration entry points are the exact CPU/GPU enumerators
    assert "enumerate_models_gpu" in code and "enumerate_models" in code
    for banned in ("import pymc", "MC3(", "metropolis", "gibbs"):
        assert banned.lower() not in code.lower()


def test_checkpoints_cannot_be_committed():
    gitignore = (ROOT / ".gitignore").read_text()
    assert "*.ckpt.npz" in gitignore
    code = "\n".join(_src(c) for c in _code_cells(_nb()))
    assert ".ckpt.npz" in code  # notebook uses the ignored suffix


def test_expected_seed_and_dataset_constants():
    code = "\n".join(_src(c) for c in _code_cells(_nb()))
    assert "EXPECTED_SEED = 20260724" in code
    assert "TOP_K_MODELS = 10000" in code
    assert "colab.research.google.com" in "\n".join(
        _src(c) for c in _nb()["cells"] if c["cell_type"] == "markdown")
