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


def test_drive_optional_and_disabled_by_default():
    code_cells = [_src(c) for c in _code_cells(_nb())]
    code = "\n".join(code_cells)
    assert re.search(r"^USE_GOOGLE_DRIVE\s*=\s*False", code, re.M)
    # Drive functionality may appear only inside a guarded (indented) branch
    # of a cell that tests USE_GOOGLE_DRIVE — never unconditionally.
    for src in code_cells:
        for line in src.splitlines():
            if line.lstrip().startswith("#"):
                continue  # comments may mention Drive (e.g. "never touches")
            if "drive.mount" in line or "/content/drive" in line:
                assert line[:1] in (" ", "\t"), f"unguarded Drive use: {line!r}"
                assert "if USE_GOOGLE_DRIVE" in src, "Drive use outside guard"
    assert re.search(r"^RESTORE_CHECKPOINT_UPLOAD\s*=\s*False", code, re.M)


def test_local_storage_outside_git_clone():
    code = "\n".join(_src(c) for c in _code_cells(_nb()))
    m = re.search(r"^LOCAL_RUN_ROOT\s*=\s*'([^']+)'", code, re.M)
    assert m, "LOCAL_RUN_ROOT missing"
    root = m.group(1)
    assert root.startswith("/content/")
    assert not root.startswith("/content/gpubma"), "run root inside the clone"
    assert "def export_checkpoint_bundle" in code
    assert "def download_checkpoint_bundle" in code
    assert "def restore_checkpoint_bundle" in code
    assert "def _safe_extract_zip" in code


def test_checkpoint_bundle_functions_execute_safely(tmp_path=None):
    """Exec the checkpoint-tools and restore cells and verify: empty export
    is a no-op, ZIP path traversal is rejected, and an incompatible bundle
    manifest is rejected before any checkpoint file is restored."""
    import shutil, tempfile, zipfile
    work = Path(tempfile.mkdtemp(prefix="gpubma-nbtest-"))
    try:
        cells = [_src(c) for c in _code_cells(_nb())]
        tools = [s for s in cells if "def export_checkpoint_bundle" in s]
        restore = [s for s in cells if "def restore_checkpoint_bundle" in s]
        assert len(tools) == 1 and len(restore) == 1
        ns = {
            "PASS1_CKPT": work / "enum.ckpt.npz",
            "PASS2_CKPT": work / "pass2.ckpt.npz",
            "CKPT_DIR": work, "TMP_DIR": work, "WORK_DIR": work,
            "PARQUET_SHA256": "a" * 64, "METADATA_SHA256": "b" * 64,
            "RESOLVED_COMMIT": "c" * 40, "P": 30, "n": 2000, "G": 2000.0,
            "MODEL_PRIOR": ("betabinomial", 1.0, 1.0), "DTYPE": "float64",
            "TOP_K_MODELS": 10000, "GRID_POINTS": 257,
            "COMPUTE_COEF_DENSITIES": True,
        }
        exec(tools[0], ns)
        exec(restore[0], ns)
        assert ns["export_checkpoint_bundle"]() is None  # nothing to export
        evil = work / "evil.zip"
        with zipfile.ZipFile(evil, "w") as z:
            z.writestr("../escape.txt", "x")
        try:
            ns["_safe_extract_zip"](evil, work / "dest")
            raise AssertionError("path traversal was accepted")
        except ValueError:
            pass
        wrong = work / "wrong.zip"
        with zipfile.ZipFile(wrong, "w") as z:
            z.writestr("checkpoint_bundle_manifest.json", json.dumps(
                {"schema_version": 1, "parquet_sha256": "MISMATCH",
                 "stages": {}}))
        try:
            ns["restore_checkpoint_bundle"](wrong)
            raise AssertionError("incompatible bundle was accepted")
        except ValueError as exc:
            assert "INCOMPATIBLE" in str(exc)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_expected_seed_and_dataset_constants():
    code = "\n".join(_src(c) for c in _code_cells(_nb()))
    assert "EXPECTED_SEED = 20260724" in code
    assert "TOP_K_MODELS = 10000" in code
    assert "colab.research.google.com" in "\n".join(
        _src(c) for c in _nb()["cells"] if c["cell_type"] == "markdown")
