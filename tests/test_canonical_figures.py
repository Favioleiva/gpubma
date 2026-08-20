import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

import pytest

from gpubma.plots import (
    CANONICAL_FIGURE_FILENAMES,
    CanonicalFigureError,
    _load_inputs,
    _write_manifest,
    generate_canonical_figures,
)


@pytest.fixture(scope="module")
def generated(repo_root):
    work = Path(tempfile.mkdtemp(prefix="gpubma-canonical-plots-"))
    source = repo_root / "reports" / "artifacts" / "panel_30_center15_exact_results.zip"
    try:
        manifest = generate_canonical_figures(source, work, dpi=20)
        yield source, work, manifest
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_generates_complete_canonical_set_and_manifest(generated):
    _, output, manifest_path = generated
    actual_pngs = sorted(p.name for p in output.glob("*.png"))
    assert actual_pngs == sorted(CANONICAL_FIGURE_FILENAMES)
    assert set(p.name for p in output.iterdir()) == set(actual_pngs) | {
        "panel_30_center15_figure_manifest.json"
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["figures"]
    assert [r["filename"] for r in records] == actual_pngs
    assert len(records) == 23
    for record in records:
        path = output / record["filename"]
        assert path.stat().st_size == record["size_bytes"] > 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_manifest_rejects_unexpected_png(generated):
    source, output, _ = generated
    extra = output / "unexpected.png"
    extra.write_bytes(b"not a canonical figure")
    try:
        with pytest.raises(CanonicalFigureError, match="extra=.*unexpected.png"):
            _write_manifest(_load_inputs(source), output)
    finally:
        extra.unlink()


def test_accepts_extracted_results_directory(generated):
    source, _, _ = generated
    extracted = Path(tempfile.mkdtemp(prefix="gpubma-extracted-results-"))
    try:
        with ZipFile(source) as archive:
            archive.extractall(extracted)
        data = _load_inputs(extracted)
        assert data.dataset == "panel_30_center15"
        assert data.n_models == 1_073_741_824
    finally:
        shutil.rmtree(extracted, ignore_errors=True)
