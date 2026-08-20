import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

import pytest

import gpubma.plots as plot_module
from gpubma.plots import (
    CANONICAL_FIGURE_FILENAMES,
    CanonicalFigureError,
    _load_inputs,
    _write_manifest,
    generate_canonical_figures,
    resolve_variable_names,
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


def test_variable_name_fallback_and_stored_names(generated):
    source, _, _ = generated
    assert resolve_variable_names(4) == ["x1", "x2", "x3", "x4"]
    assert _load_inputs(source).pip["variable"].tolist() == [
        f"x{i}" for i in range(1, 31)
    ]


def test_custom_variable_names_keep_position_and_appear_in_figures(
    generated, monkeypatch
):
    source, _, _ = generated
    custom = [
        "Initial income",
        "Mining employment",
        "W × Mining employment",
        "Capital stock",
        "W × Capital stock",
    ] + [f"Scientific control ({i})" for i in range(6, 31)]
    output = Path(tempfile.mkdtemp(prefix="gpubma-custom-labels-"))
    seen_text = {}
    original_save = plot_module._save

    def save_spy(fig, path, dpi, plt):
        figure_text = set()
        for item in fig.findobj():
            get_text = getattr(item, "get_text", None)
            if get_text is not None:
                figure_text.add(get_text().replace("\n", " "))
        seen_text[path.name] = figure_text
        original_save(fig, path, dpi, plt)

    monkeypatch.setattr(plot_module, "_save", save_spy)
    try:
        manifest_path = generate_canonical_figures(
            source, output, variable_names=custom, dpi=20
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["variable_names"] == custom
        for filename in (
            "pip.png",
            "varmap_equal_width.png",
            "varmap_proportional.png",
            "coefsummary_ordered.png",
            "coefsummary_ordered_exact_quantiles.png",
            "coefridgeline_ordered.png",
            "corrmap.png",
        ):
            assert set(custom).issubset(seen_text[filename]), filename
        labelled = _load_inputs(source, variable_names=custom)
        top_positions = (-labelled.pip["pip"].to_numpy()).argsort()[:10]
        top_names = [custom[position] for position in top_positions]
        assert all(
            any(name in text for text in seen_text["coefdensity_top10.png"])
            for name in top_names
        )
        for rank, (position, name) in enumerate(zip(top_positions, top_names), start=1):
            filename = f"coefdensity_slide_{rank:02d}_x{position + 1}.png"
            assert name in seen_text[filename], filename
        assert sorted(p.name for p in output.glob("*.png")) == sorted(
            CANONICAL_FIGURE_FILENAMES
        )
    finally:
        shutil.rmtree(output, ignore_errors=True)


@pytest.mark.parametrize("count", [3, 5])
def test_custom_variable_name_count_must_match_p(count):
    with pytest.raises(ValueError, match=f"exactly 4 names; received {count}"):
        resolve_variable_names(4, variable_names=[f"name {i}" for i in range(count)])
