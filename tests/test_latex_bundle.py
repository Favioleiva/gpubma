import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import pytest

import gpubma
from gpubma.latex import (
    generate_fe_comparison_table,
    generate_latex_bundle,
    generate_master_manifest,
)
from gpubma.plots import CANONICAL_8_FIGURE_FILENAMES, CanonicalFigureError


@pytest.fixture(scope="module")
def bundle_output(repo_root):
    work = Path(tempfile.mkdtemp(prefix="gpubma-latex-bundle-"))
    source = repo_root / "reports" / "artifacts" / "panel_30_center15_exact_results.zip"
    run_id = "test_run_c8_twfe"
    try:
        bundle_path = generate_latex_bundle(
            source,
            work,
            run_id=run_id,
            figure_path="Chapter2/figs/gpubma/test_run_c8_twfe",
            label_prefix="fig:c8-twfe",
            source="Author's calculations using GPUBMA.",
            style="phd_thesis_fl",
            role="main_text_canonical",
            dpi=20,
            suppress_titles=True,
        )
        yield source, work, bundle_path, run_id
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_bundle_structure_and_figure_generation(bundle_output):
    _, _, bundle_path, run_id = bundle_output
    assert bundle_path.exists()
    assert (bundle_path / "figures").is_dir()
    assert (bundle_path / "latex" / "individual").is_dir()
    assert (bundle_path / "metadata").is_dir()

    actual_pngs = sorted(p.name for p in (bundle_path / "figures").glob("*.png"))
    assert actual_pngs == sorted(CANONICAL_8_FIGURE_FILENAMES)
    assert len(actual_pngs) == 8

    for fname in actual_pngs:
        p = bundle_path / "figures" / fname
        assert p.stat().st_size > 0


def test_individual_latex_snippets(bundle_output):
    _, _, bundle_path, _ = bundle_output
    indiv_dir = bundle_path / "latex" / "individual"
    tex_files = sorted(p.name for p in indiv_dir.glob("*.tex"))
    assert len(tex_files) == 8

    for tex_file in tex_files:
        content = (indiv_dir / tex_file).read_text(encoding="utf-8")
        assert "\\begin{figure}[htbp!]" in content
        assert "\\end{figure}" in content
        assert "\\caption[" in content
        assert "\\label{fig:c8-twfe-" in content
        assert "Chapter2/figs/gpubma/test_run_c8_twfe" in content
        assert "\\\\" not in content.split("includegraphics")[1].split("}")[0]  # No Windows backslashes in path
        assert "Note:" in content
        assert "Source: Author's calculations using GPUBMA." in content


def test_canonical_panel_and_combined_files(bundle_output):
    _, _, bundle_path, _ = bundle_output
    panel_file = bundle_path / "latex" / "canonical_panel.tex"
    assert panel_file.exists()
    panel_content = panel_file.read_text(encoding="utf-8")
    assert "\\begin{figure}[htbp!]" in panel_content
    assert "\\end{figure}" in panel_content
    assert "\\begin{subfigure}" in panel_content
    assert "\\label{fig:c8-twfe-canonical-panel}" in panel_content

    main_text = bundle_path / "latex" / "main_text.tex"
    assert main_text.exists()
    assert "\\begin{figure}" in main_text.read_text(encoding="utf-8")

    appendix = bundle_path / "latex" / "appendix.tex"
    assert appendix.exists()
    assert "\\begin{figure}" in appendix.read_text(encoding="utf-8")


def test_bundle_metadata_and_manifests(bundle_output):
    _, _, bundle_path, run_id = bundle_output
    meta_dir = bundle_path / "metadata"
    assert (meta_dir / "figure_manifest.csv").exists()
    assert (meta_dir / "figure_manifest.json").exists()
    assert (meta_dir / "run_info.json").exists()

    manifest_json = json.loads((meta_dir / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest_json["run_id"] == run_id
    assert manifest_json["role"] == "main_text_canonical"
    assert len(manifest_json["figures"]) == 8

    for record in manifest_json["figures"]:
        fname = record["filename"]
        fig_path = bundle_path / "figures" / fname
        assert fig_path.exists()
        assert hashlib.sha256(fig_path.read_bytes()).hexdigest() == record["sha256"]
        assert record["qa_status"] == "PASS"

    run_info = json.loads((meta_dir / "run_info.json").read_text(encoding="utf-8"))
    assert run_info["run_id"] == run_id
    assert run_info["precision"] == "float64"


def test_master_manifest_generation(bundle_output):
    _, work, bundle_path, _ = bundle_output
    master_csv = work / "master_manifest.csv"
    master_json = work / "master_manifest.json"

    generate_master_manifest([bundle_path], master_csv, master_json)
    assert master_csv.exists()
    assert master_json.exists()

    records = json.loads(master_json.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["all_8_figures_present"] is True
    assert records[0]["all_latex_snippets_present"] is True
    assert records[0]["qa_status"] == "PASS"


def test_fe_comparison_table_generation():
    import pandas as pd
    estimates = {}
    metadata = {}
    for col in ["Pooled", "Time_FE", "District_FE", "TwoWay_FE"]:
        estimates[col] = pd.DataFrame({
            "regressor": [
                "ln1p_capital_physical",
                "ln1p_mining_employment_total",
                "d_dt",
                "translog_k2",
                "translog_l2",
                "translog_d2",
                "translog_kl",
                "translog_kd",
                "translog_ld",
            ],
            "pip": [1.0, 1.0, 1.0, 1.0, 0.05, 1.0, 0.99, 0.96, 1.0],
            "post_mean": [0.01, 0.53, 0.94, 0.04, -0.0001, -0.15, -0.03, -0.02, 0.06],
            "post_sd": [0.01, 0.01, 0.02, 0.004, 0.001, 0.009, 0.004, 0.007, 0.009],
        })
        metadata[col] = {
            "n_obs": 21156,
            "n_districts": 235,
            "total_models": 49807360,
            "expected_model_size": 17.88,
            "expected_translog_terms": 4.98,
            "p_cobb_douglas": 1e-100,
        }

    work = Path(tempfile.mkdtemp(prefix="gpubma-table-"))
    try:
        table_path = work / "fixed_effects_comparison_table.tex"
        generate_fe_comparison_table(estimates, metadata, table_path)
        assert table_path.exists()
        content = table_path.read_text(encoding="utf-8")
        assert "\\begin{table}[htbp!]" in content
        assert "\\caption[" in content
        assert "Dependent Variable:" in content
        assert "Panel A: Primary Production Factors" in content
        assert "Panel D: Model Space Statistics" in content
        assert "21,156" in content
        assert "49,807,360" in content
    finally:
        shutil.rmtree(work, ignore_errors=True)
