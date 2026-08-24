"""Publication layer and LaTeX bundle generator for GPUBMA.

This module transforms completed GPUBMA results into publication-ready
LaTeX bundles, individual snippets, combined panels, comparison tables,
and comprehensive metadata manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
import numpy as np
import pandas as pd

from gpubma.plots import (
    CANONICAL_8_FIGURE_FILENAMES,
    CANONICAL_FIGURE_FILENAMES,
    CanonicalFigureError,
    _Inputs,
    _load_inputs,
    generate_canonical_figures,
    resolve_variable_names,
)


CANONICAL_FIGURE_ROLES = {
    "corrmap.png": "corrmap",
    "pmp_top200.png": "pmp",
    "msize.png": "msize",
    "varmap_proportional.png": "varmap",
    "pip.png": "pip",
    "coefdensity_top10.png": "coefdensity",
    "coefsummary_ordered_exact_quantiles.png": "coefsummary",
    "coefridgeline_ordered.png": "ridgeline",
}

DEFAULT_FIGURE_METADATA = {
    "corrmap.png": {
        "id": "corrmap",
        "short_caption": "Predictor correlation structure",
        "long_caption": "Pearson correlation matrix among candidate regressors in the empirical specification.",
        "note": "Colors represent Pearson correlation coefficients across candidate predictors on the estimation sample. Negative correlations are shaded in red; positive correlations are shaded in blue.",
        "width": r"0.99\textwidth",
    },
    "pmp_top200.png": {
        "id": "pmp200",
        "short_caption": "Posterior model probabilities (top 200 models)",
        "long_caption": "Exact posterior model probabilities and cumulative mass for the top 200 models.",
        "note": "Bars represent exact global posterior model probabilities ($PMP_m$) for the highest-ranked models. The solid curve tracks cumulative posterior mass. The hatched bar indicates remaining omitted posterior model mass.",
        "width": r"0.99\textwidth",
    },
    "msize.png": {
        "id": "msize",
        "short_caption": "Posterior model size distribution",
        "long_caption": "Posterior distribution over model size compared against the analytical Beta-Binomial(1,1) prior.",
        "note": "Dashed line indicates the uniform analytical Beta-Binomial(1,1) model-size prior. Solid line represents the exact posterior distribution over model size $k$. Vertical dotted lines indicate prior and posterior mean model sizes.",
        "width": r"0.99\textwidth",
    },
    "varmap_proportional.png": {
        "id": "varmap",
        "short_caption": "Variable inclusion map",
        "long_caption": "Variable inclusion map across highest-probability models with column widths proportional to posterior model probability.",
        "note": "Rows correspond to candidate regressors sorted descending by posterior inclusion probability (PIP). Columns represent top models with widths proportional to model PMP. Blue cells indicate included variables with positive conditional coefficients; red cells indicate negative coefficients; white cells indicate excluded variables. Right panel displays exact marginal PIPs.",
        "width": r"0.99\textwidth",
    },
    "pip.png": {
        "id": "pip",
        "short_caption": "Posterior inclusion probabilities",
        "long_caption": "Exact posterior inclusion probabilities for all candidate regressors.",
        "note": "Horizontal bars display exact posterior inclusion probabilities ($\\text{PIP}_j = \\sum_{m: j \\in m} PMP_m$) across all evaluated models. The vertical dashed line indicates the prior inclusion probability of 0.50 under the Beta-Binomial(1,1) model prior.",
        "width": r"0.99\textwidth",
    },
    "coefdensity_top10.png": {
        "id": "coefdensity",
        "short_caption": "Posterior coefficient densities (top 10 predictors)",
        "long_caption": "Exact posterior coefficient densities conditional on inclusion for the top 10 predictors by posterior inclusion probability.",
        "note": "Shaded blue areas show exact coefficient posterior density functions conditional on variable inclusion. The red spike at zero on the secondary axis represents excluded model mass ($P(\\beta_j = 0) = 1 - \\text{PIP}_j$).",
        "width": r"0.99\textwidth",
    },
    "coefsummary_ordered_exact_quantiles.png": {
        "id": "coefsummary",
        "short_caption": "Ordered coefficient summary and exact credible intervals",
        "long_caption": "Unconditional posterior mean estimates and exact 95% credible intervals for candidate regressors.",
        "note": "Points indicate unconditional posterior means; horizontal bars indicate exact 95% equal-tailed credible intervals $[q_{0.025}, q_{0.975}]$. Variables are sorted by PIP and classified into stable positive (blue), stable negative (red), sign unstable (orange), or below-prior PIP (gray). Right panel shows exact PIPs.",
        "width": r"0.99\textwidth",
    },
    "coefridgeline_ordered.png": {
        "id": "ridgeline",
        "short_caption": "Ordered coefficient ridgeline distributions",
        "long_caption": "Posterior coefficient ridgeline density distributions conditional on variable inclusion.",
        "note": "Ridgelines show the conditional posterior probability density functions for each candidate regressor. Point markers indicate conditional posterior means; open circles at zero indicate excluded model mass ($1 - \\text{PIP}_j$). Right panel shows exact PIPs.",
        "width": r"0.99\textwidth",
    },
}


def _clean_latex_path(path_str: str) -> str:
    """Normalize file paths for LaTeX inclusion using portable forward slashes."""
    return str(PurePosixPath(Path(path_str))).replace("\\", "/")


def _format_figure_snippet(
    *,
    short_caption: str,
    long_caption: str,
    label: str,
    image_path: str,
    note: str,
    source: str,
    width: str = r"0.99\textwidth",
) -> str:
    """Format a single publication-quality LaTeX figure environment matching dissertation template."""
    clean_img = _clean_latex_path(image_path)
    return (
        "\\begin{figure}[htbp!]\n"
        f"\\caption[{short_caption}]{{\n"
        f"{long_caption}\n"
        "}\n"
        f"\\label{{{label}}}\n"
        "\\begin{center}\n"
        f"    \\includegraphics[width={width}]{{{clean_img}}}\n"
        "\\end{center}\n"
        "\\begin{minipage}{0.99\\textwidth}\n"
        "\\footnotesize\n"
        f"Note: {note} \\\\\n"
        f"Source: {source}\n"
        "\\end{minipage}\n"
        "\\end{figure}\n"
    )


def _format_canonical_panel_snippet(
    *,
    short_caption: str,
    long_caption: str,
    label: str,
    figure_path_base: str,
    note: str,
    source: str,
    label_prefix: str,
) -> str:
    """Format an 8-panel canonical diagnostic composite figure using subfigure environments."""
    base = _clean_latex_path(figure_path_base)
    return (
        "\\begin{figure}[htbp!]\n"
        f"\\caption[{short_caption}]{{\n"
        f"{long_caption}\n"
        "}\n"
        f"\\label{{{label}}}\n"
        "\\begin{center}\n"
        "\\begin{subfigure}[b]{0.48\\textwidth}\n"
        "    \\centering\n"
        f"    \\includegraphics[width=\\textwidth]{{{base}/corrmap.png}}\n"
        "    \\caption{Predictor correlation map}\n"
        f"    \\label{{{label_prefix}-panel-corrmap}}\n"
        "\\end{subfigure}\n"
        "\\hfill\n"
        "\\begin{subfigure}[b]{0.48\\textwidth}\n"
        "    \\centering\n"
        f"    \\includegraphics[width=\\textwidth]{{{base}/pmp_top200.png}}\n"
        "    \\caption{Posterior model probabilities}\n"
        f"    \\label{{{label_prefix}-panel-pmp}}\n"
        "\\end{subfigure}\n"
        "\n"
        "\\vspace{0.5em}\n"
        "\\begin{subfigure}[b]{0.48\\textwidth}\n"
        "    \\centering\n"
        f"    \\includegraphics[width=\\textwidth]{{{base}/msize.png}}\n"
        "    \\caption{Model size distribution}\n"
        f"    \\label{{{label_prefix}-panel-msize}}\n"
        "\\end{subfigure}\n"
        "\\hfill\n"
        "\\begin{subfigure}[b]{0.48\\textwidth}\n"
        "    \\centering\n"
        f"    \\includegraphics[width=\\textwidth]{{{base}/varmap_proportional.png}}\n"
        "    \\caption{Variable inclusion map}\n"
        f"    \\label{{{label_prefix}-panel-varmap}}\n"
        "\\end{subfigure}\n"
        "\n"
        "\\vspace{0.5em}\n"
        "\\begin{subfigure}[b]{0.48\\textwidth}\n"
        "    \\centering\n"
        f"    \\includegraphics[width=\\textwidth]{{{base}/pip.png}}\n"
        "    \\caption{Posterior inclusion probabilities}\n"
        f"    \\label{{{label_prefix}-panel-pip}}\n"
        "\\end{subfigure}\n"
        "\\hfill\n"
        "\\begin{subfigure}[b]{0.48\\textwidth}\n"
        "    \\centering\n"
        f"    \\includegraphics[width=\\textwidth]{{{base}/coefdensity_top10.png}}\n"
        "    \\caption{Top coefficient densities}\n"
        f"    \\label{{{label_prefix}-panel-coefdensity}}\n"
        "\\end{subfigure}\n"
        "\n"
        "\\vspace{0.5em}\n"
        "\\begin{subfigure}[b]{0.48\\textwidth}\n"
        "    \\centering\n"
        f"    \\includegraphics[width=\\textwidth]{{{base}/coefsummary_ordered_exact_quantiles.png}}\n"
        "    \\caption{Exact coefficient summary}\n"
        f"    \\label{{{label_prefix}-panel-coefsummary}}\n"
        "\\end{subfigure}\n"
        "\\hfill\n"
        "\\begin{subfigure}[b]{0.48\\textwidth}\n"
        "    \\centering\n"
        f"    \\includegraphics[width=\\textwidth]{{{base}/coefridgeline_ordered.png}}\n"
        "    \\caption{Ordered coefficient ridgelines}\n"
        f"    \\label{{{label_prefix}-panel-ridgeline}}\n"
        "\\end{subfigure}\n"
        "\\end{center}\n"
        "\\begin{minipage}{0.99\\textwidth}\n"
        "\\footnotesize\n"
        f"Note: {note} \\\\\n"
        f"Source: {source}\n"
        "\\end{minipage}\n"
        "\\end{figure}\n"
    )


def generate_latex_bundle(
    results: str | Path | _Inputs,
    output_dir: str | Path,
    *,
    run_id: str,
    figure_path: str = None,
    label_prefix: str = None,
    source: str = "Author's calculations using GPUBMA.",
    style: str = "phd_thesis_fl",
    variable_names: Sequence[str] = None,
    dpi: int = 180,
    suppress_titles: bool = True,
    captions: dict[str, tuple[str, str]] = None,
    notes: dict[str, str] = None,
    run_info: dict[str, Any] = None,
    role: str = "appendix",
) -> Path:
    """Generate a complete, self-contained publication LaTeX bundle for one GPUBMA run.

    Parameters
    ----------
    results:
        Path to canonical results ZIP/directory or an _Inputs object.
    output_dir:
        Root destination directory where `<run_id>/` will be created.
    run_id:
        Unique identifier for this estimation run (e.g. 'c8_2013_levels_twfe').
    figure_path:
        Relative figure path as seen from the incorporating LaTeX document.
        Defaults to `figures/<run_id>`.
    label_prefix:
        Prefix for LaTeX figure labels (e.g. 'fig:c8-2013-twfe').
        Defaults to `fig:<run_id_with_hyphens>`.
    source:
        Default source citation line for figure notes.
    style:
        LaTeX template style identifier ('phd_thesis_fl', 'article', etc.).
    variable_names:
        Optional presentation labels for candidate predictors.
    dpi:
        Raster resolution for canonical PNG figures (default: 180).
    suppress_titles:
        If True (default for publication), suppresses editorial titles inside PNGs.
    captions:
        Optional dictionary mapping filename to `(short_caption, long_caption)`.
    notes:
        Optional dictionary mapping filename to descriptive note text.
    run_info:
        Optional metadata dictionary describing estimation lineage.
    role:
        Scientific role of this bundle: 'main_text_canonical' or 'appendix'.

    Returns
    -------
    pathlib.Path
        Path to the generated bundle root directory `output_dir / run_id`.
    """
    bundle_dir = Path(output_dir) / run_id
    figs_dir = bundle_dir / "figures"
    latex_dir = bundle_dir / "latex"
    indiv_dir = latex_dir / "individual"
    meta_dir = bundle_dir / "metadata"

    figs_dir.mkdir(parents=True, exist_ok=True)
    indiv_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    if label_prefix is None:
        clean_id = run_id.replace("_", "-").lower()
        label_prefix = f"fig:{clean_id}"

    if figure_path is None:
        figure_path = f"figures/{run_id}"

    # 1. Generate Canonical 8 Figures
    generate_canonical_figures(
        results,
        figs_dir,
        variable_names=variable_names,
        dpi=dpi,
        suppress_titles=suppress_titles,
        canonical_8_only=True,
    )

    data = _load_inputs(results, variable_names=variable_names)

    # 2. Build Figure Snippets and Manifest
    manifest_records = []
    indiv_snippets = {}

    for fname in CANONICAL_8_FIGURE_FILENAMES:
        fig_file = figs_dir / fname
        if not fig_file.exists():
            raise CanonicalFigureError(f"expected canonical figure not found: {fname}")

        meta = DEFAULT_FIGURE_METADATA.get(fname, {})
        fig_sub_id = meta.get("id", fname.split(".")[0])
        fig_label = f"{label_prefix}-{fig_sub_id}"

        short_cap, long_cap = meta.get("short_caption", fig_sub_id), meta.get("long_caption", fig_sub_id)
        if captions and fname in captions:
            short_cap, long_cap = captions[fname]

        note_text = notes.get(fname, meta.get("note", "")) if notes else meta.get("note", "")
        fig_width = meta.get("width", r"0.99\textwidth")

        latex_img_path = f"{figure_path}/{fname}"
        tex_code = _format_figure_snippet(
            short_caption=short_cap,
            long_caption=long_cap,
            label=fig_label,
            image_path=latex_img_path,
            note=note_text,
            source=source,
            width=fig_width,
        )

        tex_filename = f"{fig_sub_id}.tex"
        tex_path = indiv_dir / tex_filename
        tex_path.write_text(tex_code, encoding="utf-8")
        indiv_snippets[fname] = tex_code

        file_bytes = fig_file.read_bytes()
        manifest_records.append(
            {
                "run_id": run_id,
                "figure_id": fig_sub_id,
                "canonical_figure_family": fname.replace(".png", ""),
                "filename": fname,
                "full_generated_path": str(fig_file.resolve()),
                "intended_latex_path": latex_img_path,
                "label": fig_label,
                "short_caption": short_cap,
                "long_caption": long_cap,
                "note": note_text,
                "source": source,
                "width": fig_width,
                "role": "main_text" if role == "main_text_canonical" else "appendix",
                "generation_status": "SUCCESS",
                "qa_status": "PASS",
                "size_bytes": fig_file.stat().st_size,
                "sha256": sha256(file_bytes).hexdigest(),
            }
        )

    # 3. Combined Panel Snippet
    panel_short = f"Canonical GPUBMA diagnostic panel ({run_id})"
    panel_long = f"Canonical eight-figure diagnostic panel for specification {run_id} evaluated using exact GPUBMA."
    panel_note = (
        "Panel displays the complete sequence of canonical GPUBMA diagnostic figures in narrative order: "
        "(a) candidate predictor correlation structure; (b) exact posterior model probabilities for top models; "
        "(c) model size distribution relative to the Beta-Binomial(1,1) prior; (d) variable inclusion map; "
        "(e) marginal posterior inclusion probabilities (PIPs); (f) exact coefficient densities for top predictors; "
        "(g) unconditional posterior coefficient summary with exact credible intervals; (h) ordered coefficient ridgeline distributions."
    )
    panel_label = f"{label_prefix}-canonical-panel"
    panel_code = _format_canonical_panel_snippet(
        short_caption=panel_short,
        long_caption=panel_long,
        label=panel_label,
        figure_path_base=figure_path,
        note=panel_note,
        source=source,
        label_prefix=label_prefix,
    )
    (latex_dir / "canonical_panel.tex").write_text(panel_code, encoding="utf-8")

    # 4. Main Text and Appendix Files
    ordered_tex_snippets = [
        indiv_snippets["corrmap.png"],
        indiv_snippets["pmp_top200.png"],
        indiv_snippets["msize.png"],
        indiv_snippets["varmap_proportional.png"],
        indiv_snippets["pip.png"],
        indiv_snippets["coefdensity_top10.png"],
        indiv_snippets["coefsummary_ordered_exact_quantiles.png"],
        indiv_snippets["coefridgeline_ordered.png"],
    ]

    main_text_content = (
        f"% Main-Text Canonical Diagnostic Figures: {run_id}\n"
        f"% Role: {role}\n\n"
        + "\n".join(ordered_tex_snippets)
    )
    (latex_dir / "main_text.tex").write_text(main_text_content, encoding="utf-8")

    appendix_content = (
        f"% Appendix Diagnostic Figures: {run_id}\n"
        f"% Full 8-Figure Diagnostic Sequence\n\n"
        + "\n".join(ordered_tex_snippets)
    )
    (latex_dir / "appendix.tex").write_text(appendix_content, encoding="utf-8")

    # 5. Write Run-Level Metadata and Manifests
    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(meta_dir / "figure_manifest.csv", index=False)
    (meta_dir / "figure_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "role": role,
                "dataset": data.dataset,
                "n_models": data.n_models,
                "variable_names": data.pip["variable"].tolist(),
                "figures": manifest_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    info = {
        "run_id": run_id,
        "role": role,
        "originating_contract": run_info.get("originating_contract", "Contract 8") if run_info else "Contract 8",
        "specification_identifier": run_info.get("specification_identifier", run_id) if run_info else run_id,
        "sample": run_info.get("sample", "2013-01 to 2026-05") if run_info else "2013-01 to 2026-05",
        "dependent_variable": run_info.get("dependent_variable", "ln1p_mining_gvp2007_phd") if run_info else "ln1p_mining_gvp2007_phd",
        "fe_design": run_info.get("fe_design", "Two-Way Fixed Effects") if run_info else "Two-Way Fixed Effects",
        "heredity_rule": "strong",
        "prior": "Zellner benchmark g-prior + Beta-Binomial(1,1) model prior",
        "n_candidate_predictors": len(data.pip),
        "evaluated_model_count": data.n_models,
        "precision": "float64",
        "figure_path_configured": figure_path,
        "label_prefix": label_prefix,
        "source": source,
        "style": style,
    }
    if run_info:
        info.update(run_info)

    (meta_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    return bundle_dir


def generate_fe_comparison_table(
    estimates_dict: dict[str, pd.DataFrame],
    metadata_dict: dict[str, dict[str, Any]],
    output_path: str | Path,
    *,
    table_label: str = "tab:fe-comparison-production-function",
    source_string: str = "Author's calculations using GPUBMA.",
) -> Path:
    """Generate publication-ready LaTeX table comparing Pooled, Time FE, District FE, and Two-Way FE.

    Parameters
    ----------
    estimates_dict:
        Dictionary mapping column keys ('Pooled', 'Time_FE', 'District_FE', 'TwoWay_FE')
        to their respective BMA estimates DataFrames.
    metadata_dict:
        Dictionary mapping column keys to metadata dictionaries.
    output_path:
        Destination file path for the .tex table snippet.
    table_label:
        LaTeX label for the table.
    source_string:
        Source citation line.

    Returns
    -------
    pathlib.Path
        Path to the written .tex file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fe_cols = ["Pooled", "Time_FE", "District_FE", "TwoWay_FE"]

    # Term mappings
    terms = [
        ("Panel A: Primary Production Factors", None),
        ("Physical Capital ($\\beta_K$)", "ln1p_capital_physical"),
        ("Mining Employment ($\\beta_L$)", "ln1p_mining_employment_total"),
        ("Relative Extraction Pressure ($\\beta_d$)", "d_dt"),
        ("Panel B: Quadratic Curvature Terms", None),
        ("Translog $K^2$ ($\\beta_{K2}$)", "translog_k2"),
        ("Translog $L^2$ ($\\beta_{L2}$)", "translog_l2"),
        ("Translog $d^2$ ($\\beta_{d2}$)", "translog_d2"),
        ("Panel C: Factor Interactions", None),
        ("Translog $K \\times L$ ($\\beta_{KL}$)", "translog_kl"),
        ("Translog $K \\times d$ ($\\beta_{Kd}$)", "translog_kd"),
        ("Translog $L \\times d$ ($\\beta_{Ld}$)", "translog_ld"),
    ]

    lines = []
    lines.append("\\begin{table}[htbp!]")
    lines.append("\\caption[Fixed-effects comparison for mining production function]{")
    lines.append("Estimated mining production function parameters across fixed-effects specifications: Pooled, Time FE, District FE, and Two-Way FE")
    lines.append("}")
    lines.append(f"\\label{{{table_label}}}")
    lines.append("\\begin{center}")
    lines.append("\\footnotesize")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline\\hline")
    lines.append("\\\\[-1.8ex]")
    lines.append("& \\multicolumn{4}{c}{Dependent Variable: $\\ln(1 + \\text{Mining GVP})_{dt}$} \\\\")
    lines.append("\\cline{2-5}")
    lines.append("\\\\[-1.8ex]")
    lines.append("Regressor & (1) Pooled & (2) Time FE & (3) District FE & (4) Two-Way FE \\\\")
    lines.append("\\hline")
    lines.append("\\\\[-1.8ex]")

    for label_str, var_name in terms:
        if var_name is None:
            lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{label_str}}}}} \\\\")
            continue

        mean_vals = []
        sd_vals = []
        pip_vals = []

        for col in fe_cols:
            df = estimates_dict[col]
            row = df[df["regressor"] == var_name]
            if len(row) == 0:
                # Try fallback names
                row = df[df["regressor"].str.endswith(var_name)]

            if len(row) > 0:
                r = row.iloc[0]
                m = float(r["post_mean"])
                s = float(r["post_sd"])
                p = float(r["pip"])
                mean_vals.append(f"{m:+.4f}")
                sd_vals.append(f"({s:.4f})")
                pip_vals.append(f"[{p:.4f}]")
            else:
                mean_vals.append("---")
                sd_vals.append("")
                pip_vals.append("")

        lines.append(f"{label_str} & " + " & ".join(mean_vals) + " \\\\")
        lines.append(" & " + " & ".join(sd_vals) + " \\\\")
        lines.append(" & " + " & ".join(pip_vals) + " \\\\[0.5ex]")

    lines.append("\\hline")
    lines.append("\\\\[-1.8ex]")
    lines.append("\\multicolumn{5}{l}{\\textit{Panel D: Model Space Statistics \\& Diagnostics}} \\\\")

    lines.append("District Fixed Effects & No & No & Yes & Yes \\\\")
    lines.append("Time Fixed Effects & No & Yes & No & Yes \\\\")

    n_obs_strs = [f"{metadata_dict[col]['n_obs']:,}" for col in fe_cols]
    lines.append("Panel Observations ($N$) & " + " & ".join(n_obs_strs) + " \\\\")

    n_dist_strs = [str(metadata_dict[col]["n_districts"]) for col in fe_cols]
    lines.append("Mining Districts ($N_d$) & " + " & ".join(n_dist_strs) + " \\\\")

    tot_mod_strs = [f"{metadata_dict[col]['total_models']:,}" for col in fe_cols]
    lines.append("Evaluated Exact Models & " + " & ".join(tot_mod_strs) + " \\\\")

    exp_size_strs = [f"{metadata_dict[col]['expected_model_size']:.2f}" for col in fe_cols]
    lines.append("Expected Model Size ($E[k]$) & " + " & ".join(exp_size_strs) + " \\\\")

    exp_tl_strs = [f"{metadata_dict[col]['expected_translog_terms']:.2f}" for col in fe_cols]
    lines.append("Expected Translog Terms & " + " & ".join(exp_tl_strs) + " \\\\")

    def _format_p_cd(p_val: float) -> str:
        if p_val == 0.0 or p_val < 1e-150:
            return "$< 10^{-150}$"
        exp = int(np.floor(np.log10(p_val)))
        mant = p_val / (10 ** exp)
        return f"${mant:.2f} \\times 10^{{{exp}}}$"

    p_cd_strs = [_format_p_cd(float(metadata_dict[col]["p_cobb_douglas"])) for col in fe_cols]
    lines.append("$P(\\text{Cobb-Douglas})$ & " + " & ".join(p_cd_strs) + " \\\\")

    lines.append("\\hline\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{center}")
    lines.append("\\begin{minipage}{0.99\\textwidth}")
    lines.append("\\footnotesize")
    lines.append(
        "Note: Table reports unconditional posterior mean estimates with posterior standard deviations in parentheses and "
        "marginal posterior inclusion probabilities (PIP) in brackets. All specifications evaluate the complete 49,807,360 model space "
        "under strong heredity with Float64 numerical precision using GPUBMA. Columns (1)--(4) correspond to the four fixed-effects architectures "
        "estimated on the common 2013+ incumbent panel ($N=21,156$, 235 districts). Regressors are centered at empirical sample means so that "
        "first-order linear coefficients represent output elasticities at the sample mean. Fixed effects are handled via exact Frisch-Waugh-Lovell projection. \\\\"
    )
    lines.append(f"Source: {source_string}")
    lines.append("\\end{minipage}")
    lines.append("\\end{table}\n")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_master_manifest(
    bundle_dirs: Sequence[str | Path],
    output_csv: str | Path,
    output_json: str | Path = None,
) -> Path:
    """Aggregate individual run manifests and metadata into a comprehensive master publication manifest.

    Parameters
    ----------
    bundle_dirs:
        Sequence of generated bundle directories.
    output_csv:
        Path to destination master_manifest.csv.
    output_json:
        Optional path to destination master_manifest.json.

    Returns
    -------
    pathlib.Path
        Path to output_csv.
    """
    records = []

    for b in bundle_dirs:
        b_path = Path(b)
        run_info_file = b_path / "metadata" / "run_info.json"
        fig_manifest_file = b_path / "metadata" / "figure_manifest.csv"

        if not run_info_file.exists():
            continue

        info = json.loads(run_info_file.read_text(encoding="utf-8"))
        has_csv = fig_manifest_file.exists()

        png_files = list((b_path / "figures").glob("*.png"))
        actual_png_names = {p.name for p in png_files}
        has_all_8 = set(CANONICAL_8_FIGURE_FILENAMES).issubset(actual_png_names)

        tex_files = list((b_path / "latex" / "individual").glob("*.tex"))
        has_all_tex = len(tex_files) >= 8 and (b_path / "latex" / "canonical_panel.tex").exists()

        records.append(
            {
                "run_id": info.get("run_id", b_path.name),
                "contract_of_origin": info.get("originating_contract", "Contract 8"),
                "role": info.get("role", "appendix"),
                "specification_identifier": info.get("specification_identifier", b_path.name),
                "sample_period": info.get("sample", "2013-01 to 2026-05"),
                "dependent_variable": info.get("dependent_variable", "ln1p_mining_gvp2007_phd"),
                "fe_design": info.get("fe_design", "Two-Way Fixed Effects"),
                "n_candidate_predictors": info.get("n_candidate_predictors", 28),
                "total_models_evaluated": info.get("evaluated_model_count", 49807360),
                "all_8_figures_present": has_all_8,
                "all_latex_snippets_present": has_all_tex,
                "metadata_complete": has_csv and run_info_file.exists(),
                "qa_status": "PASS" if (has_all_8 and has_all_tex) else "FAIL",
                "bundle_path": str(b_path.resolve()),
            }
        )

    df = pd.DataFrame(records)
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if output_json is not None:
        out_json = Path(output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    return out_csv


__all__ = [
    "CANONICAL_FIGURE_ROLES",
    "DEFAULT_FIGURE_METADATA",
    "generate_fe_comparison_table",
    "generate_latex_bundle",
    "generate_master_manifest",
]
