"""Reproducible canonical figures from an existing GPUBMA results archive.

This module only visualizes already-computed results.  It never calls the CPU
or GPU enumerators and does not recompute posterior quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import sys
import textwrap
from zipfile import ZipFile

import numpy as np
import pandas as pd


CANONICAL_FIGURE_FILENAMES = (
    "coefdensity_slide_01_x1.png",
    "coefdensity_slide_02_x2.png",
    "coefdensity_slide_03_x3.png",
    "coefdensity_slide_04_x4.png",
    "coefdensity_slide_05_x5.png",
    "coefdensity_slide_06_x8.png",
    "coefdensity_slide_07_x9.png",
    "coefdensity_slide_08_x6.png",
    "coefdensity_slide_09_x7.png",
    "coefdensity_slide_10_x11.png",
    "coefdensity_top10.png",
    "coefridgeline_ordered.png",
    "coefsummary_ordered.png",
    "coefsummary_ordered_exact_quantiles.png",
    "corrmap.png",
    "msize.png",
    "pip.png",
    "pmp_top100.png",
    "pmp_top20.png",
    "pmp_top200.png",
    "pmp_top50.png",
    "varmap_equal_width.png",
    "varmap_proportional.png",
)

_PMP_COUNTS = (20, 50, 100, 200)
_COLORS = {
    "Stable positive": "#2166ac",
    "Stable negative": "#b2182b",
    "Sign unstable": "#e08214",
    "Below-prior PIP": "0.45",
}


class CanonicalFigureError(ValueError):
    """Raised when computed results cannot produce the canonical figure set."""


def resolve_variable_names(
    p: int,
    *,
    stored_names=None,
    variable_names=None,
) -> list[str]:
    """Resolve one presentation label for each statistical position.

    An explicit ``variable_names`` override takes precedence. Otherwise,
    names stored with the results are preserved; if neither is available,
    the backward-compatible labels ``x1, ..., xp`` are returned.
    """
    if not isinstance(p, int) or p < 1:
        raise ValueError(f"p must be a positive integer; got {p!r}")
    candidate = variable_names if variable_names is not None else stored_names
    source = "variable_names" if variable_names is not None else "stored names"
    if candidate is None:
        return [f"x{i + 1}" for i in range(p)]
    if isinstance(candidate, (str, bytes)):
        raise ValueError(f"{source} must be a sequence of exactly {p} strings")
    names = list(candidate)
    if len(names) != p:
        raise ValueError(
            f"{source} must contain exactly {p} names; received {len(names)}"
        )
    invalid = [i for i, name in enumerate(names) if not isinstance(name, str) or not name.strip()]
    if invalid:
        raise ValueError(f"{source} contains invalid labels at positions {invalid}")
    if len(set(names)) != p:
        raise ValueError(f"{source} must contain {p} unique names")
    return names


@dataclass
class _Inputs:
    dataset: str
    n_models: int
    pip: pd.DataFrame
    top: pd.DataFrame
    size: pd.DataFrame
    densities: pd.DataFrame
    varmap: pd.DataFrame
    coefsummary: pd.DataFrame
    coefsummary_exact: pd.DataFrame
    coefridge: pd.DataFrame
    corr: pd.DataFrame


class _ResultReader:
    def __init__(self, source: str | Path):
        self.source = Path(source)
        self._zip = ZipFile(self.source) if self.source.is_file() else None
        if self._zip is None and not self.source.is_dir():
            raise FileNotFoundError(f"results source does not exist: {self.source}")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def _find(self, relative: str) -> str | Path:
        if self._zip is None:
            direct = self.source / relative
            if direct.exists():
                return direct
            matches = list(self.source.rglob(PurePosixPath(relative).name))
            if len(matches) == 1:
                return matches[0]
            raise FileNotFoundError(f"required result not found: {relative}")
        names = self._zip.namelist()
        if relative in names:
            return relative
        matches = [n for n in names if n.endswith("/" + PurePosixPath(relative).name)]
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(f"required result not found in ZIP: {relative}")

    def bytes(self, relative: str) -> bytes:
        target = self._find(relative)
        if self._zip is None:
            return Path(target).read_bytes()
        return self._zip.read(str(target))

    def json(self, relative: str) -> dict:
        return json.loads(self.bytes(relative).decode("utf-8"))

    def parquet(self, relative: str) -> pd.DataFrame:
        target = self._find(relative)
        if self._zip is None:
            return pd.read_parquet(target)
        return pd.read_parquet(BytesIO(self._zip.read(str(target))))


def _require_columns(frame: pd.DataFrame, name: str, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise CanonicalFigureError(f"{name} is missing columns: {sorted(missing)}")


def _load_inputs(source: str | Path, *, variable_names=None) -> _Inputs:
    reader = _ResultReader(source)
    try:
        summary = reader.json("results/panel_30_center15_exact_summary.json")
        pip = reader.parquet("results/panel_30_center15_pip.parquet")
        top = reader.parquet("results/panel_30_center15_top_models.parquet")
        size = reader.parquet("results/panel_30_center15_model_size.parquet")
        densities = reader.parquet("results/coefdensity_grids.parquet")
        varmap = reader.parquet("results/varmap_coverage90.parquet")
        coefsummary = reader.parquet("results/coefsummary_ordered.parquet")
        coefsummary_exact = reader.parquet(
            "results/coefsummary_ordered_exact_quantiles.parquet"
        )
        coefridge = reader.parquet("results/coefridgeline_ordered.parquet")
        corr = reader.parquet("results/corrmap_matrix.parquet")
    finally:
        reader.close()

    _require_columns(pip, "PIP table", {"variable", "pip", "coef_mean", "coef_sd"})
    _require_columns(
        top,
        "top-model table",
        {"rank", "model_id", "model_size", "pmp", "cumulative_pmp"},
    )
    _require_columns(size, "model-size table", {"k", "posterior"})
    _require_columns(
        densities,
        "density table",
        {"predictor", "predictor_index", "x", "conditional_density", "pip"},
    )
    _require_columns(
        varmap,
        "varmap table",
        {
            "predictor",
            "predictor_index",
            "pip",
            "model_rank",
            "model_pmp_global",
            "included",
            "conditional_coefficient",
            "coverage_target",
            "achieved_coverage",
            "other_model_mass",
        },
    )
    _require_columns(
        coefsummary,
        "coefficient summary",
        {
            "display_order",
            "predictor",
            "pip",
            "posterior_mean_unconditional",
            "posterior_sd_unconditional",
            "sign_class",
        },
    )
    _require_columns(
        coefsummary_exact,
        "exact coefficient summary",
        {
            "display_order",
            "predictor",
            "pip",
            "posterior_mean_unconditional",
            "posterior_q025_exact",
            "posterior_q975_exact",
            "sign_class",
        },
    )
    _require_columns(
        coefridge,
        "coefficient ridgeline summary",
        {
            "display_order",
            "predictor",
            "pip",
            "posterior_mean_conditional_on_inclusion",
            "sign_class",
        },
    )

    p = len(pip)
    if p < 10:
        raise CanonicalFigureError("the canonical set requires at least 10 predictors")
    if len(top) < max(_PMP_COUNTS):
        raise CanonicalFigureError("the canonical set requires at least 200 top models")
    if set(pip["variable"]) != set(densities["predictor"]):
        raise CanonicalFigureError("PIP and density predictor sets differ")

    pip = pip.reset_index(drop=True).copy()
    stored_names = pip["variable"].astype(str).tolist()
    if len(set(stored_names)) != p:
        raise CanonicalFigureError("stored predictor names must be unique")
    names = resolve_variable_names(
        p,
        stored_names=stored_names,
        variable_names=variable_names,
    )
    name_by_position = dict(enumerate(names))
    name_by_stored = dict(zip(stored_names, names))
    pip["variable"] = names

    densities = densities.copy()
    density_positions = densities["predictor_index"].astype(int)
    if not density_positions.between(0, p - 1).all():
        raise CanonicalFigureError("density predictor_index is outside 0..p-1")
    densities["predictor"] = density_positions.map(name_by_position)

    varmap = varmap.copy()
    varmap_positions = varmap["predictor_index"].astype(int)
    if not varmap_positions.between(0, p - 1).all():
        raise CanonicalFigureError("varmap predictor_index is outside 0..p-1")
    varmap["predictor"] = varmap_positions.map(name_by_position)

    def relabel_summary(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
        frame = frame.copy()
        unknown = set(frame["predictor"]).difference(name_by_stored)
        if unknown:
            raise CanonicalFigureError(
                f"{table_name} contains unknown predictor labels: {sorted(unknown)}"
            )
        frame["predictor"] = frame["predictor"].map(name_by_stored)
        return frame

    coefsummary = relabel_summary(coefsummary, "coefficient summary")
    coefsummary_exact = relabel_summary(
        coefsummary_exact, "exact coefficient summary"
    )
    coefridge = relabel_summary(coefridge, "coefficient ridgeline summary")

    corr = corr.copy()
    missing_corr = [name for name in stored_names if name not in corr.columns]
    if missing_corr:
        raise CanonicalFigureError(
            f"correlation table is missing predictors: {missing_corr}"
        )
    corr = corr.rename(columns=name_by_stored)
    corr["predictor"] = corr["predictor"].map(name_by_stored)
    if corr["predictor"].isna().any():
        raise CanonicalFigureError("correlation row labels do not match PIP positions")

    return _Inputs(
        dataset=str(summary.get("dataset", "gpubma_results")),
        n_models=int(summary["n_models"]),
        pip=pip,
        top=top.sort_values("rank"),
        size=size.sort_values("k"),
        densities=densities.sort_values(["predictor_index", "x"]),
        varmap=varmap.sort_values(["model_rank", "predictor_index"]),
        coefsummary=coefsummary.sort_values("display_order"),
        coefsummary_exact=coefsummary_exact.sort_values("display_order"),
        coefridge=coefridge.sort_values("display_order"),
        corr=corr,
    )


def _label_layout(labels) -> tuple[float, float]:
    """Return modest extra width and left margin for scientific labels."""
    longest = max(len(str(label)) for label in labels)
    extra_width = min(6.0, max(0.0, (longest - 8) * 0.065))
    left_margin = min(0.42, max(0.12, 0.10 + longest * 0.006))
    return extra_width, left_margin


def _save(fig, path: Path, dpi: int, plt) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if not path.exists() or path.stat().st_size == 0:
        raise CanonicalFigureError(f"empty or missing figure: {path.name}")


def _plot_pip(data: _Inputs, out: Path, dpi: int, plt) -> None:
    frame = data.pip
    predictors = frame["variable"].to_numpy()
    pip = frame["pip"].to_numpy(float)
    order = np.argsort(pip)
    p = len(frame)
    extra_width, left_margin = _label_layout(predictors)
    fig, ax = plt.subplots(figsize=(7 + extra_width, 0.28 * p + 1.5))
    ax.barh(np.arange(p), pip[order], color="#1f77b4")
    ax.axvline(0.5, color="0.4", ls="--", label="prior inclusion probability = 0.50 (beta-binomial(1,1))")
    ax.set_yticks(np.arange(p), predictors[order], fontsize=8)
    for i, j in enumerate(order):
        ax.text(min(pip[j] + 0.008, 1.06), i, f"{pip[j]:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel(f"posterior inclusion probability (exact, all {data.n_models:,} models)")
    ax.set_title(f"Posterior inclusion probabilities, p = {p}")
    ax.legend(fontsize=8, loc="lower right")
    fig.subplots_adjust(left=left_margin)
    _save(fig, out / "pip.png", dpi, plt)


def _plot_pmp(data: _Inputs, count: int, out: Path, dpi: int, plt) -> None:
    t = data.top.head(count)
    pmp = t["pmp"].to_numpy(float)
    cumulative = t["cumulative_pmp"].to_numpy(float)
    other = float(np.clip(1.0 - pmp.sum(), 0.0, 1.0))
    width = max(10.0, 0.055 * count + 3.0)
    fig, ax = plt.subplots(figsize=(width, 5.6))
    xs = np.arange(1, count + 1)
    ax.bar(xs, pmp, width=0.85, color="#1f77b4", linewidth=0,
           label=f"exact global PMP (denominator: all {data.n_models:,} models)")
    other_x = count + max(3.0, 0.025 * count)
    ax.bar([other_x], [other], width=max(1.0, 0.008 * count), color="0.6",
           hatch="//", edgecolor="0.25", linewidth=0.6,
           label=f"Other models (exact omitted mass = {other:.6f})")
    ax2 = ax.twinx()
    ax2.plot(xs, cumulative, color="#d62728", marker=".", markersize=2.3,
             linewidth=1, label="cumulative global PMP")
    ax2.set_ylabel("cumulative global PMP")
    ax2.set_ylim(0, 1.02)
    ax2.grid(False)
    ax.set_xlim(0, other_x + max(2.0, 0.015 * count))
    ax.set_xlabel("model rank (descending global PMP)")
    ax.set_ylabel("posterior model probability")
    ax.set_title(f"Posterior model probabilities (top {count} models)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=3, fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.925, top=0.90, bottom=0.20)
    _save(fig, out / f"pmp_top{count}.png", dpi, plt)


def _plot_model_size(data: _Inputs, out: Path, dpi: int, plt) -> None:
    size = data.size["k"].to_numpy(int)
    posterior = data.size["posterior"].to_numpy(float)
    prior = np.full(len(size), 1.0 / len(size))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(size, prior, "o--", color="0.45", label="prior (analytical)")
    ax.plot(size, posterior, "o-", color="#1f77b4", label="posterior (exact)")
    ax.axvline(float(size @ prior), color="0.45", ls=":", label=f"prior mean = {size @ prior:.2f}")
    mean = float(size @ posterior)
    ax.axvline(mean, color="#d62728", ls=":", label=f"posterior mean = {mean:.3f}")
    ax.set_xticks(size)
    ax.set_xlabel("model size (number of included optional predictors)")
    ax.set_ylabel("probability")
    ax.set_title("Model-size distribution (exact posterior and beta-binomial(1,1) prior)")
    ax.legend(fontsize=8)
    _save(fig, out / "msize.png", dpi, plt)


def _density_arrays(data: _Inputs) -> tuple[np.ndarray, np.ndarray, list[str]]:
    predictors = data.pip["variable"].tolist()
    groups = {name: g.sort_values("x") for name, g in data.densities.groupby("predictor")}
    grid = np.vstack([groups[name]["x"].to_numpy(float) for name in predictors])
    density = np.vstack([groups[name]["conditional_density"].to_numpy(float) for name in predictors])
    return grid, density, predictors


def _plot_densities(data: _Inputs, out: Path, dpi: int, plt) -> None:
    grid, density, predictors = _density_arrays(data)
    pip = data.pip["pip"].to_numpy(float)
    show = np.argsort(-pip)[:10]
    fig, axes = plt.subplots(2, 5, figsize=(16.75, 5.8), squeeze=False)
    for position, j in enumerate(show):
        ax = axes[position // 5][position % 5]
        ax.plot(grid[j], density[j], color="#1f77b4", linewidth=1.35)
        ax.fill_between(grid[j], density[j], color="#1f77b4", alpha=0.22)
        ax.axvline(0.0, color="0.72", linewidth=0.8, linestyle=":")
        ax.tick_params(axis="both", labelsize=7)
        ax.set_title(
            f"{textwrap.fill(predictors[j], width=24)}\nPIP = {pip[j]:.2f}",
            fontsize=9,
        )
        ax2 = ax.twinx()
        ax2.set_ylim(0, 1)
        zero = float(np.clip(1.0 - pip[j], 0.0, 1.0))
        if zero > 1e-6:
            ax2.vlines(0.0, 0.0, zero, color="#d62728", linewidth=2)
            ax2.plot([0.0], [zero], marker="v", markersize=5.5, color="#d62728")
        if position % 5 != 4:
            ax2.tick_params(axis="y", right=False, labelright=False)
    fig.suptitle("Posterior coefficient densities\nBlue: exact density conditional on inclusion; red: excluded-model mass", y=1.015)
    fig.tight_layout()
    _save(fig, out / "coefdensity_top10.png", dpi, plt)

    for rank, j in enumerate(show, start=1):
        fig, ax = plt.subplots(figsize=(13.333, 7.5))
        ax.plot(grid[j], density[j], color="#1f77b4", linewidth=2.4)
        ax.fill_between(grid[j], density[j], color="#1f77b4", alpha=0.20)
        ax.axvline(0.0, color="0.70", linewidth=1.2, linestyle=":")
        ax.set_xlabel(r"Coefficient value $\beta_j$", fontsize=16)
        ax.set_ylabel("Density | included", fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(axis="y", alpha=0.20)
        ax2 = ax.twinx()
        ax2.set_ylim(0, 1)
        ax2.set_ylabel(r"$P(\beta_j=0)=1-\mathrm{PIP}_j$", fontsize=16, color="#d62728")
        zero = float(np.clip(1.0 - pip[j], 0.0, 1.0))
        if zero > 1e-6:
            ax2.vlines(0.0, 0.0, zero, color="#d62728", linewidth=4)
            ax2.plot([0.0], [zero], marker="v", markersize=9, color="#d62728")
        fig.suptitle(textwrap.fill(predictors[j], width=48), fontsize=24, y=0.97)
        fig.text(0.5, 0.915, f"Posterior coefficient density | PIP = {pip[j]:.2f} | P(β=0) = {zero:.2f}",
                 ha="center", fontsize=14)
        fig.text(0.01, 0.015, f"Rank by PIP: {rank} of 10 shown", fontsize=11, color="0.35")
        fig.tight_layout(rect=[0.03, 0.05, 0.97, 0.88])
        # Filenames identify the statistical position, while visible labels
        # may contain spaces, Unicode, or other presentation characters.
        _save(fig, out / f"coefdensity_slide_{rank:02d}_x{j + 1}.png", dpi, plt)


def _plot_varmap(data: _Inputs, proportional: bool, out: Path, dpi: int, plt) -> None:
    frame = data.varmap
    predictors = data.pip["variable"].tolist()
    pip_map = data.pip.set_index("variable")["pip"]
    models = frame[["model_rank", "model_pmp_global"]].drop_duplicates().sort_values("model_rank")
    ranks = models["model_rank"].to_numpy()
    pmp = models["model_pmp_global"].to_numpy(float)
    widths = pmp if proportional else np.full(len(pmp), pmp.sum() / len(pmp))
    edges = np.concatenate(([0.0], np.cumsum(widths)))
    row_order = sorted(predictors, key=lambda name: -float(pip_map[name]))
    lookup = frame.set_index(["model_rank", "predictor"])
    p = len(predictors)
    extra_width, left_margin = _label_layout(predictors)
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(12.8 + extra_width, 11.8), sharey=True,
                                  gridspec_kw={"width_ratios": [5.6, 1.1], "wspace": 0.20})
    for edge in range(p + 1):
        ax.axhline(edge, color="0.85", linewidth=0.45, zorder=0)
        axp.axhline(edge, color="0.88", linewidth=0.45, zorder=0)
    for ri, name in enumerate(row_order):
        for ci, rank in enumerate(ranks):
            row = lookup.loc[(rank, name)]
            if not bool(row["included"]):
                continue
            value = float(row["conditional_coefficient"])
            color = "#ffee99" if abs(value) < 1e-12 else ("#2166ac" if value > 0 else "#b2182b")
            ax.add_patch(plt.Rectangle((edges[ci], ri), widths[ci], 1.0, facecolor=color, edgecolor="none"))
    for edge in edges[1:-1]:
        ax.axvline(edge, color="white", linewidth=0.35, alpha=0.9, zorder=3)
    omitted = float(np.clip(1.0 - widths.sum(), 0.0, 1.0))
    ax.add_patch(plt.Rectangle((edges[-1], 0), omitted, p, facecolor="0.86", edgecolor="0.45", hatch="///"))
    if omitted > 0.025:
        ax.text(edges[-1] + omitted / 2, p / 2,
                f"Other models\nexact mass = {omitted:.4f}", rotation=90,
                ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(p, 0)
    ax.set_yticks(np.arange(p) + 0.5, row_order, fontsize=8)
    ax.set_xlabel("cumulative global PMP" if proportional else "displayed models at equal width")
    target = float(frame["coverage_target"].iloc[0])
    achieved = float(frame["achieved_coverage"].iloc[0])
    ax.set_title(f"Variable-inclusion map: K = {len(ranks)} models, cumulative PMP {achieved:.4f} ≥ {target:.2f}\nblue = positive; red = negative; white = excluded")
    ordered_pip = np.array([pip_map[name] for name in row_order], float)
    axp.barh(np.arange(p) + 0.5, ordered_pip, color="#1f77b4", height=0.78)
    axp.set_xlim(0, 1.10)
    axp.set_xlabel("PIP (exact)")
    axp.tick_params(axis="y", labelleft=False)
    axp.set_ylim(p, 0)
    for ri, value in enumerate(ordered_pip):
        axp.text(min(float(value) + 0.012, 1.065), ri + 0.5, f"{value:.2f}",
                 va="center", fontsize=7, clip_on=False)
    fig.subplots_adjust(left=max(0.18, left_margin), right=0.96, top=0.92, bottom=0.08)
    name = "varmap_proportional.png" if proportional else "varmap_equal_width.png"
    _save(fig, out / name, dpi, plt)


def _plot_coefsummary(data: _Inputs, exact: bool, out: Path, dpi: int, plt) -> None:
    frame = data.coefsummary_exact if exact else data.coefsummary
    predictors = frame["predictor"].tolist()
    mean = frame["posterior_mean_unconditional"].to_numpy(float)
    pip = frame["pip"].to_numpy(float)
    classes = frame["sign_class"].tolist()
    colors = [_COLORS.get(c, "0.45") for c in classes]
    if exact:
        low = frame["posterior_q025_exact"].to_numpy(float)
        high = frame["posterior_q975_exact"].to_numpy(float)
        interval_label = "exact 95% equal-tail credible interval"
        filename = "coefsummary_ordered_exact_quantiles.png"
    else:
        sd = frame["posterior_sd_unconditional"].to_numpy(float)
        low, high = mean - 1.96 * sd, mean + 1.96 * sd
        interval_label = "mean ± 1.96 posterior SD"
        filename = "coefsummary_ordered.png"
    p = len(frame)
    y = np.arange(p)
    extra_width, left_margin = _label_layout(predictors)
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(12.8 + extra_width, 0.34 * p + 2.6), sharey=True,
                                  gridspec_kw={"width_ratios": [4.8, 1.35], "wspace": 0.08})
    n_high = int(np.count_nonzero(pip >= 0.5))
    for yi in y:
        ax.hlines(yi, low[yi], high[yi], color=colors[yi], linewidth=1.5)
        ax.plot(mean[yi], yi, "o", color=colors[yi], markersize=5.5)
    for edge in np.arange(p + 1) - 0.5:
        ax.axhline(edge, color="0.88", linewidth=0.45, zorder=0)
        axp.axhline(edge, color="0.88", linewidth=0.45, zorder=0)
    if 0 < n_high < p:
        ax.axhline(n_high - 0.5, color="#1f77b4", linewidth=1.25)
        axp.axhline(n_high - 0.5, color="#1f77b4", linewidth=1.25)
    ax.axvline(0, color="0.35", ls="--", linewidth=0.9)
    ax.set_yticks(y, predictors, fontsize=9)
    ax.set_ylim(p - 0.5, -0.5)
    ax.set_xlabel(f"Unconditional posterior mean and {interval_label}")
    ax.set_ylabel("Candidate regressor")
    ax.grid(axis="x", alpha=0.18)
    axp.barh(y, pip, color=colors, height=0.72)
    axp.axvline(0.5, color="0.45", ls=":", linewidth=0.8)
    axp.set_xlim(0, 1.12)
    axp.set_xlabel("PIP (exact)")
    axp.tick_params(axis="y", labelleft=False)
    for yi, value, color in zip(y, pip, colors):
        axp.text(min(float(value) + 0.015, 1.075), yi, f"{value:.2f}",
                 ha="left", va="center", fontsize=8, color=color, clip_on=False)
    title = "Exact BMA" if exact else "BMA"
    fig.suptitle(
        f"{title} coefficient summary and posterior inclusion probabilities\n"
        "variables grouped relative to the prior inclusion probability of 0.50",
        fontsize=13,
        y=0.985,
    )
    fig.subplots_adjust(left=max(0.20, left_margin), right=0.95, top=0.91, bottom=0.10)
    _save(fig, out / filename, dpi, plt)


def _plot_ridgeline(data: _Inputs, out: Path, dpi: int, plt) -> None:
    grid, density, predictor_order = _density_arrays(data)
    index = {name: i for i, name in enumerate(predictor_order)}
    frame = data.coefridge
    predictors = frame["predictor"].tolist()
    pip = frame["pip"].to_numpy(float)
    classes = frame["sign_class"].tolist()
    colors = [_COLORS.get(c, "0.45") for c in classes]
    means = frame["posterior_mean_conditional_on_inclusion"].to_numpy(float)
    p = len(frame)
    n_high = int(np.count_nonzero(pip >= 0.5))
    extra_width, _ = _label_layout(predictors)
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(13.5 + extra_width, 0.38 * p + 2.8), sharey=True,
                                  gridspec_kw={"width_ratios": [5.8, 1.35], "wspace": 0.08})
    for yi, (name, color) in enumerate(zip(predictors, colors)):
        j = index[name]
        dens = np.clip(density[j], 0, np.inf)
        dens = dens / max(float(dens.max()), 1e-300)
        ridge = yi - 0.82 * dens
        visible = np.flatnonzero(dens >= 0.015)
        label_x = float(grid[j][visible[0]] if visible.size else grid[j][0])
        ax.fill_between(grid[j], yi, ridge, color=color, alpha=0.32)
        ax.plot(grid[j], ridge, color=color, linewidth=1.25)
        ax.text(label_x - 0.012 * (grid.max() - grid.min()), yi - 0.20, name,
                ha="right", va="center", fontsize=8.5, color=color)
        ax.plot(means[yi], yi - 0.04, "o", markersize=4.8, color=color)
        ax.scatter([0.0], [yi], s=8 + (1 - pip[yi]) * 82, facecolor="white", edgecolor=color, linewidth=1.1)
    for edge in np.arange(p + 1) - 0.5:
        ax.axhline(edge, color="0.90", linewidth=0.45, zorder=0)
        axp.axhline(edge, color="0.90", linewidth=0.45, zorder=0)
    if 0 < n_high < p:
        ax.axhline(n_high - 0.5, color="#1f77b4", linewidth=1.25)
        axp.axhline(n_high - 0.5, color="#1f77b4", linewidth=1.25)
    ax.axvline(0, color="0.25", ls="--", linewidth=0.9)
    ax.set_ylim(p - 0.5, -1.15)
    ax.set_yticks([])
    ax.set_xlabel("Coefficient value; ridgelines show conditional posterior shape")
    axp.barh(np.arange(p), pip, color=colors, height=0.68)
    axp.axvline(0.5, color="0.45", ls=":", linewidth=0.9)
    axp.set_xlim(0, 1.13)
    axp.set_xlabel("PIP (exact)")
    axp.tick_params(axis="y", labelleft=False)
    for yi, value, color in zip(np.arange(p), pip, colors):
        axp.text(min(float(value) + 0.015, 1.085), yi, f"{value:.2f}",
                 ha="left", va="center", fontsize=8, color=color, clip_on=False)
    fig.suptitle("Exact BMA coefficient densities and posterior inclusion probabilities\nridgelines are conditional on inclusion; circles at zero represent excluded-model mass", fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.10, right=0.95, top=0.91, bottom=0.10)
    _save(fig, out / "coefridgeline_ordered.png", dpi, plt)


def _plot_corrmap(data: _Inputs, out: Path, dpi: int, plt) -> None:
    frame = data.corr
    predictors = frame["predictor"].tolist()
    corr = frame[predictors].to_numpy(float)
    p = len(predictors)
    extra_width, left_margin = _label_layout(predictors)
    fig, ax = plt.subplots(figsize=(9 + extra_width, 8 + 0.25 * extra_width))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    if p == 30:
        ax.axhline(14.5, color="#1f77b4", linewidth=1.2)
        ax.axvline(14.5, color="#1f77b4", linewidth=1.2)
    ax.set_xticks(np.arange(p), predictors, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(p), predictors, fontsize=7)
    ax.set_title(f"Predictor correlation structure of {data.dataset}")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Pearson correlation")
    fig.subplots_adjust(left=left_margin, bottom=min(0.42, left_margin + 0.04))
    _save(fig, out / "corrmap.png", dpi, plt)


def _write_manifest(data: _Inputs, output_dir: Path) -> Path:
    actual = {p.name for p in output_dir.glob("*.png")}
    expected = set(CANONICAL_FIGURE_FILENAMES)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise CanonicalFigureError(
            f"canonical PNG set mismatch; missing={missing}, extra={extra}"
        )
    records = []
    for path in sorted(output_dir.glob("*.png")):
        records.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = output_dir / "panel_30_center15_figure_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": data.dataset,
                "n_models": data.n_models,
                "variable_names": data.pip["variable"].tolist(),
                "figures": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def generate_canonical_figures(
    results: str | Path,
    output_dir: str | Path,
    *,
    variable_names=None,
    dpi: int = 180,
) -> Path:
    """Generate the 23 canonical PNGs from already-computed GPUBMA results.

    Parameters
    ----------
    results:
        Path to the canonical results ZIP or an extracted results directory.
    output_dir:
        Destination for the PNG files and final SHA-256 manifest.
    variable_names:
        Optional presentation labels in exact statistical position order.
        Spaces and Unicode are accepted. Stored result labels are used when
        omitted, with ``x1, ..., xp`` as the resolver's final fallback.
    dpi:
        Raster resolution. The canonical default is 180; tests may use a lower
        value to keep lightweight fixtures fast.

    Returns
    -------
    pathlib.Path
        Path to ``panel_30_center15_figure_manifest.json``.
    """
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")
    try:
        import matplotlib

        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "canonical figures require matplotlib; install gpubma[plots]"
        ) from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = _load_inputs(results, variable_names=variable_names)
    style = {
        "figure.dpi": 120,
        "savefig.dpi": dpi,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
    }
    with plt.rc_context(style):
        _plot_pip(data, output, dpi, plt)
        for count in _PMP_COUNTS:
            _plot_pmp(data, count, output, dpi, plt)
        _plot_model_size(data, output, dpi, plt)
        _plot_varmap(data, True, output, dpi, plt)
        _plot_varmap(data, False, output, dpi, plt)
        _plot_densities(data, output, dpi, plt)
        _plot_coefsummary(data, False, output, dpi, plt)
        _plot_coefsummary(data, True, output, dpi, plt)
        _plot_ridgeline(data, output, dpi, plt)
        _plot_corrmap(data, output, dpi, plt)
    return _write_manifest(data, output)


__all__ = [
    "CANONICAL_FIGURE_FILENAMES",
    "CanonicalFigureError",
    "generate_canonical_figures",
    "resolve_variable_names",
]
