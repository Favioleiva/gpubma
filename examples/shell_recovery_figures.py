"""Render title-free/caption-free figures for post-BFG random shell recovery."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

CAPS = (10_000, 100_000, 1_000_000)
COLORS = ["#b17b27", "#2563a6", "#168579"]


def render_shell_recovery_figures(
    benchmark_dir: Path,
    exact_ref_dir: Path,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Render the five publication-quality title-free figures for post-BFG shell recovery.

    Guarantees:
    - Zero titles and zero captions burned into image files.
    - Exact reproduction of validated layouts, styling, and numerical curves.
    - Returns metadata catalog for notebook Markdown rendering.
    """
    benchmark_dir = Path(benchmark_dir)
    exact_ref_dir = Path(exact_ref_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allrows = pd.read_csv(benchmark_dir / "RANDOM_SHELL_CAP_SENSITIVITY_BY_SHELL.csv")
    primary = allrows[allrows.cap.eq(1_000_000)].set_index("k")
    h = np.load(benchmark_dir / "RECONSTRUCTED_HISTOGRAMS.npz")
    exacth = np.load(exact_ref_dir / "exact_shell_histograms.npz")
    edges = h["edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    ref = pd.read_csv(exact_ref_dir / "exhaustive_reticular_summary.csv")
    sens = pd.read_csv(benchmark_dir / "RANDOM_SHELL_CAP_SENSITIVITY.csv").set_index("cap")

    catalog: List[Dict[str, Any]] = []

    def _save(fig: plt.Figure, name: str, title: str, note: str) -> None:
        assert not fig._suptitle and all(not a.get_title() for a in fig.axes)
        target = output_dir / name
        fig.savefig(target, dpi=180, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        catalog.append(dict(filename=name, title=title, note=note, path=str(target)))

    # Figures 1 & 2: Reconstructed ridgelines (alone and vs exact)
    for compare, name, title in [
        (False, "post_bfg_random_reticular_ridgeline.png", "Population-weighted post-BFG shell recovery"),
        (True, "post_bfg_random_vs_exact_reticular_ridgeline.png", "Post-BFG reconstruction against the frozen exact shells"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 15), layout="constrained")
        for k in range(31):
            r = ref.iloc[k]
            ax.hlines(k, 100, 1140, color="#e7eaee", lw=0.4, zorder=0)
            datasets = [(h["cap_1000000"][k], "#168579", 0.3)]
            if compare:
                datasets.insert(0, (exacth["counts"][k].astype(float), "#68717c", 0.2))
            for data, color, alpha in datasets:
                if k in [0, 30]:
                    ax.vlines(r.min_log_numerator, k, k + 0.65, color=color, lw=1.4)
                    continue
                curve = gaussian_filter1d(data, 3)
                support = (centers >= r.min_log_numerator) & (centers <= r.max_log_numerator)
                xx = np.r_[r.min_log_numerator, centers[support], r.max_log_numerator]
                yy = np.r_[0, curve[support] / curve.max() * 0.72, 0]
                ax.fill_between(xx, k, k + yy, color=color, alpha=alpha)
                ax.plot(xx, k + yy, color=color, lw=0.8)
            n = int(primary.loc[k, "n_k_random"])
            label = f"{n:,}" if n else "census"
            ax.text(1150, k + 0.07, label, ha="left", va="center", fontsize=7)

        ax.plot(ref.max_log_numerator, ref.k, color="#ce6a24", marker=".", ms=3, lw=0.7)
        ax.scatter([ref.loc[15, "max_log_numerator"]], [15], marker="*", s=110, color="#b51435", zorder=5)

        handles = [
            Line2D([], [], color="#168579", lw=3, label="Weighted full-shell reconstruction"),
            Line2D([], [], color="#ce6a24", lw=1, marker=".", label="Shell champion found by BFG"),
            Line2D([], [], color="#b51435", marker="*", linestyle="", markersize=10, label="Exact global MAP"),
        ]
        if compare:
            handles.insert(0, Line2D([], [], color="#68717c", lw=3, label="Frozen exact full-shell density"))
        ax.legend(handles=handles, loc="upper left", framealpha=0.95)
        ax.text(1150, 31, "Random n", ha="left", fontsize=8)
        ax.set(
            ylim=(-0.5, 31.5),
            xlim=(80, 1245),
            yticks=range(31),
            xlabel="Log unnormalized model evidence / log numerator",
            ylabel="Candidate model size k",
        )
        note = (
            "Cap=1,000,000 with the unchanged 1% rule. Known BFG models have weight 1/N; random-remainder models "
            "have weight (N-D)/(N*n). Both components reconstruct the full shell; targeted discoveries are not treated as random. "
            "Gaussian smoothing sigma=3 bins and independent unit-peak normalization are display-only, on the frozen 2,000-bin grid. "
            "These are score densities, not posterior mass. Reference support endpoints bound rendering; no support recovery is claimed. "
            "Right labels give new random samples; the exact wings are censuses. k=3 and k=27 retain only 41 samples. "
            "All shell champions were already found by BFG and appear as overlays."
        )
        _save(fig, name, title, note)

    # Figure 3: Error by k
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained")
    for cap, color in zip(CAPS, COLORS):
        d = allrows[allrows.cap.eq(cap) & allrows.n_k_random.gt(0)]
        axs[0].plot(d.k, d.KS * 100, ".-", color=color, label=f"Cap {cap:,}", lw=1.2)
        axs[1].plot(d.k, d.Wasserstein_1, ".-", color=color, label=f"Cap {cap:,}", lw=1.2)
    for ax in axs:
        ax.set(xlabel="Candidate model size k", yscale="log", xticks=range(3, 28, 3))
        ax.grid(alpha=0.15)
        ax.legend()
    axs[0].set_ylabel("Exact KS distance (percentage points)")
    axs[1].set_ylabel("Exact Wasserstein-1 (score units)")
    _save(
        fig,
        "random_shell_error_by_k.png",
        "Reconstruction error by shell and cap",
        "Exact weighted-CDF KS and Wasserstein-1, not histogram approximations. Logarithmic vertical axes; exact-wing zero errors "
        "are omitted. Coincident curves identify shells with the same unchanged 1%-limited sample. Differences across caps "
        "use nested samples from one seed. Endpoint shells dominate worst-case error even at the 1M cap.",
    )

    # Figure 4: Cap sensitivity vs runtime
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    for ax, central, allcol, label in [
        (axs[0], "central_KS_mean", "KS_median", "KS distance"),
        (axs[1], "central_W1_mean", "W1_median", "Wasserstein-1 (score units)"),
    ]:
        ax.plot(sens.recovery_runtime_s, sens[central], color="#243b53", lw=1.1, label="Mean, common central shells k=13–17")
        ax.plot(sens.recovery_runtime_s, sens[allcol], color="#9ca3af", lw=1.1, ls="--", label="Median, all 25 interior shells")
        for cap, c in zip(CAPS, COLORS):
            r = sens.loc[cap]
            ax.scatter(r.recovery_runtime_s, r[central], s=50, color=c, zorder=4)
            ax.annotate(f"{cap//1000:,}k", xy=(r.recovery_runtime_s, r[central]), xytext=(8, 8), textcoords="offset points", color=c, fontsize=9)
        ax.set(xscale="log", yscale="log", xlabel="Measured cumulative recovery runtime (s)", ylabel=label)
        ax.margins(x=0.25, y=0.25)
        ax.grid(alpha=0.15)
        ax.legend(loc="lower left", fontsize=8)
    _save(
        fig,
        "random_shell_cap_sensitivity.png",
        "Distribution accuracy versus incremental recovery cost",
        "Costs are measured enclosing wall times for the nested cap ladder, including generation, GPU work, structural checks and export. "
        "Dots show the mean over the same central shells k=13–17, where every cap binds; dashed lines show all-interior medians. "
        "100k is proposed as the practical default for human review. One million gives a further measurable precision gain at greater cost. "
        "The timing replay repeats the saved sample and contributes no additional unique models.",
    )

    # Figure 5: Sample fraction by k
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for cap, c in zip(CAPS, COLORS):
        d = allrows[allrows.cap.eq(cap) & allrows.n_k_random.gt(0)]
        ax.plot(d.k, d.sample_fraction * 100, ".-", color=c, label=f"Cap {cap:,}", lw=1.2)
    ax.set(xlabel="Candidate model size k", ylabel="New random sample / full shell (%)", yscale="log", xticks=range(3, 28, 2))
    ax.grid(alpha=0.15)
    ax.legend(loc="lower left")
    _save(
        fig,
        "random_shell_sample_fraction_by_k.png",
        "Sample fractions under the 1% rule and precision caps",
        "Fraction n/N, with exact-wing zero samples omitted. Ceiling to an integer can put a small shell just above 1%. "
        "Random draws exclude every BFG-discovered model. Capping reduces the fraction in large shells without changing uniform sampling "
        "from their undiscovered remainders.",
    )

    return catalog
