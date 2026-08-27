"""Lightweight diagnostics and plotting wrappers for BFG results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from gpubma.bfg.results import BFGResult


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for BFG plotting functions. "
            "Install matplotlib via `pip install matplotlib`."
        ) from err


def plot_convergence(result: BFGResult, save_path: Optional[str] = None, show: bool = True):
    """Plot convergence of log Z_hat across progressive evaluation checkpoints."""
    plt = _require_matplotlib()
    if not result.checkpoints:
        print("[BFG Diagnostics] No intermediate checkpoints recorded to plot convergence.")
        return None

    evals = [c["eval_count"] for c in result.checkpoints]
    log_zs = [c["log_Z_hat"] for c in result.checkpoints]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(evals, log_zs, marker="o", color="#1f77b4", lw=2, label=r"$\log \widehat{Z}$")
    ax.set_xlabel("Cumulative Model Evaluations", fontsize=11)
    ax.set_ylabel(r"Reconstructed $\log Z$", fontsize=11)
    ax.set_title("BFG Denominator Convergence", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_model_size(result: BFGResult, save_path: Optional[str] = None, show: bool = True):
    """Plot reconstructed posterior model size distribution P(k | y)."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    k_indices = result.model_size_posterior.index
    probs = result.model_size_posterior.values

    ax.bar(k_indices, probs, color="#2ca02c", edgecolor="black", alpha=0.8, width=0.8)
    ax.set_xlabel("Model Size (k)", fontsize=11)
    ax.set_ylabel(r"$\widehat{P}(k \mid y)$", fontsize=11)
    ax.set_title("Posterior Model Size Distribution", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_pips(result: BFGResult, save_path: Optional[str] = None, show: bool = True):
    """Plot Posterior Inclusion Probabilities across candidate predictors."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)

    vars_list = result.pips.index
    pips = result.pips.values

    ax.bar(range(len(vars_list)), pips, color="#d62728", edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(vars_list)))
    ax.set_xticklabels(vars_list, rotation=45, ha="right", fontsize=9)
    ax.axhline(0.5, color="black", linestyle="--", lw=1, alpha=0.7, label="Median Probability Threshold")
    ax.set_ylabel("Posterior Inclusion Probability (PIP)", fontsize=11)
    ax.set_title("Candidate Predictor Inclusion Probabilities", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(frameon=True)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_allocation(result: BFGResult, save_path: Optional[str] = None, show: bool = True):
    """Plot budget allocated and evaluated across lattice sizes."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    k_vals = list(result.lattice_results.keys())
    evals = [result.lattice_results[k].evaluated_count for k in k_vals]

    ax.bar(k_vals, evals, color="#9467bd", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Model Size (k)", fontsize=11)
    ax.set_ylabel("Evaluated Models", fontsize=11)
    ax.set_title("Evaluation Volume Across Model Lattices", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig
