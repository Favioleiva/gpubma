"""Canonical exact-reference figures: saved inputs only, external notebook captions.

Ported from the accepted exact p30 and historical publication renderers.
No scoring, estimation, enumeration or BFG search occurs in this module.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
BLUE='#2166ac';RED='#b2182b';INK='#0f172a';ORANGE='#ea580c'

FIGURE_CATALOG = [{'filename': '00_predictor_correlogram.png', 'title': 'Canonical p30 predictor geometry', 'note': 'Observed candidate correlations, n = 2,000; original metadata order. Controls w1, w2 excluded. Not a causal or posterior map. Blue labels: causal DGP predictors x1–x15; red labels: structural-zero proxies x16–x30.'}, {'filename': '01_exhaustive_model_space_cloud.png', 'title': 'Full-enumeration model-space cloud', 'note': 'All 1,073,741,824 models contribute once. Exact counts in 2,000 common score bins; no random thinning. Lines: exact support.'}, {'filename': '02_exhaustive_reticular_ridgeline.png', 'title': 'Exact log BMA numerator across all reticula', 'note': 'Full-shell histograms; Gaussian smoothing σ = 3 bins for display. Unit-peak ridge height is not posterior mass. Exact finite support; singleton spikes.'}, {'filename': '03_exhaustive_reticular_frontiers.png', 'title': 'Exact lower and upper reticular frontiers', 'note': 'Left: exact lower reticular frontier. Right: exact upper reticular frontier. Every shell is fully enumerated. The MAP and true model both have k=15, but their compositions differ by two predictor indicators. Short marker callouts are anchored at the exact scores.'}, {'filename': '04_exact_global_variable_map.png', 'title': 'Exact global variable / model structure', 'note': 'Widths use exact global PMP; inclusion sidebar sums all 2³⁰ models. Cell colors describe model-specific OLS signs. Always-in controls excluded.'}, {'filename': '05_exact_top_models_composition_weights.png', 'title': 'Exact top models: composition and global model probabilities', 'note': 'Shared row coordinates preserve rank / composition / weight alignment. Candidates ordered by exact global PIP; w1 and w2 always included.'}, {'filename': '06_exact_global_model_size_distribution.png', 'title': 'Exact global posterior model-size distribution', 'note': 'Candidate size excludes the intercept and always-in controls w1, w2. Posterior probabilities aggregate the entire model universe.'}, {'filename': '07_exact_global_coefficient_structure.png', 'title': 'Exact BMA coefficient structure: archived full-universe mixture', 'note': 'Verified A100 exact-mixture artifact reused on its 257-point grids. Curves condition on inclusion; exclusion atoms at zero omitted. Height is unit-peak. Rows read top to bottom: blue positive effects (largest first), red negative effects (most negative first), then gray predictors (PIP descending). Original colors and inclusion rule are unchanged.'}, {'filename': '08_exact_true_model_vs_map.png', 'title': 'Causal generating model versus exact posterior champion', 'note': 'Known synthetic truth is distinct from posterior optimality. Red outlines identify differing predictor indicators; controls w1, w2 are common. Hamming distance = 2; x15 is replaced by proxy x30 in the MAP.'}]

def render_suite(reference_dir, output_dir, *, inspect_figure=None):
    """Export nine caption-free PNGs and return their external text catalog.

    inspect_figure(name, fig) may inspect artists before export for regression QA.
    Input and output directories must differ. No scientific input file is written.
    """
    reference_dir=Path(reference_dir);output_dir=Path(output_dir)
    if reference_dir.resolve()==output_dir.resolve():raise ValueError('Separate reference inputs from figure exports')
    output_dir.mkdir(parents=True,exist_ok=True)
    checks={'figures':{},'scientific_inputs_read_only':True}
    catalog={entry['filename']:dict(entry) for entry in FIGURE_CATALOG}
    def save(fig,name,note):
        assert not fig.texts, 'Canvas title/caption text forbidden'
        assert all(not ax.get_title(loc=loc) for ax in fig.axes for loc in ['left','center','right']), 'Axes titles forbidden'
        if inspect_figure is not None:inspect_figure(name,fig)
        checks['figures'][name]={'axes':len(fig.axes),'embedded_titles':0,'embedded_captions':0}
        target=output_dir/name
        temporary=output_dir/(target.stem+'.tmp.png')
        fig.savefig(temporary,dpi=220,bbox_inches='tight',facecolor='white')
        temporary.replace(target)
        plt.close(fig)
    names=[f'x{i}' for i in range(1,31)]
    top=pd.read_csv(reference_dir/'exact_top10000_models.csv')
    shell=pd.read_csv(reference_dir/'exhaustive_reticular_summary.csv')
    pip=pd.read_csv(reference_dir/'exact_global_pips.csv').exact_global_pip.to_numpy()
    truth=json.loads((reference_dir/'exact_true_model.json').read_text())
    champion=json.loads((reference_dir/'exact_map.json').read_text())
    meta=json.loads((reference_dir/'exact_benchmark_metadata.json').read_text())
    histogram=np.load(reference_dir/'exact_shell_histograms.npz');edges=histogram['edges'];counts=histogram['counts'];centers=(edges[:-1]+edges[1:])/2
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 14, 'axes.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    corr = pd.read_csv(reference_dir / 'predictor_correlations.csv', index_col=0).to_numpy()
    fig, ax = plt.subplots(figsize=(14, 13))
    v = np.ma.masked_where(np.triu(np.ones_like(corr), 1).astype(bool), corr)
    im = ax.imshow(v, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(30), names, rotation=40, ha='right', fontsize=9)
    ax.set_yticks(range(30), names, fontsize=9)
    for i in range(30):
        for j in range(i + 1):
            ax.text(j, i, f'{corr[i, j]:+.2f}', ha='center', va='center', fontsize=5.6, color='white' if abs(corr[i, j]) > 0.62 else 'black')
    ax.axhline(14.5, color='#2c3e50', lw=1.8)
    ax.plot([14.5, 14.5], [14.5, 29.5], color='#2c3e50', lw=1.8)
    for j, t in enumerate(ax.get_xticklabels()):
        t.set_color(BLUE if j < 15 else RED)
    for j, t in enumerate(ax.get_yticklabels()):
        t.set_color(BLUE if j < 15 else RED)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035, label='Correlation coefficient ρ')
    fig.subplots_adjust(bottom=0.12)
    save(fig, '00_predictor_correlogram.png', 'Observed candidate correlations, n = 2,000; original metadata order. Controls w1, w2 excluded. Not a causal or posterior map.')
    plt.rcParams['font.family'] = 'DejaVu Serif'
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.pcolormesh(np.arange(32) - 0.5, edges, np.ma.masked_equal(counts.T, 0), cmap='Blues', norm=LogNorm(vmin=1, vmax=counts.max()), shading='flat')
    ax.plot(shell.k, shell.min_log_numerator, color=BLUE, lw=1)
    ax.plot(shell.k, shell.max_log_numerator, color=ORANGE, lw=1)
    ax.set(xlabel='Candidate model size k', ylabel='Log posterior model numerator', xlim=(-0.5, 30.5))
    fig.colorbar(im, ax=ax, label='Exact model count per shell × score bin')
    fig.subplots_adjust(bottom=0.14)
    save(fig, '01_exhaustive_model_space_cloud.png', 'All 1,073,741,824 models contribute once. Exact counts in 2,000 common score bins; no random thinning. Lines: exact support.')
    fig, ax = plt.subplots(figsize=(13, 16))
    support = []
    for row in shell.itertuples():
        k = row.k
        lo = row.min_log_numerator
        hi = row.max_log_numerator
        color = RED if k == 15 else '#0284c7' if k <= 7 or k >= 23 else ORANGE
        if lo == hi:
            ax.plot([lo, lo], [k, k + 0.6], color=color, lw=1.3)
            support.append([k, lo, hi, lo, hi])
            continue
        smooth = gaussian_filter1d(counts[k].astype(float), 3)
        inside = (centers > lo) & (centers < hi)
        x = np.r_[lo, centers[inside], hi]
        d = np.r_[0, smooth[inside], 0]
        d = d / d.max() * 0.85 if d.max() else d
        ax.fill_between(x, k, k + d, color=color, alpha=0.14)
        ax.plot(x, k + d, color=INK, lw=1.15)
        ax.plot([lo, hi], [k, k], color='#cbd5e1', lw=0.6)
        support.append([k, lo, hi, float(x[0]), float(x[-1])])
    ax.plot(shell.min_log_numerator, shell.k, 'o--', ms=3, color='#2563eb', lw=1.2, label='Exact lower frontier')
    ax.plot(shell.max_log_numerator, shell.k, 'D-', ms=3, color=ORANGE, lw=1.2, label='Exact upper frontier / shell champions')
    ax.scatter([top.iloc[0].log_score], [15], marker='*', s=180, color=RED, zorder=5, label='Exact global MAP')
    ax.set(yticks=range(31), ylim=(-0.5, 31), xlabel='Log unnormalized model evidence / log numerator', ylabel='Candidate model size k')
    ax.legend(loc='upper left', fontsize=9)
    fig.subplots_adjust(bottom=0.07)
    save(fig, '02_exhaustive_reticular_ridgeline.png', 'Full-shell histograms; Gaussian smoothing σ = 3 bins for display. Unit-peak ridge height is not posterior mass. Exact finite support; singleton spikes.')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={'width_ratios': [1, 1]})
    left, right = axes
    left.plot(shell.k, shell.min_log_numerator, 'o--', color='#2166ac', lw=1.6, ms=4, label='Exact shell minimum L(k)')
    right.plot(shell.k, shell.max_log_numerator, 'D-', color='#ea580c', lw=1.6, ms=4, label='Exact shell champion U(k)')
    handles = [right.lines[0]]
    for model, marker, color, offset, label in [(champion, '*', '#b2182b', (-26, 16), f"Exact global MAP · k=15 · rank 1\nlog numerator = {champion['log_score']:.6f}"), (truth, 'x', '#0f172a', (26, -16), f"True causal model M* · k=15 · rank 8\nlog numerator = {truth['log_score']:.6f}")]:
        glyph = DrawingArea(18, 18, 0, 0)
        glyph.add_artist(Line2D([9], [9], marker=marker, color=color, markersize=12 if marker == '*' else 7, linestyle='None'))
        right.add_artist(AnnotationBbox(glyph, (15, model['log_score']), xybox=offset, xycoords='data', boxcoords='offset points', frameon=False, pad=0, arrowprops={'arrowstyle': '-', 'color': color, 'lw': 0.85, 'shrinkA': 1, 'shrinkB': 0}))
        handles.append(Line2D([], [], marker=marker, color=color, markersize=12 if marker == '*' else 7, linestyle='None', label=label))
    for ax, title in zip(axes, ['Exact lower reticular frontier', 'Exact upper reticular frontier']):
        ax.set(xlabel='Candidate model size k', ylabel='Log posterior model numerator', xlim=(-0.7, 30.7), ylim=(65, 1240), xticks=range(0, 31, 5))
        ax.grid(alpha=0.18)
    left.legend(loc='upper left', fontsize=9, framealpha=1)
    right.legend(handles=handles, loc='lower right', fontsize=9, framealpha=1, labelspacing=1.1)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.9, bottom=0.2, wspace=0.23)
    note = 'Every shell is fully enumerated. The MAP and true model both have k=15, but their compositions differ by two predictor indicators.'
    save(fig, '03_exhaustive_reticular_frontiers.png', note)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    order = np.argsort(-pip, kind='stable')
    fits = json.loads((reference_dir / 'model_sign_display.json').read_text())
    fig, (ax, side) = plt.subplots(1, 2, figsize=(13.2, 11.2), gridspec_kw={'width_ratios': [5.2, 1.4], 'wspace': 0.12}, sharey=True)
    left = 0.0
    for row, fit in zip(top.head(50).itertuples(), fits):
        for y, j in enumerate(order):
            name = names[j]
            if name in fit['names']:
                sign = fit['beta'][fit['names'].index(name)]
                ax.add_patch(Rectangle((left, y - 0.5), row.pmp, 1, facecolor=BLUE if sign >= 0 else RED, edgecolor='white', lw=0.2))
        left += row.pmp
    ax.add_patch(Rectangle((left, -0.5), 1 - left, 30, facecolor='#eeeeee', hatch='///', edgecolor='#999999', lw=0.3))
    ax.set(xlim=(0, 1), ylim=(29.5, -0.5), yticks=range(30), yticklabels=[names[j] for j in order], xlabel='Cumulative exact global PMP; top 50 models shown')
    side.barh(range(30), pip[order], height=0.75, color='#1f77b4')
    side.set(xlim=(0, 1.16), xlabel='Exact global PIP')
    side.axvline(0.5, color='gray', ls='--', lw=1)
    side.tick_params(axis='y', left=False, labelleft=False)
    for y, value in enumerate(pip[order]):
        side.text(value + 0.015, y, f'{value:.3f}', va='center', fontsize=8)
    ax.legend(handles=[Patch(color=BLUE, label='Positive OLS coefficient'), Patch(color=RED, label='Negative OLS coefficient'), Patch(facecolor='#eee', hatch='///', label='Remaining global model mass')], loc='upper center', bbox_to_anchor=(0.55, -0.07), ncol=2, fontsize=8)
    fig.subplots_adjust(bottom=0.13)
    save(fig, '04_exact_global_variable_map.png', 'Widths use exact global PMP; inclusion sidebar sums all 2³⁰ models. Cell colors describe model-specific OLS signs. Always-in controls excluded.')
    selected = top.head(15)
    matrix = np.array([[bool(int(m) >> j & 1) for j in order] for m in selected.model_id])
    fig, (ax, side) = plt.subplots(1, 2, figsize=(14, 7.5), sharey=True, gridspec_kw={'width_ratios': [5.2, 1.7], 'wspace': 0.03})
    ax.imshow(matrix, aspect='auto', cmap=ListedColormap(['white', BLUE]), vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(30), [names[j] for j in order], rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(15), [f'Rank {r.exact_global_rank} · k={r.model_size}' + (' · M*' if int(r.model_id) == 32767 else '') for r in selected.itertuples()])
    side.barh(range(15), selected.pmp, height=0.75, color='#1f77b4')
    side.set_xlim(0, 0.56)
    side.tick_params(axis='y', left=False, labelleft=False)
    side.set_xlabel('Exact global PMP')
    for y, v in enumerate(selected.pmp):
        side.text(v + 0.005, y, f'{v:.4f}', va='center', fontsize=9)
    fig.subplots_adjust(bottom=0.15, top=0.91, left=0.14)
    save(fig, '05_exact_top_models_composition_weights.png', 'Shared row coordinates preserve rank / composition / weight alignment. Candidates ordered by exact global PIP; w1 and w2 always included.')
    size = pd.read_csv(reference_dir / 'exact_model_size_distribution.csv')
    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    ax.plot(size.k, size.prior_probability, 'o--', color='gray', lw=1.6, ms=6.5, label='Beta-binomial (1,1) size prior')
    ax.plot(size.k, size.exact_posterior_probability, 'o-', color='#1f77b4', lw=2.2, ms=7, label='Exact posterior size probability')
    ax.axvline(meta['posterior_expected_model_size'], color=RED, ls='--', lw=1.8, label=f"Posterior mean = {meta['posterior_expected_model_size']:.3f}")
    ax.axvline(15, color='black', lw=1, label='MAP size = true-model size = 15')
    ax.set(xlabel='Candidate model size k', ylabel='Probability')
    ax.grid(axis='y', alpha=0.2)
    ax.legend(fontsize=9)
    fig.subplots_adjust(bottom=0.19)
    save(fig, '06_exact_global_model_size_distribution.png', 'Candidate size excludes the intercept and always-in controls w1, w2. Posterior probabilities aggregate the entire model universe.')
    density = np.load(reference_dir / 'coefficient_densities.npz')
    np.testing.assert_allclose(density['pip'], pip, rtol=0, atol=1e-10)
    grid = density['grid']
    curves = density['conditional_density']
    signed = trapezoid(grid * curves, grid, axis=1)
    positive = sorted([j for j in range(30) if pip[j] >= 0.5 and signed[j] >= 0], key=lambda j: (-signed[j], j))
    negative = sorted([j for j in range(30) if pip[j] >= 0.5 and signed[j] < 0], key=lambda j: (signed[j], j))
    gray = sorted([j for j in range(30) if pip[j] < 0.5], key=lambda j: (-pip[j], j))
    coefficient_order = positive + negative + gray
    checks['coefficient_order_top_to_bottom'] = [names[j] for j in coefficient_order]
    checks['coefficient_blocks'] = {'blue': [names[j] for j in positive], 'red': [names[j] for j in negative], 'gray': [names[j] for j in gray]}
    checks['coefficient_sort_quantity'] = 'Unchanged original grid first moment integral(x * conditional_density dx); gray classification retains PIP < 0.5.'
    fig, (ax, side) = plt.subplots(1, 2, figsize=(13.6, 12), sharey=True, gridspec_kw={'width_ratios': [5.6, 1.4], 'wspace': 0.1})
    for row, j in enumerate(coefficient_order):
        y = 29 - row
        x = grid[j]
        d = curves[j]
        d = d / d.max() * 0.78 if d.max() else d
        sign = trapezoid(grid[j] * curves[j], grid[j])
        color = '#7f7f7f' if pip[j] < 0.5 else '#1f77b4' if sign >= 0 else '#d62728'
        ax.fill_between(x, y, y + d, color=color, alpha=0.32)
        ax.plot(x, y + d, color=color, lw=1.4)
    ax.axvline(0, color='gray', lw=0.8)
    ax.set(yticks=range(30), yticklabels=[names[j] for j in coefficient_order[::-1]], xlabel='Coefficient value, conditional on predictor inclusion', ylim=(-0.5, 30))
    side.barh(range(29, -1, -1), pip[coefficient_order], height=0.75, color='#1f77b4')
    side.set(xlim=(0, 1.16), xlabel='Exact global PIP')
    side.tick_params(axis='y', left=False, labelleft=False)
    for row, j in enumerate(coefficient_order):
        side.text(pip[j] + 0.01, 29 - row, f'{pip[j]:.3f}', va='center', fontsize=8)
    fig.subplots_adjust(bottom=0.075)
    save(fig, '07_exact_global_coefficient_structure.png', 'Verified A100 exact-mixture artifact reused on its 257-point grids. Curves condition on inclusion; exclusion atoms at zero omitted. Height is unit-peak.')
    fig, ax = plt.subplots(figsize=(14, 4.8))
    masks = [int(top.iloc[0].model_id), 32767]
    matrix = np.array([[bool(m >> j & 1) for j in range(30)] for m in masks])
    ax.imshow(matrix, aspect='auto', cmap=ListedColormap(['white', BLUE]), vmin=0, vmax=1)
    ax.set_xticks(range(30), names, rotation=45, ha='right')
    ax.set_yticks([0, 1], [f'Exact MAP · rank 1\nk = 15 · PMP {top.iloc[0].pmp:.6f}', f"True M* · rank 8\nk = 15 · PMP {truth['pmp']:.6f}"])
    for j in [14, 29]:
        ax.add_patch(Rectangle((j - 0.5, -0.5), 1, 2, fill=False, edgecolor=RED, lw=2))
    fig.subplots_adjust(left=0.18, bottom=0.21, top=0.8)
    save(fig, '08_exact_true_model_vs_map.png', 'Known synthetic truth is distinct from posterior optimality. Red outlines identify differing predictor indicators; controls w1, w2 are common.')

    assert len(checks['figures'])==9
    (output_dir/'figure_presentation.json').write_text(json.dumps({'catalog':list(catalog.values()),'validation':checks},indent=2),encoding='utf-8')
    return list(catalog.values()),checks

def display_suite(output_dir, catalog=FIGURE_CATALOG):
    """Notebook order is always external Markdown title → PNG → Markdown note."""
    from IPython.display import Markdown, Image, display
    for entry in catalog:
        display(Markdown('### '+entry['title']))
        display(Image(filename=str(Path(output_dir)/entry['filename'])))
        display(Markdown('*Notes:* '+entry['note']))
