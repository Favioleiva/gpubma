"""BFG figures using recovered model-space and publication BMA design grammar.

No global posterior normalization or posterior coefficient uncertainty is
computed here. All model and inclusion weights are conditional on evaluated E.
See docs/canonical_figures.md for design provenance and intentional adaptations.
"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import json
import math
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from .figure_data import prepare_discovered_set, observed_ridge
from gpubma.priors.model_priors import log_model_prior_function

BLUE, RED, GRAY = '#1f77b4', '#d62728', '#7f7f7f'
POS, NEG, INK = '#2166ac', '#b2182b', '#0f172a'
ORANGE, SPINE = '#ea580c', '#94a3b8'
CONDITIONAL = 'Weights normalize over BFG-evaluated models E only; not global PMP/PIP.'
STAGE_COLORS = {'EXACT_WING': POS, 'FORWARD_GENEALOGY': NEG,
                'BACKWARD_GENEALOGY': '#9a3412', 'BEAM': ORANGE,
                'RANDOM_BULK': GRAY, 'RANDOM_TAIL': '#94a3b8', 'ELITE_HIT': RED}
FILENAMES = ('01_evaluated_model_space_cloud.png', '02_progressive_model_space_cloud.png',
             '03_reticular_log_numerator_ridgeline.png', '04_variable_inclusion_model_map.png',
             '05_top_models_composition_weights.png', '06_discovered_model_size_distribution.png',
             '07_conditional_coefficient_ridgeline.png', '08_champion_trajectory.png',
             '09_reticular_search_map.png', '10_champion_genealogy.png',
             '11_observed_reticular_frontiers.png')


def _label(value, width=26):
    return textwrap.fill(str(value), width, break_long_words=True)


def _caption(fig, text=CONDITIONAL):
    fig.supxlabel(text, fontsize=9, color=INK)


def _axes(ax, grid=False):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(SPINE)
    ax.tick_params(labelsize=9.5)
    if grid:
        ax.grid(axis='y', alpha=.20)
        ax.set_axisbelow(True)


def _stages(d, ax, limit=None, size=5):
    n = len(d['ids']) if limit is None else limit
    stages = d['frame'].plot_stage.to_numpy()[:n]
    for stage in dict.fromkeys(stages):
        use = np.flatnonzero(stages == stage)
        ax.scatter(d['sizes'][use], d['scores'][use], s=size,
                   c=STAGE_COLORS.get(stage, INK), alpha=.35, linewidths=0,
                   label=stage.replace('_', ' ').title(), rasterized=True)


def _clouds(d):
    with plt.rc_context({'font.family': 'serif'}):
        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
        _stages(d, ax)
        ax.set(xlabel='Candidate model size k', ylabel='Log posterior model numerator',
               title='BFG-Evaluated Model-Space Cloud', xlim=(-.5, d['p']+.5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)); _axes(ax, True)
        ax.legend(loc='best', fontsize=8, framealpha=.92, edgecolor='.8', markerscale=2)
        _caption(fig, f'Each point is one BFG-evaluated model. {len(d["ids"]):,} / {1 << d["p"]:,} models evaluated; no exhaustive-coverage claim.')
        # Cumulative snapshots at REAL stage-transition boundaries; never invent stages.
        stages = d['frame'].plot_stage.to_numpy()
        bounds = np.r_[np.flatnonzero(stages[1:] != stages[:-1])+1, len(stages)]
        first_stage_ends = {}
        for stop in bounds:
            first_stage_ends.setdefault(stages[stop-1], int(stop))
        # Show the first complete visit to distinct stages, then the final ledger.
        # This avoids near-identical snapshots from brief repeated elite visits.
        stops = sorted(set(list(first_stage_ends.values())[:4] + [len(stages)]))
        progressive, axs = plt.subplots(1, len(stops), figsize=(4*len(stops), 4.5),
                                         sharey=True, squeeze=False, layout='constrained')
        for axp, stop in zip(axs[0], stops):
            _stages(d, axp, int(stop), 2)
            axp.set_title(f'First {stop:,} evaluations\nThrough {stages[stop-1].replace("_", " ").title()}', fontsize=9)
            axp.set(xlabel='Candidate model size k', xlim=(-.5, d['p']+.5))
            axp.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5)); _axes(axp, True)
        axs[0, 0].set_ylabel('Log posterior model numerator')
        progressive.suptitle('Progressive Revelation of the BFG-Evaluated Cloud', fontsize=13)
        _caption(progressive, 'Actual cumulative ledger prefixes at stage boundaries; repeated stage visits remain in their original order.')
    return fig, progressive


def _ridges(d):
    with plt.rc_context({'font.family': 'serif'}):
        fig, ax = plt.subplots(figsize=(12, max(7, .43*(d['p']+1)+2)), layout='constrained')
        rows = []
        best = d['top'][0]
        lower, upper = np.full(d['p']+1, np.nan), np.full(d['p']+1, np.nan)
        counts = np.bincount(d['sizes'], minlength=d['p']+1)
        for k in np.unique(d['sizes']):
            indices = np.flatnonzero(d['sizes'] == k)
            r = observed_ridge(d['scores'][indices], grid_size=300)
            lower[k], upper[k] = r['xmin'], r['xmax']
            wing_only = (d['frame'].iloc[indices].plot_stage == 'EXACT_WING').all()
            fill, line = ('#0284c7', INK) if wing_only else (ORANGE, '#9a3412')
            if k == d['sizes'][best]:
                fill, line = '#dc2626', '#991b1b'
            baseline, = ax.plot([r['xmin'], r['xmax']], [k, k], color='#cbd5e1', lw=.5, alpha=.60)
            baseline.set_gid(f'support:{k}')
            if r['kind'] == 'kde':
                h = r['height'].copy(); h[[0, -1]] = 0
                ax.fill_between(r['x'], k, k+.85*h, color=fill, alpha=.14, linewidth=0)
                curve, = ax.plot(r['x'], k+.85*h, c=line, lw=1.15)
                curve.set_gid(f'ridge:{k}')
            else:
                coll = ax.vlines(r['x'], k, k+.65*r['height'], color=line, lw=1.15)
                coll.set_gid(f'spikes:{k}')
            rows.append(dict(k=int(k), evaluated_count=len(indices), observed_min=r['xmin'],
                             observed_max=r['xmax'], representation=r['kind'],
                             champion_model_id=str(d['ids'][indices[np.argmax(d['scores'][indices])]])))
        ks = np.arange(d['p']+1)
        ax.plot(lower, ks, '--o', c='#2563eb', lw=1.2, ms=3, alpha=.85, label='Observed lower frontier $L_E(k)$')
        ax.plot(upper, ks, '-D', c=ORANGE, lw=1.2, ms=3, alpha=.9, label='Observed upper frontier / shell champions $U_E(k)$')
        ax.scatter([d['scores'][best]], [d['sizes'][best]], marker='*', s=210, c=RED,
                   edgecolor='white', lw=1, zorder=20, label=f'Best discovered model (k={d["sizes"][best]})')
        tick_k = ks[::max(1, math.ceil(len(ks)/25))]
        ax.set_yticks(tick_k, [f'k = {k}   [n={counts[k]:,}]' for k in tick_k])
        ax.set_ylim(-.6, d['p']+1.3)
        ax.set(xlabel='Log unnormalized model evidence / log numerator', ylabel='Candidate model size k [evaluated count n]',
               title='Log BMA Numerator Across BFG-Explored Reticula')
        _axes(ax)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1.0), fontsize=8, framealpha=.92)
        _caption(fig, 'Each ridge is restricted to the observed min–max support of evaluated models in that reticula.\n'
                 'Unweighted score shapes, unit peak; sparse shells are spikes. Frontiers do not certify the full model space.')
    return fig, pd.DataFrame(rows)


def _frontiers(d, support):
    with plt.rc_context({'font.family': 'serif'}):
        fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
        # Missing reticula stay NaN so a line cannot imply their observation.
        indexed = support.set_index('k').reindex(range(d['p']+1))
        ax.plot(indexed.index, indexed.observed_min, '--o', color='#2563eb', lw=1.2,
                ms=3, label='Observed lower frontier')
        ax.plot(indexed.index, indexed.observed_max, '-D', color=ORANGE, lw=1.2,
                ms=3, label='Observed upper frontier / shell champions')
        best = d['top'][0]
        ax.scatter(d['sizes'][best], d['scores'][best], marker='*', s=180,
                   color=RED, edgecolor='white', zorder=4, label='Best discovered model')
        ax.set(xlabel='Candidate model size k', ylabel='Log posterior model numerator',
               title='Observed Reticular Evidence Frontiers')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)); _axes(ax, True)
        ax.legend(fontsize=9, framealpha=.92)
        _caption(fig, 'Minimum and maximum among BFG-evaluated models in each reticula.\n'
                 'Observed frontiers are not exhaustive bounds on the full model space.')
    return fig


def _display_variables(d, report):
    cols = d['columns'].tolist()
    names = [d['names'][j] for j in cols]
    weights = d['variable_weights'][cols].tolist()
    # FE dummy columns never become inclusion discoveries or coefficient rows.
    controls = [] if report is None else [v for v in report['controls'] if not any(v in g for g in report['fe_groups'].values())]
    return names+controls, np.array(weights+[1.]*len(controls)), len(names)


def _cell_colors(d, report, indices, names):
    arr = np.zeros((len(indices), len(names)), dtype=int)
    for row, i in enumerate(indices):
        for col, name in enumerate(names):
            if name not in d['names']:
                arr[row, col] = 4  # always-in, not discovery
            elif d['inclusion'][i, d['names'].index(name)]:
                v = np.nan if report is None else report['coef'][i, report['names'].index(name)]
                arr[row, col] = 1 if v > 0 else 2 if v < 0 else 3
    return arr


def _structure(d, report):
    names, weights, candidate_count = _display_variables(d, report)
    top = d['top']
    color_list = ['white', POS, NEG, GRAY, INK]
    cmap = ListedColormap(color_list)
    heights = max(5.5, .37*len(names)+2)
    fig, (ax, side) = plt.subplots(1, 2, figsize=(13.2, heights), sharey=True,
                                 gridspec_kw={'width_ratios': [5.2, 1.4], 'wspace': .12}, layout='constrained')
    # Proportional model map uses up to 50 models, normalized over ALL E.
    shown = np.argsort(-d['scores'], kind='stable')[:50]
    widths = d['weights'][shown]
    edges = np.r_[0, np.cumsum(widths)]
    cellcolors = _cell_colors(d, report, shown, names)
    for row in range(len(names)):
        for col in range(len(shown)):
            ax.add_patch(Rectangle((edges[col], row), widths[col], 1,
                                   facecolor=color_list[cellcolors[col, row]], edgecolor='none'))
    for x in edges[1:-1]:
        ax.axvline(x, color='white', lw=.35, alpha=.9)
    remainder = max(0., 1-float(widths.sum()))
    ax.add_patch(Rectangle((edges[-1], 0), remainder, len(names), facecolor='.86', edgecolor='.45', hatch='///'))
    if remainder > .04:
        ax.text(edges[-1]+remainder/2, len(names)/2, f'Other evaluated models\nweight = {remainder:.3f}',
                rotation=90, ha='center', va='center', fontsize=8)
    for edge in range(len(names)+1):
        ax.axhline(edge, color='.85', lw=.45)
        side.axhline(edge, color='.88', lw=.45, zorder=0)
    ax.set_yticks(np.arange(len(names))+.5, [_label(v)+(' [always in]' if i >= candidate_count else '') for i, v in enumerate(names)])
    ax.set(xlim=(0, 1), ylim=(len(names), 0), xlabel='Cumulative discovered-set model weight')
    ax.set_title('Proportional model composition', fontsize=11)
    y = np.arange(len(names))+.5
    side.barh(y, weights, color=[BLUE]*candidate_count+[INK]*(len(names)-candidate_count), height=.75)
    side.set(xlim=(0, 1.16), xlabel='Inclusion weight within E')
    side.tick_params(axis='y', labelleft=False)
    for yi, w in zip(y, weights):
        side.text(min(w+.02, 1.07), yi, f'{w:.2f}', va='center', fontsize=8.5, weight='bold', color=BLUE if yi < candidate_count else INK)
    for a in (ax, side):
        _axes(a)
    fig.suptitle('Discovered-Set Variable Inclusion Weights and Model Structure', fontsize=14)
    _caption(fig, 'Blue/red: positive/negative OLS coefficient; gray: coefficient unavailable; dark: always-in control.\n'+CONDITIONAL)
    # Model rows and weight bars share the exact same axis, including inversion.
    pair, (matrix, bars) = plt.subplots(1, 2, figsize=(max(12, .43*len(names)+5), max(5.5, .37*len(top)+2.5)),
                                       sharey=True, gridspec_kw={'width_ratios': [5.2, 1.8], 'wspace': .08}, layout='constrained')
    matrix.imshow(_cell_colors(d, report, top, names), cmap=cmap, vmin=0, vmax=4,
                  interpolation='nearest', aspect='auto', origin='upper')
    matrix.set_xticks(range(len(names)), [_label(v, 18)+(' [A]' if j >= candidate_count else '') for j, v in enumerate(names)], rotation=45, ha='right')
    matrix.set_yticks(range(len(top)), [f'({rank+1})   k={d["sizes"][i]}' for rank, i in enumerate(top)])
    matrix.set_title('Top-model composition (columns ordered by inclusion weight)', fontsize=10)
    for edge in np.arange(len(top)+1)-.5:
        matrix.axhline(edge, color='.85', lw=.45)
    bars.barh(range(len(top)), d['weights'][top], color=BLUE, height=.75)
    for row, i in enumerate(top):
        bars.text(d['weights'][i]+max(d['weights'][top])*.025, row, f'{d["weights"][i]:.4f}', va='center', fontsize=8.5)
    bars.set_xlim(0, max(d['weights'][top])*1.25)
    bars.tick_params(axis='y', labelleft=False)
    bars.set_xlabel('Discovered-set model weight')
    bars.set_title('Same model rows', fontsize=10)
    matrix.set_ylim(len(top)-.5, -.5)
    for a in (matrix, bars):
        _axes(a)
    pair.suptitle('Discovered-Set Model Weights Aligned with Composition', fontsize=14)
    _caption(pair, 'White: excluded; blue/red: OLS sign; gray: unavailable; [A]: always in (excluded from k).\n'+CONDITIONAL)
    return fig, pair


def _size(d, result):
    fig, ax = plt.subplots(figsize=(11.8, 4.4), layout='constrained')
    ks = np.arange(d['p']+1)
    prior = result.reproducibility.get('config', {}).get('model_prior')
    prior_weights = np.full(d['p']+1, np.nan)
    if prior is not None:
        lp, description = log_model_prior_function(tuple(prior) if isinstance(prior, list) else prior, d['p'])
        prior_weights = np.exp([lp(int(k))+math.lgamma(d['p']+1)-math.lgamma(k+1)-math.lgamma(d['p']-k+1) for k in ks])
        ax.plot(ks, prior_weights, 'o--', c='.45', ms=6.5, lw=1.6, label='Configured prior: '+description)
    ax.plot(ks, d['size_weights'], 'o-', c=BLUE, ms=7, lw=2.2, label=f'Discovered-set size weights (mean = {d["expected_size"]:.2f})')
    ax.axvline(d['expected_size'], c=RED, ls='--', lw=1.8, label='Discovered-set mean size')
    ax.axvline(d['champion_size'], c=INK, ls=':', lw=1.5, label=f'Best discovered model size = {d["champion_size"]}')
    ax.set(xlabel='Number of included candidate predictors k (always-in controls excluded)',
           ylabel='Model-size weight', title='Discovered-Set Model Size Distribution')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); _axes(ax, True)
    ax.legend(fontsize=8.5, framealpha=.92, edgecolor='.8')
    _caption(fig, 'Conditional size weights describe E; the configured prior describes the full candidate model space.')
    return fig, pd.DataFrame({'k': ks, 'discovered_set_weight': d['size_weights'], 'configured_prior_weight': prior_weights})


def _coefficients(d, report):
    names, inclusion, n_candidates = _display_variables(d, report)
    rows = []
    for name, incl in zip(names, inclusion):
        values = report['coef'][:, report['names'].index(name)]
        valid = np.isfinite(values)
        weight = d['weights'][valid]
        available = float(weight.sum())
        mean = float(np.average(values[valid], weights=weight)) if available > 0 else np.nan
        rows.append(dict(variable=name, inclusion_weight=incl, available_fit_weight=available,
                         mean_ols_conditional=mean, always_in=name in report['controls']))
    rows.sort(key=lambda r: (r['always_in'], r['inclusion_weight'] < .5,
                             -r['mean_ols_conditional'] if np.isfinite(r['mean_ols_conditional']) else np.inf))
    fig, (ax, side) = plt.subplots(1, 2, figsize=(13.6, max(6, .43*len(rows)+2.2)), sharey=True,
                                 gridspec_kw={'width_ratios': [5.6, 1.4], 'wspace': .10}, layout='constrained')
    for yi, row in enumerate(rows):
        values = report['coef'][:, report['names'].index(row['variable'])]
        valid = np.isfinite(values)
        weights = d['weights'][valid]
        values = values[valid]
        color = INK if row['always_in'] else GRAY if row['inclusion_weight'] < .5 else BLUE if row['mean_ols_conditional'] >= 0 else RED
        if len(values) and weights.sum() > 0:
            if len(np.unique(values)) >= 8 and values.min() < values.max():
                h, edges = np.histogram(values, bins=min(40, max(8, int(np.sqrt(len(values))))), weights=weights)
                h = .78*h/h.max()
                ax.stairs(yi-h, edges, baseline=yi, fill=True, color=color, alpha=.32)
                ax.stairs(yi-h, edges, baseline=None, color=color, lw=1.4)
            else:
                unique, inv = np.unique(values, return_inverse=True)
                mass = np.bincount(inv, weights=weights)
                ax.vlines(unique, yi, yi-.78*mass/mass.max(), color=color, lw=1.4)
            ax.plot(row['mean_ols_conditional'], yi-.05, 'o', ms=5.2, color=color)
        else:
            ax.text(.02, yi, 'No valid coefficient fit', transform=ax.get_yaxis_transform(), fontsize=8)
        if not row['always_in']:
            ax.scatter([0], [yi], s=10+85*(1-row['inclusion_weight']), facecolor='white', edgecolor=color, lw=1.3, zorder=4)
        side.barh(yi, row['inclusion_weight'], color=color, height=.72)
        side.text(row['inclusion_weight']+.02, yi, f'{row["inclusion_weight"]:.2f}', va='center', color=color, fontsize=8.5, weight='bold')
    ax.set_yticks(range(len(rows)), [_label(r['variable'])+(' [always in]' if r['always_in'] else '') for r in rows])
    ax.set_ylim(len(rows)-.4, -1.1)
    ax.axvline(0, color='.30', ls='--', lw=1)
    ax.set_xlabel('OLS coefficient point estimate (empirical variation across evaluated models)')
    side.set(xlim=(0, 1.16), xlabel='Inclusion within E')
    side.tick_params(axis='y', labelleft=False)
    for a in (ax, side):
        _axes(a)
    fig.suptitle('Discovered-Set Coefficient Structure, Conditional on Inclusion', fontsize=14)
    _caption(fig, 'Weighted empirical OLS point estimates among available full-rank fits; not posterior draws or confidence densities.\n'
             'Blue/red: mean sign; gray: inclusion < 0.50 (display grouping only); dark: always in. Open circles encode exclusion weight.')
    return fig, pd.DataFrame(rows)


def _trajectory(d):
    incumbent = np.maximum.accumulate(d['scores'])
    hits = np.flatnonzero(np.r_[True, np.diff(incumbent) > 0])
    last = int(d['order'][hits[-1]])
    early = last < len(d['ids'])*.25 and len(d['ids']) > 10
    fig, axs = plt.subplots(1, 2 if early else 1, figsize=(11.8, 4.5), squeeze=False,
                            gridspec_kw={'width_ratios': [3, 1]} if early else {}, layout='constrained')
    for j, ax in enumerate(axs[0]):
        ax.step(d['order'], incumbent, where='post', color=POS, lw=1.6)
        ax.scatter(d['order'][hits], incumbent[hits], color=RED, s=28, zorder=4)
        ax.set_xlabel('Cumulative unique evaluations'); _axes(ax, True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3 if early and j == 1 else 6))
        if early and j == 0:
            ax.set_xlim(.8, max(3, last*1.3))
            ax.set_title('Improvement detail', fontsize=11)
        else:
            ax.set_title('Full evaluation budget' if early else 'Incumbent improvements', fontsize=11)
    axs[0, 0].set_ylabel('Best log numerator found so far')
    fig.suptitle('BFG Champion Trajectory', fontsize=14)
    _caption(fig, f'{len(hits)} incumbent records. Last strict improvement: evaluation {last:,}; run ends at {len(d["ids"]):,}. No global MAP certificate.')
    return fig


def _genealogy(d):
    champions = [np.flatnonzero(d['sizes'] == k)[np.argmax(d['scores'][d['sizes'] == k])] for k in np.unique(d['sizes'])]
    lookup = {m: i for i, m in enumerate(d['ids'])}
    nodes = set(champions)
    edges = []
    rows = []
    for i in champions:
        parent = d['frame'].iloc[i].parent_id
        parent = None if parent is None or pd.isna(parent) else int(parent)
        if parent in lookup:
            pi = lookup[parent]
            nodes.add(pi); edges.append((pi, i))
        rows.append(dict(k=int(d['sizes'][i]), model_id=str(d['ids'][i]),
                         model=', '.join(d['names'][j] for j in np.flatnonzero(d['inclusion'][i])) or '(no candidates)',
                         log_numerator=d['scores'][i], retained_parent_id='' if parent is None else str(parent),
                         parent_evaluated=parent in lookup, discovery_stage=d['frame'].iloc[i].plot_stage))
    fig, ax = plt.subplots(figsize=(11.8, 5.5), layout='constrained')
    for parent, child in edges:
        ax.annotate('', xy=(d['sizes'][child], d['scores'][child]), xytext=(d['sizes'][parent], d['scores'][parent]),
                    arrowprops=dict(arrowstyle='->', color=SPINE, lw=1.2, shrinkA=5, shrinkB=5))
    others = sorted(nodes-set(champions))
    if others:
        ax.scatter(d['sizes'][others], d['scores'][others], s=30, facecolor='white', edgecolor=SPINE, label='Retained predecessor')
    ax.scatter(d['sizes'][champions], d['scores'][champions], c=POS, s=35, label='Shell champion', zorder=3)
    best = d['top'][0]
    ax.scatter(d['sizes'][best], d['scores'][best], marker='*', s=200, c=RED, edgecolor='white', label='Best discovered model', zorder=4)
    ax.set(xlabel='Candidate model size k', ylabel='Log posterior model numerator', title='Champion Genealogy in the Evaluated Set')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); _axes(ax, True)
    ax.legend(fontsize=9, framealpha=.92)
    _caption(fig, f'{len(edges)} retained predecessor edges to shell champions. Missing edges are not inferred.\n'
             'CSV supplies masks, composition, parent and stage. Shell champions need not form one dynasty.')
    return fig, pd.DataFrame(rows)


def canonical_bfg_figures(result, top_k=12, report=None):
    """Return rebuilt canonical views; coefficients require explicit refit data."""
    d = prepare_discovered_set(result, top_k)
    if report is not None and (report['data']['ids'] != d['ids'] or not np.array_equal(report['data']['scores'], d['scores'])):
        raise ValueError('Coefficient report does not belong to this discovery ledger')
    figures, tables = OrderedDict(), {}
    style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
             'axes.labelsize': 10.5, 'figure.facecolor': 'white', 'axes.facecolor': 'white'}
    with plt.rc_context(style):
        figures[FILENAMES[0]], figures[FILENAMES[1]] = _clouds(d)
        figures[FILENAMES[2]], tables['ridge_observed_support'] = _ridges(d)
        figures[FILENAMES[3]], figures[FILENAMES[4]] = _structure(d, report)
        figures[FILENAMES[5]], tables['model_size_weights'] = _size(d, result)
        if report is not None:
            figures[FILENAMES[6]], tables['coefficient_summary'] = _coefficients(d, report)
        figures[FILENAMES[7]] = _trajectory(d)
        fig, ax = plt.subplots(figsize=(11.8, 4.8), layout='constrained')
        stages = d['frame'].plot_stage.to_numpy()
        for stage in dict.fromkeys(stages):
            mask = stages == stage
            ax.scatter(d['order'][mask], d['sizes'][mask], s=7, c=STAGE_COLORS.get(stage, INK),
                       alpha=.5, linewidths=0, label=stage.replace('_', ' ').title())
        ax.set(xlabel='Unique evaluation order', ylabel='Candidate model size k', title='Reticular Search Map: Actual Stage Provenance')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)); _axes(ax)
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1), framealpha=.92)
        _caption(fig, 'Each evaluated model appears once, labeled by its retained first registration stage.')
        figures[FILENAMES[8]] = fig
        figures[FILENAMES[9]], tables['champion_genealogy'] = _genealogy(d)
        figures[FILENAMES[10]] = _frontiers(d, tables['ridge_observed_support'])
    tables['variable_weights'] = pd.DataFrame({'variable': d['names'], 'discovered_set_inclusion_weight': d['variable_weights']})
    tables['top_models'] = pd.DataFrame({'rank': np.arange(1, len(d['top'])+1),
                                       'model_id': [str(d['ids'][i]) for i in d['top']],
                                       'k': d['sizes'][d['top']], 'log_numerator': d['scores'][d['top']],
                                       'discovered_set_model_weight': d['weights'][d['top']]})
    return figures, tables


def save_canonical_figures(figures, tables, output_dir, dpi=300):
    """PNG only; refuse reuse of a previous output directory."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    paths = []
    for name, fig in figures.items():
        if name not in FILENAMES:
            raise ValueError('Unknown canonical figure name')
        path = directory / name
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        paths.append(path)
    for name, table in tables.items():
        table.to_csv(directory / (name+'.csv'), index=False)
    (directory / 'figure_semantics.json').write_text(json.dumps({
        'normalization': 'all evaluated models E, not selected top K',
        'global_posterior_calibration': False,
        'ridge_measure': 'unweighted observed log numerators; unit peak, not posterior mass',
        'ridge_support': 'exact per-shell observed min and max; no extended baseline',
        'ridge_kde': 'Silverman, minimum 8 observations and 3 distinct scores; otherwise empirical spikes',
        'coefficient_measure': 'discovered-weighted empirical OLS point estimates, conditional on inclusion and available full-rank fit',
        'always_in': 'not candidate discoveries; not counted in k',
        'genealogy': 'retained predecessor edges only', 'format': 'PNG only', 'dpi': dpi}, indent=2), encoding='utf-8')
    return paths
