"""Descriptive OLS refits of discovered models, separate from BFG scoring.

These are classical per-model estimates, not model-averaged posterior moments.
Conventional standard errors assume a full-rank homoskedastic linear model.
They do not account for model selection, clustering, or posterior uncertainty.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .figure_data import prepare_discovered_set
from gpubma.priors.model_priors import log_model_prior_function


def refit_discovered_models(result, y, X, always_in=None, fe_groups=None):
    """Fit only E for requested descriptive diagnostics; never score new masks.

    X and always_in must be named dataframes in the original observation order.
    fe_groups optionally maps an FE label to explicitly supplied always-in dummy
    columns. Coding/reference levels are supplied by the user, never inferred.
    Singular models retain scores/weights but have unavailable coefficient/SE
    entries. Excluded coefficients stay NaN, never invented zero observations.
    """
    d = prepare_discovered_set(result)
    if not isinstance(X, pd.DataFrame) or list(X.columns) != d['names']:
        raise ValueError('X columns must exactly match the scored candidate order')
    controls = pd.DataFrame(index=X.index) if always_in is None else always_in
    if not isinstance(controls, pd.DataFrame) or not controls.index.equals(X.index):
        raise ValueError('Always-in controls must share the original X row index')
    if isinstance(y, pd.Series) and not y.index.equals(X.index):
        raise ValueError('y must share the original X row index')
    names = ['Intercept'] + list(controls.columns) + d['names']
    if len(names) != len(set(names)):
        raise ValueError('Intercept, controls and candidates need distinct names')
    y = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones(len(X)), controls.to_numpy(dtype=float)])
    values = X.to_numpy(dtype=float)
    if y.shape != (len(X),) or len(y) != result.n_obs:
        raise ValueError('Original observation count/response shape required')
    if not all(np.isfinite(a).all() for a in (y, A, values)):
        raise ValueError('Nonfinite regression data')
    fe_groups = {} if fe_groups is None else dict(fe_groups)
    assigned = [v for cols in fe_groups.values() for v in cols]
    if (len(assigned) != len(set(assigned)) or any(not cols for cols in fe_groups.values())
            or not set(assigned) <= set(controls.columns)):
        raise ValueError('FE groups must identify disjoint, nonempty always-in dummy lists')
    if assigned and not np.isin(controls[assigned].to_numpy(), [0, 1]).all():
        raise ValueError('Declared FE columns must be explicitly coded 0/1 dummies')
    config = result.reproducibility.get('config', {})
    prior = config.get('model_prior')
    log_prior = None
    if prior is not None:
        log_prior, _ = log_model_prior_function(tuple(prior) if isinstance(prior, list) else prior, d['p'])
    beta = np.full((len(d['ids']), len(names)), np.nan)
    se = np.full_like(beta, np.nan)
    rank_order = np.argsort(-d['scores'], kind='stable')
    ranks = np.empty(len(rank_order), dtype=int)
    ranks[rank_order] = np.arange(1, len(ranks)+1)
    diagnostics = []
    tss = float(np.sum((y-y.mean())**2))
    for i, mid in enumerate(d['ids']):
        candidate_idx = np.flatnonzero(d['inclusion'][i])
        cols = np.r_[np.arange(A.shape[1]), A.shape[1]+candidate_idx]
        design = np.column_stack([A, values[:, candidate_idx]])
        coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        residual = y-design@coef
        sse = float(residual@residual)
        n, dim = design.shape
        df = n-rank
        full = rank == dim
        if full:
            beta[i, cols] = coef
            if df > 0:
                _, triangular = np.linalg.qr(design, mode='reduced')
                inv_r = np.linalg.solve(triangular, np.eye(dim))
                se[i, cols] = np.sqrt((sse/df)*np.sum(inv_r**2, axis=1))
        # Perfect-fit Gaussian MLE has no finite maximizer for sigma^2.
        ll = -.5*n*(math.log(2*math.pi)+1+math.log(sse/n)) if sse > np.finfo(float).eps*max(1., tss)*n else np.nan
        n_parameters = rank+1  # includes intercept, supplied controls and variance
        lp = float(log_prior(int(d['sizes'][i]))) if log_prior else np.nan
        diagnostics.append(dict(
            model_id=str(mid), bfg_rank=int(ranks[i]), observations=n,
            candidate_regressors=d['p'], model_size=int(d['sizes'][i]),
            always_in_count=controls.shape[1], design_rank=int(rank), residual_df=int(df),
            coefficient_status='OLS_FULL_RANK' if full else 'UNAVAILABLE_RANK_DEFICIENT',
            r_squared=1-sse/tss if tss > 0 else np.nan,
            adjusted_r_squared=1-(sse/df)/(tss/(n-1)) if tss > 0 and df > 0 and n > 1 else np.nan,
            rmse=math.sqrt(sse/n), residual_standard_error=math.sqrt(sse/df) if df > 0 else np.nan,
            log_likelihood=ll, aic=-2*ll+2*n_parameters,
            bic=-2*ll+math.log(n)*n_parameters,
            hqic=-2*ll+2*math.log(math.log(n))*n_parameters if n > 1 else np.nan,
            log_model_prior=lp, log_marginal_likelihood_relative=d['scores'][i]-lp,
            log_posterior_model_numerator=d['scores'][i], discovered_set_weight=d['weights'][i]))
    return dict(names=names, controls=list(controls.columns), fe_groups=fe_groups,
                coef=beta, se=se, diagnostics=pd.DataFrame(diagnostics), data=d,
                coefficient_measure='OLS point estimates across E; not posterior draws')


def _tex(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(value))


def top5_regression_table(report, outcome='y'):
    """Return booktabs LaTeX and a string dataframe with identical row ordering."""
    d = report['data']
    top = np.argsort(-d['scores'], kind='stable')[:5]
    hidden = {v for group in report['fe_groups'].values() for v in group}
    reference = report.get('benchmark_reference')
    rows = []
    for j, name in enumerate(report['names']):
        if name in hidden:
            continue
        label = name + (' [always in]' if name in report['controls'] else '')
        vals, ses = [], []
        for i in top:
            excluded = name in d['names'] and not d['inclusion'][i, d['names'].index(name)]
            b, s = report['coef'][i, j], report['se'][i, j]
            vals.append('---' if excluded else f'{b:.4f}' if np.isfinite(b) else 'NA')
            ses.append('' if excluded else f'({s:.4f})' if np.isfinite(s) else '(NA)')
        rows.append((label, vals))
        if any(ses):
            rows.append(('', ses))
    split = len(rows)
    for label in report['fe_groups']:
        rows.append((label + ' FE', ['Yes']*len(top)))
    labels = {
        'observations': 'Observations', 'candidate_regressors': 'Candidate regressors',
        'model_size': 'Model size (candidates)', 'always_in_count': 'Always-in controls',
        'design_rank': 'Design rank (including intercept)', 'residual_df': 'Residual degrees of freedom',
        'r_squared': 'R-squared', 'adjusted_r_squared': 'Adjusted R-squared', 'rmse': 'RMSE',
        'residual_standard_error': 'Residual standard error', 'log_likelihood': 'Gaussian OLS log likelihood',
        'aic': 'AIC', 'bic': 'BIC', 'hqic': 'HQIC', 'bfg_rank': 'BFG rank',
        'log_marginal_likelihood_relative': 'Log marginal likelihood (relative)',
        'log_model_prior': 'Log model prior', 'log_posterior_model_numerator': 'Log posterior model numerator',
        'discovered_set_weight': 'Discovered-set normalized model weight'}
    for key, label in labels.items():
        vals = report['diagnostics'].iloc[top][key]
        rows.append((label, [f'{v:.4f}' if np.isfinite(v) and key not in ('observations','candidate_regressors','model_size','always_in_count','design_rank','residual_df','bfg_rank') else str(int(v)) if np.isfinite(v) else 'NA' for v in vals]))
    if reference:
        exact_rank = {int(r['model_id']): int(r['rank']) for r in reference['top_models']}
        rows.append(('Exact/global BMA rank (reference)', [str(exact_rank.get(d['ids'][i], 'NA')) for i in top]))
    # If fewer than five models exist, preserve rank positions without fabrication.
    preview = pd.DataFrame([[name]+vals+['NA']*(5-len(top)) for name, vals in rows],
                           columns=['Regressor / diagnostic']+[f'({i})' for i in range(1, 6)])
    headers = [f'({j+1})' for j in range(5)]
    benchmark_rows = []
    if reference:
        if d['names'] != reference['candidates'] or report['controls'] != reference['always_in']:
            raise ValueError('Synthetic reference requires original candidates and always-in controls')
        for j, i in enumerate(top):
            marker = ('T' if d['ids'][i] == int(reference['true_model']['model_id']) else '')
            marker += 'M' if d['ids'][i] == int(reference['map_model']['model_id']) else ''
            if marker:
                headers[j] += r'\textsuperscript{' + marker + '}'
        from gpubma.benchmarks.canonical_p30 import reference_panel
        class Ledger:
            candidate_names = d['names']
            records = d['frame'].to_dict('records')
        benchmark_rows = reference_panel(Ledger(), reference)
    header = ' & '+' & '.join(headers)+r' \\'
    long = len(rows)+len(benchmark_rows) > 45
    if long:
        lines = [r'\begingroup\footnotesize', r'\begin{longtable}{lrrrrr}',
                 r'\caption{Top five BFG-discovered models: '+_tex(outcome)+r'}\label{tab:bfg-top5}\\',
                 r'\toprule', header, r'\midrule', r'\endfirsthead',
                 r'\toprule', header, r'\midrule', r'\endhead']
    else:
        lines = [r'\begin{table}[htbp]', r'\centering', r'\footnotesize',
                 r'\caption{Top five BFG-discovered models: ' + _tex(outcome) + '}',
                 r'\label{tab:bfg-top5}', r'\begin{tabular}{lrrrrr}', r'\toprule', header, r'\midrule']
    for index, row in preview.iterrows():
        if index == split:
            lines += [r'\midrule']
        ending = r' \\*' if index+1 < len(preview) and preview.iloc[index+1, 0] == '' else r' \\'
        lines.append(' & '.join(_tex(v) for v in row) + ending)
    if benchmark_rows:
        lines += [r'\midrule', r'\multicolumn{6}{l}{\textit{Synthetic benchmark reference (archived exact enumeration)}} \\*']
        for label, value in benchmark_rows:
            lines.append(_tex(label)+r' & \multicolumn{5}{p{0.51\linewidth}}{'+_tex(value)+r'} \\')
    lines += [r'\bottomrule', r'\end{longtable}' if long else r'\end{tabular}', r'\par\smallskip',
              r'\noindent\begin{minipage}{0.98\linewidth}\scriptsize',
              r'\textit{Notes:} OLS coefficients; conventional homoskedastic standard errors in parentheses.',
              r'No selection adjustment or posterior uncertainty is implied. Excluded predictors: ---; undefined or unavailable: NA.',
              r'Candidate model size excludes the intercept and always-in controls. Declared FE dummies are suppressed.',
              r'AIC/BIC/HQIC count design rank plus the variance parameter. RMSE uses $N$; residual standard error uses residual degrees of freedom.',
              r'Log marginal likelihood and numerator use the BFG common-constant convention; the former subtracts the actual log model prior.',
              r'Model weights normalize over all evaluated models $E$, not only these five, and are not global PMP.',
              r'\textsuperscript{T}: known synthetic true model; \textsuperscript{M}: archived exact MAP (synthetic reference only).'
              if reference else r'No known true model or exact/global MAP is asserted for this dataset.',
              r'BFG ranks index the evaluated set; exact/global ranks and PMPs in the benchmark panel are archived references, not BFG estimates.' if reference else '',
              r'\end{minipage}', r'\endgroup' if long else r'\end{table}', '']
    return '\n'.join(lines), preview


def write_top5(report, output_dir, outcome='y'):
    tex, table = top5_regression_table(report, outcome)
    output_dir = Path(output_dir)
    with (output_dir / 'bfg_top5_regression_comparison.tex').open('x', encoding='utf-8') as f:
        f.write(tex)
    table.to_csv(output_dir / 'bfg_top5_regression_comparison.csv', index=False, mode='x')
    report['diagnostics'].to_csv(output_dir / 'evaluated_model_diagnostics.csv', index=False, mode='x')
    if report.get('benchmark_reference'):
        from gpubma.benchmarks.canonical_p30 import reference_panel
        d = report['data']
        class Ledger:
            candidate_names = d['names']
            records = d['frame'].to_dict('records')
        pd.DataFrame(reference_panel(Ledger(), report['benchmark_reference']), columns=['benchmark_field', 'value']).to_csv(
            output_dir/'synthetic_benchmark_panel.csv', index=False, mode='x')
    return table
