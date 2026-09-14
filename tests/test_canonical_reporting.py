"""Synthetic-only regressions for conditional reporting and plot geometry."""
import copy
import math
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gpubma import fit_bfg, BFGConfig
from gpubma.bfg.figure_data import prepare_discovered_set, observed_ridge
from gpubma.bfg.figures import canonical_bfg_figures, save_canonical_figures
from gpubma.bfg.model_report import refit_discovered_models, top5_regression_table


@pytest.fixture(scope='module')
def fixture():
    rng = np.random.default_rng(260907)
    X = pd.DataFrame(rng.normal(size=(50, 5)), columns=['long_variable_name_'+str(j) for j in range(5)])
    y = pd.Series(X.iloc[:, 0]+rng.normal(size=50), name='outcome')
    controls = pd.DataFrame({'control': rng.normal(size=50), 'district_B': np.arange(50)%2}, index=X.index)
    result = fit_bfg(y=y, X=X, always_in=controls, config=BFGConfig(device='cpu', budget_models=25, seed=731, beam_width=5))
    report = refit_discovered_models(result, y, X, controls, {'District': ['district_B']})
    return result, report, y, X, controls


def test_denominator_and_always_in(fixture):
    result, r, *_ = fixture
    d = prepare_discovered_set(result, top_k=5)
    assert d['weights'].sum() == pytest.approx(1)
    assert d['weights'][d['top']].sum() < 1
    for j in range(d['p']):
        assert d['variable_weights'][j] == pytest.approx(sum(w for mid, w in zip(d['ids'], d['weights']) if mid & (1 << j)))
    assert r['diagnostics'].always_in_count.eq(2).all()
    assert d['p'] == 5


@pytest.mark.parametrize('values', [[1], [1, 1, 1], [2, 3, 5], list(range(20))])
def test_finite_ridge_support(values):
    r = observed_ridge(values)
    assert min(r['x']) == min(values) and max(r['x']) == max(values)
    assert r['kind'] == ('kde' if len(values) >= 8 and len(set(values)) >= 3 else 'empirical')


def test_ols_diagnostics_against_direct_reference(fixture):
    _, report, y, X, controls = fixture
    d = report['data']; i = d['top'][0]
    design = np.column_stack([np.ones(len(X)), controls, X.iloc[:, np.flatnonzero(d['inclusion'][i])]])
    beta = np.linalg.solve(design.T@design, design.T@y)
    residual = y-design@beta
    sse = residual@residual; n, q = design.shape
    se = np.sqrt(np.diag(np.linalg.inv(design.T@design))*sse/(n-q))
    available = np.isfinite(report['coef'][i])
    np.testing.assert_allclose(report['coef'][i, available], beta, atol=1e-10)
    np.testing.assert_allclose(report['se'][i, available], se, atol=1e-10)
    row = report['diagnostics'].iloc[i]
    assert row.rmse == pytest.approx(math.sqrt(sse/n))
    assert row.residual_standard_error == pytest.approx(math.sqrt(sse/(n-q)))
    assert row.aic == pytest.approx(-2*row.log_likelihood+2*(q+1))
    assert row.log_marginal_likelihood_relative+row.log_model_prior == pytest.approx(d['scores'][i])


def test_booktabs_rank_order_and_fe_suppression(fixture):
    result, report, *_ = fixture
    tex, table = top5_regression_table(report, 'outcome_%')
    assert r'\toprule' in tex and r'\midrule' in tex and r'\bottomrule' in tex
    assert r'\begin{tabular}{lrrrrr}' in tex
    assert 'district' not in tex.lower() or 'District FE' in tex
    assert 'district\\_B' not in tex
    assert 'District FE' in tex and 'control [always in]' in tex
    assert r'outcome\_\%' in tex
    rankrow = table[table.iloc[:, 0] == 'BFG rank'].iloc[0, 1:].tolist()
    assert rankrow == ['1', '2', '3', '4', '5']
    assert '---' in tex


def test_bad_data_and_fe_metadata_fail(fixture):
    result, _, y, X, controls = fixture
    with pytest.raises(ValueError):
        refit_discovered_models(result, y, X.iloc[::-1], controls)
    with pytest.raises(ValueError):
        refit_discovered_models(result, y, X, controls, {'Bad': ['not_a_control']})
    with pytest.raises(ValueError):
        refit_discovered_models(result, y, X, controls, {'Bad': ['control']})


def test_rank_deficiency_no_fabricated_uncertainty(fixture):
    result, _, y, X, controls = fixture
    bad = controls.assign(duplicate=controls['control'])
    report = refit_discovered_models(result, y, X, bad)
    assert np.isnan(report['coef']).all()
    assert np.isnan(report['se']).all()


def test_full_figure_geometry_and_png_only(fixture, tmp_path):
    result, report, *_ = fixture
    before = copy.deepcopy(result.records)
    figs, tables = canonical_bfg_figures(result, report=report)
    assert len(figs) == 11
    matrix, bars = figs['05_top_models_composition_weights.png'].axes
    assert matrix.get_shared_y_axes().joined(matrix, bars)
    centers = [bar.get_y()+bar.get_height()/2 for bar in bars.patches]
    np.testing.assert_allclose(centers, matrix.get_yticks())
    ridge = figs['03_reticular_log_numerator_ridgeline.png'].axes[0]
    support = tables['ridge_observed_support'].set_index('k')
    for line in ridge.lines:
        if (line.get_gid() or '').startswith(('ridge:', 'support:')):
            k = int(line.get_gid().split(':')[1])
            xs = line.get_xdata()
            assert min(xs) == support.loc[k, 'observed_min']
            assert max(xs) == support.loc[k, 'observed_max']
    for fig in figs.values():
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        caption = fig._supxlabel.get_window_extent(renderer)
        for ax in fig.axes:
            assert caption.y1 < ax.get_window_extent(renderer).y0
    paths = save_canonical_figures(figs, tables, tmp_path/'new', dpi=60)
    assert len(paths) == 11 and not list((tmp_path/'new').glob('*.pdf'))
    with pytest.raises(FileExistsError):
        save_canonical_figures(figs, tables, tmp_path/'new')
    assert result.records == before
    plt.close('all')


def test_prior_is_actual_uniform_size(fixture):
    result, report, *_ = fixture
    figs, tables = canonical_bfg_figures(result, report=report)
    np.testing.assert_allclose(tables['model_size_weights'].configured_prior_weight, 1/6)
    plt.close('all')


def test_no_coefficients_without_original_data(fixture):
    figs, _ = canonical_bfg_figures(fixture[0])
    assert len(figs) == 10
    assert '07_conditional_coefficient_ridgeline.png' not in figs
    plt.close('all')
