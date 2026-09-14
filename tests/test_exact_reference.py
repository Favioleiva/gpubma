from pathlib import Path
import sys, json, shutil
import numpy as np, pandas as pd, pytest
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use("Agg")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"examples"))
from exact_reference_workflow import load_reference, export_tables
import exact_reference_figures as module
INPUT=ROOT/"benchmark/exact_p30/reference"
shell=pd.read_csv(INPUT/"exhaustive_reticular_summary.csv")
pip=pd.read_csv(INPUT/"exact_global_pips.csv").exact_global_pip.to_numpy()
top=pd.read_csv(INPUT/"exact_top10000_models.csv")
density=np.load(INPUT/"coefficient_densities.npz")
checks=[]

def inspect(name,fig):
    assert not fig.texts
    assert all(not ax.get_title(loc=loc) for ax in fig.axes for loc in ['left','center','right'])
    axes=fig.axes
    if name.startswith('00'):
        actual=np.ma.asarray(axes[0].images[0].get_array())
        expected=pd.read_csv(INPUT/'predictor_correlations.csv',index_col=0).to_numpy()
        np.testing.assert_array_equal(actual.data,expected)
    if name.startswith('01'):
        for line,values in zip(axes[0].lines,[shell.min_log_numerator,shell.max_log_numerator]):
            np.testing.assert_array_equal(line.get_xdata(),shell.k);np.testing.assert_array_equal(line.get_ydata(),values)
    if name.startswith('02'):
        for k in range(31):
            lo,hi=shell.iloc[k][['min_log_numerator','max_log_numerator']]
            assert any(np.min(line.get_xdata())==lo and np.max(line.get_xdata())==hi for line in axes[0].lines)
    if name.startswith('03'):
        assert len(axes)==2 and not any(ax.child_axes for ax in axes)
        for ax,values in zip(axes,[shell.min_log_numerator,shell.max_log_numerator]):
            np.testing.assert_array_equal(ax.lines[0].get_ydata(),values)
        anchors=[a.xy for a in axes[1].artists]
        assert anchors==[(15,json.loads((INPUT/'exact_map.json').read_text())['log_score']),(15,json.loads((INPUT/'exact_true_model.json').read_text())['log_score'])]
    if name.startswith('04'):
        np.testing.assert_array_equal([p.get_width() for p in axes[1].patches],pip[np.argsort(-pip,kind='stable')])
    if name.startswith('05'):
        np.testing.assert_array_equal([p.get_width() for p in axes[1].patches],top.head(15).pmp)
    if name.startswith('06'):
        s=pd.read_csv(INPUT/'exact_model_size_distribution.csv')
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(),s.prior_probability)
        np.testing.assert_array_equal(axes[0].lines[1].get_ydata(),s.exact_posterior_probability)
    if name.startswith('07'):
        grid=density['grid'];curves=density['conditional_density'];signed=trapezoid(grid*curves,grid,axis=1)
        blue=sorted([j for j in range(30) if pip[j]>=.5 and signed[j]>=0],key=lambda j:(-signed[j],j))
        red=sorted([j for j in range(30) if pip[j]>=.5 and signed[j]<0],key=lambda j:(signed[j],j))
        gray=sorted([j for j in range(30) if pip[j]<.5],key=lambda j:(-pip[j],j))
        order=blue+red+gray
        labels=[t.get_text() for t in axes[0].get_yticklabels()][::-1]
        assert labels==[f'x{j+1}' for j in order]
        for row,j in enumerate(order):
            line=axes[0].lines[row]
            np.testing.assert_array_equal(line.get_xdata(),grid[j])
            expected=29-row+curves[j]/curves[j].max()*.78
            np.testing.assert_array_equal(line.get_ydata(),expected)
            expected_color='#7f7f7f' if pip[j]<.5 else ('#1f77b4' if signed[j]>=0 else '#d62728')
            assert line.get_color()==expected_color
        np.testing.assert_array_equal([p.get_width() for p in axes[1].patches],pip[order])
        np.testing.assert_allclose([p.get_y()+p.get_height()/2 for p in axes[1].patches],range(29,-1,-1),atol=1e-15)
    if name.startswith('08'):
        expected=np.array([[bool(m>>j&1) for j in range(30)] for m in [536887295,32767]])
        np.testing.assert_array_equal(axes[0].images[0].get_array(),expected)
    checks.append(name)

def test_all_nine_figures_preserve_exact_scientific_artists(tmp_path):
    load_reference(INPUT)
    catalog,validation=module.render_suite(INPUT,tmp_path,inspect_figure=inspect)
    assert len(catalog)==9 and len(list(tmp_path.glob("*.png")))==9
    table=export_tables(INPUT,tmp_path).read_text()
    assert "0.01881348830750715" in table and "0.4715187997893168" in table
    assert "one variable substitution" in table

def test_reference_corruption_fails_closed(tmp_path):
    copy=tmp_path/"reference"
    shutil.copytree(INPUT,copy)
    with (copy/"exact_map.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError,match="checksum mismatch"):
        load_reference(copy)
