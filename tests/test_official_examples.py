"""Public-data regression tests; no enumeration or scientific-campaign inputs."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import pandas as pd
import pytest
from gpubma.bfg.example_workflow import (
    PUBLIC_DATA, budget_summary, ensure_public_data, fit_example,
    load_example_data, resolve_device)
from gpubma.bfg.model_report import refit_discovered_models, top5_regression_table
from gpubma.benchmarks.canonical_p30 import load_reference, reference_panel

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('dataset', list(PUBLIC_DATA))
def test_canonical_public_bytes(dataset):
    rel, digest = PUBLIC_DATA[dataset]
    path = ROOT/rel
    assert ensure_public_data(dataset, path) == path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_checksum_substitution_rejected(tmp_path):
    bad = tmp_path/'substitute.parquet'
    bad.write_bytes(b'not the canonical dataset')
    with pytest.raises(ValueError, match='checksum'):
        ensure_public_data('canonical_p30', bad)


@pytest.mark.parametrize('budget', [0, -1, 2.5, True, '5000'])
def test_bad_budgets_rejected(budget):
    with pytest.raises(ValueError):
        budget_summary(30, budget)


def test_budget_fraction_and_cap():
    assert budget_summary(30, 5000)['maximum_percent'] == pytest.approx(.0004656612873077393)
    assert budget_summary(2, 5000, 4)['actual_fraction'] == 1
    with pytest.raises(AssertionError):
        budget_summary(30, 5000, 5001)


def test_auto_device_dispatch(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_device('auto') == 'cpu'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert resolve_device('auto') == 'cuda'
    assert resolve_device('cpu') == 'cpu'


def test_three_loaders_preserve_public_stata_rows(tmp_path):
    source = ROOT/PUBLIC_DATA['grunfeld'][0]
    frame = pd.read_stata(source, convert_categoricals=False)
    expected = load_example_data(source, 'invest', ['mvalue', 'kstock'])
    assert len(expected[0]) == 200 and expected[2] is None
    for extension in ['csv', 'parquet']:
        path = tmp_path/f'grunfeld.{extension}'
        if extension == 'csv':
            # Expand Stata float32 to its exact float64 value before text export;
            # pandas' float32 pretty-print otherwise rounds the test input.
            frame.astype({c: 'float64' for c in frame.select_dtypes(include='number')}).to_csv(path, index=False)
        else:
            frame.to_parquet(path, index=False)
        y, X, controls = load_example_data(path, 'invest', ['mvalue', 'kstock'])
        np.testing.assert_array_equal(y, expected[0])
        np.testing.assert_array_equal(X, expected[1])
        assert controls is None


def test_loader_no_implicit_sample_changes(tmp_path):
    path = tmp_path/'bad.csv'
    path.write_text('y,x,x\n1,2,3\n')
    with pytest.raises(ValueError, match='Duplicate'):
        load_example_data(path, 'y', ['x'])
    path.write_text('y,x\n1,\n')
    with pytest.raises(ValueError, match='nonfinite'):
        load_example_data(path, 'y', ['x'])
    with pytest.raises(ValueError, match='distinct'):
        load_example_data(path, 'y', ['x'], ['x'])


def test_canonical_p30_controls_and_sample():
    y, X, controls = load_example_data(ROOT/PUBLIC_DATA['canonical_p30'][0], 'y',
                                      [f'x{i}' for i in range(1,31)], ['w1','w2'])
    assert len(y) == 2000 and X.shape == (2000,30)
    assert list(controls.columns) == ['w1','w2']


def test_stata_budget_early_completion_and_table():
    y, X, controls = load_example_data(ROOT/PUBLIC_DATA['grunfeld'][0], 'invest', ['mvalue','kstock'])
    result, metrics = fit_example(y, X, controls, budget=5000, device='cpu')
    assert result.n_models_evaluated == 4 <= result.budget_models == 5000
    assert len({r['model_id'] for r in result.records}) == 4
    assert metrics['actual_fraction'] == 1
    report = refit_discovered_models(result, y, X, controls)
    tex, preview = top5_regression_table(report, 'invest')
    assert preview['(5)'].eq('NA').all()
    assert 'Synthetic benchmark reference' not in tex
    assert r'\textsuperscript{M}' not in tex
    assert preview[preview.iloc[:,0]=='BFG rank'].iloc[0,1:].tolist() == ['1','2','3','4','NA']


def test_archived_truth_is_distinct_from_map_and_first_hits_are_observed():
    reference = load_reference()
    assert reference['true_model']['model_id'] == 32767
    assert int(reference['map_model']['model_id']) == 536887295
    assert reference['true_model']['exact_posterior_rank'] == 8
    assert reference['true_model']['hamming_from_top_model'] == 2
    assert reference['log_score_atol'] == 1e-9
    ledger = SimpleNamespace(candidate_names=reference['candidates'], records=[])
    panel = dict(reference_panel(ledger, reference))
    assert panel['BFG found true model'] == 'NOT FOUND'
    assert panel['N_TRUE (unique evaluations)'] == 'Not reached'
    assert panel['True model is the exact/global MAP'] == 'NO'
    ledger.records = [{'model_id':32767,'discovery_order':17}]
    assert dict(reference_panel(ledger, reference))['N_TRUE (unique evaluations)'] == '17'
