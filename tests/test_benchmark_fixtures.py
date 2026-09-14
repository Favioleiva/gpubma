"""Fixture integrity and 16-evaluation public-fixture software smokes, not new benchmarks."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from gpubma import fit_bfg

B=Path(__file__).parents[1]/'benchmark'

@pytest.mark.parametrize('p',[12,18,30])
def test_public_fixture(p):
    spec=next(r for r in json.loads((B/'fixtures.json').read_text()) if r['p']==p)
    path=B/spec['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
    with np.load(path,allow_pickle=False) as f:
        assert set(f.files)=={'X','y','always_in'}
        assert f['X'].shape==(spec['n'],p)
        result=fit_bfg(f['y'],f['X'],always_in=f['always_in'],budget_models=16,
                       device='cpu',g=spec['g'],verbose=False)
    assert result.n_models_evaluated==16
    assert result.total_universe_models==int(spec['universe_size'])

def test_historical_tables_consistent():
    rows=pd.read_csv(B/'recorded/scaling_discovery.csv')
    assert len(rows)==30
    assert rows.N_MAP.notna().sum()==30
    assert rows.true_discovered.sum()==29
    rows=pd.read_csv(B/'recorded/seed_results.csv')
    assert set(rows.p)=={12,18,30}
    assert (rows.unique_evaluated<=rows.budget).all()
    assert {'BFG','Baseline_Random','Baseline_Greedy'}<=set(rows.algorithm)
    ranks=pd.read_csv(B/'recorded/recorded_rank_observations.csv.gz')
    assert {'Benchmark_A','Benchmark_B','Benchmark_C'}==set(ranks.benchmark)
