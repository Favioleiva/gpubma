"""Bounded software regressions on existing synthetic fixture recipes only."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from gpubma import BFGConfig, BFGResult, fit_bfg
from gpubma.bfg.checkpoint import CheckpointManager
from gpubma.bfg.scorer import BFGScorer
from gpubma.bfg.registry import EliteRegistry
from gpubma.bfg.genealogy import GenealogicalSearch

def fixture(p=5):
    # Existing corrective-resume fixture recipe; no scientific dataset input.
    rng = np.random.default_rng(81)
    X = rng.normal(size=(40, p))
    return X, rng.normal(size=40)

@pytest.mark.parametrize('budget', [1, 2, 5, 10, 31, 32, 100])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_hard_budget_unique_deterministic(budget, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    X, y = fixture()
    args = dict(budget_models=budget, device=device, verbose=False)
    a, b = fit_bfg(y, X, **args), fit_bfg(y, X, **args)
    assert a.records == b.records
    ids = a.discovery_table().model_id.tolist()
    assert len(ids) == len(set(ids)) == a.n_models_evaluated <= budget
    assert sum(a.search_diagnostics()['first_registration_stage_counts'].values()) == len(ids)
    assert all(r['initial_provenance'] for r in a.records)
    assert a.hardware['backend'] == ('gpu' if device == 'cuda' else 'cpu')
    assert a.best_model['model_id'] == a.top_models(1).iloc[0].model_id
    assert len(a.model_families()) == len(ids)
    assert np.all(np.diff(a.champion_path().log_score) > 0)

def test_discovered_quantities_and_serialization(tmp_path):
    X, y = fixture()
    a = fit_bfg(y, X, config=BFGConfig(budget_models=20, device='cpu'))
    assert abs(a.discovered_set_model_weights().sum()-1) < 1e-12
    direct = np.zeros(5)
    for mid, weight in a.discovered_set_model_weights().items():
        for j in range(5):
            direct[j] += weight * bool(mid & (1 << j))
    np.testing.assert_allclose(direct, a.discovered_set_pips(), rtol=0, atol=1e-12)
    for forbidden in ['pips', 'pmp', 'posterior_moments', 'posterior_mean', 'log_Z', 'map_pmp']:
        assert not hasattr(a, forbidden)
    dest = tmp_path/'result.json'
    a.save_json(dest)
    b = BFGResult.load_json(dest)
    assert a.to_dict() == b.to_dict()
    assert isinstance(json.loads(dest.read_text())['records'][0]['model_id'], str)
    with pytest.raises(FileExistsError):
        a.save_json(dest)
    # Corruption cannot silently create duplicate scientific records.
    bad = a.to_dict(); bad['records'][1]['model_id'] = bad['records'][0]['model_id']
    dest.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        BFGResult.load_json(dest)

@pytest.mark.parametrize('stage', ['wings', 'genealogy', 'sampling', 'final_reconnaissance'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_interrupted_resume_matches_full(tmp_path, monkeypatch, stage, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(260907)
    X = rng.normal(size=(90,12)); y = X[:,0] + rng.normal(size=90)
    args = dict(device=device, budget_models=1600, wing_max_size=20,
                beam_width=1, verbose=False, seed=731)
    full = fit_bfg(y,X,checkpoint_dir=tmp_path/'whole',**args)
    save = CheckpointManager.save_checkpoint
    class Interrupted(RuntimeError):
        pass
    def interrupt(self, *a, **kw):
        path = save(self,*a,**kw)
        if kw['metadata']['stage'] == stage and (stage != 'final_reconnaissance' or
                kw['metadata']['execution']['final_reconnaissance']['next_k'] >= 4):
            raise Interrupted()
        return path
    with monkeypatch.context() as patch:
        patch.setattr(CheckpointManager,'save_checkpoint',interrupt)
        with pytest.raises(Interrupted):
            fit_bfg(y,X,checkpoint_dir=tmp_path/'split',**args)
    resumed = fit_bfg(y,X,checkpoint_dir=tmp_path/'split',resume=True,**args)
    assert resumed.records == full.records
    assert resumed.diagnostics['eval_calls'] == full.diagnostics['eval_calls']
    np.testing.assert_array_equal(resumed.discovered_set_model_weights(),full.discovered_set_model_weights())

def test_checkpoint_mismatch_corruption_and_completion(tmp_path):
    X,y=fixture()
    args=dict(device='cpu',budget_models=100,checkpoint_dir=tmp_path,checkpoints=[10000])
    a=fit_bfg(y,X,**args)
    assert a.checkpoints[-1]['metadata']['stage']=='complete'
    assert fit_bfg(y,X,resume=True,**args).records==a.records
    with pytest.raises(ValueError,match='fingerprint'):
        fit_bfg(y+.01*np.arange(len(y)),X,resume=True,**args)
    pointer=json.loads((tmp_path/'latest_checkpoint.json').read_text())
    assert not Path(pointer['path']).is_absolute()
    cache=tmp_path/pointer['path']/'evaluation_cache.npz'
    cache.write_bytes(cache.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        fit_bfg(y,X,resume=True,**args)
    with pytest.raises(ValueError,match='committed checkpoint'):
        fit_bfg(y,X,resume=True,device='cpu',checkpoint_dir=tmp_path/'empty')

@pytest.mark.parametrize('bad', [{'budget_models':1.5},{'budget_models':True},
    {'budget_semantics':'sampling_only'},{'acesm_beta':3.5}, {'budget_seconds':1},
    {'device':'invalid'}, {'precision':'float32'}, {'resume':True}])
def test_invalid_config_rejected(bad):
    with pytest.raises((ValueError,TypeError)):
        BFGConfig(**bad)

def test_config_options_never_silently_ignored():
    X,y=fixture()
    with pytest.raises(TypeError,match='OR'):
        fit_bfg(y,X,config=BFGConfig(),budget_models=10)

@pytest.mark.parametrize('p',[64,90])
def test_wide_bitmask_gpu_cpu_and_json(p,tmp_path):
    # Small software masks only; not a p=90 scientific discovery experiment.
    rng=np.random.default_rng(81)
    X=rng.normal(size=(100,p)); y=rng.normal(size=100)
    X-=X.mean(0); y-=y.mean()
    args=dict(X_r=X,y_r=y,df_resid=99,g=100,log_model_prior=lambda k:0.)
    ids=[0,1 << (p-1),(1 << (p-1)) | 1]
    cpu=BFGScorer(device='cpu',**args); expected=cpu.score_batch(ids)
    assert cpu.eval_order==ids
    if torch.cuda.is_available():
        gpu=BFGScorer(device='cuda',**args)
        actual=gpu.score_batch(ids)
        np.testing.assert_allclose(list(actual.values()),list(expected.values()),rtol=0,atol=1e-10)
    rows=[dict(model_id=m,log_score=expected[m],discovery_order=i+1,
               parent_id=None,initial_provenance='RANDOM_BULK') for i,m in enumerate(ids)]
    result=BFGResult([f'x{j}' for j in range(p)],100,'y',rows,3,0.,{}, {}, {})
    path=tmp_path/'wide.json'; result.save_json(path)
    assert BFGResult.load_json(path).records==rows
    assert result.discovery_table().model_id.tolist()==ids

def test_budget_exhausted_beam_seed():
    X,y=fixture()
    scorer=BFGScorer(X_r=X,y_r=y,df_resid=39,g=40,log_model_prior=lambda k:0.,device='cpu',max_eval_budget=1)
    scorer.score_single(0)
    search=GenealogicalSearch(scorer,EliteRegistry(p=5))
    search.forward_beam([31],beam_width=1)
    search.backward_beam([31],beam_width=1)
    assert scorer.eval_order==[0]

def test_cli(tmp_path):
    X,y=fixture()
    frame=pd.DataFrame(X,columns=[f'x{j}' for j in range(5)]);frame['y']=y
    data=tmp_path/'input.csv';frame.to_csv(data,index=False)
    target=tmp_path/'result.json'
    args=[sys.executable,'-m','gpubma.bfg.cli','--data',str(data),'--outcome','y',
          '--candidates','x0,x1,x2,x3,x4','--budget-models','20','--device','cpu','--out',str(target)]
    # Explicit staged src for child process; isolated wheel validation tests installed CLI separately.
    import os
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).parents[1]/'src'))
    proc=subprocess.run(args,capture_output=True,text=True,env=env)
    assert proc.returncode==0,proc.stderr
    assert BFGResult.load_json(target).n_models_evaluated<=20
    assert subprocess.run(args,capture_output=True,env=env).returncode!=0

def test_no_experimental_default_imports():
    assert 'gpubma.bfg.acesm' not in sys.modules
    assert 'gpubma.bfg.reconstruction' not in sys.modules

def test_cpu_fallback_when_cuda_unavailable(monkeypatch):
    import gpubma.bfg.scorer as module
    monkeypatch.setattr(module, 'torch_cuda_available', lambda: (False, 'simulated absence'))
    X,y=fixture()
    result=fit_bfg(y,X,device='cuda',budget_models=5)
    assert result.hardware['backend']=='cpu'
    assert result.hardware['requested_device']=='cuda'

def test_mutated_config_validated_at_entry():
    config=BFGConfig(device='cpu');config.budget_semantics='sampling_only'
    X,y=fixture()
    with pytest.raises(ValueError):
        fit_bfg(y,X,config=config)
