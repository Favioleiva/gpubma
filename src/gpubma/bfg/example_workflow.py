"""Shared portable data, device, budget and reporting workflow for BFG examples."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import urllib.request
import numpy as np
import pandas as pd
import torch
from gpubma import BFGConfig, fit_bfg
from .figures import canonical_bfg_figures, save_canonical_figures
from .model_report import refit_discovered_models, write_top5

PUBLIC_DATA = {
    'canonical_p30': ('data/synthetic/panel_30_center15.parquet', '7b468ca0c09249a83b05638b53c884bc444fdd535ba5f698b8fa92c24f7dd6e0'),
    'grunfeld': ('data/public/grunfeld.dta', 'be844fe819788af680c8c7bb44f0a86f6726fb440081bd2de4a102679ebbc939'),
}


def resolve_device(device='auto'):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ('cpu', 'cuda') and not (isinstance(device, str) and device.startswith('cuda:')):
        raise ValueError('DEVICE must be auto, cpu, cuda or cuda:<index>')
    return device


def print_environment(device):
    selected = resolve_device(device)
    info = {'Python': platform.python_version(), 'gpubma': importlib.metadata.version('gpubma'),
            'PyTorch': torch.__version__, 'CUDA available': torch.cuda.is_available(),
            'GPU name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
            'selected device': selected}
    for key, value in info.items():
        print(f'{key}: {value}')
    return info


def ensure_public_data(dataset, destination=None):
    """Use unchanged canonical bytes locally, or retrieve that exact public file."""
    rel, expected = PUBLIC_DATA[dataset]
    path = Path(destination) if destination is not None else Path(rel)
    if not path.is_file():
        url = 'https://raw.githubusercontent.com/Favioleiva/gpubma/main/'+rel
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Public dataset hash differs from the recorded canonical bytes')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(raw)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Dataset checksum mismatch; do not substitute another dataset')
    return path


def load_example_data(path, target, candidates, always_in=()):
    path = Path(path)
    if not candidates or len([target, *candidates, *always_in]) != len(set([target, *candidates, *always_in])):
        raise ValueError('Target, candidate and always-in names must be distinct; candidates cannot be empty')
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            names = next(csv.reader(f))
        if len(names) != len(set(names)):
            raise ValueError('Duplicate CSV header')
        data = pd.read_csv(path, float_precision='round_trip')
    elif path.suffix.lower() == '.parquet':
        data = pd.read_parquet(path)
    elif path.suffix.lower() == '.dta':
        data = pd.read_stata(path, convert_categoricals=False)
    else:
        raise ValueError('Supported data formats: CSV, Parquet, Stata .dta')
    selected = [target, *candidates, *always_in]
    if data.columns.duplicated().any() or not set(selected) <= set(data.columns):
        raise ValueError('Missing or duplicate columns')
    numeric = data[selected].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError('Resolve nonfinite/missing data explicitly; no automatic row deletion')
    return numeric[target], numeric[list(candidates)], numeric[list(always_in)] if always_in else None


def budget_summary(p, budget, actual=None):
    if type(budget) is not int or budget <= 0:
        raise ValueError('BUDGET must be a positive integer')
    universe = 1 << p
    values = dict(candidate_predictors=p, universe=universe, requested_budget=budget,
                  maximum_fraction=min(budget, universe)/universe,
                  maximum_percent=100*min(budget, universe)/universe)
    if actual is not None:
        assert actual <= budget, 'Shared BFG hard budget exceeded'
        assert actual <= universe
        values.update(actual_unique_evaluations=actual, actual_fraction=actual/universe, actual_percent=100*actual/universe)
    return values


def fit_example(y, X, controls, *, budget=5000, seed=12345, device='auto', outcome='y'):
    """Map BUDGET directly to the one existing global scorer budget."""
    before = budget_summary(X.shape[1], budget)
    print(f'Candidate predictors p = {before["candidate_predictors"]}')
    print(f'Full model universe = {before["universe"]:,} models')
    print(f'BFG evaluation budget = {budget:,} unique model evaluations')
    print(f'Maximum explored fraction = {before["maximum_percent"]:.6f}%')
    start = time.perf_counter()
    result = fit_bfg(y=y, X=X, always_in=controls, outcome_name=outcome,
                    config=BFGConfig(budget_models=budget, seed=seed, device=resolve_device(device), beam_width=5))
    wall = time.perf_counter()-start
    after = budget_summary(X.shape[1], budget, result.n_models_evaluated)
    print(f'Requested BUDGET: {budget:,}; actual unique evaluations: {result.n_models_evaluated:,}')
    print(f'Actual fraction evaluated: {after["actual_percent"]:.6f}%')
    print(f'BFG algorithm runtime: {result.elapsed_seconds:.6f} s; fit-call wall time: {wall:.6f} s')
    return result, dict(**after, algorithm_seconds=result.elapsed_seconds, fit_call_seconds=wall,
                        stage_runtimes='REFERENCE NOT AVAILABLE: not retained by this engine',
                        device=result.hardware, seed=seed)


def export_example(result, y, X, controls, output_dir, *, outcome='y', benchmark_reference=None):
    """Generate figures and OLS reporting after search; time separately."""
    start = time.perf_counter()
    report = refit_discovered_models(result, y, X, controls)
    report['benchmark_reference'] = benchmark_reference
    figures, tables = canonical_bfg_figures(result, report=report)
    paths = save_canonical_figures(figures, tables, output_dir)
    preview = write_top5(report, output_dir, outcome=outcome)
    result.save_json(Path(output_dir)/'discovery_result.json')
    reporting_seconds = time.perf_counter()-start
    print(f'Post-search figure/table reporting: {reporting_seconds:.3f} s')
    return figures, tables, preview, reporting_seconds
