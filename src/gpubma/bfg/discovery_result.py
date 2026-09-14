"""Discovery ledger. No global posterior estimators or oracle-only fields."""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp

@dataclass
class BFGResult:
    """Unique scored models in one-based evaluation order.

    The best discovered model is a MAP candidate, not certified global MAP.
    JSON masks are decimal strings to avoid JavaScript integer truncation.
    """
    candidate_names: list[str]
    n_obs: int
    outcome: str
    records: list[dict]
    budget_models: int
    elapsed_seconds: float
    hardware: dict
    reproducibility: dict
    diagnostics: dict
    checkpoints: list[dict] = field(default_factory=list, repr=False)

    def __post_init__(self):
        ids = [r['model_id'] for r in self.records]
        if not ids or len(set(ids)) != len(ids) or len(ids) > self.budget_models:
            raise ValueError('Invalid unique-evaluation ledger or hard budget')
        if any(type(m) is not int or not 0 <= m < self.total_universe_models for m in ids):
            raise ValueError('Invalid model bitmask')
        if not np.isfinite([r['log_score'] for r in self.records]).all():
            raise ValueError('Nonfinite evaluated score')
        if [r['discovery_order'] for r in self.records] != list(range(1, len(ids)+1)):
            raise ValueError('Invalid evaluation order')

    @classmethod
    def from_search(cls, *, scorer, registry, names, n_obs, outcome, config,
                    elapsed_seconds, fingerprint, checkpoints, exact_wings):
        import scipy
        import torch
        if set(scorer.eval_order) != set(registry.records):
            raise ValueError('Scorer and provenance ledger disagree')
        records = []
        for order, mid in enumerate(scorer.eval_order, 1):
            rec = registry.records[mid].to_dict()
            rec['registration_order'] = rec['discovery_order']
            rec['discovery_order'] = order
            rec['variable_names'] = [v for j, v in enumerate(names) if mid & (1 << j)]
            records.append(rec)
        settings = json.loads(json.dumps(config.to_dict()))
        for key in ('checkpoint_dir', 'resume', 'checkpoints', 'verbose'):
            settings.pop(key, None)
        return cls(names, n_obs, outcome, records, config.budget_models,
                   elapsed_seconds,
                   {'backend': scorer.backend, 'device_name': scorer.device_name,
                    'requested_device': config.device, 'precision': 'float64'},
                   {'schema': 'bfg-discovery-v1', 'input_code_config_sha256': fingerprint,
                    'config': settings, 'python': platform.python_version(),
                    'numpy': np.__version__, 'scipy': scipy.__version__,
                    'torch': torch.__version__, 'platform': platform.system()},
                   {'cache_hits': scorer.n_cache_hits, 'eval_calls': scorer.n_eval_calls,
                    'exact_wings': exact_wings,
                    'global_posterior_calibration': 'NOT_ESTABLISHED'}, checkpoints)

    @property
    def n_predictors(self):
        return len(self.candidate_names)

    @property
    def total_universe_models(self):
        return 1 << self.n_predictors

    @property
    def n_models_evaluated(self):
        return len(self.records)

    @property
    def fraction_explored(self):
        return self.n_models_evaluated / self.total_universe_models

    @property
    def best_model(self):
        return deepcopy(max(self.records, key=lambda r: r['log_score']))

    @property
    def best_log_score(self):
        return self.best_model['log_score']

    @property
    def log_Z_seen(self):
        """Log score mass of the discovered set, not an estimate of global Z."""
        return float(logsumexp([r['log_score'] for r in self.records]))

    @staticmethod
    def _table(records):
        df = pd.DataFrame(deepcopy(records))
        for name in ('model_id', 'parent_id'):
            if records and name in records[0]:
                df[name] = pd.Series([r.get(name) for r in records], dtype=object)
        return df

    def discovery_table(self):
        """One row per scored model, in exact scorer evaluation order."""
        return self._table(self.records)

    def top_models(self, k=20):
        """Rank discovered models; ties retain evaluation order."""
        if type(k) is not int or k < 1:
            raise ValueError('k must be a positive integer')
        rows = sorted(self.records, key=lambda r: -r['log_score'])[:k]
        df = self._table(rows)
        df.insert(0, 'discovered_rank', range(1, len(rows)+1))
        return df

    def discovered_set_model_weights(self):
        """Y_M / Z_seen: NOT globally calibrated model probabilities."""
        return pd.Series(np.exp(np.array([r['log_score'] for r in self.records])-self.log_Z_seen),
                         index=pd.Index([r['model_id'] for r in self.records], dtype=object),
                         name='discovered_set_normalized_weight')

    def discovered_set_pips(self):
        """Conditional inclusion weights, NOT global PIPs; bias can have either sign."""
        values = np.zeros(self.n_predictors)
        for record, weight in zip(self.records, self.discovered_set_model_weights()):
            values += weight * np.array([bool(record['model_id'] & (1 << j))
                                         for j in range(self.n_predictors)])
        return pd.Series(values, index=self.candidate_names,
                         name='discovered_set_inclusion_weight')

    def genealogy(self):
        """First registered traversal edges with scored endpoints; not all edges.

        In a backward move the source is the larger model. No missing lineage
        is synthesized. These edges describe search, not causality.
        """
        known = {r['model_id'] for r in self.records}
        edges = []
        for r in self.records:
            source, target = r['parent_id'], r['model_id']
            if source is not None and source in known and source != target:
                edges.append({'source_model_id': str(source), 'target_model_id': str(target),
                              'direction': 'add' if source.bit_count() < target.bit_count() else 'remove',
                              'stage': r['initial_provenance']})
        return pd.DataFrame(edges, columns=['source_model_id', 'target_model_id', 'direction', 'stage'])

    parent_child_edges = genealogy

    def model_families(self):
        """Descriptive components of retained traversal edges, not posterior modes.

        Search through the empty model can connect many branches. Isolated
        models are included; there is no single-dynasty assumption.
        """
        parents = {r['model_id']: r['model_id'] for r in self.records}
        def root(m):
            while parents[m] != m:
                parents[m] = parents[parents[m]]
                m = parents[m]
            return m
        for e in self.genealogy().to_dict('records'):
            a, b = root(int(e['source_model_id'])), root(int(e['target_model_id']))
            if a != b:
                parents[max(a,b)] = min(a,b)
        return pd.DataFrame([{'model_id': str(r['model_id']), 'family_id': str(root(r['model_id']))}
                             for r in self.records])

    def champion_path(self):
        """Strict best-score improvements, not exact-oracle ranks."""
        best = -float('inf')
        rows = []
        for r in self.records:
            if r['log_score'] > best:
                rows.append(r)
                best = r['log_score']
        return self._table(rows)

    def search_diagnostics(self):
        return dict(deepcopy(self.diagnostics), budget_models=self.budget_models,
                    n_unique_scored=self.n_models_evaluated,
                    universe_size=self.total_universe_models,
                    fraction_explored=self.fraction_explored,
                    elapsed_seconds=self.elapsed_seconds,
                    first_registration_stage_counts=dict(Counter(
                        r['initial_provenance'] for r in self.records)))

    def summary(self):
        return (f'BFG discovery: {self.n_models_evaluated:,}/{self.total_universe_models:,} models '
                f'({100*self.fraction_explored:.6g}%), budget={self.budget_models:,}\n'
                f'Best discovered model={self.best_model["model_id"]}, log score={self.best_log_score:.9g}\n'
                f'Elapsed={self.elapsed_seconds:.3f}s; backend={self.hardware["backend"]}\n'
                'Global MAP certification and global Z/PMP/PIP are not provided.')

    def to_dict(self):
        rows = deepcopy(self.records)
        for r in rows:
            r['model_id'] = str(r['model_id'])
            if r['parent_id'] is not None:
                r['parent_id'] = str(r['parent_id'])
        return {'schema': 'bfg-discovery-v1', 'candidate_names': self.candidate_names,
                'n_obs': self.n_obs, 'outcome': self.outcome, 'records': rows,
                'budget_models': self.budget_models, 'elapsed_seconds': self.elapsed_seconds,
                'hardware': self.hardware, 'reproducibility': self.reproducibility,
                'diagnostics': self.diagnostics}

    def save_json(self, path):
        """Create a result file exclusively. Checkpoint payloads remain separate."""
        with Path(path).open('x', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, allow_nan=False)
            f.write('\n')

    @classmethod
    def load_json(cls, path):
        obj = json.loads(Path(path).read_text(encoding='utf-8'))
        if obj.pop('schema') != 'bfg-discovery-v1':
            raise ValueError('Unsupported discovery result schema')
        for r in obj['records']:
            r['model_id'] = int(r['model_id'])
            if r['parent_id'] is not None:
                r['parent_id'] = int(r['parent_id'])
        return cls(**obj)
