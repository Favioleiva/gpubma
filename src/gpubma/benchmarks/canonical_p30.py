"""Read-only archived exact references; never run an exact enumeration."""
from importlib.resources import files
import json
import pandas as pd


def load_reference():
    return json.loads(files('gpubma.benchmarks').joinpath('canonical_p30_reference.json').read_text(encoding='utf-8'))


def reference_panel(result, reference):
    if result.candidate_names != reference['candidates']:
        raise ValueError('Canonical reference requires the original ordered x1..x30 candidate set')
    records = {int(row['model_id']): row for row in result.records}
    true, champion = reference['true_model'], reference['map_model']
    tid, mid = int(true['model_id']), int(champion['model_id'])
    t, m = records.get(tid), records.get(mid)
    variables = lambda mask: ', '.join(name for j, name in enumerate(result.candidate_names) if mask & (1 << j))
    return [
        ('True model M* (mask)', str(tid)), ('True model variables', variables(tid)),
        ('True model size k*', str(true['model_size'])), ('True model exact/global BMA rank', str(true['exact_posterior_rank'])),
        ('True model exact/global PMP (reference)', f'{true["pmp"]:.12g}'),
        ('BFG found true model', 'FOUND' if t else 'NOT FOUND'), ('N_TRUE (unique evaluations)', str(t['discovery_order']) if t else 'Not reached'),
        ('Exact/global MAP (mask)', str(mid)), ('Exact/global MAP variables', variables(mid)),
        ('Exact/global MAP rank (reference)', '1'), ('Exact/global MAP size', str(champion['model_size'])),
        ('Exact/global MAP PMP (reference)', f'{champion["pmp"]:.12g}'),
        ('BFG found exact/global MAP', 'FOUND' if m else 'NOT FOUND'),
        ('N_MAP (unique evaluations)', str(m['discovery_order']) if m else 'Not reached'),
        ('Hamming distance M* vs exact MAP', str(true['hamming_from_top_model'])),
        ('True model is the exact/global MAP', 'YES' if tid == mid else 'NO')]


def compare_reference(result, reference=None):
    reference = load_reference() if reference is None else reference
    if result.candidate_names != reference['candidates']:
        raise ValueError('Wrong candidate set for canonical reference')
    known = {int(r['model_id']): r for r in reference['top_models']}
    seen = {int(r['model_id']): r for r in result.records}
    mid, tid = int(reference['map_model']['model_id']), int(reference['true_model']['model_id'])
    rows = []
    def add(metric, ref, observed, status, note=''):
        rows.append(dict(metric=metric, reference=ref, observed=observed, classification=status, note=note))
    add('MAP model identity', mid, result.best_model['model_id'], 'EXACTLY REPLICATED' if int(result.best_model['model_id']) == mid else 'NOT REPLICATED')
    delta = abs(float(seen[mid]['log_score'])-float(reference['map_model']['log_score'])) if mid in seen else None
    add('MAP log numerator', reference['map_model']['log_score'], seen[mid]['log_score'] if mid in seen else None,
        'NUMERICALLY REPLICATED WITHIN TOLERANCE' if delta is not None and delta <= reference['log_score_atol'] else 'NOT REPLICATED',
        f'absolute difference={delta}; fixed tolerance={reference["log_score_atol"]}')
    exact10 = {int(r['model_id']) for r in reference['top_models'][:10]}
    found10 = len(exact10 & set(seen))
    ranked10 = {int(r['model_id']) for r in sorted(result.records, key=lambda r:-r['log_score'])[:10]}
    add('Exact top-10 discovered', 10, found10, 'EXACTLY REPLICATED' if found10 == 10 else 'NOT REPLICATED')
    add('Top-10 rank-list overlap', 10, len(exact10 & ranked10), 'EXACTLY REPLICATED' if exact10 == ranked10 else 'NOT REPLICATED')
    add('Historical milestone top-10 recall at 5000', reference['historical_search_top10_recall'], found10/10,
        'BEHAVIORALLY REPLICATED' if found10/10 >= reference['historical_search_top10_recall'] else 'NOT REPLICATED',
        reference['historical_search_note'])
    add('True model explicitly discovered', tid, tid if tid in seen else None, 'BEHAVIORALLY REPLICATED' if tid in seen else 'NOT REPLICATED')
    for metric, mask in [('N_MAP', mid), ('N_TRUE', tid)]:
        add(metric, None, seen[mask]['discovery_order'] if mask in seen else None, 'REFERENCE NOT AVAILABLE',
            'New run first-hit count; no verified historical canonical first-hit target was found.')
    for metric, value in [('requested_budget', result.budget_models), ('unique_models', result.n_models_evaluated),
                          ('fraction_of_universe', result.fraction_explored), ('algorithm_runtime_seconds', result.elapsed_seconds)]:
        add(metric, None, value, 'REFERENCE NOT AVAILABLE', 'Run descriptor, not a claim to match another engine or hardware runtime.')
    return pd.DataFrame(rows)
