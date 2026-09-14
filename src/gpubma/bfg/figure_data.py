"""Validated discovered-set ledger and finite-support display arrays."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import gaussian_kde
MIN_KDE_SAMPLES = 8
MIN_KDE_DISTINCT = 3

def prepare_discovered_set(result, top_k=12):
    """Build display arrays from all evaluated scores; never renormalize top K.

    A model's log_score is the scorer's log unnormalized numerator, with the
    scorer's common-constant convention. No exact oracle or global Z is used.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    if not hasattr(result, 'discovery_table'):
        raise TypeError('Use the discovery BFGResult, not the legacy posterior-result API')
    frame = result.discovery_table().copy(deep=True)
    names = list(result.candidate_names)
    required = {'model_id', 'log_score', 'model_size', 'discovery_order'}
    if not names or len(set(names)) != len(names) or not required <= set(frame):
        raise ValueError('Missing discovery columns or invalid variable names')
    frame = frame.sort_values('discovery_order', kind='stable').reset_index(drop=True)
    ids = [int(m) for m in frame.model_id]
    scores = frame.log_score.to_numpy(dtype=float)
    order = frame.discovery_order.to_numpy()
    p = len(names)
    if (not ids or len(set(ids)) != len(ids) or not np.isfinite(scores).all()
            or any(m < 0 or m >= (1 << p) for m in ids)
            or not np.array_equal(order, np.arange(1, len(ids)+1))):
        raise ValueError('Expected a complete finite unique-evaluation ledger')
    sizes = np.array([m.bit_count() for m in ids], dtype=int)
    if not np.array_equal(sizes, frame.model_size.to_numpy()):
        raise ValueError('Model size disagrees with its inclusion mask')
    # Object dtype prevents pandas rounding nullable or wide integer masks.
    frame['model_id'] = pd.Series(ids, dtype=object)
    if 'parent_id' not in frame:
        frame['parent_id'] = pd.Series([None]*len(ids), dtype=object)
    stage = frame.get('initial_provenance', pd.Series(['']*len(frame))).fillna('')
    fallback = frame.get('provenance', pd.Series(['UNKNOWN']*len(frame))).fillna('UNKNOWN')
    frame['plot_stage'] = stage.where(stage != '', fallback).astype(str)
    weight = np.exp(scores - logsumexp(scores))
    inclusion = np.array([[(mid >> j) & 1 for j in range(p)] for mid in ids], dtype=bool)
    variable_weight = weight @ inclusion
    columns = np.argsort(-variable_weight, kind='stable')
    top = np.argsort(-scores, kind='stable')[:min(top_k, len(ids))]
    size_weight = np.bincount(sizes, weights=weight, minlength=p+1)
    return dict(frame=frame, names=names, ids=ids, scores=scores, sizes=sizes,
                order=order, weights=weight, inclusion=inclusion,
                variable_weights=variable_weight, columns=columns, top=top,
                size_weights=size_weight, p=p,
                expected_size=float(np.arange(p+1) @ size_weight),
                champion_size=int(sizes[top[0]]))


def observed_ridge(values, grid_size=256):
    """Unweighted evaluated-score shape, restricted to exact observed support.

    KDE requires >=8 observations and >=3 distinct values. It is evaluated
    ONLY from observed min to max and scaled to unit peak for display. This
    is not a posterior density and ridge areas must not be compared as mass.
    Sparse/constant/singular shells retain empirical spikes instead.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Ridge requires nonempty finite one-dimensional scores')
    lo, hi = float(values.min()), float(values.max())
    distinct, count = np.unique(values, return_counts=True)
    if len(values) >= MIN_KDE_SAMPLES and len(distinct) >= MIN_KDE_DISTINCT:
        try:
            grid = np.linspace(lo, hi, grid_size)
            density = gaussian_kde(values, bw_method="silverman")(grid)  # historical Silverman display bandwidth
            if np.isfinite(density).all() and density.max() > 0:
                return dict(kind='kde', x=grid, height=density/density.max(),
                            xmin=lo, xmax=hi, n=len(values))
        except (np.linalg.LinAlgError, ValueError):
            pass
    return dict(kind='empirical', x=distinct, height=count/count.max(),
                xmin=lo, xmax=hi, n=len(values))

