"""Fixed-effects design construction.

Two approaches are implemented and compared in Phase 1:

1. Reference approach — explicit dummy variables (``fe_method="dummies"``).
   Base category: the FIRST level in sorted order of each factor is dropped.
   The intercept is always included, so the design is full rank by
   construction (no dummy-variable trap). Fixed effects belong to the
   always-included block and appear in every candidate model.

2. Candidate production approach — residualization (``fe_method="within"``).
   One-way: subtract group means. Two-way (balanced panels only):
   x_it - mean_i - mean_t + grand_mean, which is the exact two-way projection
   for balanced panels. Unbalanced two-way panels would require iterative
   demeaning and are rejected with an explicit error in this phase.

Both approaches produce the SAME residualized data (Frisch-Waugh-Lovell), and
therefore the same OLS slopes. Equality of Bayesian model scores additionally
requires that the effective residual degrees of freedom and the g value are
kept consistent — see docs/FIXED_EFFECTS_DESIGN.md for the statistical
discussion. This module reports the absorbed rank so callers can do that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VALID_FIXED_EFFECTS = ("individual", "time")


def _factor_column(fixed_effect: str, entity_col: str, time_col: str) -> str:
    if fixed_effect == "individual":
        if entity_col is None:
            raise ValueError("entity_col is required for individual fixed effects")
        return entity_col
    if fixed_effect == "time":
        if time_col is None:
            raise ValueError("time_col is required for time fixed effects")
        return time_col
    raise ValueError(
        f"unsupported fixed effect {fixed_effect!r}; Phase 1 supports {VALID_FIXED_EFFECTS}"
    )


def dummy_design(data: pd.DataFrame, fixed_effects, entity_col=None, time_col=None):
    """Explicit dummy-variable design matrix for the given fixed effects.

    Returns ``(D, info)`` where D is an (n, m) float64 matrix of 0/1 dummies
    (base categories dropped) and info documents base categories and rank.
    """
    columns, names, base_categories = [], [], {}
    for fe in fixed_effects:
        col = _factor_column(fe, entity_col, time_col)
        levels = sorted(pd.unique(data[col]))
        base_categories[fe] = levels[0]
        for level in levels[1:]:
            columns.append((data[col] == level).to_numpy(dtype=np.float64))
            names.append(f"{fe}[{col}={level}]")
    D = np.column_stack(columns) if columns else np.empty((len(data), 0), dtype=np.float64)
    info = {
        "method": "dummies",
        "base_categories": {k: str(v) for k, v in base_categories.items()},
        "dummy_names": names,
        "n_dummies": len(names),
    }
    return D, info


def _check_balanced(data: pd.DataFrame, entity_col: str, time_col: str) -> None:
    counts = data.groupby(entity_col, observed=True)[time_col].count()
    n_periods = data[time_col].nunique()
    if not (counts == n_periods).all():
        raise ValueError(
            "two-way within transform requires a balanced panel in Phase 1; "
            "found unbalanced individual/time cells"
        )


def within_transform(values: np.ndarray, data: pd.DataFrame, fixed_effects,
                     entity_col=None, time_col=None):
    """Residualize the columns of ``values`` on the given fixed effects.

    Returns ``(transformed, absorbed_rank)`` where ``absorbed_rank`` is the
    rank of the span of {intercept, FE dummies} absorbed by the transform:
      individual only: N_i;  time only: N_t;  both (balanced): N_i + N_t - 1.
    """
    fixed_effects = list(fixed_effects)
    for fe in fixed_effects:
        _factor_column(fe, entity_col, time_col)  # validate names early
    values = np.asarray(values, dtype=np.float64)
    squeeze = values.ndim == 1
    V = values.reshape(len(values), -1).copy()

    frame = pd.DataFrame(V, index=data.index)

    if fixed_effects == ["individual"]:
        out = V - frame.groupby(data[entity_col], observed=True).transform("mean").to_numpy()
        rank = data[entity_col].nunique()
    elif fixed_effects == ["time"]:
        out = V - frame.groupby(data[time_col], observed=True).transform("mean").to_numpy()
        rank = data[time_col].nunique()
    elif sorted(fixed_effects) == ["individual", "time"]:
        _check_balanced(data, entity_col, time_col)
        mean_i = frame.groupby(data[entity_col], observed=True).transform("mean").to_numpy()
        mean_t = frame.groupby(data[time_col], observed=True).transform("mean").to_numpy()
        out = V - mean_i - mean_t + V.mean(axis=0, keepdims=True)
        rank = data[entity_col].nunique() + data[time_col].nunique() - 1
    else:
        raise ValueError(f"unsupported fixed_effects combination: {fixed_effects}")

    return (out.ravel() if squeeze else out), int(rank)


def build_always_block(data: pd.DataFrame, controls, fixed_effects, fe_method,
                       entity_col=None, time_col=None,
                       y: np.ndarray = None, X: np.ndarray = None):
    """Assemble the always-included block and pre-transform y and X.

    Returns a dict with:
      y_work, X_work : possibly within-transformed outcome and predictors
      A              : always-included design to residualize on (may have 0 cols)
      absorbed_rank  : rank absorbed by a within transform (0 for dummies)
      base_rank      : rank of A (validated == A.shape[1], no dummy trap)
      info           : documentation of the construction
    """
    n = len(data)
    W = (
        data[list(controls)].to_numpy(dtype=np.float64)
        if controls
        else np.empty((n, 0), dtype=np.float64)
    )
    intercept = np.ones((n, 1), dtype=np.float64)
    info = {"controls": list(controls), "fixed_effects": list(fixed_effects or []),
            "fe_method": fe_method if fixed_effects else None}

    if not fixed_effects:
        A = np.hstack([intercept, W])
        y_work, X_work, absorbed = y, X, 0
    elif fe_method == "dummies":
        D, dinfo = dummy_design(data, fixed_effects, entity_col, time_col)
        info.update(dinfo)
        A = np.hstack([intercept, D, W])
        y_work, X_work, absorbed = y, X, 0
    elif fe_method == "within":
        y_work, absorbed = within_transform(y, data, fixed_effects, entity_col, time_col)
        X_work, _ = within_transform(X, data, fixed_effects, entity_col, time_col)
        if W.shape[1]:
            W, _ = within_transform(W, data, fixed_effects, entity_col, time_col)
        # The intercept (and FE means) are absorbed by the transform; the
        # always block that remains is only the transformed controls.
        A = W
        info["method"] = "within"
        info["absorbed_rank"] = absorbed
    else:
        raise ValueError(f"unsupported fe_method: {fe_method!r}")

    base_rank = int(np.linalg.matrix_rank(A)) if A.shape[1] else 0
    if base_rank != A.shape[1]:
        raise ValueError(
            f"always-included block is rank deficient (rank {base_rank} < "
            f"{A.shape[1]} columns); check for a dummy-variable trap or "
            "collinear controls"
        )
    return {
        "y_work": np.asarray(y_work, dtype=np.float64),
        "X_work": np.asarray(X_work, dtype=np.float64),
        "A": A,
        "absorbed_rank": absorbed if fixed_effects and fe_method == "within" else 0,
        "base_rank": base_rank,
        "info": info,
    }
