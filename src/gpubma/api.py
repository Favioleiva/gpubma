"""Functional API: :func:`bma_regress`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gpubma.cpu.enumeration import enumerate_models
from gpubma.fixed_effects.design import build_always_block
from gpubma.priors.gpriors import resolve_g
from gpubma.priors.model_priors import log_model_prior_function
from gpubma.result import BMAResult

_MAX_ENUMERATION_PREDICTORS = 20  # Phase 1 safety limit (2^20 = 1,048,576 models)


def _residualize(M: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if Q.shape[1] == 0:
        return M
    return M - Q @ (Q.T @ M)


def bma_regress(
    data: pd.DataFrame,
    outcome: str,
    predictors,
    *,
    controls=None,
    fixed_effects=None,
    fe_method: str = "dummies",
    entity_col: str = None,
    time_col: str = None,
    always_prior: str = "shrink",
    backend: str = "cpu",
    method: str = "enumeration",
    precision: str = "float64",
    g="benchmark",
    model_prior=("betabinomial", 1.0, 1.0),
    top_k: int = 10,
    compute_coefficients: bool = True,
    deterministic: bool = True,
) -> BMAResult:
    """Exhaustive Bayesian Model Averaging for Gaussian linear regression.

    Optional ``predictors`` define the 2^p model space. ``controls`` and
    ``fixed_effects`` are always included and never change the model count.

    ``always_prior`` selects how always-included slopes are treated:

    - "shrink" (default): the Zellner g-prior covers optional AND
      always-included slopes jointly; only the intercept is flat and
      df = n - 1. This matches Stata ``bmaregress`` — verified 2026-07-15
      against StataNow/SE 19.5 exports on six designs (max abs diff ~1e-11).
      Requires ``fe_method="dummies"`` when fixed effects are present.
    - "flat": always-included coefficients get flat (improper) priors and
      df = n - rank(always block). This is gpubma's conditional convention;
      it is NOT Stata's. It is the only coherent choice for
      ``fe_method="within"`` because absorption implies flat fixed effects.
    """
    # ---- validation ------------------------------------------------------
    if method != "enumeration":
        raise ValueError(
            "Phase 1 supports method='enumeration' only; MC3/sampling must "
            "not be substituted without explicit authorization (CLAUDE.md rule 4)"
        )
    if precision != "float64":
        raise ValueError(
            "Phase 1 reference requires precision='float64'; float32 must "
            "never be used silently (CLAUDE.md rule 6)"
        )
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"unsupported backend {backend!r}")
    predictors = list(predictors)
    controls = list(controls) if controls else []
    fixed_effects = list(fixed_effects) if fixed_effects else []
    p = len(predictors)
    if p == 0:
        raise ValueError("at least one optional predictor is required")
    if p > _MAX_ENUMERATION_PREDICTORS:
        raise ValueError(
            f"{p} optional predictors imply 2^{p} = {2**p:,} models; Phase 1 "
            f"caps exhaustive enumeration at {_MAX_ENUMERATION_PREDICTORS} "
            "predictors. The production-scale enumerator arrives in a later phase."
        )
    missing = [c for c in [outcome, *predictors, *controls] if c not in data.columns]
    if missing:
        raise KeyError(f"columns not found in data: {missing}")
    overlap = set(predictors) & set(controls)
    if overlap:
        raise ValueError(f"columns cannot be both predictor and control: {sorted(overlap)}")

    frame = data[[outcome, *predictors, *controls]
                 + [c for c in (entity_col, time_col) if c is not None]]
    if frame[[outcome, *predictors, *controls]].isna().any().any():
        raise ValueError("missing values are not supported in Phase 1")

    y = data[outcome].to_numpy(dtype=np.float64)
    X = data[predictors].to_numpy(dtype=np.float64)
    n = len(y)

    if always_prior not in ("shrink", "flat"):
        raise ValueError(f"unsupported always_prior {always_prior!r}")
    if always_prior == "shrink" and fixed_effects and fe_method == "within":
        raise ValueError(
            "always_prior='shrink' (Stata convention) requires explicit "
            "fixed-effect dummies; fe_method='within' absorbs the fixed "
            "effects with an implicitly flat prior. Use fe_method='dummies' "
            "or always_prior='flat'."
        )

    # ---- always-included block and residualization -----------------------
    block = build_always_block(
        data, controls, fixed_effects, fe_method,
        entity_col=entity_col, time_col=time_col, y=y, X=X,
    )
    A = block["A"]
    if A.shape[1]:
        Q, _ = np.linalg.qr(A)
        y_r = _residualize(block["y_work"][:, None], Q).ravel()
        X_r = _residualize(block["X_work"], Q)
    else:
        y_r, X_r = block["y_work"], block["X_work"]

    if always_prior == "shrink":
        # Joint g-prior over optional + always slopes; intercept-only flat.
        yc = y - y.mean()
        score_kwargs = {
            "tss_norm": float(yc @ yc),
            "k_always": block["base_rank"] - 1,  # always slopes excl. intercept
        }
        df_resid = n - 1
    else:
        score_kwargs = {}
        df_resid = n - block["base_rank"] - block["absorbed_rank"]

    # validate the model count identity N = 2^p explicitly
    n_models_expected = 1 << p
    assert n_models_expected == 2**p

    g_spec = resolve_g(g, n_obs=n, n_predictors=p)
    log_prior_fn, prior_desc = log_model_prior_function(model_prior, p)
    block["info"]["always_prior"] = always_prior

    # ---- scoring ---------------------------------------------------------
    hardware = {}
    if backend == "gpu":
        from gpubma.gpu.batch_scorer import gpu_score_all_models, gpu_hardware_info

        gpu_out = gpu_score_all_models(
            X_r, y_r, df_resid=df_resid, g=g_spec.g, log_model_prior=log_prior_fn,
            **score_kwargs,
        )
        hardware = gpu_hardware_info()
        # Coefficient moments are still computed on CPU in Phase 1.
        cpu_out = enumerate_models(
            X_r, y_r, df_resid=df_resid, g=g_spec.g, log_model_prior=log_prior_fn,
            compute_coefficients=compute_coefficients, top_k=top_k, **score_kwargs,
        )
        max_diff = float(np.max(np.abs(gpu_out["log_scores"] - cpu_out["log_scores"])))
        out = cpu_out
        out["log_scores"] = gpu_out["log_scores"]
        out["runtime"] = {
            **cpu_out["runtime"],
            "backend": "gpu(scores)+cpu(coefficients)",
            "gpu": gpu_out["runtime"],
            "gpu_vs_cpu_max_abs_logscore_diff": max_diff,
        }
        notes = [
            "backend='gpu' scores models on the GPU (float64) and computes "
            "coefficient moments on the CPU in Phase 1",
            f"GPU/CPU log-score max abs difference: {max_diff:.3e}",
        ]
    else:
        out = enumerate_models(
            X_r, y_r, df_resid=df_resid, g=g_spec.g, log_model_prior=log_prior_fn,
            compute_coefficients=compute_coefficients, top_k=top_k, **score_kwargs,
        )
        notes = []
    if always_prior == "shrink":
        notes.append(
            "always_prior='shrink': joint g-prior over optional and "
            "always-included slopes — matches Stata bmaregress (verified "
            "2026-07-15 vs StataNow/SE 19.5 on six designs, see "
            "reports/comparison_report.md)"
        )
    else:
        notes.append(
            "always_prior='flat': gpubma's conditional convention "
            "(flat always block, df = n - rank); this is NOT Stata's "
            "parameterization (see docs/STATISTICAL_SPECIFICATION.md)"
        )

    return BMAResult(
        outcome=outcome,
        predictor_names=predictors,
        n_obs=n,
        n_predictors=p,
        n_models_expected=out["n_models_expected"],
        n_models_evaluated=out["n_models_evaluated"],
        df_resid=df_resid,
        g_spec=g_spec,
        model_prior_description=prior_desc,
        fixed_effects_info=block["info"],
        backend=backend,
        precision=precision,
        pip=out["pip"],
        pmp=out["pmp"],
        masks=out["masks"],
        log_scores=out["log_scores"],
        coef_mean=out["coef_mean"],
        coef_sd=out["coef_sd"],
        mean_model_size=out["mean_model_size"],
        size_distribution=out["size_distribution"],
        _top_models=out["top_models"],
        runtime=out["runtime"],
        hardware=hardware,
        notes=notes,
    )
