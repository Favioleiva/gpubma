"""Reduced CPU/GPU parity tests on the canonical panel_30_center15 rows.

Deterministic reduced problems built from the frozen Parquet artifact and
documented variable subsets (never random draws):

- p = 10: x1-x5 (true block A) + x16-x20 (their proxies) — maximal
  true-vs-proxy competition; 1,024 models.
- p = 15: x1-x15 (all true regressors); 32,768 models.
- p = 20: x1-x10 (blocks A, B) + x16-x25 (their proxies); 1,048,576 models.

Each reduction is scored under the Stata-verified shrink convention
(residualize on [1, w1, w2], df = n - 1, tss_norm = centered y'y,
k_always = 2, g = max(n, p^2) = 2000, beta-binomial(1,1) model prior) by
BOTH the trusted CPU reference (gpubma.cpu.enumeration.enumerate_models)
and the GPU enumerator (gpubma.gpu.enumerator.enumerate_models_gpu with
keep_scores=True), and compared on: exact model count, every per-model
log score (plus named special models: null, full, singletons, central-
layer patterns), the log normalizer, normalized PMPs, PIPs, the model-
size posterior, BMA coefficient moments (p <= 15; at p = 20 the CPU
coefficient pass is impractical — the repo ladder precedent uses
scores-only aggregates there), and the top-10 model ranking.

Tolerances (float64): the two implementations share the identical formula
and differ only in Cholesky/solve kernel order (SciPy vs cuSOLVER batched).
With k <= 20, n = 2,000 and residualized-Gram condition ~67 the expected
divergence is ~1e-12 (measured on the p = 12..24 ladder); we assert
1e-9 absolute on log scores / coefficients (the repo's established
comparison tolerance) and 1e-10 on normalized probabilities. Do NOT widen
these to make a failure pass — investigate instead (CLAUDE.md rule 9).

This module is local-validation only: it never enumerates p = 30.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from gpubma.cpu.enumeration import enumerate_models
from gpubma.gpu.batch_scorer import torch_cuda_available
from gpubma.priors.model_priors import log_model_prior_function

CUDA_OK, CUDA_MSG = torch_cuda_available()
requires_cuda = pytest.mark.skipif(
    not CUDA_OK, reason=f"CUDA unavailable, GPU parity skipped explicitly: {CUDA_MSG}"
)

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "synthetic" / "panel_30_center15.parquet"

# documented deterministic variable subsets (canonical rows, no resampling)
SUBSETS = {
    10: [f"x{j}" for j in range(1, 6)] + [f"x{j}" for j in range(16, 21)],
    15: [f"x{j}" for j in range(1, 16)],
    20: [f"x{j}" for j in range(1, 11)] + [f"x{j}" for j in range(16, 26)],
}

ATOL_LOGSCORE = 1e-9   # per-model log scores and log normalizer
ATOL_PROB = 1e-10      # normalized PMPs, PIPs, size posterior
ATOL_COEF = 1e-9       # BMA coefficient means / sds
TOP_K = 10


def _special_masks(p: int) -> dict[str, int]:
    """Deterministic named model IDs: null, full, singletons, central-layer."""
    k = p // 2
    masks = {"null": 0, "full": (1 << p) - 1}
    for j in range(p):
        masks[f"singleton_x{j}"] = 1 << j
    masks[f"central_first_{k}"] = (1 << k) - 1
    masks[f"central_last_{k}"] = ((1 << k) - 1) << (p - k)
    masks[f"central_even_bits_{(p + 1) // 2}"] = sum(1 << j for j in range(0, p, 2))
    return masks


@pytest.fixture(scope="module")
def canon() -> pd.DataFrame:
    return pd.read_parquet(PARQUET)


@pytest.fixture(scope="module", params=sorted(SUBSETS), ids=lambda p: f"p{p}")
def parity(request, canon):
    """Run CPU reference and GPU enumerator once per reduction."""
    if not CUDA_OK:
        pytest.skip(f"CUDA unavailable, GPU parity skipped explicitly: {CUDA_MSG}")
    p = request.param
    cols = SUBSETS[p]
    y = canon["y"].to_numpy(np.float64)
    n = len(y)
    X = canon[cols].to_numpy(np.float64)
    A = np.column_stack([np.ones(n), canon[["w1", "w2"]].to_numpy(np.float64)])
    Q, _ = np.linalg.qr(A)
    y_r = y - Q @ (Q.T @ y)
    X_r = X - Q @ (Q.T @ X)
    yc = y - y.mean()
    conv = dict(df_resid=n - 1, tss_norm=float(yc @ yc), k_always=2)
    g = float(max(n, p * p))  # = 2000 for every subset here
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), p)
    with_coef = p <= 15  # ladder precedent: scores-only aggregates at p = 20

    cpu = enumerate_models(X_r, y_r, g=g, log_model_prior=log_prior,
                           compute_coefficients=with_coef, top_k=TOP_K, **conv)

    from gpubma.gpu.enumerator import enumerate_models_gpu
    gpu = enumerate_models_gpu(X_r, y_r, g=g, log_model_prior=log_prior,
                               compute_coefficients=with_coef, top_k=TOP_K,
                               keep_scores=True, **conv)
    return {"p": p, "cpu": cpu, "gpu": gpu, "with_coef": with_coef}


@requires_cuda
def test_exact_model_count(parity):
    p = parity["p"]
    assert parity["cpu"]["n_models_evaluated"] == 2**p
    assert parity["gpu"]["n_models_evaluated"] == 2**p
    assert parity["gpu"]["n_models_expected"] == 2**p


@requires_cuda
def test_per_model_scores_all_and_special(parity):
    cpu_s = parity["cpu"]["log_scores"]          # mask order
    gpu_s = parity["gpu"]["log_scores"]          # mask order (validated inside)
    assert cpu_s.shape == gpu_s.shape == (2 ** parity["p"],)
    diff = np.abs(cpu_s - gpu_s)
    assert diff.max() < ATOL_LOGSCORE, (
        f"p={parity['p']}: max per-model log-score diff {diff.max():.3e} "
        f"at mask {int(diff.argmax())}")
    for name, mask in _special_masks(parity["p"]).items():
        d = abs(float(cpu_s[mask] - gpu_s[mask]))
        assert d < ATOL_LOGSCORE, f"{name} (mask {mask}): diff {d:.3e}"


@requires_cuda
def test_log_normalizer(parity):
    assert abs(parity["cpu"]["log_normalizer"]
               - parity["gpu"]["log_normalizer"]) < ATOL_LOGSCORE


@requires_cuda
def test_normalized_posterior_model_probabilities(parity):
    cpu_pmp = parity["cpu"]["pmp"]
    gpu_pmp = np.exp(parity["gpu"]["log_scores"]
                     - logsumexp(parity["gpu"]["log_scores"]))
    assert abs(gpu_pmp.sum() - 1.0) < 1e-12
    diff = np.abs(cpu_pmp - gpu_pmp)
    assert diff.max() < ATOL_PROB, f"max PMP diff {diff.max():.3e}"


@requires_cuda
def test_posterior_inclusion_probabilities(parity):
    diff = np.abs(parity["cpu"]["pip"] - parity["gpu"]["pip"])
    assert diff.max() < ATOL_PROB, f"max PIP diff {diff.max():.3e}"
    assert (parity["gpu"]["pip"] >= 0).all() and (parity["gpu"]["pip"] <= 1).all()


@requires_cuda
def test_model_size_posterior(parity):
    c, gsd = parity["cpu"]["size_distribution"], parity["gpu"]["size_distribution"]
    assert abs(c.sum() - 1.0) < 1e-12 and abs(gsd.sum() - 1.0) < 1e-9
    assert np.abs(c - gsd).max() < ATOL_PROB
    assert abs(parity["cpu"]["mean_model_size"]
               - parity["gpu"]["mean_model_size"]) < 1e-8


@requires_cuda
def test_bma_coefficient_moments(parity):
    if not parity["with_coef"]:
        pytest.skip("p = 20: CPU coefficient pass impractical; scores-only "
                    "aggregates per ladder precedent (p <= 15 covers moments)")
    for key, atol in (("coef_mean", ATOL_COEF), ("coef_sd", ATOL_COEF)):
        diff = np.abs(parity["cpu"][key] - parity["gpu"][key])
        assert diff.max() < atol, f"max {key} diff {diff.max():.3e}"


@requires_cuda
def test_top_model_ranking(parity):
    cpu_top, gpu_top = parity["cpu"]["top_models"], parity["gpu"]["top_models"]
    assert len(cpu_top) == len(gpu_top) == TOP_K
    assert [m["mask"] for m in cpu_top] == [m["mask"] for m in gpu_top]
    for a, b in zip(cpu_top, gpu_top):
        assert abs(a["pmp"] - b["pmp"]) < ATOL_PROB
        assert abs(a["log_score"] - b["log_score"]) < ATOL_LOGSCORE
