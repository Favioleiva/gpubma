"""Unit tests for structured and strong-heredity GPU BMA enumerator."""

import math
import numpy as np
import pytest
import torch

from gpubma.gpu.structured import (
    enumerate_structured_models_gpu,
    get_translog_heredity_masks,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_translog_heredity_combinatorics():
    masks_by_size, count = get_translog_heredity_masks()
    assert count == 1337
    assert sum(len(v) for v in masks_by_size.values()) == 1337


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_structured_gpubma_toy():
    np.random.seed(42)
    n = 200
    n_struct = 14
    n_ctrl = 4
    p = n_struct + n_ctrl

    X = np.random.randn(n, p)
    # y generated from structural linear terms
    y = 0.5 * X[:, 0] + 0.3 * X[:, 2] + 0.8 * X[:, 3] + 0.1 * np.random.randn(n)

    # Demeaned
    y -= np.mean(y)
    X -= np.mean(X, axis=0)

    masks_by_size, count = get_translog_heredity_masks()
    log_prior = [-math.log(p + 1)] * (p + 1)

    result = enumerate_structured_models_gpu(
        X=X,
        y=y,
        df_resid=n - 1,
        g=float(n),
        log_model_prior_by_size=log_prior,
        structural_masks_by_size=masks_by_size,
        n_structural=n_struct,
        n_controls=n_ctrl,
        batch_size=1024,
    )

    assert result.backend == "gpu"
    assert result.precision == "float64"
    assert len(result.pip) == p
    assert result.n_models_evaluated == 1337 * (1 << n_ctrl) - 1
    # Check that true factors have high PIP
    assert result.pip[0] > 0.95  # P
    assert result.pip[2] > 0.95  # L
    assert result.pip[3] > 0.95  # R
