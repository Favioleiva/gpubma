"""GPU tests: real float64 execution when CUDA is available; explicit,
reasoned skips when it is not (never a silent pass)."""

import numpy as np
import pytest

from gpubma.gpu.batch_scorer import torch_cuda_available

CUDA_OK, CUDA_MSG = torch_cuda_available()

requires_cuda = pytest.mark.skipif(
    not CUDA_OK, reason=f"CUDA unavailable, GPU tests skipped explicitly: {CUDA_MSG}"
)


@requires_cuda
def test_actual_float64_gpu_execution():
    import torch

    a = torch.randn(128, 128, dtype=torch.float64, device="cuda")
    b = torch.randn(128, 128, dtype=torch.float64, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    assert c.dtype == torch.float64
    cpu = a.cpu().numpy() @ b.cpu().numpy()
    assert abs(float(c.sum().item()) - float(cpu.sum())) < 1e-8


@requires_cuda
def test_gpu_scores_match_cpu_reference(panel8):
    from gpubma.api import bma_regress

    predictors = [f"x{j}" for j in range(1, 9)]
    r_cpu = bma_regress(data=panel8, outcome="y", predictors=predictors,
                        controls=["w1", "w2"], backend="cpu")
    r_gpu = bma_regress(data=panel8, outcome="y", predictors=predictors,
                        controls=["w1", "w2"], backend="gpu")
    np.testing.assert_allclose(r_gpu.log_scores, r_cpu.log_scores, atol=1e-9)
    assert r_gpu.runtime["gpu"]["precision"] == "float64"


@requires_cuda
def test_gpu_scorer_uses_float64_not_float32(panel8):
    """Rule 6: never silently float32. The scorer builds tensors from float64
    numpy arrays; verify the device result differs from CPU by << float32 eps."""
    from gpubma.api import bma_regress

    predictors = [f"x{j}" for j in range(1, 9)]
    r_gpu = bma_regress(data=panel8, outcome="y", predictors=predictors, backend="gpu")
    diff = r_gpu.runtime["gpu_vs_cpu_max_abs_logscore_diff"]
    assert diff < 1e-9, f"difference {diff} inconsistent with float64 execution"


def test_skip_reason_is_explicit_when_cuda_unavailable():
    """This test always runs: the skip machinery itself must be honest."""
    if CUDA_OK:
        assert CUDA_MSG.startswith("torch")
    else:
        assert "CUDA" in CUDA_MSG or "torch" in CUDA_MSG
