"""Strategy routing without enumerating large spaces or changing BFG internals."""
import numpy as np
import pandas as pd
import pytest
import torch

from gpubma import MAX_EXACT_P, GPUBMARegressor, bma_regress, resolve_search


def frame(p, n=80):
    rng = np.random.default_rng(751)
    data = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{j}" for j in range(p)])
    data["y"] = data.x0 + rng.normal(size=n)
    data["control"] = rng.normal(size=n)
    data["district"] = np.arange(n) % 4
    return data, [f"x{j}" for j in range(p)]


@pytest.mark.parametrize("p", [1, 30, 31, 32])
def test_exact_policy(p):
    assert MAX_EXACT_P == 32
    assert resolve_search(p) == resolve_search(p, "exact") == "exact"
    assert resolve_search(p, "bfg") == "bfg"


def test_above_exact_boundary():
    assert resolve_search(33) == "bfg"
    with pytest.raises(ValueError, match="at most 32 selectable regressors.*p=33.*search='bfg'"):
        resolve_search(33, "exact")


@pytest.mark.parametrize("p", [0, -1, 1.5, True])
def test_invalid_dimension(p):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_search(p)


@pytest.mark.parametrize("p", [1, 30, 31, 32])
@pytest.mark.parametrize("n", [80, 140])
def test_public_exact_route_counts_only_candidates(monkeypatch, p, n):
    data, names = frame(p, n)
    seen = {}

    class ReachedExact(Exception):
        pass

    def exact(X, y, **options):
        seen.update(p=X.shape[1], n=len(y), **options)
        raise ReachedExact

    monkeypatch.setattr("gpubma.gpu.enumerator.enumerate_models_gpu", exact)
    with pytest.raises(ReachedExact):
        bma_regress(data, "y", names, controls=["control"],
                    fixed_effects=["individual"], entity_col="district", search="auto")
    assert seen["p"] == p and seen["n"] == n
    assert seen["k_always"] == 4  # control plus three FE dummies; intercept is flat
    assert seen["df_resid"] == n - 1
    assert seen["g"] == max(n, p**2)


@pytest.mark.parametrize("p,search", [(33, "auto"), (1, "bfg"), (32, "bfg")])
def test_public_bfg_route_reuses_existing_engine(monkeypatch, p, search):
    data, names = frame(p)
    expected = object()

    def existing_bfg(y, X, **options):
        assert X.shape == (80, p)
        np.testing.assert_array_equal(X, data[names].to_numpy())
        np.testing.assert_array_equal(options["always_in"], data[["control"]].to_numpy())
        assert options["g"] == "benchmark"
        assert options["model_prior"] == ("betabinomial", 1.0, 1.0)
        assert options["always_prior"] == "shrink"
        assert options["budget_models"] == 100
        assert options["device"] == "cuda"
        return expected

    monkeypatch.setattr("gpubma.bfg.fit_bfg", existing_bfg)
    assert bma_regress(data, "y", names, controls=["control"], search=search,
                       bfg_options={"budget_models": 100}) is expected


def test_unsupported_exact_fails_before_data_access():
    with pytest.raises(ValueError, match="Received p=33"):
        bma_regress(pd.DataFrame(), "y", range(33), search="exact")


def test_estimator_and_method_alias(monkeypatch):
    data, names = frame(2)
    marker = object()
    monkeypatch.setattr("gpubma.bfg.fit_bfg", lambda *args, **kwargs: marker)
    assert bma_regress(data, "y", names, method="bfg") is marker
    assert GPUBMARegressor(names, search="bfg").fit(data, "y").result_ is marker
    with pytest.raises(ValueError, match="Conflicting"):
        bma_regress(data, "y", names, method="exact", search="bfg")


def test_options_and_backend_fail_loudly():
    data, names = frame(2)
    with pytest.raises(ValueError, match="GPU enumeration"):
        bma_regress(data, "y", names, search="exact", backend="cpu")
    with pytest.raises(ValueError, match="bfg_options"):
        bma_regress(data, "y", names, search="exact", bfg_options={"seed": 1})
    with pytest.raises(ValueError, match="exact_options"):
        bma_regress(data, "y", names, search="bfg", exact_options={"max_chunk": 2})
    with pytest.raises(ValueError, match="cannot override"):
        bma_regress(data, "y", names, search="bfg", bfg_options={"g": 4})
    with pytest.raises(ValueError, match="explicit FE dummies"):
        bma_regress(data, "y", names, search="bfg", fixed_effects=["individual"],
                    entity_col="district", fe_method="within", always_prior="flat")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_p33_auto_matches_direct_bfg_numerically():
    from gpubma import fit_bfg
    rng = np.random.default_rng(751)
    X = rng.normal(size=(90, 33)); y = X[:, 0] + rng.normal(size=90)
    names = [f"x{j}" for j in range(33)]
    data = pd.DataFrame(X, columns=names); data["y"] = y
    options = dict(budget_models=2048, wing_max_size=1, beam_width=1,
                   elite_calibration_size=4, recon_sample_per_lattice=2, verbose=False)
    direct = fit_bfg(y, X, candidate_names=names, device="cuda", **options)
    routed = bma_regress(data, "y", names, search="auto", bfg_options=options)
    assert type(routed) is type(direct)
    assert routed.n_models_evaluated == direct.n_models_evaluated == 2048
    assert routed.log_Z == direct.log_Z
    assert routed.map_model_id == direct.map_model_id
    for name in ("pips", "posterior_mean", "posterior_sd", "model_size_posterior"):
        np.testing.assert_array_equal(getattr(routed, name), getattr(direct, name))
