import numpy as np
import pytest

from gpubma.api import bma_regress

PREDICTORS = [f"x{j}" for j in range(1, 9)]


@pytest.fixture(scope="module")
def result(panel8):
    return bma_regress(data=panel8, outcome="y", predictors=PREDICTORS,
                       controls=["w1", "w2"])


def test_exactly_256_models_evaluated(result):
    assert result.n_models_expected == 256
    assert result.n_models_evaluated == 256
    assert len(result.log_scores) == 256


def test_posterior_probabilities_sum_to_one(result):
    assert result.pmp.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(result.pmp >= 0)


def test_pip_bounds(result):
    assert np.all(result.pip >= 0.0) and np.all(result.pip <= 1.0 + 1e-15)


def test_model_size_distribution_sums_to_one(result):
    assert result.size_distribution.sum() == pytest.approx(1.0, abs=1e-12)
    assert result.mean_model_size == pytest.approx(
        float(np.arange(9) @ result.size_distribution), abs=1e-12)


def test_repeatable_results(panel8, result):
    again = bma_regress(data=panel8, outcome="y", predictors=PREDICTORS,
                        controls=["w1", "w2"])
    assert np.array_equal(result.log_scores, again.log_scores)
    assert np.array_equal(result.pip, again.pip)
    assert np.array_equal(result.coef_mean, again.coef_mean)


def test_true_predictors_recovered(result):
    """DGP has beta != 0 for x1..x5 with |beta| >= 0.25 and n=1000."""
    pip = result.inclusion_probabilities()
    assert (pip[[f"x{j}" for j in range(1, 6)]] > 0.95).all()
    assert (pip[[f"x{j}" for j in range(6, 9)]] < 0.5).all()


def test_coefficient_signs_and_magnitudes(result):
    coef = result.coefficients().set_index("predictor")["post_mean"]
    truth = {"x1": 1.5, "x2": -1.0, "x3": 0.75, "x4": 0.5, "x5": 0.25}
    for name, beta in truth.items():
        assert coef[name] == pytest.approx(beta, abs=0.15)
    assert np.all(result.coef_sd >= 0)


def test_mc3_substitution_rejected(panel8):
    with pytest.raises(ValueError, match="enumeration"):
        bma_regress(data=panel8, outcome="y", predictors=PREDICTORS, method="mc3")


def test_silent_float32_rejected(panel8):
    with pytest.raises(ValueError, match="float64"):
        bma_regress(data=panel8, outcome="y", predictors=PREDICTORS, precision="float32")
