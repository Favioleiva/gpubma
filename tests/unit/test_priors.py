import math

import numpy as np
import pytest

from gpubma.priors.gpriors import resolve_g
from gpubma.priors.model_priors import log_model_prior_function


def test_benchmark_g():
    assert resolve_g("benchmark", 1000, 8).g == 1000.0   # max(1000, 64)
    assert resolve_g("benchmark", 100, 30).g == 900.0    # max(100, 900)
    assert resolve_g("uip", 500, 8).g == 500.0
    assert resolve_g(123.0, 10, 2).g == 123.0
    with pytest.raises(ValueError):
        resolve_g(-1.0, 10, 2)


def test_benchmark_g_is_provisional():
    assert resolve_g("benchmark", 1000, 8).provisional is True
    assert "PROVISIONAL" in resolve_g("benchmark", 1000, 8).describe()


@pytest.mark.parametrize("p", [1, 4, 8])
@pytest.mark.parametrize("prior", [("betabinomial", 1.0, 1.0),
                                   ("betabinomial", 2.0, 5.0), "uniform"])
def test_model_prior_sums_to_one_over_model_space(p, prior):
    log_prior, _ = log_model_prior_function(prior, p)
    total = sum(math.comb(p, k) * math.exp(log_prior(k)) for k in range(p + 1))
    assert total == pytest.approx(1.0, rel=1e-12)


def test_betabinomial_uniform_over_sizes():
    p = 8
    log_prior, _ = log_model_prior_function(("betabinomial", 1.0, 1.0), p)
    size_mass = [math.comb(p, k) * math.exp(log_prior(k)) for k in range(p + 1)]
    np.testing.assert_allclose(size_mass, [1.0 / (p + 1)] * (p + 1), rtol=1e-12)
