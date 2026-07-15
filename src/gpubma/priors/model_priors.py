"""Model-space priors over the 2^p candidate models.

Supported:

- ("betabinomial", a, b): the beta-binomial model prior. The inclusion
  probability theta has a Beta(a, b) prior shared by all optional predictors,
  giving, for a model gamma with k included predictors out of p,

      p(gamma) = B(a + k, b + p - k) / B(a, b)

  With a = b = 1 this is uniform over model SIZE (each size has total prior
  probability 1/(p+1)), the default here and a common robust choice
  (Ley & Steel 2009, J. Applied Econometrics 24:651-674).

- "uniform": uniform over the 2^p models, p(gamma) = 2^{-p}.

The returned callable maps model size k -> log prior probability (exact, not
merely proportional; enumeration normalizes anyway, but exactness is free).
"""

from __future__ import annotations

import math
from typing import Callable


def log_model_prior_function(model_prior, n_predictors: int) -> tuple[Callable[[int], float], str]:
    p = n_predictors
    if model_prior == "uniform":
        c = -p * math.log(2.0)
        return (lambda k: c), "uniform over 2^p models"
    if isinstance(model_prior, tuple) and model_prior[0] == "betabinomial":
        _, a, b = model_prior
        if a <= 0 or b <= 0:
            raise ValueError("beta-binomial hyperparameters must be positive")
        lbeta_ab = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

        def log_prior(k: int, _a=float(a), _b=float(b), _p=p, _l=lbeta_ab) -> float:
            return (
                math.lgamma(_a + k)
                + math.lgamma(_b + _p - k)
                - math.lgamma(_a + _b + _p)
                - _l
            )

        return log_prior, f"beta-binomial(a={a:g}, b={b:g})"
    raise ValueError(f"unsupported model_prior: {model_prior!r}")
