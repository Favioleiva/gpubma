"""Estimator-style interface mirroring :func:`gpubma.bma_regress`."""

from __future__ import annotations

from gpubma.api import bma_regress


class GPUBMARegressor:
    """Scikit-learn-flavoured wrapper around :func:`bma_regress`.

    Example
    -------
    >>> est = GPUBMARegressor(predictors=[f"x{j}" for j in range(1, 9)])
    >>> est.fit(df, outcome="y")
    >>> est.result_.summary()
    """

    def __init__(
        self,
        predictors=None,
        *,
        controls=None,
        fixed_effects=None,
        fe_method="dummies",
        entity_col=None,
        time_col=None,
        always_prior="shrink",
        backend="cpu",
        method="enumeration",
        precision="float64",
        g="benchmark",
        model_prior=("betabinomial", 1.0, 1.0),
        top_k=10,
        deterministic=True,
    ):
        self.predictors = predictors
        self.controls = controls
        self.fixed_effects = fixed_effects
        self.fe_method = fe_method
        self.entity_col = entity_col
        self.time_col = time_col
        self.always_prior = always_prior
        self.backend = backend
        self.method = method
        self.precision = precision
        self.g = g
        self.model_prior = model_prior
        self.top_k = top_k
        self.deterministic = deterministic
        self.result_ = None

    def fit(self, data, outcome):
        self.result_ = bma_regress(
            data=data,
            outcome=outcome,
            predictors=self.predictors,
            controls=self.controls,
            fixed_effects=self.fixed_effects,
            fe_method=self.fe_method,
            entity_col=self.entity_col,
            time_col=self.time_col,
            always_prior=self.always_prior,
            backend=self.backend,
            method=self.method,
            precision=self.precision,
            g=self.g,
            model_prior=self.model_prior,
            top_k=self.top_k,
            deterministic=self.deterministic,
        )
        return self

    def summary(self):
        self._check_fitted()
        return self.result_.summary()

    def _check_fitted(self):
        if self.result_ is None:
            raise RuntimeError("call fit() before requesting results")
