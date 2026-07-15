"""Zellner g-prior specifications.

PROVISIONAL FORMULATION
-----------------------
The reference implementation uses the Zellner g-prior on the slopes of the
optional predictors, conditional on the always-included block::

    beta_gamma | sigma^2  ~  N( 0,  g * sigma^2 * (X_gamma' X_gamma)^{-1} )

with a flat (improper) prior on the always-included coefficients and
p(sigma^2) proportional to 1/sigma^2, after the optional predictors and the
outcome have been residualized on the always-included block.

The default fixed value is the "benchmark" (BRIC) choice of
Fernandez, Ley & Steel (2001, Journal of Econometrics 100:381-427):

    g = max(n, p^2)

Stata's ``bmaregress`` documentation (Stata 18/19, [BMA] bmaregress) also
describes its default as a fixed g = max(n, p^2). HOWEVER, exact parity with
Stata's parameterization (in particular the handling of always-included
predictors, degrees of freedom, and the marginal-likelihood constant) has NOT
been verified against real Stata output yet. Until then this module is
labelled provisional and results must not be described as Stata-compatible.
See STATUS.md ("Unresolved statistical questions").
"""

from __future__ import annotations

from dataclasses import dataclass

PROVISIONAL = True

SOURCES = [
    "Fernandez, Ley & Steel (2001), 'Benchmark priors for Bayesian model "
    "averaging', Journal of Econometrics 100:381-427 (benchmark g = max(n, p^2)).",
    "Liang, Paulo, Molina, Clyde & Berger (2008), JASA 103:410-423 "
    "(g-prior Bayes factor formula).",
    "StataCorp, [BMA] bmaregress manual (default fixed g = max(n, p^2)); "
    "exact parameterization unverified against Stata output.",
]


@dataclass(frozen=True)
class GPriorSpec:
    """A resolved fixed-g prior. ``kind`` records how g was chosen."""

    kind: str
    g: float
    provisional: bool = PROVISIONAL

    def describe(self) -> str:
        tag = " [PROVISIONAL, not verified against Stata]" if self.provisional else ""
        return f"Zellner g-prior, kind={self.kind}, g={self.g:g}{tag}"


def resolve_g(g, n_obs: int, n_predictors: int) -> GPriorSpec:
    """Resolve the user-supplied ``g`` argument into a fixed numeric g.

    ``g`` may be:
      - "benchmark" (default): g = max(n_obs, n_predictors**2)  [FLS 2001, BRIC]
      - "uip": unit-information prior, g = n_obs  [Kass & Wasserman 1995]
      - a positive float: used directly.
    """
    if g == "benchmark":
        return GPriorSpec("benchmark(max(n,p^2))", float(max(n_obs, n_predictors**2)))
    if g == "uip":
        return GPriorSpec("unit-information(n)", float(n_obs))
    gv = float(g)
    if gv <= 0:
        raise ValueError(f"g must be positive, got {gv}")
    return GPriorSpec("fixed", gv)
