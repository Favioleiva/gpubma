"""Zellner g-prior specifications.

FORMULATION (verified against Stata)
------------------------------------
The reference implementation places a Zellner g-prior on regression slopes
with p(sigma^2) proportional to 1/sigma^2 and a flat prior on the intercept::

    beta | sigma^2  ~  N( 0,  g * sigma^2 * (X' X)^{-1} )

Under the default ``always_prior="shrink"`` convention the prior covers the
optional predictors AND the always-included slopes jointly (only the
intercept is flat, df = n - 1). This was VERIFIED on 2026-07-15 against real
Stata ``bmaregress`` batch exports (StataNow/SE 19.5, bmaregress 1.0.2) on
six designs — no fixed effects, individual/time/two-way fixed-effect dummies,
and the Grunfeld data — with maximum absolute differences ~1e-11 in PIPs,
posterior means, and posterior standard deviations
(see reports/comparison_report.md).

The alternative ``always_prior="flat"`` convention (flat always block,
df = n - rank of the always block) is gpubma's own conditional treatment; it
is NOT Stata's and remains available for within-transformed fixed effects.

The default fixed value is the "benchmark" (BRIC) choice of
Fernandez, Ley & Steel (2001, Journal of Econometrics 100:381-427):

    g = max(n, p^2)

which is also Stata's documented default (confirmed on the executed runs:
``e(g)`` = 1000 = max(1000, 64) was displayed as "g = 1,000").
"""

from __future__ import annotations

from dataclasses import dataclass

PROVISIONAL = False  # verified vs StataNow/SE 19.5 on 2026-07-15 (shrink convention)

SOURCES = [
    "Fernandez, Ley & Steel (2001), 'Benchmark priors for Bayesian model "
    "averaging', Journal of Econometrics 100:381-427 (benchmark g = max(n, p^2)).",
    "Liang, Paulo, Molina, Clyde & Berger (2008), JASA 103:410-423 "
    "(g-prior Bayes factor formula).",
    "StataCorp, [BMA] bmaregress manual; parameterization verified against "
    "executed StataNow/SE 19.5 batch output on 2026-07-15 "
    "(always_prior='shrink' convention).",
]


@dataclass(frozen=True)
class GPriorSpec:
    """A resolved fixed-g prior. ``kind`` records how g was chosen."""

    kind: str
    g: float
    provisional: bool = PROVISIONAL

    def describe(self) -> str:
        tag = (" [PROVISIONAL, not verified against Stata]" if self.provisional
               else " [verified vs Stata bmaregress under always_prior='shrink']")
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
