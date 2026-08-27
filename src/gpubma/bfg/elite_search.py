"""GPU Elite Search: finite-population sequential upper-tail discovery and calibration."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from gpubma.bfg.registry import EliteRegistry, ModelProvenance
from gpubma.bfg.sampling import LatticeSampler
from gpubma.bfg.scorer import BFGScorer


class ThresholdEstimator:
    """Estimates upper-tail score thresholds from random calibration samples."""

    @staticmethod
    def raw_empirical_quantile(scores: np.ndarray, q: float) -> float:
        """Method A: Empirical (1 - q) quantile."""
        return float(np.quantile(scores, 1.0 - q, method="weibull"))

    @staticmethod
    def finite_population_quantile(scores: np.ndarray, q: float, N_k: int) -> float:
        """Method B: Finite-population adjusted empirical quantile."""
        r = len(scores)
        rank_idx = int(math.ceil((1.0 - q) * r))
        rank_idx = max(0, min(r - 1, rank_idx - 1))
        sorted_scores = np.sort(scores)
        return float(sorted_scores[rank_idx])

    @staticmethod
    def conservative_quantile(scores: np.ndarray, q: float, alpha: float = 0.05) -> float:
        """Method C: Conservative lower confidence bound quantile."""
        r = len(scores)
        z = 1.6448536269514722 if alpha == 0.05 else 1.959963984540054
        se_rank = math.sqrt(r * q * (1.0 - q))
        target_rank = int(math.floor(r * (1.0 - q) - z * se_rank))
        target_rank = max(0, min(r - 1, target_rank))
        sorted_scores = np.sort(scores)
        return float(sorted_scores[target_rank])


def estimate_tail_prevalence(
    h_k: int,
    m_k: int,
    r_k: int,
    N_k: int,
) -> Tuple[float, float, float, float]:
    """Estimate total population count of models exceeding threshold in lattice k.

    Returns
    -------
    Tuple[float, float, float, float]
        (H_hat, se_H_hat, total_H_hat, p_hat)
    """
    if m_k <= 0:
        return 0.0, 0.0, 0.0, 0.0

    N_rem = float(N_k - r_k)
    p_hat = float(h_k) / float(m_k)
    H_hat = N_rem * p_hat

    if m_k > 1:
        fpc = max(0.0, 1.0 - (float(m_k) / max(N_rem, 1.0)))
        var_p = fpc * (p_hat * (1.0 - p_hat)) / float(m_k - 1)
        var_H = (N_rem ** 2) * var_p
        se_H = math.sqrt(max(0.0, var_H))
    else:
        se_H = float("nan")

    return H_hat, se_H, H_hat, p_hat


@dataclass
class EliteSearchResult:
    """Output container for GPU Elite Search within a single lattice."""
    k: int
    r_k: int
    m_k: int
    threshold_tau: float
    retained_model_ids: List[int]
    retained_scores: List[float]
    calibration_scores: np.ndarray
    h_hat_prevalence: float
    se_h_hat: float
    runtime_seconds: float


class GPUEliteSearch:
    """Orchestrator for calibration and sequential upper-tail discovery."""

    def __init__(self, scorer: BFGScorer, registry: EliteRegistry):
        self.scorer = scorer
        self.registry = registry
        self.sampler = LatticeSampler(p=scorer.p)

    def run_lattice_elite_search(
        self,
        k: int,
        r_k: int,
        m_k: int,
        target_q: float = 0.05,
        threshold_method: str = "raw",
        rng: Optional[np.random.Generator] = None,
        batch_size: int = 16384,
    ) -> EliteSearchResult:
        """Execute Stage A Calibration + Stage B Sequential Discovery in lattice k."""
        t0 = time.perf_counter()
        if rng is None:
            rng = np.random.default_rng(20260715)

        N_k = math.comb(self.scorer.p, k)
        known_ids = self.registry.get_discovered_ids(k)

        # Stage A: Random Calibration
        calib_ids = self.sampler.sample_combinations(
            k=k, n_samples=r_k, rng=rng, exclude_set=known_ids
        )
        calib_dict = self.scorer.score_batch(calib_ids, chunk_size=batch_size)
        calib_scores = np.array([calib_dict[m] for m in calib_ids], dtype=np.float64)

        # Register calibration sample as RANDOM_BULK / RANDOM_TAIL
        for m, s in zip(calib_ids, calib_scores):
            self.registry.register(m, s, ModelProvenance.RANDOM_BULK, source_tag=f"calib_k{k}")

        if len(calib_scores) == 0:
            return EliteSearchResult(
                k=k,
                r_k=0,
                m_k=0,
                threshold_tau=float("-inf"),
                retained_model_ids=[],
                retained_scores=[],
                calibration_scores=np.array([], dtype=np.float64),
                h_hat_prevalence=0.0,
                se_h_hat=0.0,
                runtime_seconds=time.perf_counter() - t0,
            )

        # Estimate threshold
        if threshold_method == "raw":
            tau_k = ThresholdEstimator.raw_empirical_quantile(calib_scores, target_q)
        elif threshold_method == "finite":
            tau_k = ThresholdEstimator.finite_population_quantile(calib_scores, target_q, N_k)
        elif threshold_method == "conservative":
            tau_k = ThresholdEstimator.conservative_quantile(calib_scores, target_q, alpha=0.05)
        else:
            raise ValueError(f"Unknown threshold_method '{threshold_method}'")

        # Stage B: Sequential Discovery
        exclude_seen = known_ids.union(set(calib_ids))
        search_ids = self.sampler.sample_combinations(
            k=k, n_samples=m_k, rng=rng, exclude_set=exclude_seen
        )

        retained_ids: List[int] = []
        retained_scores: List[float] = []

        if search_ids:
            search_dict = self.scorer.score_batch(search_ids, chunk_size=batch_size)
            for m in search_ids:
                s = search_dict[m]
                if s >= tau_k:
                    retained_ids.append(m)
                    retained_scores.append(s)
                    self.registry.register(m, s, ModelProvenance.ELITE_HIT, source_tag=f"elite_hit_k{k}")
                else:
                    self.registry.register(m, s, ModelProvenance.RANDOM_BULK, source_tag=f"search_k{k}")

        H_hat, se_H_hat, _, _ = estimate_tail_prevalence(
            h_k=len(retained_ids),
            m_k=len(search_ids),
            r_k=len(calib_ids),
            N_k=N_k,
        )

        return EliteSearchResult(
            k=k,
            r_k=len(calib_ids),
            m_k=len(search_ids),
            threshold_tau=tau_k,
            retained_model_ids=retained_ids,
            retained_scores=retained_scores,
            calibration_scores=calib_scores,
            h_hat_prevalence=H_hat,
            se_h_hat=se_H_hat,
            runtime_seconds=time.perf_counter() - t0,
        )
