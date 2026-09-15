"""Native 128-bit GPU Elite Search: calibration and sequential upper-tail discovery."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from gpubma.bfg.elite_search import ThresholdEstimator, estimate_tail_prevalence
from gpubma.bfg.registry import ModelProvenance
from gpubma.bfg.scorer import BFGScorer
from gpubma.bfg.bfg128.registry import WideEliteRegistry
from gpubma.bfg.bfg128.sampling import DirectUniformShellSampler128


@dataclass
class WideEliteSearchResult:
    """Output container for 128-bit GPU Elite Search in shell k."""

    k: int
    r_k: int
    m_k: int
    threshold_tau: float
    retained_model_keys: List[Tuple[int, int]]
    retained_scores: List[float]
    calibration_scores: np.ndarray
    h_hat_prevalence: float
    se_h_hat: float
    runtime_seconds: float


class WideEliteSearch:
    """Orchestrator for 128-bit calibration and sequential upper-tail discovery."""

    def __init__(self, scorer: BFGScorer, registry: WideEliteRegistry):
        self.scorer = scorer
        self.registry = registry
        self.p = scorer.p
        self.device = scorer.device

    def run_lattice_elite_search(
        self,
        k: int,
        r_k: int,
        m_k: int,
        target_q: float = 0.05,
        threshold_method: str = "raw",
        seed: int = 20260715,
        batch_size: int = 16384,
    ) -> WideEliteSearchResult:
        """Execute Stage A Calibration + Stage B Sequential Discovery in shell k."""
        t0 = time.perf_counter()
        N_k = math.comb(self.p, k)
        known_keys = self.registry.get_discovered_keys(k)

        # Stage A: Random Calibration
        calib_scores: np.ndarray = np.array([], dtype=np.float64)
        calib_keys: List[Tuple[int, int]] = []
        if r_k > 0:
            calib_sampler = DirectUniformShellSampler128(
                p=self.p,
                k=k,
                seed=seed,
                excluded_models=known_keys,
            )
            calib_batch = calib_sampler.sample(r_k, chunk_size=batch_size)
            if len(calib_batch.mask_lo) > 0:
                with torch.no_grad():
                    idx_t = torch.from_numpy(calib_batch.idx.astype(np.int64)).to(self.device)
                    sc_t = self.scorer.score_indices(idx_t, k)
                    calib_scores = sc_t.detach().cpu().numpy()

                for lo, hi, s in zip(calib_batch.mask_lo, calib_batch.mask_hi, calib_scores):
                    m_key = (int(lo), int(hi))
                    calib_keys.append(m_key)
                    self.registry.register(
                        m_key[0], m_key[1], float(s),
                        ModelProvenance.RANDOM_BULK,
                        source_tag=f"calib_k{k}",
                        k=k,
                    )

        if len(calib_scores) == 0:
            return WideEliteSearchResult(
                k=k,
                r_k=0,
                m_k=0,
                threshold_tau=float("-inf"),
                retained_model_keys=[],
                retained_scores=[],
                calibration_scores=np.array([], dtype=np.float64),
                h_hat_prevalence=0.0,
                se_h_hat=0.0,
                runtime_seconds=time.perf_counter() - t0,
            )

        # Estimate threshold tau_k
        if threshold_method == "raw":
            tau_k = ThresholdEstimator.raw_empirical_quantile(calib_scores, target_q)
        elif threshold_method == "finite":
            # For finite population quantile, clamp N_k to sys.maxsize if necessary
            N_k_int = min(N_k, 1 << 62)
            tau_k = ThresholdEstimator.finite_population_quantile(calib_scores, target_q, N_k_int)
        elif threshold_method == "conservative":
            tau_k = ThresholdEstimator.conservative_quantile(calib_scores, target_q, alpha=0.05)
        else:
            raise ValueError(f"Unknown threshold_method '{threshold_method}'")

        # Stage B: Sequential Discovery
        retained_keys: List[Tuple[int, int]] = []
        retained_scores: List[float] = []
        n_search_eval = 0

        if m_k > 0:
            search_exclude = known_keys.union(set(calib_keys))
            search_sampler = DirectUniformShellSampler128(
                p=self.p,
                k=k,
                seed=seed + 1,
                excluded_models=search_exclude,
            )
            search_batch = search_sampler.sample(m_k, chunk_size=batch_size)
            n_search_eval = len(search_batch.mask_lo)

            if len(search_batch.mask_lo) > 0:
                with torch.no_grad():
                    idx_t = torch.from_numpy(search_batch.idx.astype(np.int64)).to(self.device)
                    sc_t = self.scorer.score_indices(idx_t, k)
                    search_scores = sc_t.detach().cpu().numpy()

                for lo, hi, s in zip(search_batch.mask_lo, search_batch.mask_hi, search_scores):
                    m_key = (int(lo), int(hi))
                    sc = float(s)
                    if sc >= tau_k:
                        retained_keys.append(m_key)
                        retained_scores.append(sc)
                        self.registry.register(
                            m_key[0], m_key[1], sc,
                            ModelProvenance.ELITE_HIT,
                            source_tag=f"elite_hit_k{k}",
                            k=k,
                        )
                    else:
                        self.registry.register(
                            m_key[0], m_key[1], sc,
                            ModelProvenance.RANDOM_BULK,
                            source_tag=f"search_k{k}",
                            k=k,
                        )

        # Tail prevalence estimation
        N_k_float = float(min(N_k, 10**18))
        H_hat, se_H_hat, _, _ = estimate_tail_prevalence(
            h_k=len(retained_keys),
            m_k=n_search_eval,
            r_k=len(calib_scores),
            N_k=int(N_k_float),
        )

        return WideEliteSearchResult(
            k=k,
            r_k=len(calib_scores),
            m_k=n_search_eval,
            threshold_tau=tau_k,
            retained_model_keys=retained_keys,
            retained_scores=retained_scores,
            calibration_scores=calib_scores,
            h_hat_prevalence=H_hat,
            se_h_hat=se_H_hat,
            runtime_seconds=time.perf_counter() - t0,
        )
