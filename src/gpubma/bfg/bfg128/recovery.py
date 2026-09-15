"""Native 128-bit stratified random shell recovery with online evidence and PIP accumulation."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from scipy.special import logsumexp

from gpubma.bfg.registry import ModelProvenance
from gpubma.bfg.scorer import BFGScorer
from gpubma.bfg.bfg128.bitops import (
    pack_indices,
    popcount,
    unpack_mask,
    unpack_mask_batch,
)
from gpubma.bfg.bfg128.registry import WideEliteRegistry
from gpubma.bfg.bfg128.sampling import DirectUniformShellSampler128


@dataclass
class ShellAccumulator128:
    """Numerically stable online streaming accumulator for shell k."""

    p: int
    k: int
    N_k: int
    D_k: int
    n_recovery: int = 0
    max_score: float = float("-inf")
    sum_weights: float = 0.0
    pip_numerators: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if self.pip_numerators.size == 0:
            self.pip_numerators = np.zeros(self.p, dtype=np.float64)

    def update_batch(
        self,
        scores: np.ndarray,
        idx_batch: np.ndarray,
        weight: float,
    ) -> None:
        """Accumulate a batch of models in shell k with uniform expansion weight.

        Parameters
        ----------
        scores : ndarray of shape (B,)
            Canonical log scores.
        idx_batch : ndarray of shape (B, k)
            Active predictor indices (0 <= j < p).
        weight : float
            Expansion weight per model in this batch.
        """
        if len(scores) == 0 or weight <= 0:
            return

        batch_max = float(np.max(scores))
        if batch_max == float("-inf"):
            return

        if batch_max > self.max_score:
            if self.max_score > float("-inf"):
                factor = math.exp(self.max_score - batch_max)
                self.sum_weights *= factor
                self.pip_numerators *= factor
            self.max_score = batch_max

        # Scaled weights: weight * exp(s_i - max_score)
        scaled_w = weight * np.exp(scores - self.max_score)
        self.sum_weights += float(np.sum(scaled_w))

        # Accumulate into pip_numerators
        if self.k > 0 and idx_batch.shape[1] > 0:
            # Vectorized scatter-add for PIP numerators
            # idx_batch is (B, k), scaled_w is (B,)
            flat_idx = idx_batch.ravel()
            flat_w = np.repeat(scaled_w, self.k)
            np.add.at(self.pip_numerators, flat_idx, flat_w)

    @property
    def log_Z_k(self) -> float:
        """Estimated shell log normalizing constant log Z_hat_k."""
        if self.sum_weights <= 0 or self.max_score == float("-inf"):
            return float("-inf")
        return self.max_score + math.log(self.sum_weights)

    @property
    def pip_k(self) -> np.ndarray:
        """Conditional inclusion probabilities within shell k: PIP_k in [0, 1]."""
        if self.sum_weights <= 0:
            return np.zeros(self.p, dtype=np.float64)
        pips = self.pip_numerators / self.sum_weights
        return np.clip(pips, 0.0, 1.0)


@dataclass
class ShellRecoveryResult128:
    """Summary of 128-bit stratified recovery for shell k."""

    k: int
    N_k: int
    D_k: int
    remainder_k: int
    n_recovery: int
    binding_rule: str
    log_Z_k: float
    pip_k: np.ndarray
    champion_key: Optional[Tuple[int, int]]
    champion_score: float
    next_raw_index: int = 0
    runtime_seconds: float = 0.0


class WideShellRecovery:
    """High-throughput stratified shell recovery engine for 61 <= p <= 128."""

    def __init__(
        self,
        scorer: BFGScorer,
        registry: WideEliteRegistry,
        census_max_combinations: int = 2500,
    ):
        self.scorer = scorer
        self.registry = registry
        self.p = scorer.p
        self.device = scorer.device
        self.census_max = census_max_combinations

    def enumerate_census_shell(self, k: int) -> ShellRecoveryResult128:
        """Exhaustively enumerate, evaluate, and register all models in wing shell k."""
        t0 = time.perf_counter()
        N_k = math.comb(self.p, k)
        known_keys = self.registry.get_discovered_keys(k)

        all_keys: List[Tuple[int, int]] = []
        all_indices: List[List[int]] = []

        if k == 0:
            all_keys.append((0, 0))
            all_indices.append([])
        elif k == 1:
            for j in range(self.p):
                if j < 64:
                    all_keys.append((1 << j, 0))
                else:
                    all_keys.append((0, 1 << (j - 64)))
                all_indices.append([j])
        elif k == self.p:
            key = pack_indices(range(self.p), self.p)
            all_keys.append(key)
            all_indices.append(list(range(self.p)))
        elif k == self.p - 1:
            full_lo, full_hi = pack_indices(range(self.p), self.p)
            for j in range(self.p):
                if j < 64:
                    all_keys.append((full_lo & ~(1 << j), full_hi))
                else:
                    all_keys.append((full_lo, full_hi & ~(1 << (j - 64))))
                idx = [x for x in range(self.p) if x != j]
                all_indices.append(idx)
        else:
            for combo in itertools.combinations(range(self.p), k):
                key = pack_indices(combo, self.p)
                all_keys.append(key)
                all_indices.append(list(combo))

        # Evaluate uncached models
        scores: List[float] = []
        to_eval_keys: List[Tuple[int, int]] = []
        to_eval_idx: List[List[int]] = []
        cached_scores: Dict[Tuple[int, int], float] = {}

        for key, idx in zip(all_keys, all_indices):
            if key in self.registry.records:
                cached_scores[key] = self.registry.records[key].log_score
            else:
                to_eval_keys.append(key)
                to_eval_idx.append(idx)

        if to_eval_keys:
            if k == 0:
                idx_tensor = torch.empty((len(to_eval_keys), 0), dtype=torch.int64, device=self.device)
            else:
                idx_arr = np.array(to_eval_idx, dtype=np.int64)
                idx_tensor = torch.from_numpy(idx_arr).to(self.device)

            with torch.no_grad():
                sc_t = self.scorer.score_indices(idx_tensor, k)
                new_scores = sc_t.detach().cpu().numpy()

            for key, sc in zip(to_eval_keys, new_scores):
                s = float(sc)
                cached_scores[key] = s
                self.registry.register(
                    key[0], key[1], s,
                    ModelProvenance.EXACT_WING,
                    source_tag=f"exact_wing_k{k}",
                    k=k,
                )

        # Assemble full exact shell results
        shell_scores = np.array([cached_scores[key] for key in all_keys], dtype=np.float64)
        if k == 0:
            shell_idx = np.empty((len(all_keys), 0), dtype=np.int64)
        else:
            shell_idx = np.array(all_indices, dtype=np.int64)

        accum = ShellAccumulator128(p=self.p, k=k, N_k=N_k, D_k=N_k, n_recovery=0)
        accum.update_batch(scores=shell_scores, idx_batch=shell_idx, weight=1.0)

        best_idx = int(np.argmax(shell_scores))
        champ_key = all_keys[best_idx]
        champ_score = float(shell_scores[best_idx])

        return ShellRecoveryResult128(
            k=k,
            N_k=N_k,
            D_k=N_k,
            remainder_k=0,
            n_recovery=0,
            binding_rule="census",
            log_Z_k=accum.log_Z_k,
            pip_k=accum.pip_k,
            champion_key=champ_key,
            champion_score=champ_score,
            next_raw_index=0,
            runtime_seconds=time.perf_counter() - t0,
        )

    def recover_shell(
        self,
        k: int,
        target_evaluations: int,
        seed: int = 2026091401,
        start_raw_index: int = 0,
        batch_size: int = 16384,
    ) -> ShellRecoveryResult128:
        """Execute stratified recovery for shell k using direct uniform remainder sampling."""
        t0 = time.perf_counter()
        N_k = math.comb(self.p, k)

        # Check if shell is small enough for exact census
        if N_k <= self.census_max:
            return self.enumerate_census_shell(k)

        # Collect discovered models in this shell from registry
        discovered_dict = self.registry.by_k.get(k, {})
        D_k = len(discovered_dict)
        remainder_k = N_k - D_k

        if remainder_k < 0:
            raise ValueError(f"Discovered count D_k={D_k} exceeds shell population N_k={N_k}")

        accum = ShellAccumulator128(p=self.p, k=k, N_k=N_k, D_k=D_k)

        # 1. Accumulate BFG discoveries with weight 1.0
        if D_k > 0:
            disc_keys = list(discovered_dict.keys())
            disc_scores = np.array([discovered_dict[k_].log_score for k_ in disc_keys], dtype=np.float64)
            lo_arr = np.array([k_[0] for k_ in disc_keys], dtype=np.uint64)
            hi_arr = np.array([k_[1] for k_ in disc_keys], dtype=np.uint64)
            disc_idx = unpack_mask_batch(lo_arr, hi_arr, p=self.p, k=k)
            accum.update_batch(scores=disc_scores, idx_batch=disc_idx, weight=1.0)
            champ_key = disc_keys[int(np.argmax(disc_scores))]
            champ_score = float(np.max(disc_scores))
        else:
            champ_key = None
            champ_score = float("-inf")

        if remainder_k == 0:
            return ShellRecoveryResult128(
                k=k,
                N_k=N_k,
                D_k=D_k,
                remainder_k=0,
                n_recovery=0,
                binding_rule="fully_discovered",
                log_Z_k=accum.log_Z_k,
                pip_k=accum.pip_k,
                champion_key=champ_key,
                champion_score=champ_score,
                next_raw_index=0,
                runtime_seconds=time.perf_counter() - t0,
            )

        # 2. Plan recovery sample count
        n_recovery = min(remainder_k, target_evaluations)
        if n_recovery <= 0:
            return ShellRecoveryResult128(
                k=k,
                N_k=N_k,
                D_k=D_k,
                remainder_k=remainder_k,
                n_recovery=0,
                binding_rule="zero_budget",
                log_Z_k=accum.log_Z_k,
                pip_k=accum.pip_k,
                champion_key=champ_key,
                champion_score=champ_score,
                next_raw_index=0,
                runtime_seconds=time.perf_counter() - t0,
            )

        # Expansion weight per remainder sample model
        expansion_weight = float(remainder_k) / float(n_recovery)

        # Draw remainder samples using DirectUniformShellSampler128
        sampler = DirectUniformShellSampler128(
            p=self.p,
            k=k,
            seed=seed,
            excluded_models=set(discovered_dict.keys()),
        )

        n_sampled = 0
        current_raw_index = start_raw_index
        while n_sampled < n_recovery:
            chunk_n = min(batch_size, n_recovery - n_sampled)
            batch = sampler.sample(chunk_n, start_raw_index=current_raw_index, chunk_size=batch_size)
            current_raw_index = batch.next_raw_index
            batch_count = len(batch.mask_lo)
            if batch_count == 0:
                break

            with torch.no_grad():
                idx_t = torch.from_numpy(batch.idx.astype(np.int64)).to(self.device)
                sc_t = self.scorer.score_indices(idx_t, k)
                batch_scores = sc_t.detach().cpu().numpy()

            accum.update_batch(
                scores=batch_scores,
                idx_batch=batch.idx,
                weight=expansion_weight,
            )

            # Check if any remainder model beat the champion
            chunk_max_idx = int(np.argmax(batch_scores))
            if batch_scores[chunk_max_idx] > champ_score:
                champ_score = float(batch_scores[chunk_max_idx])
                champ_key = (int(batch.mask_lo[chunk_max_idx]), int(batch.mask_hi[chunk_max_idx]))

            n_sampled += batch_count

        binding_rule = "cap" if target_evaluations < remainder_k else "census_remainder"

        return ShellRecoveryResult128(
            k=k,
            N_k=N_k,
            D_k=D_k,
            remainder_k=remainder_k,
            n_recovery=n_sampled,
            binding_rule=binding_rule,
            log_Z_k=accum.log_Z_k,
            pip_k=accum.pip_k,
            champion_key=champ_key,
            champion_score=champ_score,
            next_raw_index=current_raw_index,
            runtime_seconds=time.perf_counter() - t0,
        )
