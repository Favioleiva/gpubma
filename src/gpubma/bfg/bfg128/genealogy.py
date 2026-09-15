"""Native 128-bit bidirectional and beam genealogical search over Boolean lattice graph."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from gpubma.bfg.registry import ModelProvenance
from gpubma.bfg.scorer import BFGScorer
from gpubma.bfg.bfg128.bitops import (
    get_children,
    get_parents,
    pack_indices,
    popcount,
    unpack_mask,
    unpack_mask_batch,
)
from gpubma.bfg.bfg128.registry import WideEliteRegistry
from gpubma.bfg.bfg128.types import Mask128


@dataclass
class SearchTrajectoryResult128:
    """Output container for a single 128-bit genealogical search trajectory."""

    algorithm: str
    path: List[Tuple[Tuple[int, int], int, float]]  # ((mask_lo, mask_hi), k, score)
    evals_at_step: List[int]
    best_model_key: Tuple[int, int]
    best_k: int
    best_score: float
    runtime_seconds: float


class WideGenealogicalSearch:
    """Genealogical model search engine operating on 128-bit Boolean lattice graph."""

    def __init__(self, scorer: BFGScorer, registry: WideEliteRegistry):
        self.scorer = scorer
        self.registry = registry
        self.p = scorer.p
        self.device = scorer.device

    def _score_models(self, models: Sequence[Tuple[int, int]], k: int) -> np.ndarray:
        """Score a list of models of identical size k."""
        if not models:
            return np.array([], dtype=np.float64)

        b_size = len(models)
        if k == 0:
            idx_tensor = torch.empty((b_size, 0), dtype=torch.int64, device=self.device)
        else:
            lo_arr = np.array([m[0] for m in models], dtype=np.uint64)
            hi_arr = np.array([m[1] for m in models], dtype=np.uint64)
            idx_arr = unpack_mask_batch(lo_arr, hi_arr, p=self.p, k=k)
            idx_tensor = torch.from_numpy(idx_arr.astype(np.int64)).to(self.device)

        with torch.no_grad():
            scores_t = self.scorer.score_indices(idx_tensor, k)
        return scores_t.detach().cpu().numpy()

    def forward_greedy(
        self,
        start_key: Tuple[int, int] = (0, 0),
        target_k: Optional[int] = None,
        max_eval_budget: Optional[int] = None,
    ) -> SearchTrajectoryResult128:
        """Run forward greedy search from start_key toward higher k."""
        t0 = time.perf_counter()
        if target_k is None:
            target_k = self.p

        curr_key = (int(start_key[0]), int(start_key[1]))
        curr_k = popcount(curr_key[0], curr_key[1])
        curr_score = float(self._score_models([curr_key], curr_k)[0])
        self.registry.register(
            curr_key[0], curr_key[1], curr_score,
            ModelProvenance.FORWARD_GENEALOGY, source_tag="greedy_fwd_root", k=curr_k
        )

        path = [(curr_key, curr_k, curr_score)]
        evals_at_step = [self.registry.total_registered()]

        while curr_k < target_k:
            if max_eval_budget is not None and self.registry.total_registered() >= max_eval_budget:
                break

            children = get_children(curr_key[0], curr_key[1], self.p)
            if not children:
                break

            child_scores = self._score_models(children, curr_k + 1)
            best_idx = int(np.argmax(child_scores))
            best_child = children[best_idx]
            best_score = float(child_scores[best_idx])

            for c, sc in zip(children, child_scores):
                self.registry.register(
                    c[0], c[1], float(sc),
                    ModelProvenance.FORWARD_GENEALOGY,
                    parent_key=curr_key,
                    generation=curr_k + 1,
                    source_tag="greedy_fwd",
                    k=curr_k + 1,
                )

            curr_key = best_child
            curr_k = curr_k + 1
            curr_score = best_score
            path.append((curr_key, curr_k, curr_score))
            evals_at_step.append(self.registry.total_registered())

        best_step = max(path, key=lambda x: x[2])
        return SearchTrajectoryResult128(
            algorithm="forward_greedy_128",
            path=path,
            evals_at_step=evals_at_step,
            best_model_key=best_step[0],
            best_k=best_step[1],
            best_score=best_step[2],
            runtime_seconds=time.perf_counter() - t0,
        )

    def backward_greedy(
        self,
        start_key: Optional[Tuple[int, int]] = None,
        target_k: int = 0,
        max_eval_budget: Optional[int] = None,
    ) -> SearchTrajectoryResult128:
        """Run backward greedy search from start_key toward lower k."""
        t0 = time.perf_counter()
        if start_key is None:
            start_key = pack_indices(range(self.p), self.p)

        curr_key = (int(start_key[0]), int(start_key[1]))
        curr_k = popcount(curr_key[0], curr_key[1])
        curr_score = float(self._score_models([curr_key], curr_k)[0])
        self.registry.register(
            curr_key[0], curr_key[1], curr_score,
            ModelProvenance.BACKWARD_GENEALOGY, source_tag="greedy_bwd_root", k=curr_k
        )

        path = [(curr_key, curr_k, curr_score)]
        evals_at_step = [self.registry.total_registered()]

        while curr_k > target_k:
            if max_eval_budget is not None and self.registry.total_registered() >= max_eval_budget:
                break

            parents = get_parents(curr_key[0], curr_key[1], self.p)
            if not parents:
                break

            parent_scores = self._score_models(parents, curr_k - 1)
            best_idx = int(np.argmax(parent_scores))
            best_parent = parents[best_idx]
            best_score = float(parent_scores[best_idx])

            for p_m, sc in zip(parents, parent_scores):
                self.registry.register(
                    p_m[0], p_m[1], float(sc),
                    ModelProvenance.BACKWARD_GENEALOGY,
                    parent_key=curr_key,
                    generation=curr_k - 1,
                    source_tag="greedy_bwd",
                    k=curr_k - 1,
                )

            curr_key = best_parent
            curr_k = curr_k - 1
            curr_score = best_score
            path.append((curr_key, curr_k, curr_score))
            evals_at_step.append(self.registry.total_registered())

        best_step = max(path, key=lambda x: x[2])
        return SearchTrajectoryResult128(
            algorithm="backward_greedy_128",
            path=path,
            evals_at_step=evals_at_step,
            best_model_key=best_step[0],
            best_k=best_step[1],
            best_score=best_step[2],
            runtime_seconds=time.perf_counter() - t0,
        )

    def forward_beam(
        self,
        start_seeds: Sequence[Tuple[int, int]],
        beam_width: int = 5,
        target_k: Optional[int] = None,
        max_eval_budget: Optional[int] = None,
    ) -> SearchTrajectoryResult128:
        """Forward beam search maintaining top B unique models per generation."""
        t0 = time.perf_counter()
        if target_k is None:
            target_k = self.p

        seed_list = list(dict.fromkeys([(int(m[0]), int(m[1])) for m in start_seeds]))
        # Score seeds by grouping by k
        seed_scores: Dict[Tuple[int, int], float] = {}
        by_k_seeds: Dict[int, List[Tuple[int, int]]] = {}
        for s in seed_list:
            sk = popcount(s[0], s[1])
            by_k_seeds.setdefault(sk, []).append(s)

        for sk, slist in by_k_seeds.items():
            scs = self._score_models(slist, sk)
            for m, sc in zip(slist, scs):
                seed_scores[m] = float(sc)
                self.registry.register(
                    m[0], m[1], float(sc), ModelProvenance.BEAM, source_tag="beam_seed", k=sk
                )

        current_beam = sorted(seed_list, key=lambda m: seed_scores[m], reverse=True)[:beam_width]
        history: List[Tuple[Tuple[int, int], int, float]] = [
            (m, popcount(m[0], m[1]), seed_scores[m]) for m in current_beam
        ]
        evals_at_step = [self.registry.total_registered()]

        while True:
            if max_eval_budget is not None and self.registry.total_registered() >= max_eval_budget:
                break
            current_k = min(popcount(m[0], m[1]) for m in current_beam)
            if current_k >= target_k:
                break

            candidate_children: Set[Tuple[int, int]] = set()
            parent_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
            for m in current_beam:
                if popcount(m[0], m[1]) == current_k:
                    for child in get_children(m[0], m[1], self.p):
                        candidate_children.add(child)
                        if child not in parent_map:
                            parent_map[child] = m

            if not candidate_children:
                break

            child_list = list(candidate_children)
            child_scores = self._score_models(child_list, current_k + 1)

            for c, sc in zip(child_list, child_scores):
                self.registry.register(
                    c[0], c[1], float(sc), ModelProvenance.BEAM,
                    parent_key=parent_map.get(c), generation=current_k + 1,
                    source_tag=f"beam_w{beam_width}", k=current_k + 1,
                )

            order = np.argsort(-child_scores)[:beam_width]
            current_beam = [child_list[i] for i in order]
            for i in order:
                history.append((child_list[i], current_k + 1, float(child_scores[i])))
            evals_at_step.append(self.registry.total_registered())

        best_item = max(history, key=lambda x: x[2])
        return SearchTrajectoryResult128(
            algorithm=f"forward_beam_w{beam_width}_128",
            path=history,
            evals_at_step=evals_at_step,
            best_model_key=best_item[0],
            best_k=best_item[1],
            best_score=best_item[2],
            runtime_seconds=time.perf_counter() - t0,
        )

    def backward_beam(
        self,
        start_seeds: Sequence[Tuple[int, int]],
        beam_width: int = 5,
        target_k: int = 0,
        max_eval_budget: Optional[int] = None,
    ) -> SearchTrajectoryResult128:
        """Backward beam search maintaining top B unique models per generation."""
        t0 = time.perf_counter()
        seed_list = list(dict.fromkeys([(int(m[0]), int(m[1])) for m in start_seeds]))
        seed_scores: Dict[Tuple[int, int], float] = {}
        by_k_seeds: Dict[int, List[Tuple[int, int]]] = {}
        for s in seed_list:
            sk = popcount(s[0], s[1])
            by_k_seeds.setdefault(sk, []).append(s)

        for sk, slist in by_k_seeds.items():
            scs = self._score_models(slist, sk)
            for m, sc in zip(slist, scs):
                seed_scores[m] = float(sc)
                self.registry.register(
                    m[0], m[1], float(sc), ModelProvenance.BEAM, source_tag="bwd_beam_seed", k=sk
                )

        current_beam = sorted(seed_list, key=lambda m: seed_scores[m], reverse=True)[:beam_width]
        history: List[Tuple[Tuple[int, int], int, float]] = [
            (m, popcount(m[0], m[1]), seed_scores[m]) for m in current_beam
        ]
        evals_at_step = [self.registry.total_registered()]

        while True:
            if max_eval_budget is not None and self.registry.total_registered() >= max_eval_budget:
                break
            current_k = max(popcount(m[0], m[1]) for m in current_beam)
            if current_k <= target_k:
                break

            candidate_parents: Set[Tuple[int, int]] = set()
            parent_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
            for m in current_beam:
                if popcount(m[0], m[1]) == current_k:
                    for p_m in get_parents(m[0], m[1], self.p):
                        candidate_parents.add(p_m)
                        if p_m not in parent_map:
                            parent_map[p_m] = m

            if not candidate_parents:
                break

            parent_list = list(candidate_parents)
            parent_scores = self._score_models(parent_list, current_k - 1)

            for p_m, sc in zip(parent_list, parent_scores):
                self.registry.register(
                    p_m[0], p_m[1], float(sc), ModelProvenance.BEAM,
                    parent_key=parent_map.get(p_m), generation=current_k - 1,
                    source_tag=f"bwd_beam_w{beam_width}", k=current_k - 1,
                )

            order = np.argsort(-parent_scores)[:beam_width]
            current_beam = [parent_list[i] for i in order]
            for i in order:
                history.append((parent_list[i], current_k - 1, float(parent_scores[i])))
            evals_at_step.append(self.registry.total_registered())

        best_item = max(history, key=lambda x: x[2])
        return SearchTrajectoryResult128(
            algorithm=f"backward_beam_w{beam_width}_128",
            path=history,
            evals_at_step=evals_at_step,
            best_model_key=best_item[0],
            best_k=best_item[1],
            best_score=best_item[2],
            runtime_seconds=time.perf_counter() - t0,
        )
