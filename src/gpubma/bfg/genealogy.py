"""Bidirectional and beam genealogical search over the BMA Boolean lattice."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from gpubma.bfg.registry import EliteRegistry, ModelProvenance
from gpubma.bfg.scorer import (
    BFGScorer,
    count_set_bits,
    get_immediate_children,
    get_immediate_parents,
    hamming_distance,
)


@dataclass
class SearchTrajectoryResult:
    """Output container for a single genealogical search execution."""
    algorithm: str
    path: List[Tuple[int, int, float]]  # (model_id, k, score)
    evals_at_step: List[int]
    best_model_id: int
    best_k: int
    best_score: float
    runtime_seconds: float


class GenealogicalSearch:
    """Genealogical model search engine operating on the Boolean lattice graph."""

    def __init__(self, scorer: BFGScorer, registry: EliteRegistry):
        self.scorer = scorer
        self.registry = registry
        self.p = scorer.p

    def forward_greedy(
        self,
        start_model_id: int = 0,
        target_k: Optional[int] = None,
    ) -> SearchTrajectoryResult:
        """Run forward greedy search from start_model_id toward higher k."""
        t0 = time.perf_counter()
        if target_k is None:
            target_k = self.p

        curr_m = start_model_id
        curr_score = self.scorer.score_single(curr_m)
        self.registry.register(curr_m, curr_score, ModelProvenance.FORWARD_GENEALOGY, source_tag="greedy_fwd_root")

        path = [(curr_m, count_set_bits(curr_m), curr_score)]
        evals_at_step = [self.scorer.n_unique_evaluated]

        while count_set_bits(curr_m) < target_k:
            children = get_immediate_children(curr_m, self.p)
            if not children:
                break
            child_scores = self.scorer.score_batch(children)
            best_child = max(children, key=lambda c: child_scores[c])
            best_score = child_scores[best_child]

            # Register all evaluated children
            for c in children:
                self.registry.register(
                    c, child_scores[c], ModelProvenance.FORWARD_GENEALOGY,
                    parent_id=curr_m, generation=count_set_bits(c), source_tag="greedy_fwd"
                )

            curr_m = best_child
            curr_score = best_score
            path.append((curr_m, count_set_bits(curr_m), curr_score))
            evals_at_step.append(self.scorer.n_unique_evaluated)

        best_step = max(path, key=lambda x: x[2])
        return SearchTrajectoryResult(
            algorithm="forward_greedy",
            path=path,
            evals_at_step=evals_at_step,
            best_model_id=best_step[0],
            best_k=best_step[1],
            best_score=best_step[2],
            runtime_seconds=time.perf_counter() - t0,
        )

    def backward_greedy(
        self,
        start_model_id: Optional[int] = None,
        target_k: int = 0,
    ) -> SearchTrajectoryResult:
        """Run backward greedy search from start_model_id toward lower k."""
        t0 = time.perf_counter()
        if start_model_id is None:
            start_model_id = (1 << self.p) - 1

        curr_m = start_model_id
        curr_score = self.scorer.score_single(curr_m)
        self.registry.register(curr_m, curr_score, ModelProvenance.BACKWARD_GENEALOGY, source_tag="greedy_bwd_root")

        path = [(curr_m, count_set_bits(curr_m), curr_score)]
        evals_at_step = [self.scorer.n_unique_evaluated]

        while count_set_bits(curr_m) > target_k:
            parents = get_immediate_parents(curr_m, self.p)
            if not parents:
                break
            parent_scores = self.scorer.score_batch(parents)
            best_parent = max(parents, key=lambda p_m: parent_scores[p_m])
            best_score = parent_scores[best_parent]

            # Register all evaluated parents
            for p_m in parents:
                self.registry.register(
                    p_m, parent_scores[p_m], ModelProvenance.BACKWARD_GENEALOGY,
                    parent_id=curr_m, generation=count_set_bits(p_m), source_tag="greedy_bwd"
                )

            curr_m = best_parent
            curr_score = best_score
            path.append((curr_m, count_set_bits(curr_m), curr_score))
            evals_at_step.append(self.scorer.n_unique_evaluated)

        best_step = max(path, key=lambda x: x[2])
        return SearchTrajectoryResult(
            algorithm="backward_greedy",
            path=path,
            evals_at_step=evals_at_step,
            best_model_id=best_step[0],
            best_k=best_step[1],
            best_score=best_step[2],
            runtime_seconds=time.perf_counter() - t0,
        )

    def forward_beam(
        self,
        start_seeds: Sequence[int],
        beam_width: int = 5,
        target_k: Optional[int] = None,
    ) -> SearchTrajectoryResult:
        """Forward beam search maintaining top B unique models per generation."""
        t0 = time.perf_counter()
        if target_k is None:
            target_k = self.p

        seed_list = list(dict.fromkeys(start_seeds))
        self.scorer.score_batch(seed_list)
        for s in seed_list:
            self.registry.register(s, self.scorer.cache[s], ModelProvenance.BEAM, source_tag="beam_seed")

        current_beam = sorted(seed_list, key=lambda m: self.scorer.cache[m], reverse=True)[:beam_width]
        history: List[Tuple[int, int, float]] = [(m, count_set_bits(m), self.scorer.cache[m]) for m in current_beam]
        evals_at_step = [self.scorer.n_unique_evaluated]

        while True:
            if self.scorer.max_eval_budget is not None and self.scorer.n_unique_evaluated >= self.scorer.max_eval_budget:
                break
            current_k = min(count_set_bits(m) for m in current_beam)
            if current_k >= target_k:
                break

            candidate_children: Set[int] = set()
            parent_map: Dict[int, int] = {}
            for m in current_beam:
                if count_set_bits(m) == current_k:
                    for child in get_immediate_children(m, self.p):
                        candidate_children.add(child)
                        if child not in parent_map:
                            parent_map[child] = m

            if not candidate_children:
                break

            child_list = list(candidate_children)
            child_scores = self.scorer.score_batch(child_list)

            for c in child_list:
                self.registry.register(
                    c, child_scores[c], ModelProvenance.BEAM,
                    parent_id=parent_map.get(c), generation=count_set_bits(c), source_tag=f"beam_w{beam_width}"
                )

            next_beam = sorted(child_list, key=lambda m: child_scores[m], reverse=True)[:beam_width]
            current_beam = next_beam
            for m in current_beam:
                history.append((m, count_set_bits(m), self.scorer.cache.get(m, float("-inf"))))
            evals_at_step.append(self.scorer.n_unique_evaluated)

        best_item = max(history, key=lambda x: x[2])
        return SearchTrajectoryResult(
            algorithm=f"forward_beam_w{beam_width}",
            path=history,
            evals_at_step=evals_at_step,
            best_model_id=best_item[0],
            best_k=best_item[1],
            best_score=best_item[2],
            runtime_seconds=time.perf_counter() - t0,
        )

    def backward_beam(
        self,
        start_seeds: Sequence[int],
        beam_width: int = 5,
        target_k: int = 0,
    ) -> SearchTrajectoryResult:
        """Backward beam search maintaining top B unique models per generation."""
        t0 = time.perf_counter()
        seed_list = list(dict.fromkeys(start_seeds))
        self.scorer.score_batch(seed_list)
        for s in seed_list:
            self.registry.register(s, self.scorer.cache[s], ModelProvenance.BEAM, source_tag="bwd_beam_seed")

        current_beam = sorted(seed_list, key=lambda m: self.scorer.cache[m], reverse=True)[:beam_width]
        history: List[Tuple[int, int, float]] = [(m, count_set_bits(m), self.scorer.cache[m]) for m in current_beam]
        evals_at_step = [self.scorer.n_unique_evaluated]

        while True:
            if self.scorer.max_eval_budget is not None and self.scorer.n_unique_evaluated >= self.scorer.max_eval_budget:
                break
            current_k = max(count_set_bits(m) for m in current_beam)
            if current_k <= target_k:
                break

            candidate_parents: Set[int] = set()
            parent_map: Dict[int, int] = {}
            for m in current_beam:
                if count_set_bits(m) == current_k:
                    for p_m in get_immediate_parents(m, self.p):
                        candidate_parents.add(p_m)
                        if p_m not in parent_map:
                            parent_map[p_m] = m

            if not candidate_parents:
                break

            parent_list = list(candidate_parents)
            parent_scores = self.scorer.score_batch(parent_list)

            for p_m in parent_list:
                self.registry.register(
                    p_m, parent_scores[p_m], ModelProvenance.BEAM,
                    parent_id=parent_map.get(p_m), generation=count_set_bits(p_m), source_tag=f"bwd_beam_w{beam_width}"
                )

            next_beam = sorted(parent_list, key=lambda m: parent_scores[m], reverse=True)[:beam_width]
            current_beam = next_beam
            for m in current_beam:
                history.append((m, count_set_bits(m), self.scorer.cache[m]))
            evals_at_step.append(self.scorer.n_unique_evaluated)

        best_item = max(history, key=lambda x: x[2])
        return SearchTrajectoryResult(
            algorithm=f"backward_beam_w{beam_width}",
            path=history,
            evals_at_step=evals_at_step,
            best_model_id=best_item[0],
            best_k=best_item[1],
            best_score=best_item[2],
            runtime_seconds=time.perf_counter() - t0,
        )
