"""Multi-path elite registry with provenance and cumulative evidence tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from gpubma.bfg.scorer import count_set_bits


class ModelProvenance(str, Enum):
    """Provenance tracking origin for discovered model structures."""
    RANDOM_BULK = "RANDOM_BULK"
    RANDOM_TAIL = "RANDOM_TAIL"
    ELITE_HIT = "ELITE_HIT"
    FORWARD_GENEALOGY = "FORWARD_GENEALOGY"
    BACKWARD_GENEALOGY = "BACKWARD_GENEALOGY"
    BEAM = "BEAM"
    EXACT_WING = "EXACT_WING"
    MULTIPLE_SOURCE = "MULTIPLE_SOURCE"


@dataclass
class ModelRecord:
    """Detailed record for a registered model in the elite registry."""
    model_id: int
    model_size: int
    log_score: float
    provenance: ModelProvenance
    parent_id: Optional[int] = None
    generation: int = 0
    discovery_order: int = 0
    source_tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_size": self.model_size,
            "log_score": self.log_score,
            "provenance": self.provenance.value,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "discovery_order": self.discovery_order,
            "source_tags": list(self.source_tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelRecord:
        data_copy = dict(data)
        data_copy["provenance"] = ModelProvenance(data_copy["provenance"])
        data_copy["source_tags"] = set(data_copy.get("source_tags", []))
        return cls(**data_copy)


class EliteRegistry:
    """Central registry tracking all discovered elite and anchor models with provenance."""

    def __init__(self, p: int):
        self.p = p
        self.records: Dict[int, ModelRecord] = {}
        self.by_k: Dict[int, Dict[int, ModelRecord]] = {k: {} for k in range(p + 1)}
        self._next_discovery_order: int = 0

    def register(
        self,
        model_id: int,
        log_score: float,
        provenance: ModelProvenance,
        parent_id: Optional[int] = None,
        generation: int = 0,
        source_tag: str = "",
        k: Optional[int] = None,
    ) -> bool:
        """Register a discovered model. If already registered, updates source tags.

        Returns
        -------
        bool
            True if newly added; False if already present.
        """
        if k is None:
            k = count_set_bits(model_id)

        if model_id in self.records:
            rec = self.records[model_id]
            if source_tag:
                rec.source_tags.add(source_tag)
            if rec.provenance != provenance:
                rec.provenance = ModelProvenance.MULTIPLE_SOURCE
            return False

        self._next_discovery_order += 1
        rec = ModelRecord(
            model_id=model_id,
            model_size=k,
            log_score=float(log_score),
            provenance=provenance,
            parent_id=parent_id,
            generation=generation,
            discovery_order=self._next_discovery_order,
            source_tags={source_tag} if source_tag else set(),
        )
        self.records[model_id] = rec
        self.by_k[k][model_id] = rec
        return True

    def register_batch(
        self,
        model_ids: Sequence[int],
        scores: Sequence[float],
        provenance: ModelProvenance,
        parent_ids: Optional[Sequence[Optional[int]]] = None,
        generation: int = 0,
        source_tag: str = "",
    ) -> int:
        """Register a collection of models in batch.

        Returns
        -------
        int
            Number of newly added models.
        """
        n_added = 0
        p_ids = parent_ids if parent_ids is not None else [None] * len(model_ids)
        for m, s, p_id in zip(model_ids, scores, p_ids):
            if self.register(
                model_id=m,
                log_score=s,
                provenance=provenance,
                parent_id=p_id,
                generation=generation,
                source_tag=source_tag,
            ):
                n_added += 1
        return n_added

    def get_known_elite_sum(self, k: int) -> Tuple[float, int, float]:
        """Compute exact LogSumExp evidence mass of known registered models in lattice k.

        Returns
        -------
        Tuple[float, int, float]
            (log_Z_known, count, max_score)
        """
        k_dict = self.by_k[k]
        if not k_dict:
            return float("-inf"), 0, float("-inf")
        scores = [rec.log_score for rec in k_dict.values()]
        return float(logsumexp(scores)), len(scores), float(max(scores))

    def get_discovered_ids(self, k: int) -> Set[int]:
        """Set of all registered model IDs in lattice k."""
        return set(self.by_k[k].keys())

    def get_discovered_scores(self, k: int) -> List[float]:
        """List of all log scores of registered models in lattice k."""
        return [rec.log_score for rec in self.by_k[k].values()]

    def get_elite_ids(self, k: int) -> Set[int]:
        """Set of directed elite/genealogy/hit model IDs in lattice k (excluding random bulk)."""
        return {
            m for m, rec in self.by_k[k].items()
            if rec.provenance != ModelProvenance.RANDOM_BULK
        }

    def get_elite_scores(self, k: int) -> List[float]:
        """List of log scores of directed elite/genealogy/hit models in lattice k (excluding random bulk)."""
        return [
            rec.log_score for rec in self.by_k[k].values()
            if rec.provenance != ModelProvenance.RANDOM_BULK
        ]

    def get_champion(self, k: Optional[int] = None) -> Optional[ModelRecord]:
        """Get highest scoring model in lattice k or globally."""
        if k is not None:
            if not self.by_k[k]:
                return None
            return max(self.by_k[k].values(), key=lambda r: r.log_score)
        if not self.records:
            return None
        return max(self.records.values(), key=lambda r: r.log_score)

    def total_registered(self) -> int:
        """Total number of registered models across all lattices."""
        return len(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        """Export registry as a pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=[
                "model_id", "model_size", "log_score", "provenance",
                "parent_id", "generation", "discovery_order", "source_tags"
            ])
        recs = list(self.records.values())
        df = pd.DataFrame({
            "model_id": [r.model_id for r in recs],
            "model_size": [r.model_size for r in recs],
            "log_score": [r.log_score for r in recs],
            "provenance": [r.provenance.value for r in recs],
            "parent_id": [r.parent_id for r in recs],
            "generation": [r.generation for r in recs],
            "discovery_order": [r.discovery_order for r in recs],
            "source_tags": [";".join(sorted(r.source_tags)) for r in recs],
        })
        return df.sort_values("discovery_order").reset_index(drop=True)

    def clear(self) -> None:
        """Clear registry."""
        self.records.clear()
        for k in range(self.p + 1):
            self.by_k[k].clear()
        self._next_discovery_order = 0
