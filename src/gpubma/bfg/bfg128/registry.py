"""Native 128-bit multi-path elite registry with (mask_lo, mask_hi) model provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from gpubma.bfg.registry import ModelProvenance
from gpubma.bfg.bfg128.types import Mask128


@dataclass
class WideModelRecord:
    """Detailed record for a registered 128-bit model."""

    mask_lo: int
    mask_hi: int
    model_size: int
    log_score: float
    provenance: ModelProvenance
    parent_key: Optional[Tuple[int, int]] = None
    generation: int = 0
    discovery_order: int = 0
    source_tags: Set[str] = field(default_factory=set)

    @property
    def key(self) -> Tuple[int, int]:
        return (self.mask_lo, self.mask_hi)

    @property
    def model_id(self) -> int:
        return (self.mask_hi << 64) | self.mask_lo

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask_lo": self.mask_lo,
            "mask_hi": self.mask_hi,
            "model_id": self.model_id,
            "model_size": self.model_size,
            "log_score": self.log_score,
            "provenance": self.provenance.value,
            "parent_lo": self.parent_key[0] if self.parent_key else None,
            "parent_hi": self.parent_key[1] if self.parent_key else None,
            "generation": self.generation,
            "discovery_order": self.discovery_order,
            "source_tags": list(self.source_tags),
        }


class WideEliteRegistry:
    """Central registry tracking discovered 128-bit models with (mask_lo, mask_hi) keys."""

    def __init__(self, p: int):
        if not (1 <= p <= 128):
            raise ValueError(f"p must be between 1 and 128, got {p}")
        self.p = p
        self.records: Dict[Tuple[int, int], WideModelRecord] = {}
        self.by_k: Dict[int, Dict[Tuple[int, int], WideModelRecord]] = {k: {} for k in range(p + 1)}
        self._next_discovery_order: int = 0

    def register(
        self,
        mask_lo: int,
        mask_hi: int,
        log_score: float,
        provenance: ModelProvenance,
        parent_key: Optional[Tuple[int, int]] = None,
        generation: int = 0,
        source_tag: str = "",
        k: Optional[int] = None,
    ) -> bool:
        """Register a discovered 128-bit model.

        Returns
        -------
        bool
            True if newly added; False if already present or non-finite score.
        """
        if not np.isfinite(log_score):
            return False

        mask_lo = int(mask_lo) & 0xFFFFFFFFFFFFFFFF
        mask_hi = int(mask_hi) & 0xFFFFFFFFFFFFFFFF
        key = (mask_lo, mask_hi)

        if k is None:
            k = mask_lo.bit_count() + mask_hi.bit_count()

        if key in self.records:
            rec = self.records[key]
            if source_tag:
                rec.source_tags.add(source_tag)
            if rec.provenance != provenance:
                rec.provenance = ModelProvenance.MULTIPLE_SOURCE
            return False

        self._next_discovery_order += 1
        rec = WideModelRecord(
            mask_lo=mask_lo,
            mask_hi=mask_hi,
            model_size=k,
            log_score=float(log_score),
            provenance=provenance,
            parent_key=parent_key,
            generation=generation,
            discovery_order=self._next_discovery_order,
            source_tags={source_tag} if source_tag else set(),
        )
        self.records[key] = rec
        self.by_k[k][key] = rec
        return True

    def register_batch(
        self,
        mask_lo: Sequence[int],
        mask_hi: Sequence[int],
        scores: Sequence[float],
        provenance: ModelProvenance,
        parent_keys: Optional[Sequence[Optional[Tuple[int, int]]]] = None,
        generation: int = 0,
        source_tag: str = "",
        k: Optional[int] = None,
    ) -> int:
        """Register a collection of 128-bit models in batch.

        Returns
        -------
        int
            Number of newly added models.
        """
        n_added = 0
        p_keys = parent_keys if parent_keys is not None else [None] * len(mask_lo)
        for lo, hi, s, p_key in zip(mask_lo, mask_hi, scores, p_keys):
            if self.register(
                mask_lo=lo,
                mask_hi=hi,
                log_score=s,
                provenance=provenance,
                parent_key=p_key,
                generation=generation,
                source_tag=source_tag,
                k=k,
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

    def get_discovered_keys(self, k: int) -> Set[Tuple[int, int]]:
        """Set of all registered model keys (mask_lo, mask_hi) in lattice k."""
        return set(self.by_k[k].keys())

    def get_discovered_scores(self, k: int) -> List[float]:
        """List of all log scores of registered models in lattice k."""
        return [rec.log_score for rec in self.by_k[k].values()]

    def get_champion(self, k: Optional[int] = None) -> Optional[WideModelRecord]:
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
                "mask_lo", "mask_hi", "model_id", "model_size", "log_score",
                "provenance", "parent_lo", "parent_hi", "generation",
                "discovery_order", "source_tags"
            ])
        recs = list(self.records.values())
        df = pd.DataFrame({
            "mask_lo": np.array([r.mask_lo for r in recs], dtype=np.uint64),
            "mask_hi": np.array([r.mask_hi for r in recs], dtype=np.uint64),
            "model_id": [str(r.model_id) for r in recs],
            "model_size": np.array([r.model_size for r in recs], dtype=np.int32),
            "log_score": np.array([r.log_score for r in recs], dtype=np.float64),
            "provenance": [r.provenance.value for r in recs],
            "parent_lo": [r.parent_key[0] if r.parent_key else None for r in recs],
            "parent_hi": [r.parent_key[1] if r.parent_key else None for r in recs],
            "generation": np.array([r.generation for r in recs], dtype=np.int32),
            "discovery_order": np.array([r.discovery_order for r in recs], dtype=np.int64),
            "source_tags": [";".join(sorted(r.source_tags)) for r in recs],
        })
        return df.sort_values("discovery_order").reset_index(drop=True)

    def clear(self) -> None:
        """Clear registry."""
        self.records.clear()
        for k in range(self.p + 1):
            self.by_k[k].clear()
        self._next_discovery_order = 0
