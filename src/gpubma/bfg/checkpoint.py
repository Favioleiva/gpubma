"""Checkpoint serialization, progressive state tracking, and resume capability for BFG."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from gpubma.bfg.registry import EliteRegistry, ModelRecord
from gpubma.bfg.scorer import BFGScorer


@dataclass
class CheckpointState:
    """Serializable snapshot of BFG execution state at a checkpoint."""
    checkpoint_id: int
    eval_count: int
    elapsed_seconds: float
    log_Z_seen: float
    exact_wing_log_mass: Dict[str, float]
    discovered_size_weights: List[float]
    discovered_set_pips: List[float]
    best_model_id: int
    best_log_score: float
    best_discovered_weight: float
    evaluated_count: int
    registry_count: int
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CheckpointState:
        return cls(**data)


class CheckpointManager:
    """Manages progressive checkpointing and lossless execution resumption."""

    def __init__(self, checkpoint_dir: Optional[Union[str, Path]] = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        checkpoint_id: int,
        eval_count: int,
        elapsed_seconds: float,
        log_Z_seen: float,
        exact_wing_log_mass: Dict[int, float],
        discovered_size_weights: np.ndarray,
        discovered_set_pips: np.ndarray,
        best_model_id: int,
        best_log_score: float,
        best_discovered_weight: float,
        scorer: BFGScorer,
        registry: EliteRegistry,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Serialize state, evaluation cache, and registry to disk."""
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint_dir specified.")

        ckpt_subdir = self.checkpoint_dir / f"checkpoint_{checkpoint_id:09d}"
        while ckpt_subdir.exists():
            checkpoint_id += 1
            ckpt_subdir = self.checkpoint_dir / f"checkpoint_{checkpoint_id:09d}"
        ckpt_subdir.mkdir(parents=True, exist_ok=False)

        state = CheckpointState(
            checkpoint_id=checkpoint_id,
            eval_count=eval_count,
            elapsed_seconds=round(elapsed_seconds, 2),
            log_Z_seen=float(log_Z_seen),
            exact_wing_log_mass={str(k): float(v) for k, v in exact_wing_log_mass.items()},
            discovered_size_weights=discovered_size_weights.tolist(),
            discovered_set_pips=discovered_set_pips.tolist(),
            best_model_id=int(best_model_id),
            best_log_score=float(best_log_score),
            best_discovered_weight=float(best_discovered_weight),
            evaluated_count=scorer.n_unique_evaluated,
            registry_count=registry.total_registered(),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metadata=dict(metadata or {}, scorer_counters={
                "n_eval_calls": scorer.n_eval_calls, "n_cache_hits": scorer.n_cache_hits}),
        )

        # 1. Save state summary JSON
        with open(ckpt_subdir / "state.json", "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2)

        # 2. Save evaluation cache as compressed NPZ
        # Decimal strings preserve Python bitmasks beyond signed int64; no pickle.
        cache_ids = np.array([str(m) for m in scorer.eval_order], dtype=str)
        cache_scores = np.array([scorer.cache[m] for m in scorer.eval_order], dtype=np.float64)
        np.savez_compressed(
            ckpt_subdir / "evaluation_cache.npz",
            model_ids=cache_ids,
            log_scores=cache_scores,
        )

        # 3. Save registry records JSON
        reg_records = [r.to_dict() for r in registry.records.values()]
        with open(ckpt_subdir / "elite_registry.json", "w", encoding="utf-8") as f:
            json.dump(reg_records, f, indent=2)

        # Seal all payloads before advancing the atomic pointer.
        digests = {name: hashlib.sha256((ckpt_subdir / name).read_bytes()).hexdigest()
                   for name in ('state.json', 'evaluation_cache.npz', 'elite_registry.json')}
        (ckpt_subdir / 'checksums.json').write_text(json.dumps(digests, sort_keys=True))
        # Update latest pointer
        latest_file = self.checkpoint_dir / "latest_checkpoint.json"
        pointer_temp = self.checkpoint_dir / "latest_checkpoint.json.tmp"
        with open(pointer_temp, "w", encoding="utf-8") as f:
            json.dump({
                "latest_checkpoint_id": checkpoint_id,
                "path": ckpt_subdir.name,
                "checksums_sha256": hashlib.sha256((ckpt_subdir / "checksums.json").read_bytes()).hexdigest(),
                "timestamp": state.timestamp,
            }, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(pointer_temp, latest_file)

        return ckpt_subdir

    def load_latest_checkpoint(
        self,
        scorer: BFGScorer,
        registry: EliteRegistry,
        expected_fingerprint: Optional[str] = None,
    ) -> Optional[CheckpointState]:
        """Load state and restore cache from latest checkpoint if available."""
        if self.checkpoint_dir is None:
            return None

        latest_file = self.checkpoint_dir / "latest_checkpoint.json"
        if not latest_file.exists():
            return None

        with open(latest_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if Path(meta['path']).name != meta['path']:
            raise ValueError('Invalid checkpoint pointer path')
        ckpt_path = self.checkpoint_dir / meta['path']
        checksum_bytes = (ckpt_path / 'checksums.json').read_bytes()
        if hashlib.sha256(checksum_bytes).hexdigest() != meta.get('checksums_sha256'):
            raise ValueError('Checkpoint checksum manifest mismatch')
        digests = json.loads(checksum_bytes)
        if set(digests) != {'state.json', 'evaluation_cache.npz', 'elite_registry.json'}:
            raise ValueError('Incomplete checkpoint manifest')
        for name, expected in digests.items():
            if hashlib.sha256((ckpt_path / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Checkpoint checksum mismatch: {name}')

        # Load state JSON
        with open(ckpt_path / "state.json", "r", encoding="utf-8") as f:
            state_data = json.load(f)
        state = CheckpointState.from_dict(state_data)
        if expected_fingerprint is not None and state.metadata.get("fingerprint") != expected_fingerprint:
            raise ValueError("Checkpoint input/configuration fingerprint mismatch or legacy incomplete checkpoint")

        # Restore evaluation cache
        with np.load(ckpt_path / "evaluation_cache.npz") as cache_npz:
            model_ids = cache_npz["model_ids"]
            log_scores = cache_npz["log_scores"]
            ids = [int(m) for m in model_ids]
            if (len(ids) != len(log_scores) or len(ids) != state.evaluated_count
                    or len(ids) != len(set(ids)) or not np.isfinite(log_scores).all()
                    or any(m < 0 or m >= (1 << scorer.p) for m in ids)
                    or (scorer.max_eval_budget is not None and len(ids) > scorer.max_eval_budget)):
                raise ValueError('Invalid checkpoint evaluation ledger')
            scorer.reset_cache()
            for m, s in zip(model_ids, log_scores):
                scorer.cache[int(m)] = float(s)
                scorer.eval_order.append(int(m))
        counters = state.metadata.get("scorer_counters", {})
        scorer.n_eval_calls = int(counters.get("n_eval_calls", 0))
        scorer.n_cache_hits = int(counters.get("n_cache_hits", 0))

        # Restore registry
        reg_json = ckpt_path / "elite_registry.json"
        if reg_json.exists():
            with open(reg_json, "r", encoding="utf-8") as f:
                records_data = json.load(f)
            registry.records.clear()
            registry.by_k = {k: {} for k in range(registry.p + 1)}
            registry._next_discovery_order = 0
            for item in records_data:
                record = ModelRecord.from_dict(item)
                registry.records[record.model_id] = record
                registry.by_k[record.model_size][record.model_id] = record
                registry._next_discovery_order = max(registry._next_discovery_order, record.discovery_order)

        if (len(registry.records) != state.registry_count
                or set(registry.records) != set(scorer.cache)
                or any(r.log_score != scorer.cache[m] for m, r in registry.records.items())):
            raise ValueError('Checkpoint registry/cache mismatch')
        return state
