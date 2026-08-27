"""Unit tests for EliteRegistry and provenance bookkeeping."""

import math
import pytest

from gpubma.bfg.registry import EliteRegistry, ModelProvenance


def test_elite_registry_operations():
    p = 10
    registry = EliteRegistry(p=p)

    # Register models
    m1 = 7   # size 3
    m2 = 11  # size 3
    m3 = 1   # size 1

    assert registry.register(m1, 100.5, ModelProvenance.FORWARD_GENEALOGY, source_tag="step1")
    assert registry.register(m2, 105.2, ModelProvenance.BEAM, source_tag="step2")
    assert registry.register(m3, 50.0, ModelProvenance.RANDOM_BULK, source_tag="calib")

    # Duplicate registration updates tag and marks multiple source
    assert not registry.register(m1, 100.5, ModelProvenance.ELITE_HIT, source_tag="hit")
    rec1 = registry.records[m1]
    assert rec1.provenance == ModelProvenance.MULTIPLE_SOURCE
    assert "step1" in rec1.source_tags
    assert "hit" in rec1.source_tags

    # Test retrieval
    log_Z_k3, count_k3, max_s3 = registry.get_known_elite_sum(3)
    assert count_k3 == 2
    assert max_s3 == 105.2
    assert math.isclose(log_Z_k3, math.log(math.exp(100.5) + math.exp(105.2)), rel_tol=1e-12)

    # Test champion
    champ3 = registry.get_champion(3)
    assert champ3.model_id == m2
    global_champ = registry.get_champion()
    assert global_champ.model_id == m2

    # DataFrame conversion
    df = registry.to_dataframe()
    assert len(df) == 3
    assert "model_id" in df.columns
    assert "provenance" in df.columns
