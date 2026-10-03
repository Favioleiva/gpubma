"""Interface regression: never silently discard explicitly supplied search options."""
import pytest

from gpubma import BFGConfig, fit_bfg


def test_config_and_explicit_options_fail_before_engine(monkeypatch):
    called = []

    def engine_fit(*args, **kwargs):
        called.append(True)
        return "unchanged-engine-result"

    monkeypatch.setattr("gpubma.bfg.engine.BFGEngine.fit", engine_fit)
    config = BFGConfig(budget_models=100)
    with pytest.raises(TypeError, match="config OR keyword search options"):
        fit_bfg(None, None, config=config, budget_models=10)
    assert not called
    # Even an explicit value equal to a default must not be silently ignored.
    with pytest.raises(TypeError, match="config OR keyword search options"):
        fit_bfg(None, None, config=config, budget_models=100_000)
    assert not called
    # Data, labels and always-in columns are interface inputs, not config knobs.
    assert fit_bfg(None, None, config=config, candidate_names=["x"],
                   always_in=None, outcome_name="outcome") == "unchanged-engine-result"
    assert fit_bfg(None, None, budget_models=10) == "unchanged-engine-result"
