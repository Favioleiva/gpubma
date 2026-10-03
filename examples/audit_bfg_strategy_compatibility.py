"""Capture deterministic BFG outputs before/after an interface-only change.

Run from a checkout with: python examples/audit_bfg_strategy_compatibility.py OUTPUT.json
Uses existing synthetic test recipes plus a bounded p=33 routing fixture.
Never reads scientific datasets. Runtime/timestamps are intentionally excluded.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gpubma import BFGConfig, fit_bfg


def deterministic_result(result):
    fields = ("log_Z", "map_model_id", "map_log_score", "map_pmp",
              "n_models_evaluated", "total_universe_models")
    record = {key: getattr(result, key) for key in fields}
    for key in ("pips", "posterior_mean", "posterior_sd", "sign_probability",
                "model_size_posterior"):
        record[key] = getattr(result, key).tolist()
    record["result_type"] = type(result).__name__
    record["top_models"] = result.top_models(10).to_json(orient="split", double_precision=15)
    return record


def capture():
    records = {}
    # Existing tests/bfg/test_bfg128_engine.py::test_dispatch_p_less_than_or_equal_60
    rng = np.random.default_rng(46)
    X = rng.standard_normal((40, 10)); y = rng.standard_normal(40)
    for device in ("cpu", "cuda"):
        records[f"existing_p10_{device}"] = deterministic_result(fit_bfg(
            y, X, budget_models=2000, seed=46, device=device, verbose=False))
    # Existing test_dispatch_61_to_128_end_to_end: identical data and configuration.
    rng = np.random.default_rng(47)
    X = rng.standard_normal((100, 67)); beta = np.zeros(67)
    beta[:3] = [2.5, -2., 1.8]; beta[64] = 2.2
    y = X @ beta + rng.standard_normal(100)
    controls = rng.standard_normal((100, 2))
    cfg = BFGConfig(budget_models=500, recon_sample_per_lattice=50,
                    seed=20260915, beam_width=3, verbose=False)
    records["existing_p67_config"] = deterministic_result(fit_bfg(y, X, always_in=controls, config=cfg))
    # Above-boundary check: bounded BFG search, not exact enumeration.
    rng = np.random.default_rng(751)
    X = rng.normal(size=(90, 33)); y = X[:, 0] + rng.normal(size=90)
    records["p33_bounded"] = deterministic_result(fit_bfg(
        y, X, budget_models=2048, device="cuda", wing_max_size=1, beam_width=1,
        elite_calibration_size=4, recon_sample_per_lattice=2, verbose=False))
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return dict(signature=str(inspect.signature(fit_bfg)), records=records,
                numeric_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0))


if __name__ == "__main__":
    output = Path(sys.argv[1])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite baseline: {output}")
    audit = capture()
    output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "numeric_sha256": audit["numeric_sha256"]}))
