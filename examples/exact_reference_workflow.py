"""Read and export the frozen p30 reference; never evaluate candidate models."""
from pathlib import Path
import hashlib
import json
import math
import shutil
import numpy as np


def load_reference(reference_dir):
    root = Path(reference_dir)
    manifest_path = root / "REFERENCE_MANIFEST.json"
    expected = (root / "REFERENCE_MANIFEST.sha256").read_text().split()[0]
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Exact-reference manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != "gpubma-exact-p30-reference-v1":
        raise ValueError("Unsupported exact-reference schema")
    for name, digest in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Reference filenames must be basenames")
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Exact-reference checksum mismatch: {name}")
    metadata = json.loads((root / "exact_benchmark_metadata.json").read_text())
    champion = json.loads((root / "exact_map.json").read_text())
    truth = json.loads((root / "exact_true_model.json").read_text())
    assert (metadata["n"], metadata["p"], metadata["g"], metadata["models"]) == (2000, 30, 2000, 2**30)
    assert metadata["always_in"] == ["w1", "w2"]
    assert metadata["precision"] == "float64"
    assert metadata["model_prior"] == ["betabinomial", 1.0, 1.0]
    assert champion["model_id"] == 536887295 and truth["model_id"] == 32767
    with np.load(root / "exact_shell_histograms.npz", allow_pickle=False) as histogram:
        np.testing.assert_array_equal(histogram["counts"].sum(axis=1), [math.comb(30, k) for k in range(31)])
        assert histogram["counts"].sum() == 2**30
    return metadata, champion, truth


def export_tables(reference_dir, output_dir):
    """Copy stored numerical tables; append a presentation-only substitution note."""
    source, destination = Path(reference_dir), Path(output_dir)
    if source.resolve() == destination.resolve():
        raise ValueError("Use a separate output directory")
    load_reference(source)
    destination.mkdir(parents=True, exist_ok=True)
    for file in source.glob("*.csv"):
        shutil.copyfile(file, destination / file.name)
    for name in ["exact_benchmark_metadata.json", "exact_map.json", "exact_true_model.json"]:
        shutil.copyfile(source / name, destination / name)
    table = (source / "exact_top5_regression_comparison.tex").read_text(encoding="utf-8")
    table = table.replace(r"\textit{Notes:}", r"\textit{Notes:} The MAP makes one variable substitution relative to $M^*$: $x_{15}$ is replaced by proxy $x_{30}$; binary inclusion Hamming distance is 2.", 1)
    target = destination / "exact_top5_regression_comparison.tex"
    target.write_text(table, encoding="utf-8")
    return target
