"""Download a reproducible snapshot of the classic Grunfeld panel dataset.

Primary source: Stata Press (the exact dataset Stata's `webuse grunfeld`
loads), so the Python and Stata sides share identical observations.
Fallback: the plm (R) copy served by the Rdatasets project.

Writes data/public/grunfeld.{csv,parquet,dta} + grunfeld_provenance.json.
The .dta artifact preserves the original Stata Press bytes when that source
is used (bit-exact provenance); CSV/Parquet are derived from it.

Usage:  python scripts/download_grunfeld.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpubma.datasets.io_utils import file_sha256, frame_sha256  # noqa: E402

OUT = ROOT / "data" / "public"

SOURCES = [
    {
        "name": "Stata Press (webuse grunfeld, Stata 18 data collection)",
        "url": "https://www.stata-press.com/data/r18/grunfeld.dta",
        "format": "dta",
    },
    {
        "name": "Rdatasets (plm::Grunfeld, R plm package)",
        "url": "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Grunfeld.csv",
        "format": "csv",
    },
]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "gpubma-phase1/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, used, errors = None, None, []
    for src in SOURCES:
        try:
            raw = fetch(src["url"])
            used = src
            break
        except Exception as exc:  # noqa: BLE001 — record and try next source
            errors.append({"source": src["name"], "url": src["url"], "error": repr(exc)})
    if raw is None:
        print("ERROR: all sources failed:", json.dumps(errors, indent=2))
        print("Do NOT recreate observations manually; document the failure instead.")
        return 1

    source_sha256 = hashlib.sha256(raw).hexdigest()
    if used["format"] == "dta":
        (OUT / "grunfeld.dta").write_bytes(raw)  # preserve original bytes
        df = pd.read_stata(io.BytesIO(raw))
    else:
        df = pd.read_csv(io.BytesIO(raw))
        df = df.rename(columns={"firm": "company", "inv": "invest", "value": "mvalue",
                                "capital": "kstock"})
        df.to_stata(OUT / "grunfeld.dta", write_index=False, version=118)

    panel_id, time_id = "company", "year"
    df.to_csv(OUT / "grunfeld.csv", index=False)
    df.to_parquet(OUT / "grunfeld.parquet", index=False)

    dup = int(df.duplicated(subset=[panel_id, time_id]).sum())
    provenance = {
        "dataset": "Grunfeld (1958) investment panel",
        "source": used["name"],
        "source_url": used["url"],
        "source_format": used["format"],
        "retrieval_date_utc": datetime.now(timezone.utc).isoformat(),
        "source_bytes_sha256": source_sha256,
        "failed_sources": errors,
        "local_files": {
            name: {"path": str(OUT / name), "sha256": file_sha256(OUT / name)}
            for name in ("grunfeld.csv", "grunfeld.parquet", "grunfeld.dta")
        },
        "content_sha256": frame_sha256(df),
        "row_count": int(len(df)),
        "column_names": list(map(str, df.columns)),
        "panel_identifier": panel_id,
        "time_identifier": time_id,
        "n_companies": int(df[panel_id].nunique()),
        "n_years": int(df[time_id].nunique()),
        "duplicate_panel_time_keys": dup,
        "missing_values_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "notes": "Observations were downloaded, never manually recreated. "
                 "When the Stata Press source is used, grunfeld.dta preserves the "
                 "original downloaded bytes; CSV/Parquet are derived from it.",
    }
    (OUT / "grunfeld_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"source: {used['name']}")
    print(f"rows: {len(df)}, columns: {list(df.columns)}")
    print(f"duplicate (company,year) keys: {dup}")
    print(f"missing values: {sum(provenance['missing_values_by_column'].values())}")
    print(f"source sha256: {source_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
