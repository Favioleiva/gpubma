"""Installed CLI: ``gpubma doctor`` and ``gpubma benchmark``."""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: gpubma {doctor,benchmark} [options]\n"
              "  doctor     hardware/software diagnostics (add --json PATH to save)\n"
              "  benchmark  run the Phase 1 benchmark ladder (--max-predictors N)")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "doctor":
        from gpubma.diagnostics.doctor import main as doctor_main
        return doctor_main(rest)
    if cmd == "benchmark":
        from gpubma.benchmarks.ladder import main as bench_main
        return bench_main(rest)
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2
