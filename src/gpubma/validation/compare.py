"""Deterministic comparison utilities.

Rule: values are NEVER rounded before comparison; rounding is used only when
formatting the report for display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ComparisonRow:
    quantity: str
    reference: float
    candidate: float
    abs_diff: float
    rel_diff: float
    tolerance: float
    tolerance_kind: str  # "abs" or "rel"
    passed: bool


@dataclass
class ComparisonReport:
    title: str
    reference_label: str
    candidate_label: str
    rows: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.rows)

    def add(self, quantity: str, reference, candidate, tolerance: float,
            tolerance_kind: str = "abs") -> ComparisonRow:
        ref = float(reference)
        cand = float(candidate)
        abs_diff = abs(ref - cand)
        denom = max(abs(ref), abs(cand))
        rel_diff = abs_diff / denom if denom > 0 else 0.0
        metric = abs_diff if tolerance_kind == "abs" else rel_diff
        row = ComparisonRow(quantity, ref, cand, abs_diff, rel_diff,
                            tolerance, tolerance_kind, bool(metric <= tolerance))
        self.rows.append(row)
        return row

    def add_arrays(self, quantity: str, reference, candidate, tolerance: float,
                   tolerance_kind: str = "abs") -> None:
        ref = np.asarray(reference, dtype=np.float64).ravel()
        cand = np.asarray(candidate, dtype=np.float64).ravel()
        if ref.shape != cand.shape:
            self.rows.append(ComparisonRow(
                f"{quantity} (shape)", float(ref.size), float(cand.size),
                math.inf, math.inf, tolerance, tolerance_kind, False))
            return
        worst = int(np.argmax(np.abs(ref - cand))) if ref.size else 0
        self.add(f"{quantity} [max@i={worst}]",
                 ref[worst] if ref.size else 0.0,
                 cand[worst] if cand.size else 0.0,
                 tolerance, tolerance_kind)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"- Reference: {self.reference_label}",
            f"- Candidate: {self.candidate_label}",
            f"- Overall: {'PASS' if self.passed else 'FAIL'}",
            "",
            "| quantity | reference | candidate | abs diff | rel diff | tolerance | pass |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in self.rows:
            lines.append(
                f"| {r.quantity} | {r.reference:.12g} | {r.candidate:.12g} "
                f"| {r.abs_diff:.3e} | {r.rel_diff:.3e} "
                f"| {r.tolerance:.1e} ({r.tolerance_kind}) "
                f"| {'PASS' if r.passed else 'FAIL'} |"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "reference": self.reference_label,
            "candidate": self.candidate_label,
            "passed": self.passed,
            "rows": [r.__dict__ for r in self.rows],
        }


def compare_quantities(title, reference_label, candidate_label, items) -> ComparisonReport:
    """items: iterable of (quantity, reference, candidate, tolerance[, kind])."""
    rep = ComparisonReport(title, reference_label, candidate_label)
    for item in items:
        quantity, ref, cand, tol, *rest = item
        kind = rest[0] if rest else "abs"
        if np.ndim(ref) > 0 or np.ndim(cand) > 0:
            rep.add_arrays(quantity, ref, cand, tol, kind)
        else:
            rep.add(quantity, ref, cand, tol, kind)
    return rep
