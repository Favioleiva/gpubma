from gpubma.validation.compare import ComparisonReport, compare_quantities


def test_comparison_report_pass_fail():
    rep = ComparisonReport("t", "ref", "cand")
    row = rep.add("a", 1.0, 1.0 + 1e-12, tolerance=1e-9)
    assert row.passed and rep.passed
    rep.add("b", 1.0, 1.1, tolerance=1e-9)
    assert not rep.passed


def test_no_rounding_before_comparison():
    rep = ComparisonReport("t", "ref", "cand")
    row = rep.add("tiny", 1.0, 1.0 + 1e-15, tolerance=0.0)
    assert not row.passed  # a rounded comparison would spuriously pass


def test_relative_tolerance_kind():
    rep = ComparisonReport("t", "ref", "cand")
    assert rep.add("r", 1000.0, 1000.5, tolerance=1e-3, tolerance_kind="rel").passed
    assert not rep.add("r2", 1000.0, 1002.0, tolerance=1e-3, tolerance_kind="rel").passed


def test_compare_quantities_arrays_and_markdown():
    rep = compare_quantities("t", "ref", "cand",
                             [("v", [1.0, 2.0], [1.0, 2.0], 0.0)])
    assert rep.passed
    md = rep.to_markdown()
    assert "| quantity |" in md and "PASS" in md
