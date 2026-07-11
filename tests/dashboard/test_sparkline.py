from __future__ import annotations

from mlops_toolbox.dashboard.sparkline import render_sparkline


def test_renders_svg_with_expected_number_of_points() -> None:
    svg = render_sparkline([1.0, 2.0, 3.0, 2.5])
    assert svg.startswith('<svg class="sparkline"')
    assert svg.count(",") >= 4  # at least 4 "x,y" coordinate pairs in the polyline
    assert "<polyline" in svg
    assert "<title>" in svg


def test_fallback_for_single_value() -> None:
    result = render_sparkline([1.0])
    assert "not enough data" in result
    assert "<svg" not in result


def test_fallback_for_empty_sequence() -> None:
    result = render_sparkline([])
    assert "not enough data" in result


def test_flat_series_does_not_divide_by_zero() -> None:
    svg = render_sparkline([5.0, 5.0, 5.0])
    assert svg.startswith("<svg")
