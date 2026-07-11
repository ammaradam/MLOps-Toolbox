from __future__ import annotations

from collections.abc import Sequence

_STROKE_CLASS = "spark-line"


def render_sparkline(values: Sequence[float], width: int = 120, height: int = 32) -> str:
    """Render a minimal single-series line chart as an inline SVG fragment.

    Colors are set via CSS classes (never an inline hex stroke) so the chart
    follows the page's light/dark theme. The returned markup is safe to embed
    directly (it contains no interpolated free-text, only numeric coordinates)
    — render it in a template with the `safe` filter. Returns a plain "not
    enough data" fallback fragment when fewer than 2 points are given.
    """
    if len(values) < 2:
        return '<span class="spark-empty">not enough data</span>'

    padding = 4
    plot_width = width - 2 * padding
    plot_height = height - 2 * padding

    lo, hi = min(values), max(values)
    span = hi - lo or 1.0  # avoid divide-by-zero when every value is identical

    def point(index: int, value: float) -> tuple[float, float]:
        x = padding + (index / (len(values) - 1)) * plot_width
        y = padding + (1 - (value - lo) / span) * plot_height
        return round(x, 2), round(y, 2)

    points = [point(i, v) for i, v in enumerate(values)]
    path = " ".join(f"{x},{y}" for x, y in points)
    end_x, end_y = points[-1]
    summary = f"{len(values)} points: {values[0]:g} to {values[-1]:g}"

    return (
        f'<svg class="sparkline" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{summary}">'
        f"<title>{summary}</title>"
        f'<polyline class="{_STROKE_CLASS}" fill="none" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" points="{path}" />'
        f'<circle class="spark-end-dot" cx="{end_x}" cy="{end_y}" r="4" stroke-width="2" />'
        "</svg>"
    )
