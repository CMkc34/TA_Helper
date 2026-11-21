"""Einfache SVG-Grafiken für Studienberichte (ohne zusätzliche Bibliothek).

Die Ausgabe ist deterministisch: gleiche Eingaben ergeben byte-gleiche SVG-Dateien.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date as Date  # noqa: N812
from html import escape
from typing import Final

import numpy as np
import pandas as pd

__all__ = ["PALETTE", "ChartSeries", "histogram_svg", "line_chart_svg"]

PALETTE: Final[tuple[str, ...]] = ("#1f4e9e", "#7a7a7a", "#d9822b", "#2e8b57", "#9c3d8f")
"""Linienfarben in Reihenfolge der Reihen (Strategie zuerst)."""

_WIDTH: Final = 900
_HEIGHT: Final = 360
_LEFT: Final = 70
_RIGHT: Final = 20
_TOP: Final = 50
_BOTTOM: Final = 40
_FONT: Final = "font-family='sans-serif' font-size='12'"


@dataclass(frozen=True)
class ChartSeries:
    """Eine Linie im Diagramm.

    Attributes:
        label: Name in der Legende.
        dates: Datum je Wert.
        values: Werte (gleiche Länge wie ``dates``).
    """

    label: str
    dates: pd.DatetimeIndex
    values: Sequence[float] | np.ndarray


def line_chart_svg(
    series: Sequence[ChartSeries],
    *,
    title: str,
    log_scale: bool = False,
    unit: str = "",
    markers: Sequence[tuple[Date, str]] = (),
) -> str:
    """Liniendiagramm über der Zeit.

    Args:
        series: Linien; die erste wird hervorgehoben.
        title: Überschrift.
        log_scale: Logarithmische y-Achse (nur positive Werte).
        unit: Einheit hinter den Achsenbeschriftungen (z. B. ``" %"``).
        markers: Senkrechte Markierungen (Datum, Beschriftung).

    Returns:
        SVG-Dokument als Text.
    """
    parts = _header(title)
    points = [
        (pd.DatetimeIndex(s.dates).as_unit("ns"), np.asarray(s.values, dtype=float)) for s in series
    ]
    filled = [(d, v) for d, v in points if len(d)]
    if not filled:
        parts.append(_text(_WIDTH / 2, _HEIGHT / 2, "Keine Daten", anchor="middle"))
        return _footer(parts)
    x_min = min(d[0] for d, _ in filled).value
    x_max = max(d[-1] for d, _ in filled).value
    finite = np.concatenate([v[np.isfinite(v)] for _, v in filled])
    if log_scale:
        finite = finite[finite > 0]
    y_min = float(finite.min()) if finite.size else 0.0
    y_max = float(finite.max()) if finite.size else 1.0
    transform = math.log10 if log_scale else float
    lo = transform(y_min) if (not log_scale or y_min > 0) else 0.0
    hi = transform(y_max) if (not log_scale or y_max > 0) else 1.0
    if hi - lo < 1e-12:
        lo, hi = lo - 1.0, hi + 1.0

    def px(ts: int) -> float:
        span = max(x_max - x_min, 1)
        return _LEFT + (ts - x_min) / span * (_WIDTH - _LEFT - _RIGHT)

    def py(value: float) -> float:
        return _TOP + (hi - transform(value)) / (hi - lo) * (_HEIGHT - _TOP - _BOTTOM)

    for tick in _y_ticks(lo, hi, log_scale):
        y = py(tick)
        parts.append(_line(_LEFT, y, _WIDTH - _RIGHT, y, "#e0e0e0"))
        parts.append(_text(_LEFT - 6, y + 4, _format(tick) + unit, anchor="end"))
    for year in _x_years(x_min, x_max):
        x = px(pd.Timestamp(year=year, month=1, day=1).value)
        parts.append(_text(x, _HEIGHT - _BOTTOM + 16, str(year), anchor="middle"))
    for day, label in markers:
        ts = pd.Timestamp(day).value
        if x_min <= ts <= x_max:
            x = px(ts)
            parts.append(_line(x, _TOP, x, _HEIGHT - _BOTTOM, "#b00020", dash=True))
            parts.append(_text(x + 4, _TOP + 12, label))
    drawn = list(enumerate(zip(series, points, strict=True)))
    for i, (s, (dates, values)) in [*drawn[1:], *drawn[:1]]:  # Strategie zuletzt (oben)
        color = PALETTE[i % len(PALETTE)]
        coords = [
            f"{px(int(ts)):.1f},{py(float(v)):.1f}"
            for ts, v in zip(dates.asi8, values, strict=True)
            if math.isfinite(v) and (not log_scale or v > 0)
        ]
        if coords:
            width = 2.0 if i == 0 else 1.2
            parts.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='{width}' "
                f"points='{' '.join(coords)}'/>"
            )
        parts.append(_legend(i, s.label, color))
    parts.append(_line(_LEFT, _HEIGHT - _BOTTOM, _WIDTH - _RIGHT, _HEIGHT - _BOTTOM, "#333"))
    return _footer(parts)


def histogram_svg(
    values: Sequence[float] | np.ndarray,
    *,
    title: str,
    bins: int = 30,
    marker: float | None = None,
    marker_label: str = "",
    unit: str = "",
) -> str:
    """Histogramm einer Verteilung.

    Args:
        values: Werte.
        title: Überschrift.
        bins: Anzahl der Klassen.
        marker: Senkrechte Markierung (z. B. tatsächlicher Wert) oder ``None``.
        marker_label: Beschriftung der Markierung.
        unit: Einheit der x-Achse.

    Returns:
        SVG-Dokument als Text.
    """
    parts = _header(title)
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        parts.append(_text(_WIDTH / 2, _HEIGHT / 2, "Keine Daten", anchor="middle"))
        return _footer(parts)
    lo, hi = float(data.min()), float(data.max())
    if marker is not None and math.isfinite(marker):
        lo, hi = min(lo, marker), max(hi, marker)
    if hi - lo < 1e-12:
        lo, hi = lo - 1.0, hi + 1.0
    counts, edges = np.histogram(data, bins=bins, range=(lo, hi))
    top = max(int(counts.max()), 1)
    plot_w = _WIDTH - _LEFT - _RIGHT
    plot_h = _HEIGHT - _TOP - _BOTTOM

    def px(value: float) -> float:
        return _LEFT + (value - lo) / (hi - lo) * plot_w

    for count, left, right in zip(counts, edges[:-1], edges[1:], strict=True):
        h = count / top * plot_h
        parts.append(
            f"<rect x='{px(left):.1f}' y='{_TOP + plot_h - h:.1f}' "
            f"width='{max(px(right) - px(left) - 1, 0.5):.1f}' height='{h:.1f}' "
            f"fill='{PALETTE[0]}' fill-opacity='0.7'/>"
        )
    for k in range(5):
        value = lo + (hi - lo) * k / 4
        parts.append(_text(px(value), _HEIGHT - _BOTTOM + 16, _format(value) + unit, "middle"))
    parts.append(_text(_LEFT - 6, _TOP + 4, str(top), anchor="end"))
    parts.append(_text(_LEFT - 6, _TOP + plot_h + 4, "0", anchor="end"))
    if marker is not None and math.isfinite(marker):
        x = px(marker)
        parts.append(_line(x, _TOP, x, _TOP + plot_h, "#b00020", dash=True))
        parts.append(_text(x + 4, _TOP + 12, marker_label))
    parts.append(_line(_LEFT, _TOP + plot_h, _WIDTH - _RIGHT, _TOP + plot_h, "#333"))
    return _footer(parts)


# ---------------------------------------------------------------------------
# Interna
# ---------------------------------------------------------------------------


def _header(title: str) -> list[str]:
    return [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{_WIDTH}' height='{_HEIGHT}' "
        f"viewBox='0 0 {_WIDTH} {_HEIGHT}'>",
        f"<rect width='{_WIDTH}' height='{_HEIGHT}' fill='white'/>",
        f"<text x='{_LEFT}' y='22' font-family='sans-serif' font-size='15' "
        f"font-weight='bold'>{escape(title)}</text>",
    ]


def _footer(parts: list[str]) -> str:
    return "\n".join([*parts, "</svg>"]) + "\n"


def _legend(index: int, label: str, color: str) -> str:
    x = _LEFT + 200 * index
    y = 38
    return (
        f"<line x1='{x}' y1='{y}' x2='{x + 18}' y2='{y}' stroke='{color}' stroke-width='3'/>"
        + _text(x + 24, y + 4, label)
    )


def _line(x1: float, y1: float, x2: float, y2: float, color: str, *, dash: bool = False) -> str:
    style = " stroke-dasharray='4 3'" if dash else ""
    return (
        f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{color}'{style}/>"
    )


def _text(x: float, y: float, text: str, anchor: str = "start") -> str:
    return f"<text x='{x:.1f}' y='{y:.1f}' {_FONT} text-anchor='{anchor}'>{escape(text)}</text>"


def _format(value: float) -> str:
    """Achsenwert mit deutschem Dezimalkomma und sinnvoller Genauigkeit."""
    magnitude = abs(value)
    digits = 0 if magnitude >= 100 else (1 if magnitude >= 10 else 2)
    text = f"{value:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def _y_ticks(lo: float, hi: float, log_scale: bool) -> list[float]:
    if log_scale:
        first, last = math.floor(lo), math.ceil(hi)
        ticks = [
            m * 10.0**e
            for e in range(first, last + 1)
            for m in (1, 2, 5)
            if lo <= math.log10(m * 10.0**e) <= hi
        ]
        return ticks or [10.0**lo, 10.0**hi]
    raw = (hi - lo) / 4
    power = 10.0 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw)
    first = math.ceil(lo / step - 1e-9)
    return [k * step for k in range(first, math.floor(hi / step + 1e-9) + 1)]


def _x_years(x_min: int, x_max: int) -> list[int]:
    first = pd.Timestamp(x_min).year + 1
    last = pd.Timestamp(x_max).year
    years = list(range(first, last + 1))
    step = max(1, math.ceil(len(years) / 12))
    return years[::step]
