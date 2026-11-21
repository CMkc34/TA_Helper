"""Tests des Studienberichts und der SVG-Grafiken (R-01)."""

from __future__ import annotations

import dataclasses
import json
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from trading_helfer.research.charts import ChartSeries, histogram_svg, line_chart_svg
from trading_helfer.research.report import (
    DRAWDOWN_FILE,
    EQUITY_FILE,
    MONTE_CARLO_FILE,
    REPORT_FILE,
    SUMMARY_FILE,
    render_markdown,
    summary_dict,
    write_report,
)
from trading_helfer.research.study import Robustness, StudyResult

SECTIONS = [
    "## Rahmen",
    "## Bewertungskriterien",
    "## In-Sample (2000–2015)",
    "## Walk-Forward (5/1 Jahre)",
    "## Robustheit",
    "## Monte-Carlo",
    "## Kosten-Stresstest",
    "## Marktphasen",
    "## Out-of-Sample (2016–2023)",
    "## Gesamtzeitraum (2000–2023)",
    "## Protokoll",
]


def test_write_report_creates_all_files(full_study: StudyResult, report_dir: Path) -> None:
    path = write_report(full_study, report_dir)
    assert path == report_dir / REPORT_FILE
    for name in (EQUITY_FILE, DRAWDOWN_FILE, MONTE_CARLO_FILE, SUMMARY_FILE):
        assert (report_dir / name).is_file()
    for name in (EQUITY_FILE, DRAWDOWN_FILE, MONTE_CARLO_FILE):
        root = ET.parse(report_dir / name).getroot()
        assert root.tag.endswith("svg")
    summary = json.loads((report_dir / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["strategy_id"] == "trend_sma_cross"
    assert summary["verdict"] == full_study.verdict.value
    assert [c["key"] for c in summary["criteria"]] == [c.key for c in full_study.criteria]
    assert summary["walk_forward_windows"] == 19
    assert summary["out_of_sample"]["start"] == "2016-01-01"


def test_report_sections_in_fixed_order(full_study: StudyResult) -> None:
    text = render_markdown(full_study)
    positions = [text.index(section) for section in SECTIONS]
    assert positions == sorted(positions)
    assert text.startswith("# Studie: SMA-Kreuzung")
    assert f"**Urteil: {full_study.verdict.value}**" in text
    assert "| Parameter | `fast=10`, `slow=40` |" in text
    assert "| Startkapital | 1.000 USD |" in text
    assert "| Kosten | 1,00 USD je Order, 0,05 % Slippage |" in text
    for criterion in full_study.criteria:
        assert criterion.name in text
    assert "| Kennzahl | Strategie | SPY | QQQ | Universum gleichgewichtet |" in text
    assert "| Kennzahl | Normal | Kosten × 2 |" in text
    assert "Dotcom-Crash 2000–2002" in text
    assert f"![Equity]({EQUITY_FILE})" in text
    robustness = text.split("## Robustheit")[1].split("## Monte-Carlo")[0]
    assert robustness.count("**") == 4  # Maß und gewählte Kombination hervorgehoben


def test_report_is_byte_identical(full_study: StudyResult, tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    write_report(full_study, first)
    write_report(full_study, second)
    for name in (REPORT_FILE, EQUITY_FILE, DRAWDOWN_FILE, MONTE_CARLO_FILE, SUMMARY_FILE):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_report_without_optional_parts(full_study: StudyResult, tmp_path: Path) -> None:
    reduced = dataclasses.replace(
        full_study,
        out_of_sample=None,
        walk_forward=None,
        monte_carlo=None,
        robustness=Robustness(None, "nicht bestimmbar", None),
        phases=full_study.phases.iloc[:0],
        log=[],
    )
    write_report(reduced, tmp_path)
    text = (tmp_path / REPORT_FILE).read_text(encoding="utf-8")
    assert not (tmp_path / MONTE_CARLO_FILE).exists()
    assert "| Out-of-Sample | nicht ausgewertet |" in text
    assert "Noch nicht ausgewertet." in text
    assert "Robustheitsmaß: **nicht bestimmbar**" in text
    assert "Keine Marktphase im ausgewerteten Zeitraum." in text
    assert "- keine Hinweise" in text
    assert text.count("Nicht möglich (siehe Protokoll).") == 2
    summary = summary_dict(reduced)
    assert summary["out_of_sample"] is None
    assert summary["monte_carlo"] is None
    assert summary["walk_forward_windows"] == 0


def test_report_phase_table_marks_missing_series(full_study: StudyResult) -> None:
    phases = full_study.phases
    keep = ~((phases["phase"] == "Corona 2020") & (phases["series"] == "Buy and Hold QQQ"))
    text = render_markdown(dataclasses.replace(full_study, phases=phases[keep]))
    corona = next(line for line in text.splitlines() if line.startswith("| Corona 2020"))
    assert "| – |" in corona


# ---------------------------------------------------------------------------
# Grafiken
# ---------------------------------------------------------------------------


def _series(label: str, values: list[float], unit: str = "ns") -> ChartSeries:
    dates = pd.DatetimeIndex(pd.date_range("2000-01-07", periods=len(values), freq="W-FRI"))
    return ChartSeries(label, dates.as_unit(unit), np.asarray(values))


def test_line_chart_mixes_time_units_and_draws_markers() -> None:
    svg = line_chart_svg(
        [_series("A", [100.0, 200.0, 400.0]), _series("B", [100.0, 110.0, 90.0], unit="us")],
        title="Test & Co",
        log_scale=True,
        markers=[(date(2000, 1, 14), "Grenze"), (date(2010, 1, 1), "außerhalb")],
    )
    root = ET.fromstring(svg)
    polylines = [el for el in root.iter() if el.tag.endswith("polyline")]
    assert len(polylines) == 2
    xs = [
        sorted(float(p.split(",")[0]) for p in line.attrib["points"].split()) for line in polylines
    ]
    assert xs[0] == xs[1]  # gleiche Daten → gleiche x-Positionen trotz ns/us
    assert "Test &amp; Co" in svg
    assert "Grenze" in svg
    assert "außerhalb" not in svg


def test_line_chart_linear_ticks_and_empty_input() -> None:
    svg = line_chart_svg([_series("DD", [0.0, -12.5, -3.0])], title="DD", unit=" %")
    assert "-10,0 %" in svg
    assert "0,00 %" in svg
    empty = line_chart_svg([_series("leer", [])], title="leer")
    assert "Keine Daten" in empty
    flat = line_chart_svg([_series("flach", [5.0, 5.0])], title="flach")
    assert "<polyline" in flat


def test_line_chart_log_scale_skips_non_positive_values() -> None:
    svg = line_chart_svg([_series("X", [-1.0, 0.0])], title="log", log_scale=True)
    assert "<polyline" not in svg


def test_histogram_with_marker_and_edge_cases() -> None:
    svg = histogram_svg([-30.0, -20.0, -20.0, -10.0], title="H", bins=3, marker=-5.0,
                        marker_label="tatsächlich", unit=" %")  # fmt: skip
    root = ET.fromstring(svg)
    rects = [el for el in root.iter() if el.tag.endswith("rect")]
    assert len(rects) == 1 + 3  # Hintergrund plus Klassen
    assert "tatsächlich" in svg
    assert "Keine Daten" in histogram_svg([float("nan")], title="leer")
    assert "<rect x=" in histogram_svg([1.0, 1.0], title="gleich")
