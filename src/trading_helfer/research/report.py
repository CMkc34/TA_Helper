"""Fester Studienbericht: Markdown plus SVG-Grafiken plus ``summary.json``.

Aufbau (immer gleich, damit Kandidaten vergleichbar sind):

1. Rahmen (Strategie, Parameter, Universum, Kosten, Zeiträume)
2. Bewertungskriterien und Urteil
3. In-Sample mit Baselines
4. Walk-Forward 5/1
5. Robustheit (Parameter-Nachbarschaft)
6. Monte-Carlo
7. Kosten-Stresstest
8. Marktphasen
9. Out-of-Sample (zuletzt)
10. Gesamtzeitraum mit Equity- und Drawdown-Grafik
11. Protokoll
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from trading_helfer.backtest.metrics import PerformanceMetrics
from trading_helfer.core.bars import DATE_COLUMN
from trading_helfer.research.charts import ChartSeries, histogram_svg, line_chart_svg
from trading_helfer.research.study import STRATEGY_LABEL, PeriodResult, StudyResult

__all__ = [
    "DRAWDOWN_FILE",
    "EQUITY_FILE",
    "MONTE_CARLO_FILE",
    "REPORT_FILE",
    "SUMMARY_FILE",
    "render_markdown",
    "summary_dict",
    "write_report",
]

REPORT_FILE: Final = "report.md"
EQUITY_FILE: Final = "equity.svg"
DRAWDOWN_FILE: Final = "drawdown.svg"
MONTE_CARLO_FILE: Final = "monte_carlo.svg"
SUMMARY_FILE: Final = "summary.json"

_MARK = {True: "✅ erfüllt", False: "❌ nicht erfüllt", None: "⏳ offen"}

_METRIC_ROWS: Final[tuple[tuple[str, str, int, str], ...]] = (
    ("total_return_pct", "Gesamtrendite", 1, " %"),
    ("cagr_pct", "CAGR", 1, " %"),
    ("volatility_pct", "Volatilität p. a.", 1, " %"),
    ("sharpe", "Sharpe", 2, ""),
    ("sortino", "Sortino", 2, ""),
    ("calmar", "Calmar", 2, ""),
    ("max_drawdown_pct", "Max Drawdown", 1, " %"),
    ("num_trades", "Trades", 0, ""),
    ("win_rate_pct", "Trefferquote", 1, " %"),
    ("profit_factor", "Profit Factor", 2, ""),
    ("exposure_pct", "Exposure", 1, " %"),
)


def write_report(study: StudyResult, out_dir: Path) -> Path:
    """Schreibt Bericht, Grafiken und Zusammenfassung nach ``out_dir``.

    Args:
        study: Ergebnis von :func:`~trading_helfer.research.study.run_study`.
        out_dir: Zielordner (wird angelegt).

    Returns:
        Pfad der Markdown-Datei.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / EQUITY_FILE, _equity_chart(study))
    _write(out_dir / DRAWDOWN_FILE, _drawdown_chart(study))
    if study.monte_carlo is not None:
        mc = study.monte_carlo
        _write(
            out_dir / MONTE_CARLO_FILE,
            histogram_svg(
                mc.max_drawdown_pct,
                title=f"Monte-Carlo: Max Drawdown aus {mc.n_simulations} Umordnungen",
                marker=mc.original_max_drawdown_pct,
                marker_label="tatsächlich",
                unit=" %",
            ),
        )
    summary = json.dumps(summary_dict(study), ensure_ascii=False, indent=2, sort_keys=True)
    _write(out_dir / SUMMARY_FILE, summary + "\n")
    path = out_dir / REPORT_FILE
    _write(path, render_markdown(study))
    return path


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def render_markdown(study: StudyResult) -> str:
    """Erzeugt den Markdown-Bericht (Grafiken als relative Links)."""
    cfg = study.config
    lines: list[str] = [
        f"# Studie: {study.strategy_name}",
        "",
        f"**Urteil: {study.verdict.value}**",
        "",
        "## Rahmen",
        "",
        "| Angabe | Wert |",
        "| --- | --- |",
        f"| Strategie | `{study.strategy_id}` Version {study.strategy_version} |",
        f"| Parameter | {_params(study.params)} |",
        f"| Zeitrahmen | {study.timeframe.value} |",
        f"| Universum | {study.universe} |",
        f"| Startkapital | {_fmt(study.initial_capital, 0)} USD |",
        f"| Kosten | {_fmt(study.costs.fee_per_order, 2)} USD je Order, "
        f"{_fmt(study.costs.slippage_pct, 2)} % Slippage |",
        f"| In-Sample | {cfg.in_sample_start} bis {study.in_sample.end} |",
        f"| Out-of-Sample | {_oos_span(study)} |",
        f"| Datenstand | {study.data_end} |",
        f"| Walk-Forward | {cfg.train_years} Jahre Training, {cfg.test_years} Jahr(e) Test, "
        "rollierend, Ziel Sharpe |",
        f"| Monte-Carlo | {cfg.mc_simulations} Umordnungen, Seed {cfg.mc_seed} |",
        "",
        "## Bewertungskriterien",
        "",
        "| Kriterium | Wert | Ergebnis |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {c.name} | {c.value} | {_MARK[c.passed]} |" for c in study.criteria]
    lines += [
        "",
        "Max Drawdown, Trades und Kosten-Stresstest beziehen sich auf den Gesamtzeitraum, das "
        "Robustheitsmaß auf die In-Sample-Optimierung "
        f"({study.robustness.source}).",
        "",
        f"## In-Sample ({cfg.in_sample_start.year}–{study.in_sample.end.year})",
        "",
        *_metrics_table(study.in_sample),
        "",
        *_walk_forward(study),
        *_robustness(study),
        *_monte_carlo(study),
        "## Kosten-Stresstest",
        "",
        *_compare_table(
            {"Normal": study.full.metrics, study.cost_stress.name: study.cost_stress.metrics}
        ),
        "",
        "## Marktphasen",
        "",
        *_phases(study.phases),
        "",
        *_out_of_sample(study),
        f"## Gesamtzeitraum ({cfg.in_sample_start.year}–{study.data_end.year})",
        "",
        *_metrics_table(study.full),
        "",
        f"![Equity]({EQUITY_FILE})",
        "",
        f"![Drawdown]({DRAWDOWN_FILE})",
        "",
        "## Protokoll",
        "",
    ]
    lines += [f"- {entry}" for entry in study.log] or ["- keine Hinweise"]
    lines += [
        "",
        "Baselines ohne Kosten; das gleichgewichtete Universum wird je Kerze neu gewichtet "
        "(Mitglieder zu Beginn der Kerze).",
        "",
    ]
    return "\n".join(lines)


def _oos_span(study: StudyResult) -> str:
    if study.out_of_sample is None:
        return "nicht ausgewertet"
    return f"{study.out_of_sample.start} bis {study.out_of_sample.end}"


def _params(params: Mapping[str, Any]) -> str:
    if not params:
        return "keine"
    return ", ".join(f"`{k}={v!r}`" for k, v in params.items())


_TRADE_ROWS: Final = frozenset({"num_trades", "win_rate_pct", "profit_factor"})


def _metrics_table(period: PeriodResult) -> list[str]:
    return _compare_table(
        {STRATEGY_LABEL: period.metrics, **period.baselines}, strategies={STRATEGY_LABEL}
    )


def _compare_table(
    columns: Mapping[str, PerformanceMetrics], *, strategies: set[str] | None = None
) -> list[str]:
    """Kennzahlen nebeneinander; Trade-Kennzahlen nur für Spalten in ``strategies``."""
    with_trades = set(columns) if strategies is None else strategies
    names = list(columns)
    lines = [
        "| Kennzahl | " + " | ".join(names) + " |",
        "| --- |" + " ---: |" * len(names),
    ]
    for attr, label, digits, unit in _METRIC_ROWS:
        cells = []
        for name in names:
            value = getattr(columns[name], attr)
            if attr in _TRADE_ROWS and name not in with_trades:
                cells.append("–")
                continue
            cells.append(_fmt(value, digits) + (unit if value is not None else ""))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _walk_forward(study: StudyResult) -> list[str]:
    cfg = study.config
    lines = [f"## Walk-Forward ({cfg.train_years}/{cfg.test_years} Jahre)", ""]
    wf = study.walk_forward
    if wf is None:
        return [*lines, "Nicht möglich (siehe Protokoll).", ""]
    names = list(study.params)
    header = ["Fenster", "Test", *names, "Sharpe Training", "Sharpe Test"]
    lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for w in wf.windows:
        cells = [
            str(w.span.index),
            f"{w.span.test_start} – {w.span.test_end}",
            *[repr(w.params.get(n)) for n in names],
            _fmt(w.train_value, 2),
            _fmt(w.test_value, 2),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    m = wf.metrics if len(wf.equity) else None
    if m is not None:
        lines += [
            "",
            f"Verkettete Testzeiträume: CAGR {_fmt(m.cagr_pct, 1)} %, Sharpe {_fmt(m.sharpe, 2)}, "
            f"Max Drawdown {_fmt(m.max_drawdown_pct, 1)} %, {len(wf.trades)} Trades.",
        ]
    return [*lines, ""]


def _robustness(study: StudyResult) -> list[str]:
    r = study.robustness
    lines = ["## Robustheit", ""]
    value = "nicht bestimmbar" if r.value is None else f"{_fmt(r.value * 100, 0)} %"
    lines.append(f"Robustheitsmaß: **{value}** ({r.source}).")
    opt = r.optimization
    if opt is not None and len(opt.grid) and len(opt.table) > 1:
        varied = [n for n, v in opt.grid.items() if len(v) > 1]
        lines += [
            "",
            "In-Sample-Sharpe der Parameter-Nachbarschaft:",
            "",
            "| " + " | ".join([*varied, "Sharpe", "Trades"]) + " |",
            "|" + " --- |" * (len(varied) + 2),
        ]
        for _, row in opt.table.iterrows():
            chosen = all(row[n] == study.params.get(n) for n in opt.grid)
            cells = [repr(_plain(row[n])) for n in varied]
            sharpe = _fmt(_float(row["objective"]), 2)
            cells += [f"**{sharpe}**" if chosen else sharpe, _fmt(_float(row["num_trades"]), 0)]
            lines.append("| " + " | ".join(cells) + " |")
    return [*lines, ""]


def _monte_carlo(study: StudyResult) -> list[str]:
    lines = ["## Monte-Carlo", ""]
    mc = study.monte_carlo
    if mc is None:
        return [*lines, "Nicht möglich (siehe Protokoll).", ""]
    dd = mc.max_drawdown_percentiles
    lines += [
        f"Umordnung der {mc.n_trades} Trades des Gesamtzeitraums, {mc.n_simulations} "
        f"Simulationen, Seed {mc.seed}.",
        "",
        "| Kennzahl | 5 % | Median | 95 % | tatsächlich |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Max Drawdown | {_fmt(dd.p5, 1)} % | {_fmt(dd.p50, 1)} % | {_fmt(dd.p95, 1)} % | "
        f"{_fmt(mc.original_max_drawdown_pct, 1)} % |",
        "",
        f"![Monte-Carlo]({MONTE_CARLO_FILE})",
        "",
    ]
    return lines


def _phases(phases: pd.DataFrame) -> list[str]:
    if phases.empty:
        return ["Keine Marktphase im ausgewerteten Zeitraum."]
    series = list(dict.fromkeys(phases["series"]))
    lines = [
        "Rendite / Max Drawdown je Phase.",
        "",
        "| Phase | " + " | ".join(series) + " |",
        "| --- |" + " ---: |" * len(series),
    ]
    for phase in dict.fromkeys(phases["phase"]):
        part = phases[phases["phase"] == phase].set_index("series")
        cells = []
        for name in series:
            if name in part.index:
                row = part.loc[name]
                cells.append(
                    f"{_fmt(float(row['return_pct']), 1)} % / "
                    f"{_fmt(float(row['max_drawdown_pct']), 1)} %"
                )
            else:
                cells.append("–")
        lines.append(f"| {phase} | " + " | ".join(cells) + " |")
    return lines


def _out_of_sample(study: StudyResult) -> list[str]:
    if study.out_of_sample is None:
        return [
            "## Out-of-Sample",
            "",
            "Noch nicht ausgewertet. Erst ansehen, wenn die Strategie feststeht "
            "(Lauf ohne `--in-sample-only`).",
            "",
        ]
    oos = study.out_of_sample
    return [
        f"## Out-of-Sample ({oos.start.year}–{oos.end.year})",
        "",
        *_metrics_table(oos),
        "",
    ]


# ---------------------------------------------------------------------------
# Grafiken
# ---------------------------------------------------------------------------


def _equity_chart(study: StudyResult) -> str:
    series = [
        ChartSeries(name, pd.DatetimeIndex(c[DATE_COLUMN]), c["equity"].to_numpy(dtype=float))
        for name, c in study.curves.items()
    ]
    markers = []
    if study.out_of_sample is not None:
        markers.append((study.out_of_sample.start, "Out-of-Sample"))
    return line_chart_svg(
        series,
        title="Equity (USD, logarithmisch)",
        log_scale=True,
        markers=markers,
    )


def _drawdown_chart(study: StudyResult) -> str:
    series = []
    for name, c in study.curves.items():
        values = c["equity"].to_numpy(dtype=float)
        start = study.initial_capital
        peak = np.maximum.accumulate(np.concatenate([[start], values]))[1:]
        dd = (values / peak - 1.0) * 100.0
        series.append(ChartSeries(name, pd.DatetimeIndex(c[DATE_COLUMN]), dd))
    return line_chart_svg(series, title="Drawdown", unit=" %")


# ---------------------------------------------------------------------------
# Zusammenfassung
# ---------------------------------------------------------------------------


def summary_dict(study: StudyResult) -> dict[str, Any]:
    """Wichtigste Ergebnisse als JSON-fähiges Dictionary (zum Vergleich der Kandidaten)."""

    def metrics(m: PerformanceMetrics | None) -> dict[str, Any] | None:
        if m is None:
            return None
        return {attr: _clean(getattr(m, attr)) for attr, *_ in _METRIC_ROWS}

    def period(p: PeriodResult | None) -> dict[str, Any] | None:
        if p is None:
            return None
        return {
            "start": p.start.isoformat(),
            "end": p.end.isoformat(),
            "strategy": metrics(p.metrics),
            "baselines": {k: metrics(v) for k, v in p.baselines.items()},
        }

    mc = study.monte_carlo
    return {
        "strategy_id": study.strategy_id,
        "strategy_version": study.strategy_version,
        "params": {k: _clean(v) for k, v in study.params.items()},
        "timeframe": study.timeframe.value,
        "universe": study.universe,
        "data_end": study.data_end.isoformat(),
        "verdict": study.verdict.value,
        "criteria": [
            {"key": c.key, "name": c.name, "value": c.value, "passed": c.passed}
            for c in study.criteria
        ],
        "in_sample": period(study.in_sample),
        "out_of_sample": period(study.out_of_sample),
        "full": period(study.full),
        "cost_stress": period(study.cost_stress),
        "robustness": {"value": _clean(study.robustness.value), "source": study.robustness.source},
        "monte_carlo": mc.summary() if mc is not None else None,
        "walk_forward_windows": len(study.walk_forward.windows) if study.walk_forward else 0,
    }


def _clean(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: float | int | None, digits: int) -> str:
    if value is None:
        return "–"
    number = float(value)
    if not math.isfinite(number):
        return "–"
    text = f"{number:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")
