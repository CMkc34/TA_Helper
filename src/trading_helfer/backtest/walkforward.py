"""Walk-Forward-Analyse: rollierend optimieren und außerhalb der Stichprobe testen.

Ablauf:

1. Der Zeitraum ``start`` bis ``end`` wird in Fenster geteilt: Training
   ``train_years`` Jahre, direkt danach Test ``test_years`` Jahre. Das nächste Fenster
   beginnt ``test_years`` später (rollierend) bzw. am selben Anfang (``anchored``). Die
   Testzeiträume grenzen lückenlos aneinander; der letzte endet bei ``end`` und kann
   kürzer sein.
2. Je Fenster: Raster-Optimierung (:func:`~trading_helfer.backtest.optimize.optimize`)
   nur mit Kursen bis zum letzten Trainingstag, danach ein Backtest mit den besten
   Parametern im Testzeitraum. Kerzen vor dem Testbeginn dienen dort nur dem Anlauf der
   Indikatoren; gehandelt wird erst im Testzeitraum.
3. Die Equity-Kurven der Testzeiträume werden zur Out-of-Sample-Kurve verkettet: jedes
   Fenster startet mit dem Startkapital und wird auf den Endstand des vorigen skaliert.
   Am Ende eines Testfensters offene Positionen sind zum letzten Close bewertet; das
   nächste Fenster startet ohne Position.

Gibt es in einem Trainingsfenster keine bewertbare Kombination, gelten die
Standardparameter der Strategie (Hinweis im Protokoll).

Beispiel::

    wf = walk_forward(SmaCross, grid, "AAPL", Timeframe.W1, "1995-01-01", "2024-12-31",
                      settings, train_years=5, test_years=1)
    print(wf.params_table(), wf.metrics.cagr_pct)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date as Date  # noqa: N812
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import (
    EQUITY_COLUMNS,
    TRADE_COLUMNS,
    BacktestError,
    BacktestResult,
    DateInput,
)
from trading_helfer.backtest.metrics import PerformanceMetrics, compute_metrics
from trading_helfer.backtest.optimize import (
    Objective,
    OptimizationError,
    OptimizationResult,
    ProgressCallback,
    _complete_grid,
    _Context,
    _make_context,
    _run,
    count_combinations,
    objective_value,
    optimize,
)
from trading_helfer.backtest.portfolio import DEFAULT_MAX_POSITIONS, Universe
from trading_helfer.core.bars import DATE_COLUMN
from trading_helfer.core.types import Timeframe
from trading_helfer.strategies.base import Strategy

if TYPE_CHECKING:
    from trading_helfer.core.config import Settings

__all__ = [
    "WINDOW_COLUMN",
    "WalkForwardResult",
    "WalkForwardWindow",
    "WindowSpan",
    "walk_forward",
    "walk_forward_windows",
]

WINDOW_COLUMN = "window"
"""Spalte mit der Fensternummer (ab 1) in der Trade-Liste."""


@dataclass(frozen=True)
class WindowSpan:
    """Zeiträume eines Walk-Forward-Fensters (alle Grenzen eingeschlossen).

    Attributes:
        index: Fensternummer ab 1.
        train_start: Erster Trainingstag.
        train_end: Letzter Trainingstag (Tag vor ``test_start``).
        test_start: Erster Testtag.
        test_end: Letzter Testtag.
    """

    index: int
    train_start: Date
    train_end: Date
    test_start: Date
    test_end: Date


@dataclass(frozen=True)
class WalkForwardWindow:
    """Ergebnis eines Fensters.

    Attributes:
        span: Zeiträume des Fensters.
        params: Im Training gewählte Parameter.
        train_value: Zielwert der gewählten Parameter im Training (``None``, wenn keine
            Kombination bewertbar war).
        test_value: Zielwert im Testzeitraum (``None``, wenn nicht definiert).
        robustness: Anteil der Nachbarparameter mit positivem Zielwert im Training.
        optimization: Vollständige Optimierung des Trainingsfensters.
        test_result: Backtest im Testzeitraum.
    """

    span: WindowSpan
    params: dict[str, Any]
    train_value: float | None
    test_value: float | None
    robustness: float | None
    optimization: OptimizationResult
    test_result: BacktestResult


@dataclass(frozen=True)
class WalkForwardResult:
    """Ergebnis einer Walk-Forward-Analyse.

    Attributes:
        strategy_id: ID der Strategie.
        objective: Zielgröße.
        timeframe: Zeitrahmen.
        train_years: Länge des Trainings in Jahren.
        test_years: Länge des Tests in Jahren.
        anchored: ``True``, wenn das Training immer bei ``start`` beginnt.
        initial_capital: Startkapital der Out-of-Sample-Kurve.
        windows: Fenster in zeitlicher Reihenfolge.
        equity: Zusammengesetzte Out-of-Sample-Equity, Spalten wie
            :data:`~trading_helfer.backtest.engine.EQUITY_COLUMNS`.
        trades: Trades aller Testzeiträume (Beträge im Kapital des jeweiligen Fensters)
            mit Spalte :data:`WINDOW_COLUMN`.
        log: Hinweise, auf Deutsch.
    """

    strategy_id: str
    objective: Objective
    timeframe: Timeframe
    train_years: int
    test_years: int
    anchored: bool
    initial_capital: float
    windows: list[WalkForwardWindow]
    equity: pd.DataFrame
    trades: pd.DataFrame
    log: list[str] = field(default_factory=list)

    @property
    def final_equity(self) -> float:
        """Endstand der Out-of-Sample-Kurve."""
        if len(self.equity) == 0:
            return self.initial_capital
        return float(self.equity["equity"].iloc[-1])

    @property
    def metrics(self) -> PerformanceMetrics:
        """Kennzahlen der Out-of-Sample-Kurve.

        Raises:
            trading_helfer.backtest.metrics.MetricsError: Keine Equity-Werte.
        """
        return compute_metrics(
            self.equity, self.trades, self.timeframe, initial_capital=self.initial_capital
        )

    def params_table(self) -> pd.DataFrame:
        """Parameter und Zielwerte je Fenster als Tabelle."""
        rows = []
        for w in self.windows:
            row: dict[str, Any] = {
                WINDOW_COLUMN: w.span.index,
                "train_start": w.span.train_start,
                "train_end": w.span.train_end,
                "test_start": w.span.test_start,
                "test_end": w.span.test_end,
            }
            row.update(w.params)
            row["train_value"] = w.train_value
            row["test_value"] = w.test_value
            row["robustness"] = w.robustness
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fenster
# ---------------------------------------------------------------------------


def walk_forward_windows(
    start: DateInput,
    end: DateInput,
    *,
    train_years: int = 5,
    test_years: int = 1,
    anchored: bool = False,
) -> list[WindowSpan]:
    """Teilt einen Zeitraum in Trainings- und Testfenster.

    Args:
        start: Erster Tag des ersten Trainings.
        end: Letzter Tag (der letzte Test endet hier, ggf. verkürzt).
        train_years: Trainingsdauer in Jahren (≥ 1).
        test_years: Testdauer und Schrittweite in Jahren (≥ 1).
        anchored: Training beginnt immer bei ``start`` und wächst mit (statt rollierend).

    Returns:
        Fenster in zeitlicher Reihenfolge; die Testzeiträume grenzen lückenlos aneinander.

    Raises:
        OptimizationError: Ungültige Längen oder Zeitraum kürzer als ein Training plus ein
            Tag Test.
    """
    if train_years < 1 or test_years < 1:
        raise OptimizationError(
            "Trainings- und Testdauer müssen mindestens 1 Jahr betragen "
            f"(erhalten: {train_years} und {test_years})."
        )
    first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    spans: list[WindowSpan] = []
    k = 0
    while True:
        train_begin = first if anchored else first + pd.DateOffset(years=k * test_years)
        test_begin = first + pd.DateOffset(years=train_years + k * test_years)
        if test_begin > last:
            break
        test_stop = min(test_begin + pd.DateOffset(years=test_years) - pd.Timedelta(days=1), last)
        spans.append(
            WindowSpan(
                index=k + 1,
                train_start=train_begin.date(),
                train_end=(test_begin - pd.Timedelta(days=1)).date(),
                test_start=test_begin.date(),
                test_end=test_stop.date(),
            )
        )
        k += 1
    if not spans:
        raise OptimizationError(
            f"Zeitraum {first.date()} bis {last.date()} ist kürzer als {train_years} Jahr(e) "
            "Training plus Test."
        )
    return spans


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------


def walk_forward(
    strategy: Strategy | type[Strategy],
    grid: Mapping[str, Sequence[Any]],
    symbol: str | None,
    timeframe: Timeframe,
    start: DateInput,
    end: DateInput,
    settings: Settings,
    *,
    train_years: int = 5,
    test_years: int = 1,
    anchored: bool = False,
    objective: Objective | str = Objective.SHARPE,
    universe: Universe | None = None,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    bars: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
    costs: CostModel | None = None,
    workers: int | None = None,
    progress: ProgressCallback | None = None,
) -> WalkForwardResult:
    """Rollierende Walk-Forward-Analyse.

    Args:
        strategy: Strategie (Instanz oder Klasse).
        grid: Parameter-Raster (z. B. aus :func:`~trading_helfer.backtest.optimize.build_grid`).
        symbol: Wertpapier; ``None`` mit ``universe``.
        timeframe: Zeitrahmen.
        start: Beginn des ersten Trainings.
        end: Ende des letzten Tests.
        settings: Einstellungen (Kapital, Kosten, risikofreier Zins, Datenbank).
        train_years: Trainingsdauer in Jahren.
        test_years: Testdauer (und Schritt) in Jahren.
        anchored: Training wächst ab ``start`` statt zu rollieren.
        objective: Zielgröße der Optimierung.
        universe: Universum für Portfolio-Backtests statt ``symbol``.
        max_positions: Maximale Anzahl Positionen im Portfolio-Backtest.
        bars: Bereinigte Tageskurse (Einzelwert: DataFrame; Universum: je Symbol).
        costs: Kostenmodell statt der Kosten aus ``settings``.
        workers: Prozesse bzw. Threads je Optimierung; ``None`` = CPU-Kerne.
        progress: Fortschritt ``(fertig, gesamt)`` über alle Fenster und Kombinationen.

    Returns:
        Fenster mit Parametern, zusammengesetzte Out-of-Sample-Equity und Trades.

    Raises:
        OptimizationError: Ungültige Fenster, Raster oder Angaben.
        BacktestError: Keine Kurse.
    """
    template = strategy() if isinstance(strategy, type) else strategy
    goal = Objective(objective)
    full_grid = _complete_grid(template, grid)
    spans = walk_forward_windows(
        start, end, train_years=train_years, test_years=test_years, anchored=anchored
    )
    base = _make_context(
        template,
        symbol,
        universe,
        timeframe,
        spans[0].train_start,
        spans[-1].test_end,
        settings,
        bars,
        costs,
        max_positions,
    )
    combos = count_combinations(full_grid)
    total = len(spans) * (combos + 1)
    log: list[str] = []
    windows: list[WalkForwardWindow] = []
    for n, span in enumerate(spans):
        offset = n * (combos + 1)

        def report(done: int, _total: int, offset: int = offset) -> None:
            if progress is not None:
                progress(offset + done, total)

        window = _run_window(base, span, full_grid, goal, workers, report, log)
        windows.append(window)
        if progress is not None:
            progress(offset + combos + 1, total)
    equity, trades = _compose(windows, settings.default_capital)
    return WalkForwardResult(
        strategy_id=template.id,
        objective=goal,
        timeframe=base.timeframe,
        train_years=train_years,
        test_years=test_years,
        anchored=anchored,
        initial_capital=settings.default_capital,
        windows=windows,
        equity=equity,
        trades=trades,
        log=log,
    )


def _run_window(
    base: _Context,
    span: WindowSpan,
    grid: Mapping[str, Sequence[Any]],
    goal: Objective,
    workers: int | None,
    progress: ProgressCallback,
    log: list[str],
) -> WalkForwardWindow:
    train_bars = _slice_bars(base.bars, span.train_end)
    opt = optimize(
        base.strategy,
        grid,
        base.symbol,
        base.timeframe,
        span.train_start,
        span.train_end,
        base.settings,
        objective=goal,
        universe=base.universe,
        max_positions=base.max_positions,
        bars=train_bars,
        costs=base.costs,
        workers=workers,
        progress=progress,
    )
    params = opt.best_params
    if params is None:
        params = base.strategy.make_params(None).to_dict()
        log.append(
            f"Fenster {span.index}: keine bewertbare Kombination im Training; "
            "Standardparameter verwendet."
        )
    test_ctx = replace(
        base, start=span.test_start, end=span.test_end, bars=_slice_bars(base.bars, span.test_end)
    )
    try:
        result = _run(test_ctx, params)
    except BacktestError as exc:
        raise BacktestError(f"Fenster {span.index} (Test): {exc}") from exc
    test_value = (
        objective_value(result, goal, risk_free_rate_pct=base.settings.risk_free_rate_pct)
        if len(result.equity)
        else None
    )
    return WalkForwardWindow(
        span=span,
        params=params,
        train_value=opt.best_value,
        test_value=test_value,
        robustness=opt.best_robustness,
        optimization=opt,
        test_result=result,
    )


def _slice_bars(bars: Any, last: Date) -> Any:
    """Kurse bis einschließlich ``last``; nichts danach erreicht Training oder Test."""
    if isinstance(bars, pd.DataFrame):
        return _slice_frame(bars, last)
    return {s: _slice_frame(f, last) for s, f in bars.items()}


def _slice_frame(frame: pd.DataFrame, last: Date) -> pd.DataFrame:
    dates = pd.to_datetime(frame[DATE_COLUMN])
    return frame.loc[(dates <= pd.Timestamp(last)).to_numpy()].reset_index(drop=True)


def _compose(
    windows: Sequence[WalkForwardWindow], initial_capital: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verkettet die Test-Equity-Kurven und sammelt die Trades."""
    parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    level = initial_capital
    for w in windows:
        result = w.test_result
        if len(result.trades):
            trades = result.trades.copy()
            trades[WINDOW_COLUMN] = w.span.index
            trade_parts.append(trades)
        if len(result.equity) == 0:
            continue
        factor = level / result.initial_capital
        part = result.equity.loc[:, list(EQUITY_COLUMNS)].copy()
        part["equity"] = part["equity"] * factor
        part["cash"] = part["cash"] * factor
        parts.append(part)
        level = float(part["equity"].iloc[-1])
    if parts:
        equity = pd.concat(parts, ignore_index=True)
        values = equity["equity"].to_numpy(dtype=float)
        peak = np.maximum.accumulate(values)
        equity["drawdown"] = values / peak - 1.0
    else:
        equity = pd.DataFrame(columns=list(EQUITY_COLUMNS))
    if trade_parts:
        trades = pd.concat(trade_parts, ignore_index=True)
    else:
        trades = pd.DataFrame(columns=[*TRADE_COLUMNS, WINDOW_COLUMN])
    return equity, trades
