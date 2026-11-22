"""Optimierungs-Tab (G-07): Eingaben, Lauf, Tabelle, Heatmap, Walk-Forward, Monte Carlo, Abbruch."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QSpinBox
from pytestqt.qtbot import QtBot

from trading_helfer.backtest.monte_carlo import MonteCarloMethod
from trading_helfer.backtest.optimize import (
    METRIC_COLUMNS,
    Objective,
    OptimizationResult,
    ParamRange,
)
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.gui import texts
from trading_helfer.gui.main_window import MainWindow
from trading_helfer.gui.widgets.optimization import (
    CANCELLED,
    OptimizationWidget,
    format_param,
    format_params,
    histogram_bins,
)
from trading_helfer.services.app_service import AppService
from trading_helfer.services.backtest_service import BacktestService, UniverseKind
from trading_helfer.services.optimization_service import (
    OptimizationMode,
    OptimizationOutcome,
    OptimizationRequest,
    OptimizationService,
)
from trading_helfer.strategies.base import BarContext, Order, Param, Params, Sizing, Strategy
from trading_helfer.strategies.registry import StrategyInfo, StrategyRegistry

TODAY = dt.date(2019, 12, 31)


def _bars(close: np.ndarray, start: str = "2014-01-02") -> pd.DataFrame:
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(close)),
            "open": opens,
            "high": np.maximum(opens, close) * 1.01,
            "low": np.minimum(opens, close) * 0.99,
            "close": close,
            "volume": 1000,
        }
    )


def _sawtooth(cycles: int = 110, half: int = 7) -> np.ndarray:
    up = np.linspace(100.0, 120.0, half, endpoint=False)
    down = np.linspace(120.0, 100.0, half, endpoint=False)
    return np.tile(np.concatenate([up, down]), cycles)


BARS = {"AAA": _bars(_sawtooth()), "MSFT": _bars(_sawtooth() * 2)}


class Cross(Strategy):
    id = "test_cross"
    name = "Test-Kreuzung"
    version = "1.0.0"
    description = "SMA-Kreuzung für GUI-Tests."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        "fast": Param(2, 1, 20, 1, "schnelle SMA"),
        "slow": Param(5, 2, 50, 1, "langsame SMA"),
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        close = bars["close"]
        return pd.DataFrame(
            {"f": close.rolling(p.fast).mean(), "s": close.rolling(p.slow).mean()},
            index=bars.index,
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and ctx.crossed_above("f", "s"):
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        if ctx.position is not None and ctx.crossed_below("f", "s"):
            return ctx.exit()
        return []


class Mixed(Cross):
    id = "test_mixed"
    name = "Gemischte Parameter"
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1, Timeframe.D3]
    params: ClassVar[dict[str, Param]] = {
        "fast": Param(3, 1, 20, 1, "schnell"),
        "slow": Param(8, 2, 50, 1, "langsam"),
        "factor": Param(1.5, 0.5, 5.0, 0.25, "Faktor"),
        "short": Param(False, description="Short erlauben"),
        "mode": Param("close", description="Kursart", choices=("close", "open")),
    }


class NoParams(Cross):
    id = "test_none"
    name = "Ohne Parameter"
    params: ClassVar[dict[str, Param]] = {}

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        return Cross.indicators(self, bars, Cross.make_params({}))


def _registry(*classes: type[Strategy]) -> StrategyRegistry:
    registry = StrategyRegistry(None, include_builtin=False)
    for cls in classes:
        registry._strategies[cls.id] = StrategyInfo(cls.id, cls, "test")
    return registry


def _loader(symbol: str) -> pd.DataFrame:
    return BARS.get(symbol, pd.DataFrame(columns=["date", "open", "high", "low", "close"]))


def _backtest_service(tmp_path: Path, *classes: type[Strategy]) -> BacktestService:
    settings = Settings(data_dir=tmp_path / "data", start_date=dt.date(2014, 1, 1))
    return BacktestService(
        settings,
        _registry(*(classes or (Cross, Mixed, NoParams))),
        bars_loader=_loader,
        symbols_loader=lambda: sorted(BARS),
    )


class _SingleWorker(OptimizationService):
    """Rechnet ohne Prozesse, damit die Tests schnell und deterministisch bleiben."""

    def run(self, request: OptimizationRequest, *args: Any, **kwargs: Any) -> OptimizationOutcome:
        request = OptimizationRequest(**{**request.__dict__, "workers": 1})
        return super().run(request, *args, **kwargs)


@pytest.fixture
def opt_service(tmp_path: Path) -> OptimizationService:
    return _SingleWorker(_backtest_service(tmp_path))


@pytest.fixture
def widget(qtbot: QtBot, opt_service: OptimizationService) -> OptimizationWidget:
    w = OptimizationWidget(opt_service, today=lambda: TODAY)
    qtbot.addWidget(w)
    return w


def _prepare(widget: OptimizationWidget) -> None:
    widget.set_symbol("AAA")
    widget.timeframe_combo.setCurrentIndex(widget.timeframe_combo.findData("1D"))
    widget.set_period(dt.date(2014, 1, 1), dt.date(2016, 12, 31))
    widget.set_range("fast", 2, 4, 1)
    widget.set_range("slow", 5, 11, 3)
    widget.mc_spin.setValue(300)


# -- Formular und Eingaben ---------------------------------------------------------------


def test_form_defaults(widget: OptimizationWidget) -> None:
    assert widget.strategy_id == "test_cross"
    assert widget.description_label.text() == "SMA-Kreuzung für GUI-Tests."
    assert widget.start_edit.date().year() == 2014
    assert widget.end_edit.date().toPython() == TODAY
    assert widget.ranges() == {}
    assert widget.combinations_label.text() == texts.OPT_COMBINATIONS.format(count="1")
    assert not widget.train_spin.isEnabled()
    assert widget.run_button.isEnabled()
    assert not widget.cancel_button.isEnabled()
    assert not widget.apply_button.isEnabled()
    assert widget.status_label.text() == texts.OPT_IDLE


def test_range_rows_follow_param_types(widget: OptimizationWidget) -> None:
    widget.select_strategy("test_mixed")
    assert [widget.timeframe_combo.itemData(i) for i in range(2)] == ["1W", "3D"]
    widget.set_range("factor", 0.5, 1.5, 0.25)
    widget.set_range("short")
    widget.set_range("mode")
    ranges = widget.ranges()
    assert ranges == {"factor": ParamRange(0.5, 1.5, 0.25), "short": None, "mode": None}
    assert widget.combinations_label.text() == texts.OPT_COMBINATIONS.format(count="20")
    widget.set_range("short", vary=False)
    assert "short" not in widget.ranges()
    with pytest.raises(KeyError):
        widget.set_range("gibt_es_nicht")


def test_combination_count_warns_and_reports_errors(widget: OptimizationWidget) -> None:
    widget.set_range("fast", 1, 20, 1)
    widget.set_range("slow", 2, 50, 1)
    assert widget.combinations_label.text() == texts.OPT_COMBINATIONS.format(count="980")
    widget.set_range("slow", 2, 50, 0.001 + 0.999)  # bleibt gültig
    widget.select_strategy("test_mixed")
    widget.set_range("factor", 0.5, 5.0, 0.0001)
    widget.set_range("fast", 1, 20, 1)
    widget.set_range("slow", 2, 50, 1)
    assert "lange dauern" in widget.combinations_label.text()
    widget.set_range("fast", 10, 2, 1)
    assert widget.combinations_label.text().startswith("Raster ungültig")


def test_strategy_without_params(widget: OptimizationWidget) -> None:
    widget.select_strategy("test_none")
    assert widget.ranges() == {}
    assert widget.combinations_label.text() == texts.OPT_COMBINATIONS.format(count="1")


def test_request_passes_all_inputs(widget: OptimizationWidget) -> None:
    _prepare(widget)
    widget.set_objective(Objective.CALMAR)
    widget.capital_spin.setValue(2500)
    widget.fee_spin.setValue(2)
    widget.fee_pct_spin.setValue(0.1)
    widget.slippage_spin.setValue(0.2)
    widget.wf_check.setChecked(True)
    assert widget.train_spin.isEnabled()
    widget.train_spin.setValue(3)
    widget.test_spin.setValue(2)
    widget.anchored_check.setChecked(True)
    widget.mc_method_combo.setCurrentIndex(1)
    widget.seed_spin.setValue(7)
    req = widget.request()
    assert req.strategy_id == "test_cross"
    assert req.ranges == {"fast": ParamRange(2, 4, 1), "slow": ParamRange(5, 11, 3)}
    assert req.universe == "AAA"
    assert req.universe_kind is UniverseKind.SYMBOL
    assert req.timeframe is Timeframe.D1
    assert (req.start, req.end) == (dt.date(2014, 1, 1), dt.date(2016, 12, 31))
    assert req.objective is Objective.CALMAR
    assert req.mode is OptimizationMode.WALK_FORWARD
    assert (req.train_years, req.test_years, req.anchored) == (3, 2, True)
    assert (req.capital, req.fee_per_order, req.fee_pct, req.slippage_pct) == (
        2500.0,
        2.0,
        0.1,
        0.2,
    )
    assert req.mc_simulations == 300
    assert req.mc_method is MonteCarloMethod.BOOTSTRAP
    assert req.seed == 7

    widget.set_index("NDX100")
    widget.max_positions_spin.setValue(4)
    req = widget.request()
    assert (req.universe, req.universe_kind, req.max_positions) == (
        "NDX100",
        UniverseKind.INDEX,
        4,
    )
    assert widget.symbol_edit.isHidden()
    with pytest.raises(KeyError):
        widget.set_index("DAX")


def test_start_hands_request_to_runner(opt_service: OptimizationService, qtbot: QtBot) -> None:
    calls: list[dict[str, Any]] = []

    def runner(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        calls.append({"fn": fn, "args": args, **kwargs})

    widget = OptimizationWidget(opt_service, runner=runner, today=lambda: TODAY)
    qtbot.addWidget(widget)
    _prepare(widget)
    widget.start()
    assert len(calls) == 1
    request = calls[0]["args"][0]
    assert request == widget.request()
    assert calls[0]["report_progress"] is True
    assert widget.is_running
    assert not widget.run_button.isEnabled()
    assert widget.cancel_button.isEnabled()
    widget.start()  # zweiter Start wird ignoriert
    assert len(calls) == 1


# -- Lauf und Ergebnis -------------------------------------------------------------------


def test_grid_run_shows_results(widget: OptimizationWidget, qtbot: QtBot) -> None:
    _prepare(widget)
    with qtbot.waitSignal(widget.finished) as blocker:
        widget.start()
    outcome = blocker.args[0]
    assert isinstance(outcome, OptimizationOutcome)
    assert widget.outcome is outcome
    assert widget.progress_bar.value() == 100
    assert not widget.is_running
    result = outcome.optimization
    assert widget.results_table.rowCount() == 9
    assert widget.results_table.columnCount() == 2 + 1 + len(METRIC_COLUMNS) + 2
    assert widget.best_label.text().startswith("Beste Kombination: ")
    assert format_params(result.best_params) in widget.best_label.text()
    assert widget.apply_button.isEnabled()
    assert "9 Kombination(en)" in widget.result_label.text()
    # Monte-Carlo-Histogramm
    hist = widget.histogram()
    assert hist is not None
    counts, edges = hist
    assert int(counts.sum()) == 300
    assert len(edges) == len(counts) + 1
    assert widget.mc_line.isVisible()
    assert "300 Simulationen" in widget.mc_label.text()
    # Umordnen: Standard ist der Drawdown, Endkapital wird zu einer Klasse
    assert widget.mc_value_combo.currentData() == "drawdown"
    assert "Drawdown" in widget.mc_plot.titleLabel.text
    widget.set_histogram_value("equity")
    counts, edges = widget.histogram()  # type: ignore[misc]
    assert int(counts.sum()) == 300
    assert "Endkapital" in widget.mc_plot.titleLabel.text
    with pytest.raises(KeyError):
        widget.set_histogram_value("gibt_es_nicht")
    # Walk-Forward-Ansicht bleibt leer
    assert widget.wf_label.text() == texts.OPT_WF_EMPTY
    assert widget.walk_forward_points() == 0


def test_result_table_is_sortable(widget: OptimizationWidget, qtbot: QtBot) -> None:
    _prepare(widget)
    with qtbot.waitSignal(widget.finished):
        widget.start()
    table = widget.outcome.optimization.table  # type: ignore[union-attr]
    widget.sort_by("objective", descending=True)
    best = table["objective"].max()
    first = widget.table_column("objective")[0]
    assert first == format_param(0) or first.replace(".", "").replace(",", ".") != ""
    shown = widget.table_column("objective")
    expected = (
        table.sort_values("objective", ascending=False)["objective"]
        .map(lambda v: f"{v:,.2f}".replace(",", "_").replace(".", ",").replace("_", "."))
        .tolist()
    )
    assert shown == expected
    assert shown[0] == f"{best:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    widget.sort_by("fast", descending=False)
    assert widget.table_column("fast") == sorted(widget.table_column("fast"), key=int)
    widget.sort_by("error", descending=False)


def test_walk_forward_run(widget: OptimizationWidget, qtbot: QtBot) -> None:
    _prepare(widget)
    widget.set_period(dt.date(2014, 1, 1), dt.date(2019, 12, 31))
    widget.wf_check.setChecked(True)
    widget.train_spin.setValue(2)
    widget.test_spin.setValue(1)
    with qtbot.waitSignal(widget.finished) as blocker:
        widget.start()
    wf = blocker.args[0].walk_forward
    assert wf is not None
    assert widget.wf_table.rowCount() == len(wf.windows) == 4
    assert widget.wf_table.columnCount() == 3 + 2 + 3
    assert widget.wf_table.item(0, 1).text() == "01.01.2014 – 31.12.2015"
    assert widget.walk_forward_points() == len(wf.equity)
    assert widget.wf_label.text().startswith("Endkapital ")
    assert "Walk-Forward mit 4 Fenster(n)" in widget.result_label.text()
    assert widget.histogram() is not None


def test_failure_is_shown(widget: OptimizationWidget, qtbot: QtBot) -> None:
    widget.set_symbol("ZZZ")
    with qtbot.waitSignal(widget.failed) as blocker:
        widget.start()
    assert "keine Kurse" in blocker.args[0]
    assert widget.status_label.text().startswith("Optimierung fehlgeschlagen")
    assert widget.run_button.isEnabled()
    assert not widget.cancel_button.isEnabled()


def test_unknown_strategy(
    widget: OptimizationWidget, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    reloads: list[bool] = []
    monkeypatch.setattr(widget._service, "reload_strategies", lambda: reloads.append(True))
    with qtbot.waitSignal(widget.failed):
        assert not widget.select_strategy("gibt_es_nicht")
    assert reloads == [True]
    assert widget.select_strategy("test_mixed")


def test_apply_best_emits_request_and_params(widget: OptimizationWidget, qtbot: QtBot) -> None:
    widget.apply_best()  # ohne Ergebnis passiert nichts
    _prepare(widget)
    with qtbot.waitSignal(widget.finished) as done:
        widget.start()
    with qtbot.waitSignal(widget.apply_requested) as blocker:
        widget.apply_button.click()
    request, params = blocker.args
    assert request is done.args[0].request
    assert params == done.args[0].optimization.best_params


# -- Heatmap -----------------------------------------------------------------------------


def _fake_outcome() -> OptimizationOutcome:
    grid = {"fast": [1, 2, 3], "slow": [10, 20], "factor": [0.5, 1.0]}
    rows = []
    for f in grid["fast"]:
        for s in grid["slow"]:
            for k in grid["factor"]:
                rows.append({"fast": f, "slow": s, "factor": k})
    table = pd.DataFrame(rows)
    # Zielwert: fast − slow/10 + factor; bestes: fast=3, slow=10, factor=1.0 → 3.0
    table["objective"] = table["fast"] - table["slow"] / 10 + table["factor"]
    table.loc[(table["fast"] == 1) & (table["slow"] == 20), "objective"] = np.nan
    for column in METRIC_COLUMNS:
        table[column] = 1.0
    table["robustness"] = 0.5
    table["error"] = ""
    best = int(table["objective"].idxmax())
    result = OptimizationResult("test_cross", Objective.SHARPE, grid, table, best)
    request = OptimizationRequest(
        "test_cross", {}, "AAA", Timeframe.W1, dt.date(2000, 1, 1), dt.date(2010, 1, 1)
    )
    return OptimizationOutcome(request=request, optimization=result, log=["Hinweis"])


def test_heatmap_with_fake_data(widget: OptimizationWidget) -> None:
    widget.set_outcome(_fake_outcome())
    assert [widget.heat_x_combo.itemData(i) for i in range(3)] == ["fast", "slow", "factor"]
    matrix = widget.heatmap_matrix()
    assert matrix is not None
    assert list(matrix.columns) == [1, 2, 3]
    assert list(matrix.index) == [10, 20]
    table = widget.heat_table
    assert (table.rowCount(), table.columnCount()) == (2, 3)
    assert table.horizontalHeaderItem(0).text() == "fast=1"
    assert table.verticalHeaderItem(1).text() == "slow=20"
    # factor fest auf der besten Kombination (1.0)
    assert widget.heatmap_cell(0, 2).text() == "3,00"  # fast=3, slow=10
    assert widget.heatmap_cell(1, 0).text() == "–"  # NaN
    strong = widget.heatmap_cell(0, 2).background().color()
    weak = widget.heatmap_cell(0, 0).background().color()  # 1 − 1 + 1 = 1
    assert strong.alpha() > weak.alpha()
    assert widget.heat_label.text() == texts.OPT_HEATMAP_FIXED
    # andere Achsen
    widget.set_heatmap_axes("factor", "fast")
    matrix = widget.heatmap_matrix()
    assert matrix is not None
    assert list(matrix.columns) == [0.5, 1.0]
    assert widget.heat_table.horizontalHeaderItem(0).text() == "factor=0,5"
    assert widget.heatmap_cell(2, 1).text() == "3,00"
    with pytest.raises(KeyError):
        widget.set_heatmap_axes("gibt_es_nicht", "fast")
    # gleiche Achse: keine Heatmap
    widget.heat_y_combo.setCurrentIndex(widget.heat_y_combo.findData("factor"))
    assert widget.heatmap_matrix() is None
    assert widget.heat_table.rowCount() == 0
    assert widget.heat_label.text() == texts.OPT_HEATMAP_NEED_TWO
    assert widget.log_view.toPlainText() == "Hinweis"
    assert widget.histogram() is None
    assert widget.mc_label.text() == texts.OPT_MC_EMPTY


def test_heat_color() -> None:
    green = OptimizationWidget.heat_color(2.0, 2.0)
    red = OptimizationWidget.heat_color(-1.0, 2.0)
    flat = OptimizationWidget.heat_color(1.0, 0.0)
    assert green.green() > green.red()
    assert red.red() > red.green()
    assert green.alpha() > red.alpha() > flat.alpha()


def test_outcome_without_best(widget: OptimizationWidget) -> None:
    outcome = _fake_outcome()
    result = outcome.optimization
    table = result.table.assign(objective=np.nan)
    empty = OptimizationResult(result.strategy_id, result.objective, result.grid, table, None)
    widget.set_outcome(OptimizationOutcome(outcome.request, empty))
    assert widget.best_label.text() == texts.OPT_BEST_NONE
    assert not widget.apply_button.isEnabled()
    assert widget.table_column("objective")[0] == "–"


def test_histogram_bins() -> None:
    counts, edges = histogram_bins(np.array([1000.0, 1000.0 + 1e-10, 1000.0]))
    assert counts.tolist() == [3]
    assert edges[0] < 1000.0 < edges[1]
    counts, edges = histogram_bins(np.linspace(0.0, 1.0, 100), bins=10)
    assert len(counts) == 10
    assert int(counts.sum()) == 100


def test_bootstrap_shows_equity_histogram(widget: OptimizationWidget, qtbot: QtBot) -> None:
    _prepare(widget)
    widget.mc_method_combo.setCurrentIndex(1)
    with qtbot.waitSignal(widget.finished):
        widget.start()
    assert widget.mc_value_combo.currentData() == "equity"
    counts, _edges = widget.histogram()  # type: ignore[misc]
    assert int(counts.sum()) == 300


def test_format_helpers() -> None:
    assert format_param(True) == "ja"
    assert format_param(False) == "nein"
    assert format_param(np.int64(5)) == "5"
    assert format_param(2.0) == "2"
    assert format_param(0.25) == "0,25"
    assert format_param("close") == "close"
    assert format_params({"a": 1, "b": 0.5}) == "a=1, b=0,5"


# -- Abbruch -----------------------------------------------------------------------------


class _DeferredRunner:
    """Merkt sich die Aufgabe; ``finish`` führt sie später aus (wie ein Hintergrundlauf)."""

    def __init__(self) -> None:
        self.task: tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]] | None = None

    def __call__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self.task = (fn, args, kwargs)

    def finish(self) -> None:
        assert self.task is not None
        fn, args, kwargs = self.task
        progress = kwargs.get("on_progress") or (lambda _p, _m: None)
        result = fn(*args, progress=progress)
        kwargs["on_result"](result)


def test_cancel_before_work_starts(opt_service: OptimizationService, qtbot: QtBot) -> None:
    runner = _DeferredRunner()
    widget = OptimizationWidget(opt_service, runner=runner, today=lambda: TODAY)
    qtbot.addWidget(widget)
    _prepare(widget)
    widget.cancel()  # ohne Lauf: nichts
    widget.start()
    widget.cancel()
    assert widget.status_label.text() == texts.OPT_CANCELLING
    assert not widget.cancel_button.isEnabled()
    widget.cancel()  # zweites Mal: nichts
    with qtbot.waitSignal(widget.cancelled):
        runner.finish()
    assert widget.status_label.text() == texts.OPT_CANCELLED
    assert widget.outcome is None
    assert not widget.is_running
    assert widget.run_button.isEnabled()
    assert widget.progress_bar.value() == 0


def test_cancel_during_run(opt_service: OptimizationService, qtbot: QtBot) -> None:
    runner = _DeferredRunner()
    widget = OptimizationWidget(opt_service, runner=runner, today=lambda: TODAY)
    qtbot.addWidget(widget)
    _prepare(widget)
    widget.set_range("fast", 1, 10, 1)
    seen: list[str] = []

    def on_progress(percent: int, message: str) -> None:
        seen.append(message)
        if "/" in message:
            widget.cancel()  # Klick auf „Abbrechen“ während des Laufs

    widget._on_progress = on_progress  # type: ignore[method-assign]
    widget.start()
    assert runner.task is not None
    fn, args, kwargs = runner.task
    kwargs["on_progress"] = on_progress
    with qtbot.waitSignal(widget.cancelled):
        kwargs["on_result"](fn(*args, progress=on_progress))
    assert sum("/" in m for m in seen) == 1
    assert widget.status_label.text() == texts.OPT_CANCELLED


def test_run_returns_cancelled_marker(widget: OptimizationWidget) -> None:
    import threading

    event = threading.Event()
    event.set()
    _prepare(widget)
    assert widget._run(widget.request(), event) == CANCELLED


# -- Hauptfenster ------------------------------------------------------------------------


@pytest.fixture
def main_window(
    qtbot: QtBot, tmp_path: Path, state_path: Path, dialogs: list[tuple[str, str]]
) -> MainWindow:
    backtest = _backtest_service(tmp_path)
    win = MainWindow(
        AppService(backtest.settings, state_path),
        backtest_service=backtest,
        optimization_service=_SingleWorker(backtest),
    )
    win.optimization._runner = lambda fn, *a, on_result=None, **k: on_result(  # type: ignore[assignment,misc]
        fn(*a, progress=lambda _p, _m: None)
    )
    qtbot.addWidget(win)
    return win


def test_tab_is_in_main_window(window: MainWindow) -> None:
    assert isinstance(window.tab_widget("optimization"), OptimizationWidget)
    assert window.optimization.scroll_area.widgetResizable()


def test_apply_to_backtest(main_window: MainWindow, qtbot: QtBot) -> None:
    opt = main_window.optimization
    _prepare(opt)
    with qtbot.waitSignal(opt.finished) as blocker:
        opt.start()
    best = blocker.args[0].optimization.best_params
    opt.apply_best()
    backtest = main_window.backtest
    assert main_window.tabs.currentWidget() is backtest
    assert backtest.strategy_id == "test_cross"
    assert backtest.param_values() == best
    req = backtest.request()
    assert req.universe == "AAA"
    assert req.timeframe is Timeframe.D1
    assert (req.start, req.end) == (dt.date(2014, 1, 1), dt.date(2016, 12, 31))
    assert main_window.statusBar().currentMessage() == texts.OPT_STATUS_APPLIED


def test_apply_index_and_unknown_strategy(main_window: MainWindow) -> None:
    request = OptimizationRequest(
        "test_mixed",
        {},
        "SP500",
        Timeframe.D1,  # Mixed kennt 1D nicht: bleibt beim vorhandenen Zeitrahmen
        dt.date(2015, 1, 1),
        dt.date(2018, 1, 1),
        universe_kind=UniverseKind.INDEX,
    )
    main_window.apply_optimization(request, {"fast": 4})
    backtest = main_window.backtest
    assert backtest.strategy_id == "test_mixed"
    assert backtest.request().universe == "SP500"
    assert backtest.param_values()["fast"] == 4
    main_window.backtest_service.reload_strategies = lambda: None  # type: ignore[method-assign]
    other = OptimizationRequest(
        "gibt_es_nicht", {}, "AAA", Timeframe.W1, dt.date(2015, 1, 1), dt.date(2018, 1, 1)
    )
    main_window.apply_optimization(other, {})
    assert "gibt_es_nicht" in backtest.status_label.text()


def test_watchlist_symbol_goes_to_optimization(main_window: MainWindow) -> None:
    main_window.tabs.setCurrentWidget(main_window.optimization)
    main_window._on_symbol_selected("MSFT")
    assert main_window.tabs.currentWidget() is main_window.optimization
    assert main_window.optimization.symbol_edit.text() == "MSFT"


def test_spin_types(widget: OptimizationWidget) -> None:
    assert isinstance(widget.train_spin, QSpinBox)
