"""Backtest-Tab (G-06): Formular, Lauf, Ergebnisanzeige, Trades → Chart, Läufe, Export."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QSpinBox
from pytestqt.qtbot import QtBot

from trading_helfer.backtest.engine import (
    EQUITY_COLUMNS,
    ORDER_COLUMNS,
    BacktestResult,
)
from trading_helfer.backtest.metrics import compute_metrics, monthly_returns
from trading_helfer.core.config import Settings
from trading_helfer.core.markers import MarkerKind
from trading_helfer.core.types import Timeframe
from trading_helfer.gui import texts
from trading_helfer.gui.main_window import MainWindow
from trading_helfer.gui.widgets.backtest import TRADE_TABLE_COLUMNS, BacktestWidget
from trading_helfer.services.app_service import AppService
from trading_helfer.services.backtest_service import (
    BacktestOutcome,
    BacktestService,
    UniverseKind,
)
from trading_helfer.services.chart_service import ChartService
from trading_helfer.strategies.base import BarContext, Order, Param, Params, Sizing, Strategy
from trading_helfer.strategies.registry import StrategyInfo, StrategyRegistry

TODAY = dt.date(2024, 6, 30)


def _bars(close: np.ndarray, start: str = "2020-01-02") -> pd.DataFrame:
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


def _sawtooth(cycles: int = 10, half: int = 7) -> np.ndarray:
    up = np.linspace(100.0, 120.0, half, endpoint=False)
    down = np.linspace(120.0, 100.0, half, endpoint=False)
    return np.tile(np.concatenate([up, down]), cycles)


BARS = {
    "AAA": _bars(_sawtooth()),
    "MSFT": _bars(_sawtooth() * 3),
    "SPY": _bars(np.linspace(100.0, 140.0, len(_sawtooth()))),
}


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
    return BARS.get(symbol, BARS["AAA"].iloc[:0])


@pytest.fixture
def bt_service(tmp_path: Path) -> BacktestService:
    return BacktestService(
        Settings(data_dir=tmp_path / "data"),
        _registry(Cross, Mixed, NoParams),
        bars_loader=_loader,
        symbols_loader=lambda: sorted(BARS),
    )


@pytest.fixture
def widget(qtbot: QtBot, bt_service: BacktestService) -> BacktestWidget:
    w = BacktestWidget(bt_service, today=lambda: TODAY)
    qtbot.addWidget(w)
    return w


def _prepare(w: BacktestWidget, symbol: str = "AAA") -> None:
    w.set_strategy("test_cross")
    w.set_symbol(symbol)
    w.set_timeframe(Timeframe.D1)
    w.set_period(dt.date(2020, 1, 1), dt.date(2021, 1, 1))


# -- Formular ------------------------------------------------------------------------


def test_form_defaults(widget: BacktestWidget, bt_service: BacktestService) -> None:
    assert widget.strategy_id == "test_cross"
    assert widget.description_label.text() == Cross.description
    assert widget.param_values() == {"fast": 2, "slow": 5}
    req = widget.request()
    assert req.universe_kind is UniverseKind.SYMBOL
    assert req.timeframe is Timeframe.D1
    assert req.end == TODAY
    assert req.start == bt_service.settings.start_date
    assert req.capital == bt_service.settings.default_capital
    assert req.fee_per_order == bt_service.settings.fee_per_order
    assert req.slippage_pct == pytest.approx(bt_service.settings.slippage_pct)
    assert req.benchmark == "SPY"
    assert req.save


def test_form_builds_request(widget: BacktestWidget) -> None:
    widget.set_strategy("test_mixed")
    assert [widget.timeframe_combo.itemText(i) for i in range(2)] == ["1W", "3D"]
    assert isinstance(widget.param_widget("fast"), QSpinBox)
    assert isinstance(widget.param_widget("factor"), QDoubleSpinBox)
    assert isinstance(widget.param_widget("short"), QCheckBox)
    assert isinstance(widget.param_widget("mode"), QComboBox)
    assert widget.param_widget("fast").toolTip() == "schnell"
    spin = widget.param_widget("slow")
    assert isinstance(spin, QSpinBox)
    assert (spin.minimum(), spin.maximum()) == (2, 50)

    widget.set_param("fast", 4)
    widget.set_param("factor", 2.25)
    widget.set_param("short", True)
    widget.set_param("mode", "open")
    widget.set_index("NDX100")
    widget.set_timeframe(Timeframe.D3)
    widget.set_period(dt.date(2001, 2, 3), dt.date(2010, 4, 5))
    widget.capital_spin.setValue(5000.0)
    widget.fee_spin.setValue(2.5)
    widget.fee_pct_spin.setValue(0.1)
    widget.slippage_spin.setValue(0.2)
    widget.benchmark_edit.setText(" qqq ")
    widget.save_checkbox.setChecked(False)

    req = widget.request()
    assert req.strategy_id == "test_mixed"
    assert req.params == {"fast": 4, "slow": 8, "factor": 2.25, "short": True, "mode": "open"}
    assert req.universe_kind is UniverseKind.INDEX
    assert req.universe == "NDX100"
    assert req.timeframe is Timeframe.D3
    assert (req.start, req.end) == (dt.date(2001, 2, 3), dt.date(2010, 4, 5))
    assert req.capital == 5000.0
    assert req.fee_per_order == 2.5
    assert req.fee_pct == pytest.approx(0.1)
    assert req.slippage_pct == pytest.approx(0.2)
    assert req.benchmark == "QQQ"
    assert not req.save
    assert req.costs().fee_per_order == 2.5

    widget.set_symbol(" msft")
    assert widget.request().universe == "MSFT"
    assert widget.request().universe_kind is UniverseKind.SYMBOL


def test_form_switches_strategy(widget: BacktestWidget) -> None:
    widget.set_strategy("test_mixed")
    widget.set_strategy("test_none")
    assert widget.param_values() == {}
    assert widget.params_form.rowCount() == 1  # Hinweis „keine Parameter“
    with pytest.raises(KeyError):
        widget.set_strategy("unbekannt")
    with pytest.raises(KeyError):
        widget.set_index("DAX")
    widget.set_strategy("test_mixed")
    with pytest.raises(KeyError):
        widget.set_timeframe(Timeframe.D1)


def test_no_strategies(qtbot: QtBot, tmp_path: Path) -> None:
    svc = BacktestService(Settings(data_dir=tmp_path), _registry(), bars_loader=_loader)
    w = BacktestWidget(svc)
    qtbot.addWidget(w)
    assert not w.run_button.isEnabled()
    assert w.status_label.text() == texts.BACKTEST_NO_STRATEGY
    with qtbot.waitSignal(w.failed):
        w.start()


# -- Lauf und Anzeige ----------------------------------------------------------------


def test_run_shows_result(qtbot: QtBot, widget: BacktestWidget) -> None:
    _prepare(widget)
    with qtbot.waitSignal(widget.finished) as signal:
        widget.start()
    outcome = signal.args[0]
    assert isinstance(outcome, BacktestOutcome)
    assert widget.outcome is outcome
    assert widget.progress_bar.value() == 100
    assert not widget.is_running
    assert "test_cross" in widget.result_label.text()
    assert texts.BACKTEST_SAVED_AS in widget.result_label.text()

    metrics = outcome.metrics
    assert metrics is not None
    assert widget.tile_text("num_trades") == str(metrics.num_trades)
    assert widget.tile_text("total_return_pct").endswith("%")
    assert widget.tile_text("benchmark.total_return_pct") != "–"
    points = widget.curve_points()
    assert points["equity"] == len(outcome.result.equity)
    assert points["benchmark"] == len(outcome.result.equity)
    assert points["drawdown"] == len(outcome.result.equity)

    assert widget.trades_table.rowCount() == len(outcome.result.trades)
    assert widget.trades_table.item(0, 0).text() == "AAA"
    assert widget.trades_table.item(0, 1).text() == texts.BACKTEST_SIDE_LONG
    assert widget.export_button.isEnabled()
    assert widget.monthly_table.rowCount() == len(outcome.monthly)
    assert "Warm-up" in widget.log_view.toPlainText() or widget.log_view.toPlainText()

    # Lauf ist gespeichert und erscheint in der Liste
    assert [r.run_id for r in widget.runs] == [outcome.run_id]
    assert widget.runs_table.rowCount() == 1
    assert widget.runs_table.item(0, 2).text() == "AAA"


def test_run_failure_is_shown(qtbot: QtBot, widget: BacktestWidget) -> None:
    _prepare(widget, symbol="")
    with qtbot.waitSignal(widget.failed) as signal:
        widget.start()
    assert "Symbol" in signal.args[0]
    assert widget.run_button.isEnabled()
    assert widget.outcome is None
    _prepare(widget, symbol="ZZZ")
    with qtbot.waitSignal(widget.failed) as signal:
        widget.start()
    assert "keine Kurse" in signal.args[0]


def _fake_outcome() -> BacktestOutcome:
    dates = pd.bdate_range("2020-01-02", "2021-12-31")
    values = 1000.0 * (1.0 + 0.0004 * np.arange(len(dates)))
    values[100:150] *= 0.9  # Drawdown
    equity = pd.DataFrame(
        {
            "date": dates,
            "equity": values,
            "cash": values,
            "exposure": 0.5,
            "drawdown": values / np.maximum.accumulate(values) - 1.0,
        },
        columns=list(EQUITY_COLUMNS),
    )
    trades = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA"],
            "side": ["LONG", "SHORT", "LONG"],
            "entry_date": pd.to_datetime(["2020-02-03", "2020-05-04", "2021-03-01"]),
            "entry_price": [10.0, 20.0, 12.0],
            "exit_date": pd.to_datetime(["2020-03-02", "2020-06-01", None]),
            "exit_price": [11.0, 21.0, np.nan],
            "qty": [10.0, 5.0, 8.0],
            "pnl": [9.0, -6.0, np.nan],
            "pnl_pct": [10.0, -5.0, np.nan],
            "fees": [2.0, 2.0, 1.0],
            "exit_reason": ["Signal", "Stop", ""],
        }
    )
    result = BacktestResult(
        strategy_id="fake",
        strategy_version="1.0.0",
        params={"fast": 3},
        symbol="AAA",
        timeframe=Timeframe.W1,
        start=dt.date(2020, 1, 1),
        end=dt.date(2021, 12, 31),
        settings={},
        initial_capital=1000.0,
        final_equity=float(values[-1]),
        trades=trades,
        equity=equity,
        orders=pd.DataFrame(columns=list(ORDER_COLUMNS)),
        log=["Zeile eins", "Zeile zwei"],
    )
    metrics = compute_metrics(equity, trades, Timeframe.W1, initial_capital=1000.0)
    return BacktestOutcome(
        result=result,
        metrics=metrics,
        universe="AAA",
        monthly=monthly_returns(equity, initial_capital=1000.0),
    )


def test_fake_outcome_display(widget: BacktestWidget) -> None:
    outcome = _fake_outcome()
    widget.set_outcome(outcome)
    assert texts.BACKTEST_SAVED_AS not in widget.result_label.text()
    assert widget.tile_text("num_trades") == "3"
    assert widget.tile_text("benchmark.total_return_pct") == "–"
    assert widget.tiles["total_return_pct"].value_label.styleSheet()  # farbig
    assert widget.curve_points()["benchmark"] == 0
    assert widget.curve_points()["equity"] == len(outcome.result.equity)

    table = widget.trades_table
    assert table.rowCount() == 3
    assert table.columnCount() == len(TRADE_TABLE_COLUMNS)
    assert table.item(1, 1).text() == texts.BACKTEST_SIDE_SHORT
    assert table.item(0, 2).text() == "03.02.2020"
    assert table.item(0, 7).text() == "9,00"
    assert table.item(1, 8).text() == "-5,00 %"
    assert table.item(2, 4).text() == ""  # offener Trade

    assert widget.monthly_table.rowCount() == 2
    jan = widget.monthly_cell(2020, 1)
    assert jan is not None
    assert jan.text()
    assert jan.background().color().alpha() > 0
    year = widget.monthly_cell(2021, 13)
    assert year is not None
    assert year.text()
    assert widget.monthly_cell(1999, 1) is None
    assert widget.log_view.toPlainText() == "Zeile eins\nZeile zwei"


def test_empty_outcome(widget: BacktestWidget) -> None:
    outcome = _fake_outcome()
    empty = BacktestOutcome(
        result=BacktestResult(
            **{
                **outcome.result.__dict__,
                "trades": outcome.result.trades.iloc[:0],
                "equity": outcome.result.equity.iloc[:0],
            }
        ),
        metrics=None,
        universe="AAA",
    )
    widget.set_outcome(empty)
    assert widget.tile_text("sharpe") == "–"
    assert widget.curve_points() == {"equity": 0, "benchmark": 0, "drawdown": 0}
    assert widget.trades_table.rowCount() == 0
    assert widget.monthly_table.rowCount() == 0
    assert not widget.export_button.isEnabled()


# -- Trades → Chart ------------------------------------------------------------------


def test_double_click_emits_markers(qtbot: QtBot, widget: BacktestWidget) -> None:
    widget.set_outcome(_fake_outcome())
    with qtbot.waitSignal(widget.trade_activated) as signal:
        widget.trades_table.cellDoubleClicked.emit(2, 0)
    symbol, timeframe, markers = signal.args
    assert symbol == "AAA"
    assert timeframe is Timeframe.W1
    # Zwei AAA-Trades: zwei Einstiege, ein Ausstieg (der zweite ist offen)
    assert [m.kind for m in markers] == [MarkerKind.BUY, MarkerKind.EXIT, MarkerKind.BUY]
    assert markers[0].date == dt.date(2020, 2, 3)
    assert markers[0].price == 10.0
    with qtbot.waitSignal(widget.trade_activated) as signal:
        widget.open_trade(1)
    assert signal.args[0] == "BBB"
    assert signal.args[2][0].kind is MarkerKind.SELL
    with qtbot.assertNotEmitted(widget.trade_activated):
        widget.open_trade(99)


def test_double_click_opens_chart_with_markers(
    qtbot: QtBot,
    tmp_path: Path,
    bt_service: BacktestService,
    dialogs: list[tuple[str, str]],
) -> None:
    service = AppService(Settings(data_dir=tmp_path / "app"), tmp_path / "state.json")
    win = MainWindow(
        service,
        chart_service=ChartService(lambda s: BARS.get(s, BARS["AAA"]).assign(adj_close=1.0)),
        backtest_service=bt_service,
    )
    qtbot.addWidget(win)
    assert win.tab_widget("backtest") is win.backtest
    win.tabs.setCurrentWidget(win.backtest)
    _prepare(win.backtest)
    with qtbot.waitSignal(win.backtest.finished, timeout=20_000):
        win.backtest.start()
    trades = win.backtest.outcome.result.trades
    assert len(trades) > 0
    with qtbot.waitSignal(win.chart.data_loaded, timeout=10_000):
        win.backtest.trades_table.cellDoubleClicked.emit(0, 3)
    assert win.tabs.currentWidget() is win.chart
    assert win.chart.symbol == "AAA"
    assert win.chart.timeframe is Timeframe.D1
    markers = win.chart.markers()
    assert len(markers) == len(trades) + int(trades["exit_date"].notna().sum())
    assert win.chart.marker_points()
    assert not dialogs


def test_watchlist_click_in_backtest_tab_sets_symbol(
    qtbot: QtBot, tmp_path: Path, bt_service: BacktestService, dialogs: list[tuple[str, str]]
) -> None:
    service = AppService(Settings(data_dir=tmp_path / "app"), tmp_path / "state.json")
    win = MainWindow(service, backtest_service=bt_service)
    qtbot.addWidget(win)
    win.tabs.setCurrentWidget(win.backtest)
    win.backtest.set_index("SP500")
    win.watchlist.add_symbol("MSFT")
    win.watchlist.symbol_list.itemClicked.emit(win.watchlist.symbol_list.item(0))
    assert win.tabs.currentWidget() is win.backtest
    assert win.backtest.request().universe == "MSFT"
    assert win.backtest.request().universe_kind is UniverseKind.SYMBOL


# -- Gespeicherte Läufe --------------------------------------------------------------


def test_runs_load_compare_delete(qtbot: QtBot, widget: BacktestWidget) -> None:
    _prepare(widget)
    with qtbot.waitSignal(widget.finished):
        widget.start()
    first = widget.outcome
    widget.set_symbol("MSFT")
    widget.set_param("slow", 6)
    with qtbot.waitSignal(widget.finished):
        widget.start()
    second = widget.outcome
    assert first is not None
    assert second is not None
    assert len(widget.runs) == 2

    # Vergleich: genau zwei markieren
    widget.select_runs([first.run_id])
    widget.compare_button.click()
    assert widget.status_label.text() == texts.BACKTEST_COMPARE_SELECT
    widget.select_runs([first.run_id, second.run_id])
    assert set(widget.selected_run_ids()) == {first.run_id, second.run_id}
    widget.compare_button.click()
    assert widget.result_tabs.currentIndex() == 3
    assert widget.compare_table.rowCount() > 5
    assert widget.compare_cell(0, 0) == "Gesamtrendite"
    assert "AAA" in widget.compare_label.text()
    assert "MSFT" in widget.compare_label.text()
    assert "slow=6" in widget.compare_label.text()

    # Laden: der erste Lauf erscheint wieder
    widget.select_runs([first.run_id])
    with qtbot.waitSignal(widget.finished) as signal:
        widget.load_button.click()
    assert signal.args[0].result.fingerprint() == first.result.fingerprint()
    assert widget.result_tabs.currentIndex() == 0
    with qtbot.waitSignal(widget.finished):
        widget.runs_table.cellDoubleClicked.emit(0, 0)

    # Löschen
    widget.select_runs([first.run_id])
    widget.delete_button.click()
    assert [r.run_id for r in widget.runs] == [second.run_id]

    # Fehler beim Laden wird angezeigt
    with qtbot.waitSignal(widget.failed):
        widget.load_run(first.run_id)
    with qtbot.waitSignal(widget.failed):
        widget.compare(first.run_id, second.run_id)


def test_run_without_saving_does_not_list(qtbot: QtBot, widget: BacktestWidget) -> None:
    _prepare(widget)
    widget.save_checkbox.setChecked(False)
    with qtbot.waitSignal(widget.finished) as signal:
        widget.start()
    assert signal.args[0].run_id is None
    widget.refresh_runs()
    assert widget.runs == []
    # Ohne Auswahl passiert nichts
    widget.load_button.click()
    widget.delete_button.click()


def test_runs_refresh_error(
    qtbot: QtBot, widget: BacktestWidget, bt_service: BacktestService
) -> None:
    def broken(limit: int | None = None) -> list[object]:
        raise RuntimeError("Datenbank gesperrt")

    bt_service.list_runs = broken  # type: ignore[method-assign]
    with qtbot.waitSignal(widget.failed) as signal:
        widget.refresh_runs()
    assert "Datenbank gesperrt" in signal.args[0]


def test_runs_loaded_on_first_show(qtbot: QtBot, widget: BacktestWidget) -> None:
    calls: list[int] = []
    original = widget.refresh_runs
    widget.refresh_runs = lambda: calls.append(1) or original()  # type: ignore[method-assign]
    widget.show()
    widget.hide()
    widget.show()
    assert calls == [1]


# -- Export --------------------------------------------------------------------------


def test_export_trades(
    widget: BacktestWidget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert widget.export_trades(tmp_path / "x.csv") is None  # noch kein Ergebnis
    widget.set_outcome(_fake_outcome())
    target = tmp_path / "trades.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    widget.export_button.click()
    assert target.exists()
    assert "3 Trades" in widget.status_label.text()
    back = pd.read_csv(target, sep=";", decimal=",", encoding="utf-8-sig")
    assert list(back["symbol"]) == ["AAA", "BBB", "AAA"]
    assert back["pnl"].iloc[0] == pytest.approx(9.0)

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    assert widget.export_trades() is None
    assert widget.export_trades(tmp_path / "fehlt" / "x.csv") is None
    assert "Export fehlgeschlagen" in widget.status_label.text()


def test_threaded_run_in_main_window(
    qtbot: QtBot, tmp_path: Path, bt_service: BacktestService, dialogs: list[tuple[str, str]]
) -> None:
    service = AppService(Settings(data_dir=tmp_path / "app"), tmp_path / "state.json")
    win = MainWindow(service, backtest_service=bt_service)
    qtbot.addWidget(win)
    _prepare(win.backtest)
    progress: list[int] = []
    win.backtest.progress_bar.valueChanged.connect(progress.append)
    with qtbot.waitSignal(win.backtest.finished, timeout=20_000):
        win.backtest.start()
        assert win.backtest.is_running
        win.backtest.start()  # zweiter Start während des Laufs wird ignoriert
    assert progress[-1] == 100
    assert len(progress) > 2
    qtbot.waitUntil(lambda: len(win.backtest.runs) == 1, timeout=10_000)
    assert not dialogs
