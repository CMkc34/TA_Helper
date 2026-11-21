"""Kennzahlen-Tab mit Fake-Service: Tabelle, Farben, Verlauf, Stichtag, Watchlist."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from trading_helfer.fundamentals.metrics import METRIC_LABELS, METRIC_NAMES
from trading_helfer.gui import texts
from trading_helfer.gui.main_window import MainWindow
from trading_helfer.gui.theme import apply_theme
from trading_helfer.gui.widgets.fundamentals import (
    COL_COMPARISON,
    COL_MEDIAN,
    COL_VALUE,
    COLUMNS,
    DEFAULT_METRIC,
    FundamentalsWidget,
)
from trading_helfer.services.app_service import AppService
from trading_helfer.services.fundamentals_service import (
    Comparison,
    FundamentalsData,
    FundamentalsService,
    MetricRow,
    compare,
    empty_history,
)

TODAY = dt.date(2024, 3, 1)
PERIODS = pd.date_range("2022-03-31", periods=8, freq="QE")


def fake_history(as_of: dt.date) -> pd.DataFrame:
    rows = []
    for i, end in enumerate(PERIODS):
        values: dict[str, float] = dict.fromkeys(METRIC_NAMES, np.nan)
        values["pe_ratio"] = 20.0 + i
        values["gross_margin"] = 0.40 + 0.01 * i
        values["roe"] = np.nan if i % 2 else 0.1  # Lücken im Verlauf
        rows.append({"period_end": end, "as_of": end + pd.Timedelta(days=30), **values})
    frame = pd.DataFrame(rows)
    return frame[frame["as_of"] <= pd.Timestamp(as_of)].reset_index(drop=True)


class FakeService(FundamentalsService):
    """Liefert feste Werte: AAA mit Sektor, NOS ohne Daten, BAD wirft einen Fehler."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[str, dt.date]] = []

    def load(self, symbol: str, as_of: dt.date) -> FundamentalsData:
        self.requests.append((symbol, as_of))
        if symbol == "BAD":
            raise RuntimeError("Datenbank gesperrt")
        if symbol == "NOS":
            rows = tuple(MetricRow(n, METRIC_LABELS[n], None, None, None) for n in METRIC_NAMES)
            return FundamentalsData(symbol, as_of, None, 0, rows, empty_history())
        values: dict[str, float | None] = dict.fromkeys(METRIC_NAMES)
        values |= {"price": 25.0, "pe_ratio": 15.0, "gross_margin": 0.30, "roe": 0.20}
        medians: dict[str, float | None] = {"pe_ratio": 20.0, "gross_margin": 0.45, "roe": 0.2}
        rows = tuple(
            MetricRow(
                n,
                METRIC_LABELS[n],
                values[n],
                medians.get(n),
                compare(n, values[n], medians.get(n)),
            )
            for n in METRIC_NAMES
        )
        return FundamentalsData(symbol, as_of, "Technologie", 12, rows, fake_history(as_of))


@pytest.fixture
def fake() -> FakeService:
    return FakeService()


@pytest.fixture
def widget(qtbot: QtBot, fake: FakeService) -> FundamentalsWidget:
    w = FundamentalsWidget(fake, today=lambda: TODAY)
    qtbot.addWidget(w)
    return w


def test_initial_state(widget: FundamentalsWidget, fake: FakeService) -> None:
    assert widget.symbol is None
    assert widget.data is None
    assert widget.as_of == TODAY
    assert widget.table.rowCount() == 0
    assert widget.message_label.text() == texts.FUNDAMENTALS_NO_SYMBOL
    widget.reload()  # ohne Symbol passiert nichts
    assert fake.requests == []
    assert widget.history_points() == 0


def test_table_filled(widget: FundamentalsWidget, qtbot: QtBot) -> None:
    with qtbot.waitSignal(widget.data_loaded):
        widget.load_symbol("AAA")
    table = widget.table
    assert table.rowCount() == len(METRIC_NAMES)
    assert table.columnCount() == len(COLUMNS)
    assert [table.horizontalHeaderItem(c).text() for c in range(4)] == list(COLUMNS)  # type: ignore[union-attr]
    assert table.item(0, 0).text() == METRIC_LABELS[METRIC_NAMES[0]]  # type: ignore[union-attr]
    assert widget.cell_text("pe_ratio", COL_VALUE) == "15,00"
    assert widget.cell_text("pe_ratio", COL_MEDIAN) == "20,00"
    assert widget.cell_text("pe_ratio", COL_COMPARISON) == texts.FUNDAMENTALS_BETTER
    assert widget.cell_text("gross_margin", COL_VALUE) == "30,0 %"
    assert widget.cell_text("gross_margin", COL_COMPARISON) == texts.FUNDAMENTALS_WORSE
    assert widget.cell_text("roe", COL_COMPARISON) == texts.FUNDAMENTALS_EQUAL
    assert widget.cell_text("price", COL_VALUE) == "25,00 $"
    assert texts.FUNDAMENTALS_HEADER.format(symbol="AAA") == widget.message_label.text()
    assert "Technologie" in widget.info_label.text()
    assert "01.03.2024" in widget.info_label.text()


def test_comparison_colors(widget: FundamentalsWidget) -> None:
    widget.load_symbol("AAA")

    def background(metric: str) -> tuple[int, int, int, int]:
        for row in range(widget.table.rowCount()):
            item = widget.table.item(row, 0)
            if item is not None and item.text() == METRIC_LABELS[metric]:
                cell = widget.table.item(row, COL_VALUE)
                assert cell is not None
                return cell.background().color().getRgb()  # type: ignore[return-value]
        raise KeyError(metric)

    better = background("pe_ratio")
    worse = background("gross_margin")
    assert better[1] > better[0]  # grün
    assert worse[0] > worse[1]  # rot
    assert better[3] > 0
    assert worse[3] > 0
    assert widget.table.item(0, COL_VALUE).background().style() == Qt.BrushStyle.NoBrush  # type: ignore[union-attr]


def test_missing_values_shown_as_dash(widget: FundamentalsWidget) -> None:
    widget.load_symbol("AAA")
    assert widget.cell_text("dividend_yield", COL_VALUE) == "–"
    assert widget.cell_text("dividend_yield", COL_MEDIAN) == "–"
    assert widget.cell_text("dividend_yield", COL_COMPARISON) == ""
    widget.load_symbol("NOS")
    for metric in METRIC_NAMES:
        assert widget.cell_text(metric, COL_VALUE) == "–"
        assert widget.cell_text(metric, COL_MEDIAN) == "–"
    assert widget.message_label.text() == texts.FUNDAMENTALS_NO_DATA.format(symbol="NOS")
    assert texts.FUNDAMENTALS_NO_SECTOR in widget.info_label.text()
    assert widget.history_points() == 0
    assert widget.history_plot.titleLabel.text == texts.FUNDAMENTALS_NO_HISTORY


def test_history_drawn(widget: FundamentalsWidget) -> None:
    widget.load_symbol("AAA")
    assert widget.selected_metric == DEFAULT_METRIC
    expected = len(fake_history(TODAY))
    assert widget.history_points() == expected
    x, y = widget.history_curve.getData()
    assert y[0] == pytest.approx(20.0)
    assert x[0] == pytest.approx(PERIODS[0].timestamp())
    assert METRIC_LABELS["pe_ratio"] in widget.history_plot.titleLabel.text

    widget.show_metric("gross_margin")
    _x, y = widget.history_curve.getData()
    assert y[0] == pytest.approx(40.0)  # Prozentwerte
    assert "(%)" in widget.history_plot.titleLabel.text

    widget.show_metric("roe")  # mit Lücken
    assert widget.history_points() == (expected + 1) // 2

    widget.show_metric("market_cap")  # nur NaN
    assert widget.history_points() == 0
    assert widget.history_plot.titleLabel.text == texts.FUNDAMENTALS_NO_HISTORY

    with pytest.raises(KeyError):
        widget.show_metric("gibt_es_nicht")
    with pytest.raises(KeyError):
        widget.cell_text("gibt_es_nicht", COL_VALUE)


def test_row_click_switches_history(widget: FundamentalsWidget) -> None:
    widget.load_symbol("AAA")
    row = list(METRIC_NAMES).index("gross_margin")
    widget.table.setCurrentCell(row, 0)
    assert widget.selected_metric == "gross_margin"
    _x, y = widget.history_curve.getData()
    assert y[-1] == pytest.approx(100 * (0.40 + 0.01 * (len(y) - 1)))
    # Auswahl bleibt beim Neuladen erhalten.
    widget.load_symbol("AAA")
    assert widget.selected_metric == "gross_margin"
    assert widget.table.currentRow() == row


def test_as_of_change_reloads(widget: FundamentalsWidget, fake: FakeService) -> None:
    widget.load_symbol("AAA")
    full = widget.history_points()
    widget.set_as_of(dt.date(2022, 12, 1))
    assert fake.requests[-1] == ("AAA", dt.date(2022, 12, 1))
    assert widget.data is not None
    assert widget.data.as_of == dt.date(2022, 12, 1)
    assert widget.history_points() < full
    widget.today_button.click()
    assert fake.requests[-1] == ("AAA", TODAY)
    assert widget.history_points() == full


def test_load_error_shown(widget: FundamentalsWidget, qtbot: QtBot) -> None:
    with qtbot.waitSignal(widget.load_failed) as blocker:
        widget.load_symbol("BAD")
    assert "Datenbank gesperrt" in blocker.args[0]
    assert "BAD" in widget.message_label.text()


def test_stale_result_ignored(qtbot: QtBot, fake: FakeService) -> None:
    pending: list[tuple[Callable[..., Any], tuple[Any, ...], Callable[[Any], None]]] = []

    def deferred(
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None] | None = None,
        name: str = "",
    ) -> None:
        assert on_result is not None
        assert name == texts.FUNDAMENTALS_TASK_NAME.format(symbol=args[0])
        pending.append((fn, args, on_result))

    w = FundamentalsWidget(fake, runner=deferred, today=lambda: TODAY)
    qtbot.addWidget(w)
    w.load_symbol("NOS")
    w.load_symbol("AAA")
    assert w.message_label.text() == texts.FUNDAMENTALS_LOADING.format(symbol="AAA")
    for fn, args, on_result in reversed(pending):
        on_result(fn(*args))
    assert w.data is not None
    assert w.data.symbol == "AAA"


def test_theme_switch_keeps_history(widget: FundamentalsWidget) -> None:
    widget.load_symbol("AAA")
    points = widget.history_points()
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    apply_theme(app, "dark")
    try:
        assert widget.history_points() == points
    finally:
        apply_theme(app, "light")


# ---------------------------------------------------------------------- Hauptfenster


def test_main_window_has_fundamentals_tab(qtbot: QtBot, service: AppService) -> None:
    win = MainWindow(service, fundamentals_service=FakeService())
    qtbot.addWidget(win)
    assert win.tab_widget("fundamentals") is win.fundamentals


def test_watchlist_selection_updates_tab(qtbot: QtBot, service: AppService) -> None:
    fake = FakeService()
    win = MainWindow(service, fundamentals_service=fake)
    qtbot.addWidget(win)
    win.tabs.setCurrentWidget(win.fundamentals)
    win.watchlist.add_symbol("AAA")
    win.watchlist.add_symbol("NOS")

    def click(symbol: str) -> None:
        for row in range(win.watchlist.symbol_list.count()):
            item = win.watchlist.symbol_list.item(row)
            if item.text().startswith(symbol):
                with qtbot.waitSignal(win.fundamentals.data_loaded, timeout=5000):
                    win.watchlist.symbol_list.itemClicked.emit(item)
                return
        raise AssertionError(symbol)

    click("AAA")
    assert win.tabs.currentWidget() is win.fundamentals  # Tab bleibt sichtbar
    assert win.fundamentals.symbol == "AAA"
    assert win.fundamentals.cell_text("pe_ratio", COL_VALUE) == "15,00"
    click("NOS")
    assert win.fundamentals.symbol == "NOS"
    assert win.fundamentals.cell_text("pe_ratio", COL_VALUE) == "–"
    assert [s for s, _d in fake.requests] == ["AAA", "NOS"]


def test_comparison_enum_values() -> None:
    assert {c.value for c in Comparison} == {"better", "worse", "equal"}
