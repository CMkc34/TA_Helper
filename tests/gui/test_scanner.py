"""Scanner-Tab mit Fake-Service: Liste, Sortierung, Doppelklick, Export, Verlauf, Filter."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from trading_helfer.core.config import Settings
from trading_helfer.core.markers import Marker, MarkerKind
from trading_helfer.core.types import SignalType, Timeframe
from trading_helfer.gui import texts
from trading_helfer.gui.main_window import MainWindow
from trading_helfer.gui.widgets.scanner import (
    COLUMN_TITLES,
    HISTORY_COLUMNS,
    ScannerWidget,
    format_number,
    parse_param_text,
)
from trading_helfer.scanner.scan import RESULT_COLUMNS, ScanError, ScanFilters
from trading_helfer.services.app_service import AppService
from trading_helfer.services.scanner_service import (
    CUSTOM_UNIVERSE,
    ScannerService,
    ScanRecord,
    ScanRequest,
    StrategyChoice,
)

STRATEGIES = [
    StrategyChoice("trend", "Trendfolge", (Timeframe.D3, Timeframe.W1), {"fast": 10}),
    StrategyChoice("daily", "Nur Monat", (Timeframe.MN1,), {}),
]


def sample_result(scan_id: str = "scan-1") -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "symbol": ["MSFT", "AAPL", "XOM", "NVDA"],
            "date": pd.to_datetime(["2024-03-08"] * 4),
            "signal": ["BUY", "BUY", "SELL", "EXIT"],
            "strength": [0.25, 1.5, float("nan"), -0.5],
            "close": [410.5, 172.25, 118.0, 875.0],
            "sector": ["Technology", "Technology", "Energy", None],
            "market_cap": [3.1e12, 2.7e12, float("nan"), 2.2e12],
            "note": ["BUY MARKET", "BUY MARKET", "SELL_SHORT MARKET", "SELL MARKET"],
            "sma_fast": [400.0, 170.0, 120.0, 880.0],
        }
    )
    frame.attrs["scan_id"] = scan_id
    frame.attrs["skipped"] = {"ZZZ": "keine Kurse"}
    return frame


class FakeScannerService(ScannerService):
    """Ohne Datenbank: liefert feste Strategien, Ergebnisse und einen Verlauf im Speicher."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(Settings(data_dir=tmp_path))
        self.requests: list[ScanRequest] = []
        self.records: list[ScanRecord] = [
            ScanRecord(
                "old",
                dt.datetime(2024, 3, 1, 18, 5),
                "trend",
                "NDX100",
                Timeframe.D3,
                dt.date(2024, 2, 29),
                1,
            )
        ]
        self.deleted: list[str] = []
        self.fail: Exception | None = None

    def strategies(self) -> list[StrategyChoice]:
        return list(STRATEGIES)

    def sectors(self) -> list[str]:
        return ["Energy", "Technology"]

    def run(self, request: ScanRequest, progress: Any = None) -> pd.DataFrame:
        self.requests.append(request)
        if self.fail is not None:
            raise self.fail
        if progress is not None:
            progress(50, "1/2")
            progress(100, "2/2")
        result = sample_result(f"scan-{len(self.requests)}")
        self.records.insert(
            0,
            ScanRecord(
                str(result.attrs["scan_id"]),
                dt.datetime(2024, 3, 9, 10, len(self.requests)),
                request.strategy_id,
                request.universe_text,
                request.timeframe,
                request.as_of,
                len(result),
            ),
        )
        return result

    def history(self, limit: int = 100) -> list[ScanRecord]:
        return list(self.records[:limit])

    def load(self, scan_id: str) -> pd.DataFrame:
        frame = sample_result(scan_id).iloc[:2][list(RESULT_COLUMNS)].reset_index(drop=True)
        frame.attrs["scan_id"] = scan_id
        return frame

    def delete(self, scan_id: str) -> None:
        self.deleted.append(scan_id)
        self.records = [r for r in self.records if r.scan_id != scan_id]


@pytest.fixture
def fake(tmp_path: Path) -> FakeScannerService:
    return FakeScannerService(tmp_path)


@pytest.fixture
def widget(qtbot: QtBot, fake: FakeScannerService) -> ScannerWidget:
    w = ScannerWidget(fake)
    qtbot.addWidget(w)
    w.refresh()
    return w


def double_click(qtbot: QtBot, table: Any, row: int) -> None:
    """Doppelklick wie mit der Maus (Qt erwartet vorher einen einfachen Klick)."""
    rect = table.visualItemRect(table.item(row, 0))
    qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())
    qtbot.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, pos=rect.center())


def column(widget: ScannerWidget, name: str) -> int:
    headers = [
        widget.table.horizontalHeaderItem(c).text() for c in range(widget.table.columnCount())
    ]
    return headers.index(COLUMN_TITLES.get(name, name))


def column_texts(widget: ScannerWidget, name: str) -> list[str]:
    col = column(widget, name)
    return [widget.table.item(r, col).text() for r in range(widget.table.rowCount())]


# -- Auswahl -------------------------------------------------------------------------


def test_choices_are_loaded(widget: ScannerWidget) -> None:
    assert widget.strategy_combo.count() == 2
    assert widget.strategy_combo.currentText() == "Trendfolge"
    assert [widget.timeframe_combo.itemText(i) for i in range(2)] == ["3D", "1W"]
    assert widget.selected_timeframe() is Timeframe.W1
    assert "fast=10" in widget.params_edit.toolTip()
    assert [widget.sector_combo.itemText(i) for i in range(3)] == [
        texts.SCANNER_SECTOR_ALL,
        "Energy",
        "Technology",
    ]
    assert widget.history_table.rowCount() == 1
    assert widget.history_table.item(0, 1).text() == "Trendfolge"
    assert widget.history_table.item(0, 2).text() == "Nasdaq-100"
    assert widget.history_table.item(0, 4).text() == "29.02.2024"
    assert widget.history_table.columnCount() == len(HISTORY_COLUMNS)
    assert widget.run_button.isEnabled()
    assert not widget.export_button.isEnabled()


def test_strategy_change_updates_timeframes(widget: ScannerWidget) -> None:
    widget.strategy_combo.setCurrentIndex(1)
    assert widget.timeframe_combo.count() == 1
    assert widget.selected_timeframe() is Timeframe.MN1
    widget.set_strategies(list(STRATEGIES))  # Wahl bleibt erhalten
    assert widget.selected_strategy() == STRATEGIES[1]


def test_without_strategies(widget: ScannerWidget) -> None:
    widget.set_strategies([])
    assert widget.selected_strategy() is None
    assert not widget.run_button.isEnabled()
    assert widget.status_label.text() == texts.SCANNER_NO_STRATEGY
    with pytest.raises(ScanError):
        widget.build_request()


def test_custom_universe_shows_symbol_field(widget: ScannerWidget) -> None:
    assert widget.symbols_edit.isHidden()
    widget.universe_combo.setCurrentIndex(widget.universe_combo.findData(CUSTOM_UNIVERSE))
    assert not widget.symbols_edit.isHidden()


def test_build_request_with_filters(widget: ScannerWidget) -> None:
    widget.universe_combo.setCurrentIndex(widget.universe_combo.findData(CUSTOM_UNIVERSE))
    widget.symbols_edit.setText("aapl, msft")
    widget.timeframe_combo.setCurrentIndex(0)
    widget.as_of_check.setChecked(True)
    widget.as_of_edit.setDate(QDate(2024, 3, 8))
    widget.params_edit.setText("fast=5; slow=20")
    widget.sector_combo.setCurrentIndex(2)
    widget.min_cap_box.setValue(10.0)
    widget.max_cap_box.setValue(0.0)
    widget.metric_combo.setCurrentIndex(widget.metric_combo.findData("pe_ratio"))
    widget.metric_min_edit.setText("")
    widget.metric_max_edit.setText("25,5")
    widget.signal_boxes[SignalType.EXIT].setChecked(False)

    request = widget.build_request()
    assert request == ScanRequest(
        strategy_id="trend",
        universe=("AAPL", "MSFT"),
        timeframe=Timeframe.D3,
        as_of=dt.date(2024, 3, 8),
        params={"fast": 5, "slow": 20},
        filters=ScanFilters(
            sectors=frozenset({"Technology"}),
            min_market_cap=10e9,
            metrics={"pe_ratio": (None, 25.5)},
            signals=frozenset({SignalType.BUY, SignalType.SELL}),
        ),
    )


def test_default_request(widget: ScannerWidget) -> None:
    request = widget.build_request()
    assert request.universe == "SP500"
    assert request.as_of is None
    assert request.params == {}
    assert request.filters == ScanFilters()


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda w: w.params_edit.setText("fast"), "NAME=WERT"),
        (lambda w: w.metric_min_edit.setText("abc"), "keine Zahl"),
        (lambda w: (w.min_cap_box.setValue(20), w.max_cap_box.setValue(10)), "Untergrenze"),
        (
            lambda w: w.universe_combo.setCurrentIndex(w.universe_combo.findData(CUSTOM_UNIVERSE)),
            "mindestens ein Symbol",
        ),
    ],
)
def test_invalid_input_is_reported(
    widget: ScannerWidget, fake: FakeScannerService, setup: Any, message: str
) -> None:
    widget.metric_combo.setCurrentIndex(1)
    setup(widget)
    widget.start_scan()
    assert fake.requests == []
    assert message in widget.status_label.text()
    assert widget.status_label.text().startswith("Eingabe ungültig")


# -- Scan und Liste ------------------------------------------------------------------


def test_scan_fills_sortable_list(widget: ScannerWidget, fake: FakeScannerService) -> None:
    progress: list[int] = []
    widget.progress_bar.valueChanged.connect(progress.append)
    widget.start_scan()

    assert len(fake.requests) == 1
    assert progress[:2] == [50, 100]
    assert widget.table.rowCount() == 4
    assert widget.table.columnCount() == len(RESULT_COLUMNS) + 1
    assert widget.table.horizontalHeaderItem(column(widget, "sma_fast")).text() == "sma_fast"
    assert widget.visible_symbols() == ["MSFT", "AAPL", "XOM", "NVDA"]
    assert column_texts(widget, "signal") == ["Kauf", "Kauf", "Short", "Ausstieg"]
    assert column_texts(widget, "strength") == ["0,25", "1,5", "", "-0,5"]
    assert column_texts(widget, "market_cap")[0] == "3100,0 Mrd."
    assert column_texts(widget, "date")[0] == "08.03.2024"
    assert column_texts(widget, "sector")[3] == ""
    assert "4 Signal(e) am 08.03.2024" in widget.status_label.text()
    assert "1 Wert(e) übersprungen" in widget.status_label.text()
    assert widget.export_button.isEnabled()
    # Verlauf wurde neu geladen, neuester Scan oben.
    assert widget.history_table.rowCount() == 2
    assert widget.history()[0].scan_id == "scan-1"


def test_sorting_by_columns(widget: ScannerWidget) -> None:
    widget.start_scan()
    widget.table.sortItems(column(widget, "symbol"), Qt.SortOrder.AscendingOrder)
    assert widget.visible_symbols() == ["AAPL", "MSFT", "NVDA", "XOM"]
    widget.table.sortItems(column(widget, "symbol"), Qt.SortOrder.DescendingOrder)
    assert widget.visible_symbols() == ["XOM", "NVDA", "MSFT", "AAPL"]

    strength = column(widget, "strength")
    widget.table.horizontalHeader().setSortIndicator(strength, Qt.SortOrder.AscendingOrder)
    assert widget.visible_symbols() == ["NVDA", "MSFT", "AAPL", "XOM"]  # leer unten
    widget.table.horizontalHeader().setSortIndicator(strength, Qt.SortOrder.DescendingOrder)
    assert widget.visible_symbols() == ["AAPL", "MSFT", "NVDA", "XOM"]  # leer weiter unten

    widget.table.sortItems(column(widget, "close"), Qt.SortOrder.AscendingOrder)
    assert widget.visible_symbols() == ["XOM", "AAPL", "MSFT", "NVDA"]  # numerisch
    widget.table.sortItems(column(widget, "signal"), Qt.SortOrder.DescendingOrder)
    assert widget.visible_symbols()[0] == "NVDA"
    widget.table.sortItems(column(widget, "sector"), Qt.SortOrder.AscendingOrder)
    assert widget.visible_symbols()[-1] == "NVDA"
    # Hinter jeder Zeile steht nach dem Sortieren der richtige Datensatz.
    for row, symbol in enumerate(widget.visible_symbols()):
        record = widget.result_row(row)
        assert record is not None
        assert record["symbol"] == symbol
    assert widget.result_row(99) is None


def test_double_click_requests_chart(qtbot: QtBot, widget: ScannerWidget) -> None:
    widget.timeframe_combo.setCurrentIndex(0)  # 3D
    widget.start_scan()
    widget.table.sortItems(column(widget, "symbol"), Qt.SortOrder.DescendingOrder)
    received: list[tuple[str, Timeframe, Marker]] = []
    widget.chart_requested.connect(lambda s, tf, m: received.append((s, tf, m)))
    widget.show()
    double_click(qtbot, widget.table, 0)
    assert len(received) == 1
    symbol, timeframe, marker = received[0]
    assert symbol == "XOM"
    assert timeframe is Timeframe.D3
    assert marker.kind is MarkerKind.SELL
    assert marker.date == dt.date(2024, 3, 8)
    widget.open_row(42)  # ohne Zeile passiert nichts
    assert len(received) == 1


def test_scan_error_is_shown(widget: ScannerWidget, fake: FakeScannerService) -> None:
    fake.fail = ScanError("Strategie unterstützt den Zeitrahmen nicht")
    widget.start_scan()
    assert widget.status_label.text().startswith("Scan fehlgeschlagen")
    assert "Zeitrahmen" in widget.status_label.text()
    assert widget.run_button.isEnabled()


def test_empty_result_message(widget: ScannerWidget, fake: FakeScannerService) -> None:
    def no_signals(request: ScanRequest, progress: Any = None) -> pd.DataFrame:
        fake.requests.append(request)
        return pd.DataFrame(columns=list(RESULT_COLUMNS))

    fake.run = no_signals  # type: ignore[method-assign]
    widget.start_scan()
    assert widget.status_label.text().startswith("Keine Signale")
    assert widget.table.rowCount() == 0
    assert not widget.export_button.isEnabled()


# -- Export --------------------------------------------------------------------------


def test_export_csv(widget: ScannerWidget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    widget.start_scan()
    target = tmp_path / "export" / "signale"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(target), "CSV-Dateien (*.csv)")
    )
    widget.ask_export()
    written = target.with_suffix(".csv")
    frame = pd.read_csv(written)
    assert list(frame["symbol"]) == ["MSFT", "AAPL", "XOM", "NVDA"]
    assert list(frame.columns) == [*RESULT_COLUMNS, "sma_fast"]
    assert frame["date"].iloc[0] == "2024-03-08"
    assert texts.SCANNER_EXPORT_DONE.format(path=written) == widget.status_label.text()


def test_export_cancel_and_failure(
    widget: ScannerWidget, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget.ask_export()  # ohne Ergebnis: nichts zu tun
    widget.start_scan()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_a, **_k: ("", ""))
    widget.ask_export()
    assert widget.status_label.text().startswith("4 Signal(e)")
    blocker = tmp_path / "datei"
    blocker.write_text("x")
    assert widget.export_csv(blocker / "scan.csv") is None
    assert widget.status_label.text().startswith("Export fehlgeschlagen")


# -- Verlauf -------------------------------------------------------------------------


def test_load_history(widget: ScannerWidget) -> None:
    widget.load_history(0)
    assert widget.visible_symbols() == ["MSFT", "AAPL"]
    assert widget.result_timeframe() is Timeframe.D3
    assert widget.result().attrs["scan_id"] == "old"
    assert widget.status_label.text().startswith("Scan vom 01.03.2024 18:05 · Trendfolge")
    widget.load_history(5)  # ungültige Zeile
    assert widget.result().attrs["scan_id"] == "old"


def test_history_buttons(widget: ScannerWidget, fake: FakeScannerService) -> None:
    widget.start_scan()
    widget.history_table.selectRow(1)
    widget.load_button.click()
    assert widget.result().attrs["scan_id"] == "old"
    widget.history_table.selectRow(0)
    widget.delete_button.click()
    assert fake.deleted == ["scan-1"]
    assert [r.scan_id for r in widget.history()] == ["old"]
    widget.delete_history(9)
    assert fake.deleted == ["scan-1"]


def test_history_double_click_and_empty(qtbot: QtBot, widget: ScannerWidget) -> None:
    widget.show()
    double_click(qtbot, widget.history_table, 0)
    assert widget.result().attrs["scan_id"] == "old"
    widget.set_history([])
    assert not widget.history_label.isHidden()
    assert not widget.load_button.isEnabled()
    assert widget.history_table.item(0, 0) is None


def test_first_show_loads_choices(qtbot: QtBot, fake: FakeScannerService) -> None:
    w = ScannerWidget(fake)
    qtbot.addWidget(w)
    assert w.strategy_combo.count() == 0
    w.show()
    assert w.strategy_combo.count() == 2


# -- Hilfsfunktionen -----------------------------------------------------------------


def test_helpers() -> None:
    assert format_number(1234.5678) == "1235"
    assert format_number(0.125) == "0,125"
    assert format_number(None) == ""
    assert format_number(float("inf")) == ""
    assert format_number("abc") == "abc"
    assert format_number(True) == "True"
    assert parse_param_text("a=1, b=true;c=x") == {"a": 1, "b": True, "c": "x"}
    assert parse_param_text("  ") == {}


# -- Hauptfenster --------------------------------------------------------------------


def test_main_window_opens_chart_with_signal(
    qtbot: QtBot, service: AppService, fake: FakeScannerService, dialogs: list[Any]
) -> None:
    win = MainWindow(service, scanner_service=fake)
    qtbot.addWidget(win)
    assert win.tab_widget("scanner") is win.scanner
    win.scanner.refresh()
    qtbot.waitUntil(lambda: win.scanner.strategy_combo.count() == 2)
    with qtbot.waitSignal(win.scanner.scan_finished, timeout=5000):
        win.scanner.start_scan()
    qtbot.waitUntil(lambda: win.scanner.table.rowCount() == 4)

    win.scanner.open_row(0)
    assert win.tabs.currentWidget() is win.chart
    assert win.chart.symbol == "MSFT"
    assert win.chart.timeframe is Timeframe.W1
    assert [m.kind for m in win.chart.markers()] == [MarkerKind.BUY]

    # Anderer Wert in der Watchlist: Signal-Marker verschwinden.
    win.watchlist.symbol_selected.emit("MSFT")
    assert len(win.chart.markers()) == 1
    win.watchlist.symbol_selected.emit("AAPL")
    assert win.chart.markers() == []
    assert dialogs == []
    qtbot.waitUntil(lambda: not win._active_workers)
