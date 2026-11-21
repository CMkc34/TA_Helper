"""Daten-Tab mit Fake-Service: Fortschritt, Abbrechen, Qualitätsbericht, Quellen, CSV-Import."""

from __future__ import annotations

import datetime as dt
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog
from pytestqt.qtbot import QtBot

from trading_helfer.core.config import ConfigError, Settings
from trading_helfer.data.providers.csv_import import CsvImportReport, RowError
from trading_helfer.data.quality import Severity
from trading_helfer.data.update import UpdateResult
from trading_helfer.gui import texts
from trading_helfer.gui.main_window import MainWindow
from trading_helfer.gui.widgets.data import (
    COL_CHECK,
    COL_DATE,
    COL_SEVERITY,
    COL_SYMBOL,
    COLUMNS,
    DataWidget,
    format_last_update,
)
from trading_helfer.services.app_service import AppService
from trading_helfer.services.data_service import (
    DataService,
    LastUpdate,
    ProviderSettings,
    mask_secret,
)

SYMBOLS = ("SPY", "QQQ", "AAA", "BBB")
STARTED = dt.datetime(2024, 3, 1, 18, 5)


def sample_report() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "date": pd.to_datetime(["2024-01-03", "2024-02-01", None]),
            "check": ["GAP", "JUMP", "MISSING_DATA"],
            "severity": ["ERROR", "WARNING", "ERROR"],
            "detail": ["2 Handelstage fehlen", "Sprung +80 %", "keine Kurse"],
        }
    )


class FakeDataService(DataService):
    """Läuft ohne Datenbank; ``block_at`` hält das Update vor diesem Symbol an."""

    def __init__(self, tmp_path: Path, *, block_at: int | None = None) -> None:
        super().__init__(Settings(data_dir=tmp_path))
        self.block_at = block_at
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.processed: list[str] = []
        self.update_calls: list[list[str]] = []
        self.fail_update = False
        self.last: LastUpdate | None = None
        self.report = sample_report()
        self.fail_report = False
        self.saved: list[tuple[list[str], str | None, bool]] = []
        self.fail_save = False
        self.imports: list[Path] = []
        self.fail_import = False
        self.key = ""

    def run_update(
        self,
        index_codes: Sequence[str],
        *,
        symbols: Sequence[str] = (),
        progress: Callable[[int, str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        today: dt.date | None = None,
    ) -> UpdateResult:
        self.update_calls.append(list(index_codes))
        if self.fail_update:
            raise RuntimeError("Quelle nicht erreichbar")
        result = UpdateResult(log_id=1, total=len(SYMBOLS))
        for i, symbol in enumerate(SYMBOLS):
            if should_cancel is not None and should_cancel():
                result.cancelled = True
                break
            if i == self.block_at:
                self.blocked.set()
                assert self.release.wait(10)
            self.processed.append(symbol)
            result.ok.append(symbol)
            if progress is not None:
                progress(round(100 * (i + 1) / len(SYMBOLS)), f"{symbol} ({i + 1}/{len(SYMBOLS)})")
        self.last = LastUpdate(STARTED, STARTED, "yfinance,tiingo", len(result.ok), 0, "ok")
        return result

    def last_update(self) -> LastUpdate | None:
        return self.last

    def quality_report(self, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        if self.fail_report:
            raise RuntimeError("Datenbank gesperrt")
        return self.report

    def import_csv(self, path: Path, *, date_format: str | None = None) -> CsvImportReport:
        self.imports.append(path)
        if self.fail_import:
            raise ValueError("Spalte close fehlt")
        report = CsvImportReport(files=[path.name], rows_read=3, rows_imported=2, symbols={"SYN"})
        report.errors.append(RowError(path.name, 3, "SYN", "Schlusskurs ≤ 0"))
        return report

    def provider_settings(self) -> ProviderSettings:
        order = tuple(self.saved[-1][0]) if self.saved else ("yfinance", "tiingo")
        return ProviderSettings(order, ("yfinance", "tiingo"), mask_secret(self.key))

    def save_provider_settings(
        self, order: Sequence[str], tiingo_key: str | None = None, *, clear_key: bool = False
    ) -> ProviderSettings:
        if self.fail_save:
            raise ConfigError("Datei schreibgeschützt")
        self.saved.append((list(order), tiingo_key, clear_key))
        if clear_key:
            self.key = ""
        elif tiingo_key:
            self.key = tiingo_key
        return self.provider_settings()


@pytest.fixture
def fake(tmp_path: Path) -> FakeDataService:
    return FakeDataService(tmp_path)


@pytest.fixture
def widget(qtbot: QtBot, fake: FakeDataService) -> DataWidget:
    view = DataWidget(fake)
    qtbot.addWidget(view)
    return view


# ---------------------------------------------------------------------- Update


def test_update_shows_progress_and_last_update(widget: DataWidget, fake: FakeDataService) -> None:
    assert widget.progress_bar.value() == 0
    assert widget.progress_label.text() == texts.DATA_UPDATE_IDLE
    assert not widget.cancel_button.isEnabled()
    seen: list[int] = []
    widget.progress_bar.valueChanged.connect(seen.append)
    results: list[UpdateResult] = []
    widget.update_finished.connect(results.append)

    widget.update_button.click()

    assert fake.update_calls == [["SP500", "NDX100"]]
    assert seen == [25, 50, 75, 100]
    assert results[0].ok == list(SYMBOLS)
    assert widget.progress_label.text() == results[0].summary()
    assert widget.last_update_label.text() == (
        "Letztes Update: 01.03.2024 18:05 · Quellen: yfinance, tiingo · ok"
    )
    assert widget.update_button.isEnabled()
    assert not widget.cancel_button.isEnabled()
    assert not widget.is_busy


def test_update_uses_selected_indices(widget: DataWidget, fake: FakeDataService) -> None:
    widget.index_boxes["SP500"].setChecked(False)
    widget.start_update()
    assert fake.update_calls == [["NDX100"]]

    widget.index_boxes["NDX100"].setChecked(False)
    widget.start_update()
    assert fake.update_calls == [["NDX100"]]
    assert widget.progress_label.text() == texts.DATA_UPDATE_NO_INDEX


def test_update_error_is_shown(widget: DataWidget, fake: FakeDataService) -> None:
    fake.fail_update = True
    widget.start_update()
    assert "Quelle nicht erreichbar" in widget.progress_label.text()
    assert widget.update_button.isEnabled()
    assert not widget.is_updating


def test_cancel_without_update_does_nothing(widget: DataWidget) -> None:
    widget.cancel_update()
    assert widget.progress_label.text() == texts.DATA_UPDATE_IDLE


def test_last_update_is_read_when_shown(
    qtbot: QtBot, widget: DataWidget, fake: FakeDataService
) -> None:
    fake.last = LastUpdate(STARTED, None, "yfinance", 0, 0, "läuft")
    widget.show()
    qtbot.waitExposed(widget)
    assert widget.last_update_label.text() == (
        "Letztes Update: 01.03.2024 18:05 · nicht abgeschlossen"
    )


def test_format_last_update() -> None:
    assert format_last_update(None) == texts.DATA_LAST_UPDATE_NONE
    entry = LastUpdate(STARTED, STARTED, "", 3, 1, "3 von 4 Symbolen aktualisiert.")
    assert format_last_update(entry).endswith("Quellen: – · 3 von 4 Symbolen aktualisiert.")


def test_update_runs_in_background_and_cancel_stops_after_current_symbol(
    qtbot: QtBot, service: AppService, dialogs: list[tuple[str, str]], tmp_path: Path
) -> None:
    fake = FakeDataService(tmp_path, block_at=1)
    window = MainWindow(service, data_service=fake)
    qtbot.addWidget(window)
    view = window.data
    assert window.tab_widget("data") is view

    view.update_button.click()
    # Das Update hängt im Hintergrund bei Symbol 2; die Ereignisschleife läuft weiter.
    qtbot.waitUntil(fake.blocked.is_set, timeout=5000)
    qtbot.waitUntil(lambda: view.progress_bar.value() == 25, timeout=5000)
    assert view.progress_label.text() == texts.DATA_UPDATE_PROGRESS.format(message="SPY (1/4)")
    assert view.is_updating
    assert not view.update_button.isEnabled()
    assert not view.quality_button.isEnabled()
    assert not view.import_button.isEnabled()
    assert view.cancel_button.isEnabled()

    # GUI bleibt bedienbar: Eingaben und Tabwechsel funktionieren, während das Update läuft.
    qtbot.keyClicks(view.key_edit, "abc")
    assert view.key_edit.text() == "abc"
    window.tabs.setCurrentWidget(window.chart)
    assert window.tabs.currentWidget() is window.chart

    view.cancel_button.click()
    assert view.progress_label.text() == texts.DATA_UPDATE_CANCELLING
    assert not view.cancel_button.isEnabled()

    with qtbot.waitSignal(view.update_finished, timeout=5000) as finished:
        fake.release.set()
    result = finished.args[0]
    assert result.cancelled
    # Das aktuelle Symbol (QQQ) wird noch fertig geladen, danach ist Schluss.
    assert fake.processed == ["SPY", "QQQ"]
    assert view.progress_label.text().startswith("Abgebrochen")
    assert view.update_button.isEnabled()
    assert not view.is_updating
    assert dialogs == []


def test_background_update_error_shows_dialog(
    qtbot: QtBot, service: AppService, dialogs: list[tuple[str, str]], tmp_path: Path
) -> None:
    fake = FakeDataService(tmp_path)
    fake.fail_update = True
    window = MainWindow(service, data_service=fake)
    qtbot.addWidget(window)
    window.data.start_update()
    qtbot.waitUntil(lambda: not window.data.is_updating, timeout=5000)
    assert "Quelle nicht erreichbar" in window.data.progress_label.text()
    qtbot.waitUntil(lambda: bool(dialogs), timeout=5000)
    assert "Quelle nicht erreichbar" in dialogs[0][1]


# ---------------------------------------------------------------------- Qualitätsbericht


def test_report_is_shown_and_filtered(widget: DataWidget) -> None:
    assert widget.table.columnCount() == len(COLUMNS)
    assert widget.quality_label.text() == texts.DATA_QUALITY_NOT_RUN
    loaded: list[pd.DataFrame] = []
    widget.report_loaded.connect(loaded.append)

    widget.quality_button.click()

    assert len(loaded) == 1
    assert widget.table.rowCount() == 3
    assert widget.table.item(0, COL_SYMBOL).text() == "AAA"
    assert widget.table.item(0, COL_DATE).text() == "03.01.2024"
    assert widget.table.item(0, COL_CHECK).text() == texts.DATA_CHECK_GAP
    assert widget.table.item(0, COL_SEVERITY).text() == texts.DATA_SEVERITY_ERROR
    assert widget.table.item(2, COL_DATE).text() == ""
    assert widget.quality_label.text() == "3 Befunde: 2 Fehler, 1 Warnungen"
    error_color = widget.table.item(0, COL_SEVERITY).foreground().color().name()
    warning_color = widget.table.item(1, COL_SEVERITY).foreground().color().name()
    assert error_color != warning_color

    widget.set_severity_filter(Severity.ERROR)
    assert widget.severity_filter() is Severity.ERROR
    assert widget.table.rowCount() == 2
    assert {widget.table.item(r, COL_SYMBOL).text() for r in range(2)} == {"AAA", "BBB"}
    assert widget.quality_label.text().endswith("2 angezeigt")

    widget.set_severity_filter(Severity.WARNING)
    assert widget.table.rowCount() == 1
    assert widget.table.item(0, COL_CHECK).text() == texts.DATA_CHECK_JUMP

    widget.set_severity_filter(None)
    assert widget.severity_filter() is None
    assert widget.table.rowCount() == 3
    assert len(widget.report) == 3


def test_clean_report(widget: DataWidget, fake: FakeDataService) -> None:
    fake.report = sample_report().iloc[0:0]
    widget.load_report()
    assert widget.table.rowCount() == 0
    assert widget.quality_label.text() == texts.DATA_QUALITY_CLEAN


def test_report_error_is_shown(widget: DataWidget, fake: FakeDataService) -> None:
    fake.fail_report = True
    widget.load_report()
    assert "Datenbank gesperrt" in widget.quality_label.text()
    assert widget.quality_button.isEnabled()


# ---------------------------------------------------------------------- Einstellungen


def provider_rows(widget: DataWidget) -> list[tuple[str, bool]]:
    rows = []
    for row in range(widget.provider_list.count()):
        item = widget.provider_list.item(row)
        rows.append((item.text(), item.checkState() == Qt.CheckState.Checked))
    return rows


def test_provider_order_and_key_are_saved(widget: DataWidget, fake: FakeDataService) -> None:
    assert provider_rows(widget) == [("yfinance", True), ("tiingo", True)]
    assert widget.key_edit.echoMode() == widget.key_edit.EchoMode.Password
    assert widget.key_edit.placeholderText() == texts.DATA_TIINGO_KEY_NONE
    assert not widget.clear_key_box.isEnabled()

    widget.provider_list.setCurrentRow(1)
    widget.up_button.click()
    assert widget.provider_order() == ["tiingo", "yfinance"]
    widget.up_button.click()  # schon oben: nichts passiert
    assert widget.provider_order() == ["tiingo", "yfinance"]

    widget.key_edit.setText("mein-tiingo-key-1234")
    widget.save_button.click()
    assert fake.saved == [(["tiingo", "yfinance"], "mein-tiingo-key-1234", False)]
    assert widget.settings_label.text() == texts.DATA_SETTINGS_SAVED
    assert widget.key_edit.text() == ""
    assert widget.key_edit.placeholderText() == "Gespeichert: ••••••••1234"
    assert "mein" not in widget.key_edit.placeholderText()
    assert widget.clear_key_box.isEnabled()

    widget.provider_list.setCurrentRow(0)
    widget.down_button.click()
    widget.provider_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    widget.clear_key_box.setChecked(True)
    widget.save_settings()
    assert fake.saved[-1] == (["tiingo"], "", True)
    assert provider_rows(widget) == [("tiingo", True), ("yfinance", False)]
    assert widget.key_edit.placeholderText() == texts.DATA_TIINGO_KEY_NONE


def test_saving_needs_one_provider(widget: DataWidget, fake: FakeDataService) -> None:
    for row in range(widget.provider_list.count()):
        widget.provider_list.item(row).setCheckState(Qt.CheckState.Unchecked)
    widget.save_settings()
    assert fake.saved == []
    assert widget.settings_label.text() == texts.DATA_SETTINGS_NO_PROVIDER


def test_saving_error_is_shown(widget: DataWidget, fake: FakeDataService) -> None:
    fake.fail_save = True
    widget.save_settings()
    assert "Datei schreibgeschützt" in widget.settings_label.text()


# ---------------------------------------------------------------------- CSV-Import


def test_csv_import_via_dialog(
    widget: DataWidget, fake: FakeDataService, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    chosen = tmp_path / "SYN.csv"
    answers: list[tuple[str, str]] = [("", ""), (str(chosen), "")]

    def ask(*_args: Any) -> tuple[str, str]:
        return answers.pop(0)

    monkeypatch.setattr(QFileDialog, "getOpenFileName", ask)
    imported: list[CsvImportReport] = []
    widget.csv_imported.connect(imported.append)

    widget.import_button.click()  # Dialog abgebrochen
    assert fake.imports == []

    widget.import_button.click()
    assert fake.imports == [chosen]
    assert len(imported) == 1
    assert widget.import_label.text() == (
        imported[0].summary() + " Abgelehnte Zeilen: SYN.csv.fehler.csv"
    )
    assert widget.import_button.isEnabled()


def test_csv_import_error_is_shown(
    widget: DataWidget, fake: FakeDataService, tmp_path: Path
) -> None:
    fake.fail_import = True
    widget.import_csv(tmp_path / "x.csv")
    assert "Spalte close fehlt" in widget.import_label.text()
    assert widget.import_button.isEnabled()


def test_busy_widget_ignores_new_jobs(widget: DataWidget, fake: FakeDataService) -> None:
    widget._set_busy(True)
    widget.start_update()
    widget.load_report()
    widget.import_csv(Path("x.csv"))
    widget.choose_and_import_csv()
    assert fake.update_calls == []
    assert fake.imports == []
    assert widget.quality_label.text() == texts.DATA_QUALITY_NOT_RUN


def test_index_labels_show_ampersand(widget: DataWidget) -> None:
    assert widget.index_boxes["SP500"].text() == "S&&P 500"
