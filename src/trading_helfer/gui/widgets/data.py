"""Daten-Tab: Datenupdate mit Fortschritt und Abbruch, Qualitätsbericht, Quellen, CSV-Import.

Das Update läuft über den übergebenen Runner im Hintergrund (``QThreadPool``), die
Oberfläche bleibt bedienbar. „Abbrechen“ setzt nur ein Flag, das das Update vor dem
nächsten Symbol abfragt; das aktuelle Symbol wird also noch fertig geladen.
"""

from __future__ import annotations

import datetime as dt
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Protocol

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from trading_helfer.core.config import ConfigError
from trading_helfer.data.providers.csv_import import CsvImportReport
from trading_helfer.data.quality import Check, Severity
from trading_helfer.data.update import UpdateResult
from trading_helfer.gui import texts
from trading_helfer.gui.widgets.chart import DOWN_COLOR
from trading_helfer.gui.worker import WorkerError
from trading_helfer.services.data_service import (
    INDEX_LABELS,
    DataService,
    LastUpdate,
    ProviderSettings,
    SeverityFilter,
    empty_report,
    filter_report,
)

__all__ = [
    "CHECK_TEXTS",
    "COLUMNS",
    "SEVERITY_TEXTS",
    "DataTaskRunner",
    "DataWidget",
    "format_last_update",
]

COLUMNS: Final[tuple[str, ...]] = (
    texts.DATA_COL_SYMBOL,
    texts.DATA_COL_DATE,
    texts.DATA_COL_CHECK,
    texts.DATA_COL_SEVERITY,
    texts.DATA_COL_DETAIL,
)
"""Spaltenüberschriften des Qualitätsberichts."""

COL_SYMBOL, COL_DATE, COL_CHECK, COL_SEVERITY, COL_DETAIL = range(5)

CHECK_TEXTS: Final[dict[str, str]] = {
    Check.GAP.value: texts.DATA_CHECK_GAP,
    Check.HIGH_LOW.value: texts.DATA_CHECK_HIGH_LOW,
    Check.NON_POSITIVE.value: texts.DATA_CHECK_NON_POSITIVE,
    Check.JUMP.value: texts.DATA_CHECK_JUMP,
    Check.MISSING_DATA.value: texts.DATA_CHECK_MISSING_DATA,
}
"""Deutsche Bezeichnung je Prüfung."""

SEVERITY_TEXTS: Final[dict[str, str]] = {
    Severity.ERROR.value: texts.DATA_SEVERITY_ERROR,
    Severity.WARNING.value: texts.DATA_SEVERITY_WARNING,
}
"""Deutsche Bezeichnung je Schwere."""

WARNING_COLOR: Final = "#f9a825"
_SEVERITY_COLORS: Final[dict[str, str]] = {
    Severity.ERROR.value: DOWN_COLOR,
    Severity.WARNING.value: WARNING_COLOR,
}
_FILTERS: Final[tuple[tuple[str, SeverityFilter], ...]] = (
    (texts.DATA_QUALITY_ALL, None),
    (texts.DATA_QUALITY_ERRORS, Severity.ERROR),
    (texts.DATA_QUALITY_WARNINGS, Severity.WARNING),
)
_PROVIDER_ROLE: Final = Qt.ItemDataRole.UserRole


class DataTaskRunner(Protocol):
    """Startet ``fn`` (z. B. im Thread-Pool) und meldet Ergebnis, Fehler und Fortschritt.

    :meth:`MainWindow.run_task` erfüllt dieses Protokoll.
    """

    def __call__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None] | None = ...,
        name: str = ...,
        report_progress: bool = ...,
        on_progress: Callable[[int, str], None] | None = ...,
        on_error: Callable[[WorkerError], None] | None = ...,
        **kwargs: Any,
    ) -> Any:
        """Startet die Aufgabe."""
        ...


def _run_inline(
    fn: Callable[..., Any],
    *args: Any,
    on_result: Callable[[Any], None] | None = None,
    name: str = "",
    report_progress: bool = False,
    on_progress: Callable[[int, str], None] | None = None,
    on_error: Callable[[WorkerError], None] | None = None,
    **kwargs: Any,
) -> None:
    """Führt ``fn`` sofort im GUI-Thread aus (Standard ohne Hauptfenster und in Tests)."""
    del name
    if report_progress:
        kwargs["progress"] = on_progress if on_progress is not None else (lambda _p, _m: None)
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        if on_error is None:
            raise
        on_error(WorkerError(exc, traceback.format_exc()))
        return
    if on_result is not None:
        on_result(result)


def _format_timestamp(value: dt.datetime) -> str:
    return value.strftime("%d.%m.%Y %H:%M")


def format_last_update(entry: LastUpdate | None) -> str:
    """Text für die Anzeige des letzten Updates aus ``update_log``."""
    if entry is None:
        return texts.DATA_LAST_UPDATE_NONE
    started = _format_timestamp(entry.started_at)
    if entry.finished_at is None:
        return texts.DATA_LAST_UPDATE_UNFINISHED.format(started=started)
    provider = ", ".join(p for p in entry.provider.split(",") if p) or "–"
    return texts.DATA_LAST_UPDATE.format(started=started, provider=provider, message=entry.message)


def _format_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%d.%m.%Y")


class DataWidget(QWidget):
    """Daten-Tab: Update starten und abbrechen, Qualitätsbericht, Quellen und CSV-Import.

    Args:
        service: Fassade für Update, Qualität, Import und Einstellungen.
        runner: Startet Aufgaben im Hintergrund, z. B. :meth:`MainWindow.run_task`;
            ohne Angabe läuft alles sofort im GUI-Thread.
    """

    update_finished = Signal(object)
    """Wird mit dem :class:`UpdateResult` gesendet, wenn ein Update endet (auch abgebrochen)."""

    report_loaded = Signal(object)
    """Wird mit dem vollständigen Qualitätsbericht (DataFrame) gesendet."""

    csv_imported = Signal(object)
    """Wird mit dem :class:`CsvImportReport` gesendet, wenn ein Import endet."""

    def __init__(
        self,
        service: DataService,
        runner: DataTaskRunner | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._runner: DataTaskRunner = runner if runner is not None else _run_inline
        self._cancel_event: threading.Event | None = None
        self._busy = False
        self._report = empty_report()
        self._last_update_loaded = False

        left = QVBoxLayout()
        left.addWidget(self._build_update_group())
        left.addWidget(self._build_quality_group(), 1)
        right = QVBoxLayout()
        right.addWidget(self._build_settings_group())
        right.addWidget(self._build_import_group())
        right.addStretch(1)

        content = QWidget()
        columns = QHBoxLayout(content)
        columns.setContentsMargins(4, 4, 4, 4)
        columns.addLayout(left, 3)
        columns.addLayout(right, 1)
        # Scrollbereich, damit der Tab die Mindestgröße des Hauptfensters nicht erhöht.
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)

        self._show_provider_settings(service.provider_settings())
        self._set_busy(False)

    # -- Aufbau ----------------------------------------------------------------------

    def _build_update_group(self) -> QGroupBox:
        group = QGroupBox(texts.DATA_UPDATE_GROUP)
        self.index_boxes: dict[str, QCheckBox] = {}
        indices = QHBoxLayout()
        for code, label in INDEX_LABELS.items():
            check = QCheckBox(label.replace("&", "&&"))  # & ist sonst ein Kürzel
            check.setChecked(True)
            self.index_boxes[code] = check
            indices.addWidget(check)
        indices.addStretch(1)

        self.update_button = QPushButton(texts.DATA_UPDATE_BUTTON)
        self.update_button.clicked.connect(self.start_update)
        self.cancel_button = QPushButton(texts.DATA_CANCEL_BUTTON)
        self.cancel_button.clicked.connect(self.cancel_update)
        indices.addWidget(self.update_button)
        indices.addWidget(self.cancel_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_label = QLabel(texts.DATA_UPDATE_IDLE)
        self.progress_label.setWordWrap(True)
        self.last_update_label = QLabel(texts.DATA_LAST_UPDATE_NONE)
        self.last_update_label.setWordWrap(True)
        self.last_update_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hint = QLabel(texts.DATA_UPDATE_BENCHMARKS)
        hint.setProperty("placeholder", True)

        box = QVBoxLayout(group)
        box.addLayout(indices)
        box.addWidget(hint)
        box.addWidget(self.progress_bar)
        box.addWidget(self.progress_label)
        box.addWidget(self.last_update_label)
        return group

    def _build_quality_group(self) -> QGroupBox:
        group = QGroupBox(texts.DATA_QUALITY_GROUP)
        self.quality_button = QPushButton(texts.DATA_QUALITY_BUTTON)
        self.quality_button.clicked.connect(self.load_report)
        self.severity_combo = QComboBox()
        for label, severity in _FILTERS:
            self.severity_combo.addItem(label, severity)
        self.severity_combo.currentIndexChanged.connect(lambda _i: self._show_report())
        self.quality_label = QLabel(texts.DATA_QUALITY_NOT_RUN)

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.quality_button)
        toolbar.addWidget(QLabel(texts.DATA_QUALITY_FILTER))
        toolbar.addWidget(self.severity_combo)
        toolbar.addWidget(self.quality_label, 1)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        for col in (COL_SYMBOL, COL_DATE, COL_CHECK, COL_SEVERITY):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_DETAIL, QHeaderView.ResizeMode.Stretch)

        box = QVBoxLayout(group)
        box.addLayout(toolbar)
        box.addWidget(self.table, 1)
        return group

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox(texts.DATA_SETTINGS_GROUP)
        self.provider_list = QListWidget()
        self.provider_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.provider_list.setMaximumHeight(90)
        self.up_button = QPushButton(texts.DATA_PROVIDER_UP)
        self.up_button.clicked.connect(lambda: self.move_provider(-1))
        self.down_button = QPushButton(texts.DATA_PROVIDER_DOWN)
        self.down_button.clicked.connect(lambda: self.move_provider(1))
        move = QHBoxLayout()
        move.addWidget(self.up_button)
        move.addWidget(self.down_button)

        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setClearButtonEnabled(True)
        self.clear_key_box = QCheckBox(texts.DATA_TIINGO_KEY_CLEAR)
        self.save_button = QPushButton(texts.DATA_SETTINGS_SAVE)
        self.save_button.clicked.connect(self.save_settings)
        self.settings_label = QLabel("")
        self.settings_label.setWordWrap(True)

        box = QVBoxLayout(group)
        box.addWidget(QLabel(texts.DATA_PROVIDERS_LABEL))
        box.addWidget(self.provider_list)
        box.addLayout(move)
        box.addWidget(QLabel(texts.DATA_TIINGO_KEY))
        box.addWidget(self.key_edit)
        box.addWidget(self.clear_key_box)
        box.addWidget(self.save_button)
        box.addWidget(self.settings_label)
        return group

    def _build_import_group(self) -> QGroupBox:
        group = QGroupBox(texts.DATA_IMPORT_GROUP)
        self.import_button = QPushButton(texts.DATA_IMPORT_BUTTON)
        self.import_button.clicked.connect(self.choose_and_import_csv)
        hint = QLabel(texts.DATA_IMPORT_HINT)
        hint.setWordWrap(True)
        hint.setProperty("placeholder", True)
        self.import_label = QLabel("")
        self.import_label.setWordWrap(True)
        box = QVBoxLayout(group)
        box.addWidget(self.import_button)
        box.addWidget(hint)
        box.addWidget(self.import_label)
        return group

    # -- Zustand ---------------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        """``True``, solange ein Update, eine Prüfung oder ein Import läuft."""
        return self._busy

    @property
    def is_updating(self) -> bool:
        """``True``, solange ein Datenupdate läuft."""
        return self._cancel_event is not None

    def _set_busy(self, busy: bool) -> None:
        # Update, Prüfung und Import nutzen dieselbe Datenbank; immer nur eine Aufgabe.
        self._busy = busy
        self.update_button.setEnabled(not busy)
        self.quality_button.setEnabled(not busy)
        self.import_button.setEnabled(not busy)
        self.cancel_button.setEnabled(self.is_updating and not self._cancel_event_set())

    def _cancel_event_set(self) -> bool:
        return self._cancel_event is not None and self._cancel_event.is_set()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 (Qt-Override)
        """Liest beim ersten Anzeigen das letzte Update."""
        super().showEvent(event)
        if not self._last_update_loaded:
            self._last_update_loaded = True
            self.refresh_last_update()

    # -- Datenupdate -----------------------------------------------------------------

    def selected_indices(self) -> list[str]:
        """Codes der angehakten Indizes in fester Reihenfolge."""
        return [code for code, box in self.index_boxes.items() if box.isChecked()]

    def start_update(self) -> None:
        """Startet das Update der gewählten Indizes im Hintergrund."""
        if self._busy:
            return
        codes = self.selected_indices()
        if not codes:
            self.progress_label.setText(texts.DATA_UPDATE_NO_INDEX)
            return
        self._cancel_event = threading.Event()
        self.progress_bar.setValue(0)
        self.progress_label.setText(texts.DATA_UPDATE_STARTED)
        self._set_busy(True)
        self._runner(
            self._service.run_update,
            codes,
            should_cancel=self._cancel_event.is_set,
            name=texts.DATA_UPDATE_TASK,
            report_progress=True,
            on_progress=self._on_progress,
            on_result=self._on_update_done,
            on_error=self._on_update_failed,
        )

    def cancel_update(self) -> None:
        """Bricht das laufende Update nach dem aktuellen Symbol ab."""
        if self._cancel_event is None or self._cancel_event.is_set():
            return
        self._cancel_event.set()
        self.cancel_button.setEnabled(False)
        self.progress_label.setText(texts.DATA_UPDATE_CANCELLING)

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        if not self._cancel_event_set():
            self.progress_label.setText(texts.DATA_UPDATE_PROGRESS.format(message=message))

    def _finish_update(self) -> None:
        self._cancel_event = None
        self._set_busy(False)

    def _on_update_done(self, result: UpdateResult) -> None:
        self._finish_update()
        if not result.cancelled:
            self.progress_bar.setValue(100)
        self.progress_label.setText(result.summary())
        self.refresh_last_update()
        self.update_finished.emit(result)

    def _on_update_failed(self, error: WorkerError) -> None:
        self._finish_update()
        self.progress_label.setText(texts.DATA_UPDATE_FAILED.format(message=error.message))
        self.refresh_last_update()

    def refresh_last_update(self) -> None:
        """Liest den letzten Eintrag aus ``update_log`` und zeigt ihn an."""
        self._runner(
            self._service.last_update,
            name=texts.DATA_LAST_UPDATE_TASK,
            on_result=self._show_last_update,
        )

    def _show_last_update(self, entry: LastUpdate | None) -> None:
        self.last_update_label.setText(format_last_update(entry))

    # -- Qualitätsbericht ------------------------------------------------------------

    def load_report(self) -> None:
        """Erstellt den Qualitätsbericht im Hintergrund."""
        if self._busy:
            return
        self._set_busy(True)
        self.quality_label.setText(texts.DATA_QUALITY_RUNNING)
        self._runner(
            self._service.quality_report,
            name=texts.DATA_QUALITY_TASK,
            on_result=self._on_report,
            on_error=self._on_report_failed,
        )

    def _on_report(self, report: pd.DataFrame) -> None:
        self._set_busy(False)
        self._report = report
        self._show_report()
        self.report_loaded.emit(report)

    def _on_report_failed(self, error: WorkerError) -> None:
        self._set_busy(False)
        self.quality_label.setText(texts.DATA_QUALITY_FAILED.format(message=error.message))

    def set_severity_filter(self, severity: SeverityFilter) -> None:
        """Zeigt nur Befunde mit dieser Schwere (``None`` = alle)."""
        index = self.severity_combo.findData(severity)
        self.severity_combo.setCurrentIndex(max(index, 0))

    def severity_filter(self) -> SeverityFilter:
        """Aktuell gewählter Schwere-Filter."""
        value = self.severity_combo.currentData()
        return Severity(value) if value is not None else None

    @property
    def report(self) -> pd.DataFrame:
        """Der zuletzt geladene, ungefilterte Qualitätsbericht."""
        return self._report

    def _show_report(self) -> None:
        shown = filter_report(self._report, self.severity_filter())
        self.table.setRowCount(len(shown))
        for row, record in enumerate(shown.itertuples(index=False)):
            severity = str(record.severity)
            values = (
                str(record.symbol),
                _format_date(record.date),
                CHECK_TEXTS.get(str(record.check), str(record.check)),
                SEVERITY_TEXTS.get(severity, severity),
                str(record.detail),
            )
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col == COL_SEVERITY and severity in _SEVERITY_COLORS:
                    item.setForeground(QBrush(QColor(_SEVERITY_COLORS[severity])))
                self.table.setItem(row, col, item)
        total = len(self._report)
        if total == 0:
            self.quality_label.setText(texts.DATA_QUALITY_CLEAN)
            return
        errors = int((self._report["severity"] == Severity.ERROR.value).sum())
        text = texts.DATA_QUALITY_SUMMARY.format(
            total=total, errors=errors, warnings=total - errors
        )
        if len(shown) != total:
            text += texts.DATA_QUALITY_SHOWN.format(shown=len(shown))
        self.quality_label.setText(text)

    # -- Einstellungen ---------------------------------------------------------------

    def _show_provider_settings(self, current: ProviderSettings) -> None:
        self.provider_list.clear()
        inactive = [p for p in current.available if p not in current.order]
        for name in (*current.order, *inactive):
            item = QListWidgetItem(name)
            item.setData(_PROVIDER_ROLE, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            active = name in current.order
            item.setCheckState(Qt.CheckState.Checked if active else Qt.CheckState.Unchecked)
            self.provider_list.addItem(item)
        self.key_edit.clear()
        self.clear_key_box.setChecked(False)
        self.clear_key_box.setEnabled(current.has_tiingo_key)
        self.key_edit.setPlaceholderText(
            texts.DATA_TIINGO_KEY_SAVED.format(masked=current.tiingo_key_masked)
            if current.has_tiingo_key
            else texts.DATA_TIINGO_KEY_NONE
        )

    def provider_order(self) -> list[str]:
        """Aktive Kursquellen in angezeigter Reihenfolge."""
        order: list[str] = []
        for row in range(self.provider_list.count()):
            item = self.provider_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                order.append(str(item.data(_PROVIDER_ROLE)))
        return order

    def move_provider(self, step: int) -> None:
        """Verschiebt die gewählte Quelle um ``step`` Plätze (−1 = nach oben)."""
        row = self.provider_list.currentRow()
        target = row + step
        if row < 0 or not 0 <= target < self.provider_list.count():
            return
        item = self.provider_list.takeItem(row)
        self.provider_list.insertItem(target, item)
        self.provider_list.setCurrentRow(target)

    def save_settings(self) -> None:
        """Speichert Reihenfolge, aktive Quellen und den Tiingo-Key in der ``config.toml``."""
        order = self.provider_order()
        if not order:
            self.settings_label.setText(texts.DATA_SETTINGS_NO_PROVIDER)
            return
        try:
            current = self._service.save_provider_settings(
                order, self.key_edit.text(), clear_key=self.clear_key_box.isChecked()
            )
        except ConfigError as exc:
            self.settings_label.setText(texts.DATA_SETTINGS_FAILED.format(message=exc))
            return
        self._show_provider_settings(current)
        self.settings_label.setText(texts.DATA_SETTINGS_SAVED)

    # -- CSV-Import ------------------------------------------------------------------

    def choose_and_import_csv(self) -> None:
        """Fragt nach einer CSV-Datei und importiert sie im Hintergrund."""
        if self._busy:
            return
        name, _filter = QFileDialog.getOpenFileName(
            self, texts.DATA_IMPORT_DIALOG, "", texts.DATA_IMPORT_FILTER
        )
        if name:
            self.import_csv(Path(name))

    def import_csv(self, path: Path) -> None:
        """Importiert ``path`` im Hintergrund und zeigt die Zusammenfassung."""
        if self._busy:
            return
        self._set_busy(True)
        self.import_label.setText(texts.DATA_IMPORT_RUNNING.format(name=path.name))
        self._runner(
            self._service.import_csv,
            path,
            name=texts.DATA_IMPORT_TASK,
            on_result=lambda report: self._on_imported(path, report),
            on_error=self._on_import_failed,
        )

    def _on_imported(self, path: Path, report: CsvImportReport) -> None:
        self._set_busy(False)
        text = report.summary()
        if report.errors:
            text += texts.DATA_IMPORT_ERRORS.format(name=path.name)
        self.import_label.setText(text)
        self.csv_imported.emit(report)

    def _on_import_failed(self, error: WorkerError) -> None:
        self._set_busy(False)
        self.import_label.setText(texts.DATA_IMPORT_FAILED.format(message=error.message))
