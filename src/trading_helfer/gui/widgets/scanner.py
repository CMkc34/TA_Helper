"""Scanner-Tab: Strategie, Universum, Zeitrahmen und Filter wählen, Signale auflisten.

Der Scan läuft über den übergebenen Runner im Hintergrund (``QThreadPool``) und meldet
Fortschritt. Die Ergebnisliste ist nach jeder Spalte sortierbar; ein Doppelklick sendet
:attr:`ScannerWidget.chart_requested`, worauf das Hauptfenster den Chart mit dem Signal
öffnet. Frühere Scans stehen im Verlauf und lassen sich wieder laden oder löschen.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd
from PySide6.QtCore import QDate, Qt, Signal
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from trading_helfer.core.types import SignalType, Timeframe
from trading_helfer.gui import texts
from trading_helfer.gui.widgets.data import DataTaskRunner, _run_inline
from trading_helfer.gui.worker import WorkerError
from trading_helfer.scanner.cli import parse_params
from trading_helfer.scanner.scan import RESULT_COLUMNS, ScanError, ScanFilters
from trading_helfer.services.scanner_service import (
    CUSTOM_UNIVERSE,
    INDEX_LABELS,
    METRIC_CHOICES,
    ScannerService,
    ScanRecord,
    ScanRequest,
    StrategyChoice,
    empty_result,
    parse_symbols,
    signal_marker,
)

__all__ = [
    "COLUMN_TITLES",
    "HISTORY_COLUMNS",
    "SIGNAL_TEXTS",
    "ScannerWidget",
    "format_number",
    "parse_param_text",
]

COLUMN_TITLES: Final[dict[str, str]] = {
    "symbol": texts.SCANNER_COL_SYMBOL,
    "date": texts.SCANNER_COL_DATE,
    "signal": texts.SCANNER_COL_SIGNAL,
    "strength": texts.SCANNER_COL_STRENGTH,
    "close": texts.SCANNER_COL_CLOSE,
    "sector": texts.SCANNER_COL_SECTOR,
    "market_cap": texts.SCANNER_COL_MARKET_CAP,
    "note": texts.SCANNER_COL_NOTE,
}
"""Überschrift je fester Ergebnisspalte; Indikatorspalten behalten ihren Namen."""

HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    texts.SCANNER_COL_CREATED,
    texts.SCANNER_COL_STRATEGY,
    texts.SCANNER_COL_UNIVERSE,
    texts.SCANNER_COL_TIMEFRAME,
    texts.SCANNER_COL_AS_OF,
    texts.SCANNER_COL_COUNT,
)
"""Spaltenüberschriften des Verlaufs."""

SIGNAL_TEXTS: Final[dict[str, str]] = {
    SignalType.BUY.value: texts.SCANNER_SIGNAL_BUY,
    SignalType.SELL.value: texts.SCANNER_SIGNAL_SELL,
    SignalType.EXIT.value: texts.SCANNER_SIGNAL_EXIT,
}
"""Deutsche Bezeichnung je Signalart."""

_SIGNAL_ORDER: Final[dict[str, int]] = {
    SignalType.BUY.value: 0,
    SignalType.SELL.value: 1,
    SignalType.EXIT.value: 2,
}
_ROW_ROLE: Final = Qt.ItemDataRole.UserRole
_PREFERRED_TIMEFRAME: Final = Timeframe.W1
_BILLION: Final = 1e9
_CAP_MAX: Final = 1e5  # Mrd. USD


def format_number(value: Any) -> str:
    """Zahl mit vier signifikanten Stellen und deutschem Dezimalkomma; leer bei NaN/None."""
    if value is None or isinstance(value, bool):
        return "" if value is None else str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.4g}".replace(".", ",")


def _format_market_cap(value: Any) -> str:
    number = _finite(value)
    if number is None:
        return ""
    return f"{number / _BILLION:.1f} Mrd.".replace(".", ",", 1)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_date(value: Any) -> str:
    if value is None or (not isinstance(value, dt.date) and pd.isna(value)):
        return ""
    return pd.Timestamp(value).strftime("%d.%m.%Y")


def parse_param_text(text: str) -> dict[str, Any]:
    """Wandelt ``NAME=WERT, NAME=WERT`` in Parameter (Zahlen und true/false erkannt).

    Raises:
        ValueError: Eintrag ohne ``=``.
    """
    items = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    return parse_params(items)


class _SortItem(QTableWidgetItem):
    """Tabellenzelle mit eigenem Sortierschlüssel; leere Werte stehen immer unten."""

    def __init__(self, text: str, key: Any) -> None:
        super().__init__(text)
        self.sort_key = key

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if not isinstance(other, _SortItem):
            return super().__lt__(other)
        mine, theirs = self.sort_key, other.sort_key
        if mine is None or theirs is None:
            # Leere Werte ans Ende, unabhängig von der Sortierrichtung.
            table = self.tableWidget()
            descending = (
                table is not None
                and table.horizontalHeader().sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
            )
            if mine is None and theirs is None:
                return False
            return (mine is None) == descending
        try:
            return bool(mine < theirs)
        except TypeError:
            return str(mine) < str(theirs)


def _cell(column: str, value: Any) -> _SortItem:
    if column == "date":
        text = _format_date(value)
        return _SortItem(text, pd.Timestamp(value).isoformat() if text else None)
    if column == "signal":
        signal = str(value) if value is not None else ""
        return _SortItem(SIGNAL_TEXTS.get(signal, signal), _SIGNAL_ORDER.get(signal, 9))
    if column == "market_cap":
        return _SortItem(_format_market_cap(value), _finite(value))
    if isinstance(value, str):
        return _SortItem(value, value.casefold())
    if value is None or pd.isna(value):
        return _SortItem("", None)
    number = _finite(value)
    if number is None:
        text = str(value)
        return _SortItem(text, text.casefold())
    item = _SortItem(format_number(number), number)
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


def _read_only_table(columns: Sequence[str]) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(list(columns))
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setAlternatingRowColors(True)
    return table


class ScannerWidget(QWidget):
    """Scanner-Tab: Scan einrichten und starten, Ergebnisliste, CSV-Export, Verlauf.

    Args:
        service: Fassade für Strategien, Scan, Verlauf und Export.
        runner: Startet Aufgaben im Hintergrund, z. B. :meth:`MainWindow.run_task`;
            ohne Angabe läuft alles sofort im GUI-Thread.
    """

    chart_requested = Signal(str, object, object)
    """Doppelklick auf ein Signal: Symbol, :class:`Timeframe` und :class:`Marker`."""

    scan_finished = Signal(object)
    """Wird mit dem Ergebnis (DataFrame) gesendet, wenn ein Scan endet."""

    def __init__(
        self,
        service: ScannerService,
        runner: DataTaskRunner | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._runner: DataTaskRunner = runner if runner is not None else _run_inline
        self._strategies: list[StrategyChoice] = []
        self._history: list[ScanRecord] = []
        self._result = empty_result()
        self._result_timeframe: Timeframe = _PREFERRED_TIMEFRAME
        self._busy = False
        self._loaded = False

        setup = QHBoxLayout()
        setup.addWidget(self._build_setup_group(), 3)
        setup.addWidget(self._build_filter_group(), 2)

        self.run_button = QPushButton(texts.SCANNER_RUN_BUTTON)
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.start_scan)
        self.export_button = QPushButton(texts.SCANNER_EXPORT_BUTTON)
        self.export_button.clicked.connect(self.ask_export)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel(texts.SCANNER_IDLE)
        self.status_label.setWordWrap(True)
        actions = QHBoxLayout()
        actions.addWidget(self.run_button)
        actions.addWidget(self.progress_bar, 1)
        actions.addWidget(self.export_button)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_result_box())
        splitter.addWidget(self._build_history_group())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        content = QWidget()
        box = QVBoxLayout(content)
        box.setContentsMargins(4, 4, 4, 4)
        box.addLayout(setup)
        box.addLayout(actions)
        box.addWidget(self.status_label)
        box.addWidget(splitter, 1)
        # Scrollbereich, damit der Tab die Mindestgröße des Hauptfensters nicht erhöht.
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)

        self._show_result(self._result)
        self._update_buttons()

    # -- Aufbau ----------------------------------------------------------------------

    def _build_setup_group(self) -> QGroupBox:
        group = QGroupBox(texts.SCANNER_SETUP_GROUP)
        self.strategy_combo = QComboBox()
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        self.timeframe_combo = QComboBox()

        self.universe_combo = QComboBox()
        for code, label in INDEX_LABELS.items():
            self.universe_combo.addItem(label, code)
        self.universe_combo.addItem(texts.SCANNER_UNIVERSE_CUSTOM, CUSTOM_UNIVERSE)
        self.universe_combo.currentIndexChanged.connect(self._on_universe_changed)
        self.symbols_edit = QLineEdit()
        self.symbols_edit.setPlaceholderText(texts.SCANNER_SYMBOLS_PLACEHOLDER)
        universe = QHBoxLayout()
        universe.addWidget(self.universe_combo)
        universe.addWidget(self.symbols_edit, 1)

        self.as_of_check = QCheckBox()
        self.as_of_check.setToolTip(texts.SCANNER_AS_OF_TOOLTIP)
        self.as_of_edit = QDateEdit(QDate.currentDate())
        self.as_of_edit.setCalendarPopup(True)
        self.as_of_edit.setDisplayFormat("dd.MM.yyyy")
        self.as_of_check.toggled.connect(self.as_of_edit.setEnabled)
        self.as_of_edit.setEnabled(False)
        latest = QLabel(texts.SCANNER_AS_OF_LATEST)
        latest.setProperty("placeholder", True)
        self.as_of_check.toggled.connect(lambda on: latest.setVisible(not on))
        as_of = QHBoxLayout()
        as_of.addWidget(self.as_of_check)
        as_of.addWidget(self.as_of_edit)
        as_of.addWidget(latest, 1)

        self.params_edit = QLineEdit()
        self.params_edit.setPlaceholderText(texts.SCANNER_PARAMS_PLACEHOLDER)

        form = QFormLayout(group)
        form.addRow(texts.SCANNER_STRATEGY, self.strategy_combo)
        form.addRow(texts.SCANNER_TIMEFRAME, self.timeframe_combo)
        form.addRow(texts.SCANNER_UNIVERSE, universe)
        form.addRow(texts.SCANNER_AS_OF, as_of)
        form.addRow(texts.SCANNER_PARAMS, self.params_edit)
        self._on_universe_changed()
        return group

    @staticmethod
    def _cap_box() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0, _CAP_MAX)
        box.setDecimals(1)
        box.setSpecialValueText("–")  # 0 = keine Grenze
        return box

    @staticmethod
    def _bound_edit(placeholder: str) -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setMaximumWidth(90)
        return edit

    def _build_filter_group(self) -> QGroupBox:
        group = QGroupBox(texts.SCANNER_FILTER_GROUP)
        self.sector_combo = QComboBox()
        self.sector_combo.addItem(texts.SCANNER_SECTOR_ALL, None)

        self.min_cap_box = self._cap_box()
        self.max_cap_box = self._cap_box()
        cap = QHBoxLayout()
        cap.addWidget(QLabel(texts.SCANNER_MIN))
        cap.addWidget(self.min_cap_box)
        cap.addWidget(QLabel(texts.SCANNER_MAX))
        cap.addWidget(self.max_cap_box)
        cap.addStretch(1)

        self.metric_combo = QComboBox()
        self.metric_combo.addItem(texts.SCANNER_METRIC_NONE, None)
        for name, label in METRIC_CHOICES.items():
            self.metric_combo.addItem(label, name)
        self.metric_min_edit = self._bound_edit(texts.SCANNER_MIN)
        self.metric_max_edit = self._bound_edit(texts.SCANNER_MAX)
        metric = QHBoxLayout()
        metric.addWidget(self.metric_combo, 1)
        metric.addWidget(self.metric_min_edit)
        metric.addWidget(self.metric_max_edit)

        self.signal_boxes: dict[SignalType, QCheckBox] = {}
        signals = QHBoxLayout()
        for signal in SignalType:
            check = QCheckBox(SIGNAL_TEXTS[signal.value])
            check.setChecked(True)
            self.signal_boxes[signal] = check
            signals.addWidget(check)
        signals.addStretch(1)

        form = QFormLayout(group)
        form.addRow(texts.SCANNER_SECTOR, self.sector_combo)
        form.addRow(texts.SCANNER_MARKET_CAP, cap)
        form.addRow(texts.SCANNER_METRIC, metric)
        form.addRow(texts.SCANNER_SIGNALS, signals)
        return group

    def _build_result_box(self) -> QWidget:
        self.table = _read_only_table([COLUMN_TITLES[c] for c in RESULT_COLUMNS])
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(lambda row, _col: self.open_row(row))
        hint = QLabel(texts.SCANNER_RESULT_HINT)
        hint.setProperty("placeholder", True)
        widget = QWidget()
        box = QVBoxLayout(widget)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.table, 1)
        box.addWidget(hint)
        return widget

    def _build_history_group(self) -> QGroupBox:
        group = QGroupBox(texts.SCANNER_HISTORY_GROUP)
        self.history_table = _read_only_table(HISTORY_COLUMNS)
        header = self.history_table.horizontalHeader()
        for col in range(len(HISTORY_COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.history_table.cellDoubleClicked.connect(lambda row, _col: self.load_history(row))
        self.history_table.itemSelectionChanged.connect(self._update_buttons)
        self.history_label = QLabel(texts.SCANNER_HISTORY_EMPTY)
        self.history_label.setProperty("placeholder", True)
        self.load_button = QPushButton(texts.SCANNER_HISTORY_LOAD)
        self.load_button.clicked.connect(lambda: self.load_history(self._history_row()))
        self.delete_button = QPushButton(texts.SCANNER_HISTORY_DELETE)
        self.delete_button.clicked.connect(lambda: self.delete_history(self._history_row()))
        buttons = QHBoxLayout()
        buttons.addWidget(self.history_label, 1)
        buttons.addWidget(self.load_button)
        buttons.addWidget(self.delete_button)
        box = QVBoxLayout(group)
        box.addWidget(self.history_table, 1)
        box.addLayout(buttons)
        return group

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 (Qt-Override)
        """Lädt beim ersten Anzeigen Strategien, Sektoren und Verlauf."""
        super().showEvent(event)
        if not self._loaded:
            self.refresh()

    # -- Auswahl ---------------------------------------------------------------------

    def refresh(self) -> None:
        """Lädt Strategien, Sektoren und Verlauf neu (im Hintergrund)."""
        self._loaded = True
        self._runner(
            self._load_choices,
            on_result=self._show_choices,
            on_error=self._on_failed,
            name=texts.SCANNER_HISTORY_TASK,
        )

    def _load_choices(self) -> tuple[list[StrategyChoice], list[str], list[ScanRecord]]:
        return self._service.strategies(), self._service.sectors(), self._service.history()

    def _show_choices(
        self, result: tuple[list[StrategyChoice], list[str], list[ScanRecord]]
    ) -> None:
        strategies, sectors, history = result
        self.set_strategies(strategies)
        self.set_sectors(sectors)
        self.set_history(history)

    def set_strategies(self, strategies: Sequence[StrategyChoice]) -> None:
        """Füllt die Strategieauswahl; die bisherige Wahl bleibt nach Möglichkeit erhalten."""
        current = self.strategy_combo.currentData()
        self._strategies = list(strategies)
        self.strategy_combo.blockSignals(True)
        self.strategy_combo.clear()
        for choice in self._strategies:
            self.strategy_combo.addItem(choice.name, choice.id)
        index = self.strategy_combo.findData(current)
        self.strategy_combo.setCurrentIndex(max(index, 0) if self._strategies else -1)
        self.strategy_combo.blockSignals(False)
        self._on_strategy_changed()
        if not self._strategies:
            self.status_label.setText(texts.SCANNER_NO_STRATEGY)

    def set_sectors(self, sectors: Sequence[str]) -> None:
        """Füllt den Sektorfilter („Alle Sektoren“ bleibt oben)."""
        current = self.sector_combo.currentData()
        self.sector_combo.clear()
        self.sector_combo.addItem(texts.SCANNER_SECTOR_ALL, None)
        for sector in sectors:
            self.sector_combo.addItem(sector, sector)
        self.sector_combo.setCurrentIndex(max(self.sector_combo.findData(current), 0))

    def selected_strategy(self) -> StrategyChoice | None:
        """Gewählte Strategie oder ``None``."""
        index = self.strategy_combo.currentIndex()
        return self._strategies[index] if 0 <= index < len(self._strategies) else None

    def selected_timeframe(self) -> Timeframe | None:
        """Gewählter Zeitrahmen oder ``None``."""
        value = self.timeframe_combo.currentData()
        return Timeframe(value) if value else None

    def _on_strategy_changed(self, _index: int = 0) -> None:
        choice = self.selected_strategy()
        current = self.selected_timeframe()
        self.timeframe_combo.clear()
        if choice is not None:
            for tf in choice.timeframes:
                self.timeframe_combo.addItem(tf.value, tf)
            preferred = current if current in choice.timeframes else _PREFERRED_TIMEFRAME
            self.timeframe_combo.setCurrentIndex(max(self.timeframe_combo.findData(preferred), 0))
            defaults = ", ".join(f"{k}={v}" for k, v in choice.defaults.items())
            self.params_edit.setToolTip(texts.SCANNER_PARAMS_TOOLTIP.format(defaults=defaults))
        self._update_buttons()

    def _on_universe_changed(self, _index: int = 0) -> None:
        custom = self.universe_combo.currentData() == CUSTOM_UNIVERSE
        self.symbols_edit.setVisible(custom)
        self.symbols_edit.setEnabled(custom)

    # -- Scan ------------------------------------------------------------------------

    def build_request(self) -> ScanRequest:
        """Liest die Eingaben als :class:`ScanRequest`.

        Raises:
            ScanError: Unvollständige oder ungültige Eingabe (Text auf Deutsch).
        """
        choice = self.selected_strategy()
        if choice is None:
            raise ScanError(texts.SCANNER_NO_STRATEGY)
        timeframe = self.selected_timeframe()
        if timeframe is None:
            raise ScanError(texts.SCANNER_NO_STRATEGY)
        code = self.universe_combo.currentData()
        universe: str | tuple[str, ...]
        if code == CUSTOM_UNIVERSE:
            universe = parse_symbols(self.symbols_edit.text())
            if not universe:
                raise ScanError("Bitte mindestens ein Symbol eingeben.")
        else:
            universe = str(code)
        as_of = self.as_of_edit.date().toPython() if self.as_of_check.isChecked() else None
        try:
            params = parse_param_text(self.params_edit.text())
        except ValueError as exc:
            raise ScanError(str(exc)) from exc
        return ScanRequest(
            strategy_id=choice.id,
            universe=universe,
            timeframe=timeframe,
            as_of=as_of if isinstance(as_of, dt.date) else None,
            params=params,
            filters=self.build_filters(),
        )

    def build_filters(self) -> ScanFilters:
        """Liest die Filter als :class:`ScanFilters`.

        Raises:
            ScanError: Grenze ist keine Zahl oder Untergrenze über Obergrenze.
        """
        sector = self.sector_combo.currentData()
        min_cap = self.min_cap_box.value() * _BILLION or None
        max_cap = self.max_cap_box.value() * _BILLION or None
        metrics: dict[str, tuple[float | None, float | None]] = {}
        metric = self.metric_combo.currentData()
        if metric is not None:
            low = _parse_bound(self.metric_min_edit.text())
            high = _parse_bound(self.metric_max_edit.text())
            if low is not None or high is not None:
                metrics[str(metric)] = (low, high)
        chosen = frozenset(s for s, box in self.signal_boxes.items() if box.isChecked())
        return ScanFilters(
            sectors=frozenset({str(sector)}) if sector else None,
            min_market_cap=min_cap,
            max_market_cap=max_cap,
            metrics=metrics,
            signals=None if len(chosen) == len(SignalType) else chosen,
        )

    def start_scan(self) -> None:
        """Startet den Scan mit den aktuellen Eingaben im Hintergrund."""
        if self._busy:
            return
        try:
            request = self.build_request()
        except ScanError as exc:
            self.status_label.setText(texts.SCANNER_INVALID.format(message=exc))
            return
        self._busy = True
        self._update_buttons()
        self.progress_bar.setValue(0)
        self.status_label.setText(texts.SCANNER_RUNNING)
        self._runner(
            self._service.run,
            request,
            on_result=lambda result: self._on_scanned(request, result),
            on_error=self._on_failed,
            name=texts.SCANNER_TASK,
            report_progress=True,
            on_progress=self._on_progress,
        )

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.status_label.setText(texts.SCANNER_PROGRESS.format(message=message))

    def _on_scanned(self, request: ScanRequest, result: pd.DataFrame) -> None:
        self._busy = False
        self.progress_bar.setValue(100)
        self._result_timeframe = request.timeframe
        self._show_result(result)
        universe = INDEX_LABELS.get(request.universe_text, request.universe_text)
        strategy = self.strategy_combo.currentText()
        if result.empty:
            text = texts.SCANNER_DONE_NONE.format(
                strategy=strategy, universe=universe, timeframe=request.timeframe.value
            )
        else:
            text = texts.SCANNER_DONE.format(
                count=len(result),
                date=_format_date(result["date"].max()),
                strategy=strategy,
                universe=universe,
                timeframe=request.timeframe.value,
            )
        skipped = result.attrs.get("skipped") or {}
        if skipped:
            text += texts.SCANNER_SKIPPED.format(count=len(skipped))
        self.status_label.setText(text)
        self._update_buttons()
        self.scan_finished.emit(result)
        self.refresh_history()

    def _on_failed(self, error: WorkerError) -> None:
        self._busy = False
        self.status_label.setText(texts.SCANNER_FAILED.format(message=error.message))
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.run_button.setEnabled(not self._busy and self.selected_strategy() is not None)
        self.export_button.setEnabled(not self._busy and not self._result.empty)
        has_history = self._history_row() >= 0
        self.load_button.setEnabled(not self._busy and has_history)
        self.delete_button.setEnabled(not self._busy and has_history)

    # -- Ergebnis --------------------------------------------------------------------

    def result(self) -> pd.DataFrame:
        """Das angezeigte Scan-Ergebnis (unsortiert, wie vom Service geliefert)."""
        return self._result

    def result_timeframe(self) -> Timeframe:
        """Zeitrahmen des angezeigten Ergebnisses."""
        return self._result_timeframe

    def _show_result(self, result: pd.DataFrame) -> None:
        self._result = result.reset_index(drop=True)
        columns = list(self._result.columns) if len(self._result.columns) else list(RESULT_COLUMNS)
        table = self.table
        table.setSortingEnabled(False)
        table.clearContents()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels([COLUMN_TITLES.get(c, c) for c in columns])
        table.setRowCount(len(self._result))
        for row, record in enumerate(self._result.itertuples(index=False)):
            for col, (column, value) in enumerate(zip(columns, record, strict=True)):
                item = _cell(column, value)
                if col == 0:
                    item.setData(_ROW_ROLE, row)
                table.setItem(row, col, item)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        if "note" in columns:
            header.setSectionResizeMode(columns.index("note"), QHeaderView.ResizeMode.Stretch)
        table.setSortingEnabled(True)
        header.setSortIndicator(-1, Qt.SortOrder.AscendingOrder)

    def result_row(self, view_row: int) -> pd.Series[Any] | None:
        """Ergebniszeile hinter der angezeigten Zeile ``view_row`` (auch nach Sortieren)."""
        item = self.table.item(view_row, 0)
        if item is None:
            return None
        index = item.data(_ROW_ROLE)
        if not isinstance(index, int) or not 0 <= index < len(self._result):
            return None
        return self._result.iloc[index]

    def visible_symbols(self) -> list[str]:
        """Symbole in der angezeigten Reihenfolge (nach Sortierung)."""
        symbols: list[str] = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            symbols.append(item.text() if item is not None else "")
        return symbols

    def open_row(self, view_row: int) -> None:
        """Sendet :attr:`chart_requested` für die angezeigte Zeile ``view_row``."""
        record = self.result_row(view_row)
        if record is None:
            return
        self.chart_requested.emit(
            str(record["symbol"]),
            self._result_timeframe,
            signal_marker({str(k): v for k, v in record.items()}),
        )

    # -- Export ----------------------------------------------------------------------

    def ask_export(self) -> None:
        """Fragt nach einem Dateinamen und speichert das Ergebnis als CSV."""
        if self._result.empty:
            return
        name, _filter = QFileDialog.getSaveFileName(
            self, texts.SCANNER_EXPORT_DIALOG, "scan.csv", texts.SCANNER_EXPORT_FILTER
        )
        if name:
            self.export_csv(Path(name))

    def export_csv(self, path: Path) -> Path | None:
        """Speichert das angezeigte Ergebnis als CSV; meldet Erfolg oder Fehler im Status."""
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        try:
            written = self._service.export_csv(self._result, path)
        except OSError as exc:
            self.status_label.setText(texts.SCANNER_EXPORT_FAILED.format(message=exc))
            return None
        self.status_label.setText(texts.SCANNER_EXPORT_DONE.format(path=written))
        return written

    # -- Verlauf ---------------------------------------------------------------------

    def refresh_history(self) -> None:
        """Lädt den Verlauf früherer Scans neu (im Hintergrund)."""
        self._runner(
            self._service.history,
            on_result=self.set_history,
            on_error=self._on_failed,
            name=texts.SCANNER_HISTORY_TASK,
        )

    def history(self) -> list[ScanRecord]:
        """Angezeigte frühere Scans, neueste zuerst."""
        return list(self._history)

    def set_history(self, records: Sequence[ScanRecord]) -> None:
        """Zeigt die früheren Scans."""
        self._history = list(records)
        names = {c.id: c.name for c in self._strategies}
        table = self.history_table
        table.setRowCount(len(self._history))
        for row, record in enumerate(self._history):
            values = (
                record.created_at.strftime("%d.%m.%Y %H:%M"),
                names.get(record.strategy_id, record.strategy_id),
                record.universe_label,
                record.timeframe.value,
                _format_date(record.as_of) or texts.SCANNER_AS_OF_LATEST,
                str(record.signal_count),
            )
            for col, text in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(text))
        self.history_label.setVisible(not self._history)
        if self._history and self._history_row() < 0:
            table.selectRow(0)
        self._update_buttons()

    def _history_row(self) -> int:
        rows = self.history_table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def load_history(self, row: int) -> None:
        """Lädt die Signale des früheren Scans in Zeile ``row`` in die Ergebnisliste."""
        if not 0 <= row < len(self._history) or self._busy:
            return
        record = self._history[row]
        self._runner(
            self._service.load,
            record.scan_id,
            on_result=lambda frame: self._on_history_loaded(record, frame),
            on_error=self._on_failed,
            name=texts.SCANNER_HISTORY_TASK,
        )

    def _on_history_loaded(self, record: ScanRecord, frame: pd.DataFrame) -> None:
        self._result_timeframe = record.timeframe
        self._show_result(frame)
        names = {c.id: c.name for c in self._strategies}
        self.status_label.setText(
            texts.SCANNER_LOADED.format(
                created=record.created_at.strftime("%d.%m.%Y %H:%M"),
                strategy=names.get(record.strategy_id, record.strategy_id),
                universe=record.universe_label,
                timeframe=record.timeframe.value,
                count=len(frame),
            )
        )
        self._update_buttons()

    def delete_history(self, row: int) -> None:
        """Löscht den früheren Scan in Zeile ``row`` und lädt den Verlauf neu."""
        if not 0 <= row < len(self._history) or self._busy:
            return
        self._runner(
            self._service.delete,
            self._history[row].scan_id,
            on_result=lambda _r: self.refresh_history(),
            on_error=self._on_failed,
            name=texts.SCANNER_HISTORY_TASK,
        )


def _parse_bound(text: str) -> float | None:
    text = text.strip().replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ScanError(f"Grenze {text!r} ist keine Zahl.") from exc
