"""Strategien-Tab: Liste aller Strategien, Details und Dialog „Strategie hinzufügen“.

Die Liste zeigt eingebaute Strategien und Plugins mit Name, Version, Zeitrahmen,
Herkunft und Status; Plugin-Dateien mit Ladefehlern erscheinen rot. Rechts stehen
Beschreibung und Parameter der gewählten Strategie. „Im Backtest öffnen“ sendet
:attr:`StrategiesWidget.open_in_backtest` mit der Strategie-ID.

Der Dialog :class:`AddStrategyDialog` liest eine Datei oder eingefügten Code, lässt
ihn im Hintergrund vom Validator prüfen, zeigt den Bericht und speichert erst nach
bestandener Prüfung in den Plugin-Ordner (bei vorhandener ID mit Rückfrage).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QFontDatabase, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from trading_helfer.gui import texts
from trading_helfer.gui.widgets.chart import DOWN_COLOR, UP_COLOR, TaskRunner
from trading_helfer.services.strategy_service import (
    BACKUP_SUFFIX,
    SaveResult,
    StrategyDetails,
    StrategyExistsError,
    StrategyRow,
    StrategySaveError,
    StrategyService,
)
from trading_helfer.strategies.validator import ValidationReport

__all__ = ["COLUMNS", "AddStrategyDialog", "StrategiesWidget"]

COLUMNS: Final[tuple[str, ...]] = (
    texts.STRATEGIES_COL_NAME,
    texts.STRATEGIES_COL_VERSION,
    texts.STRATEGIES_COL_TIMEFRAMES,
    texts.STRATEGIES_COL_ORIGIN,
    texts.STRATEGIES_COL_STATUS,
)
"""Spaltenüberschriften der Strategienliste."""

COL_NAME, COL_VERSION, COL_TIMEFRAMES, COL_ORIGIN, COL_STATUS = range(5)
_KEY_ROLE: Final = Qt.ItemDataRole.UserRole


def _run_inline(
    fn: Callable[..., Any],
    *args: Any,
    on_result: Callable[[Any], None] | None = None,
    name: str = "",
) -> None:
    del name
    result = fn(*args)
    if on_result is not None:
        on_result(result)


def _monospace() -> QFont:
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)


def _readonly_table(columns: tuple[str, ...]) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(list(columns))
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


class StrategiesWidget(QWidget):
    """Liste, Details und Hinzufügen von Strategien.

    Args:
        service: Liefert Liste und Details, prüft und speichert Plugins.
        runner: Startet lange Aufgaben, z. B. :meth:`MainWindow.run_task`; ohne Angabe
            läuft alles sofort im GUI-Thread.
        load_on_show: Liste erst laden, wenn der Tab zum ersten Mal sichtbar wird (spart
            Zeit beim Start); sonst sofort im Konstruktor.
    """

    open_in_backtest = Signal(str)
    """Wird mit der Strategie-ID gesendet, wenn „Im Backtest öffnen“ gedrückt wird."""

    strategies_changed = Signal()
    """Wird gesendet, nachdem die Liste neu geladen wurde."""

    def __init__(
        self,
        service: StrategyService,
        runner: TaskRunner | None = None,
        parent: QWidget | None = None,
        *,
        load_on_show: bool = False,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._runner: TaskRunner = runner if runner is not None else _run_inline
        self._loaded = False
        self._rows: list[StrategyRow] = []
        self._pending_select: str | None = None
        self.dialog: AddStrategyDialog | None = None

        self.add_button = QPushButton(texts.STRATEGIES_ADD)
        self.add_button.clicked.connect(self.open_add_dialog)
        self.reload_button = QPushButton(texts.STRATEGIES_RELOAD)
        self.reload_button.setToolTip(texts.STRATEGIES_RELOAD_TOOLTIP)
        self.reload_button.clicked.connect(lambda: self.refresh(reload=True))
        self.backtest_button = QPushButton(texts.STRATEGIES_OPEN_BACKTEST)
        self.backtest_button.setEnabled(False)
        self.backtest_button.clicked.connect(self._emit_open_in_backtest)
        self.summary_label = QLabel()

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.add_button)
        toolbar.addWidget(self.reload_button)
        toolbar.addWidget(self.backtest_button)
        toolbar.addStretch(1)

        self.table = _readonly_table(COLUMNS)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.doubleClicked.connect(lambda _index: self._emit_open_in_backtest())

        self.header_label = QLabel(texts.STRATEGIES_NONE_SELECTED)
        header_font = self.header_label.font()
        header_font.setBold(True)
        header_font.setPointSizeF(header_font.pointSizeF() * 1.2)
        self.header_label.setFont(header_font)
        # Lange Texte umbrechen, damit der Tab die Mindestbreite des Fensters nicht erhöht.
        self.header_label.setWordWrap(True)
        self.meta_label = QLabel()
        self.meta_label.setWordWrap(True)
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.params_table = _readonly_table(texts.STRATEGIES_PARAM_COLUMNS)
        self.params_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.no_params_label = QLabel(texts.STRATEGIES_NO_PARAMS)
        self.file_label = QLabel()
        self.file_label.setWordWrap(True)
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.error_view = QPlainTextEdit()
        self.error_view.setReadOnly(True)
        self.error_view.setFont(_monospace())

        details = QWidget()
        detail_layout = QVBoxLayout(details)
        detail_layout.addWidget(self.header_label)
        detail_layout.addWidget(self.meta_label)
        detail_layout.addWidget(self.description_label)
        detail_layout.addWidget(self.params_table, 1)
        detail_layout.addWidget(self.no_params_label)
        detail_layout.addWidget(self.error_view, 1)
        detail_layout.addWidget(self.file_label)

        # Anzahl unter der Liste statt in der Knopfleiste: hält die Mindestbreite klein.
        listing = QWidget()
        list_layout = QVBoxLayout(listing)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(self.table, 1)
        list_layout.addWidget(self.summary_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(listing)
        splitter.addWidget(details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(splitter, 1)

        self._show_details(None)
        if not load_on_show:
            self.refresh()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 (Qt-Override)
        """Lädt die Liste beim ersten Anzeigen, falls noch nicht geschehen."""
        super().showEvent(event)
        if not self._loaded:
            self.refresh()

    # -- Öffentliche API ---------------------------------------------------------------

    def rows(self) -> list[StrategyRow]:
        """Die angezeigten Zeilen in Tabellenreihenfolge."""
        return list(self._rows)

    def selected_key(self) -> str | None:
        """ID (oder Dateipfad) der gewählten Zeile."""
        row = self.table.currentRow()
        if row < 0 or not self.table.selectedItems():
            return None
        item = self.table.item(row, COL_NAME)
        return None if item is None else str(item.data(_KEY_ROLE))

    def select(self, key: str) -> bool:
        """Wählt die Zeile ``key``; ``False``, wenn es sie nicht gibt."""
        for index, row in enumerate(self._rows):
            if row.key == key:
                self.table.selectRow(index)
                return True
        return False

    def refresh(self, *, reload: bool = False, select: str | None = None) -> None:
        """Lädt die Liste (mit ``reload`` auch den Plugin-Ordner neu) im Hintergrund."""
        self._loaded = True
        self._pending_select = select if select is not None else self.selected_key()
        self._runner(
            self._load_rows, reload, on_result=self._show_rows, name=texts.STRATEGIES_TASK_LOAD
        )

    def open_add_dialog(self) -> AddStrategyDialog:
        """Öffnet den Dialog „Strategie hinzufügen“ (fensterbezogen modal)."""
        dialog = AddStrategyDialog(self._service, runner=self._runner, parent=self)
        dialog.saved.connect(lambda result: self.refresh(select=result.strategy_ids[0]))
        self.dialog = dialog
        dialog.open()
        return dialog

    # -- Laden und Anzeigen ------------------------------------------------------------

    def _load_rows(self, reload: bool) -> list[StrategyRow]:
        if reload:
            self._service.reload()
        return self._service.list_strategies()

    def _show_rows(self, rows: list[StrategyRow]) -> None:
        self._rows = list(rows)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        error_brush = QBrush(QColor(DOWN_COLOR))
        for index, row in enumerate(rows):
            values = (row.name, row.version, row.timeframes, str(row.origin), str(row.status))
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == COL_NAME:
                    item.setData(_KEY_ROLE, row.key)
                if not row.ok:
                    item.setForeground(error_brush)
                    item.setToolTip(row.message)
                self.table.setItem(index, column, item)
        loaded = sum(1 for r in rows if r.ok)
        self.summary_label.setText(texts.STRATEGIES_SUMMARY.format(count=loaded))
        self.summary_label.setToolTip(
            texts.STRATEGIES_SUMMARY_TOOLTIP.format(folder=self._service.plugins_dir)
        )
        target = self._pending_select
        self._pending_select = None
        if target is None or not self.select(target):
            self.table.clearSelection()
            self._show_details(None)
        self.strategies_changed.emit()

    def _on_selection_changed(self) -> None:
        key = self.selected_key()
        if key is None:
            self._show_details(None)
            return
        try:
            details = self._service.details(key)
        except KeyError:
            self._show_details(None)
            return
        self._show_details(details)

    def _show_details(self, details: StrategyDetails | None) -> None:
        loaded = details is not None and details.row.ok
        self.backtest_button.setEnabled(loaded)
        self.error_view.setVisible(details is not None and not details.row.ok)
        self.params_table.setVisible(loaded)
        self.no_params_label.setVisible(False)
        if details is None:
            self.header_label.setText(texts.STRATEGIES_NONE_SELECTED)
            for label in (self.meta_label, self.description_label, self.file_label):
                label.clear()
            self.params_table.setRowCount(0)
            return
        row = details.row
        path = (
            "" if details.path is None else texts.STRATEGIES_DETAIL_FILE.format(path=details.path)
        )
        self.file_label.setText(path)
        if not row.ok:
            self.header_label.setText(row.name)
            self.meta_label.setText(texts.STRATEGIES_DETAIL_ERROR)
            self.description_label.clear()
            self.error_view.setPlainText(row.message)
            self.params_table.setRowCount(0)
            return
        self.header_label.setText(
            texts.STRATEGIES_DETAIL_HEADER.format(name=row.name, version=row.version)
        )
        self.meta_label.setText(
            texts.STRATEGIES_DETAIL_META.format(
                id=row.key, timeframes=row.timeframes, origin=row.origin
            )
        )
        self.description_label.setText(details.description)
        self.params_table.setRowCount(len(details.params))
        for index, param in enumerate(details.params):
            values = (
                param.name,
                param.default,
                param.minimum,
                param.maximum,
                param.step,
                param.description,
            )
            for column, value in enumerate(values):
                self.params_table.setItem(index, column, QTableWidgetItem(value))
        self.params_table.setVisible(bool(details.params))
        self.no_params_label.setVisible(not details.params)

    def _emit_open_in_backtest(self) -> None:
        key = self.selected_key()
        if key is not None and key in self._service.registry:
            self.open_in_backtest.emit(key)


class AddStrategyDialog(QDialog):
    """Dialog: Datei wählen oder Code einfügen, prüfen, Bericht ansehen, speichern.

    Args:
        service: Prüft und speichert die Strategie.
        runner: Startet die Prüfung im Hintergrund; ohne Angabe sofort.
    """

    saved = Signal(object)
    """Wird mit dem :class:`SaveResult` gesendet, nachdem gespeichert wurde."""

    def __init__(
        self,
        service: StrategyService,
        runner: TaskRunner | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._runner: TaskRunner = runner if runner is not None else _run_inline
        self._filename = "<eingefügt>"
        self._report: ValidationReport | None = None
        self._checked_code: str | None = None
        self._checking = False
        self.result_saved: SaveResult | None = None

        self.setWindowTitle(texts.ADD_STRATEGY_TITLE)
        self.resize(900, 700)

        hint = QLabel(texts.ADD_STRATEGY_HINT)
        hint.setWordWrap(True)
        self.open_button = QPushButton(texts.ADD_STRATEGY_OPEN)
        self.open_button.clicked.connect(self.choose_file)
        self.file_label = QLabel()

        self.editor = QPlainTextEdit()
        self.editor.setFont(_monospace())
        self.editor.setPlaceholderText(texts.ADD_STRATEGY_PLACEHOLDER)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.textChanged.connect(self._on_code_changed)

        self.status_label = QLabel(texts.ADD_STRATEGY_NOT_CHECKED)
        self.status_label.setWordWrap(True)
        self.report_view = QPlainTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setFont(_monospace())

        self.validate_button = QPushButton(texts.ADD_STRATEGY_VALIDATE)
        self.validate_button.clicked.connect(self.validate)
        self.save_button = QPushButton(texts.ADD_STRATEGY_SAVE)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save)
        self.cancel_button = QPushButton(texts.ADD_STRATEGY_CANCEL)
        self.cancel_button.clicked.connect(self.reject)

        top = QHBoxLayout()
        top.addWidget(self.open_button)
        top.addWidget(self.file_label, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.validate_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.cancel_button)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.editor)
        report = QWidget()
        report_layout = QVBoxLayout(report)
        report_layout.setContentsMargins(0, 0, 0, 0)
        report_layout.addWidget(QLabel(f"<b>{texts.ADD_STRATEGY_REPORT}</b>"))
        report_layout.addWidget(self.status_label)
        report_layout.addWidget(self.report_view, 1)
        splitter.addWidget(report)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        layout.addLayout(buttons)

    # -- Öffentliche API ---------------------------------------------------------------

    @property
    def report(self) -> ValidationReport | None:
        """Bericht der letzten Prüfung (``None``, solange nicht geprüft)."""
        return self._report

    def code(self) -> str:
        """Der aktuelle Code im Editor."""
        return self.editor.toPlainText()

    def set_code(self, code: str, filename: str = "<eingefügt>") -> None:
        """Setzt den Code (z. B. aus der Zwischenablage) und verwirft den alten Bericht."""
        self.editor.setPlainText(code)
        self._filename = filename

    def choose_file(self) -> None:
        """Fragt nach einer Python-Datei und lädt sie in den Editor."""
        path, _filter = QFileDialog.getOpenFileName(
            self, texts.ADD_STRATEGY_OPEN, "", texts.ADD_STRATEGY_FILE_FILTER
        )
        if path:
            self.load_file(Path(path))

    def load_file(self, path: Path) -> bool:
        """Lädt ``path`` in den Editor; Fehler erscheinen als Meldung."""
        try:
            code = self._service.read_file(path)
        except StrategySaveError as exc:
            QMessageBox.warning(self, texts.ADD_STRATEGY_TITLE, str(exc))
            return False
        self.set_code(code, path.name)
        self.file_label.setText(str(path))
        return True

    def validate(self) -> None:
        """Prüft den Code im Hintergrund; das Ergebnis erscheint im Bericht."""
        code = self.code()
        if not code.strip():
            self._set_status(texts.ADD_STRATEGY_EMPTY, ok=False)
            return
        self._checking = True
        self._checked_code = code
        self._report = None
        self.validate_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.report_view.clear()
        self._set_status(texts.ADD_STRATEGY_CHECKING, ok=None)
        self._runner(
            self._service.validate,
            code,
            self._filename,
            on_result=self._on_report,
            name=texts.ADD_STRATEGY_TASK,
        )

    def save(self) -> bool:
        """Speichert die geprüfte Strategie; bei vorhandener ID erst nach Rückfrage."""
        report = self._report
        code = self._checked_code
        if report is None or code is None or not report.ok or code != self.code():
            return False
        try:
            result = self._service.save_plugin(code, report)
        except StrategyExistsError as exc:
            if not self._confirm_replace(exc):
                return False
            try:
                result = self._service.save_plugin(code, report, replace=True)
            except StrategySaveError as again:
                self._set_status(str(again), ok=False)
                return False
        except StrategySaveError as exc:
            self._set_status(str(exc), ok=False)
            return False
        self.result_saved = result
        if result.backups:
            backups = ", ".join(p.name for p in result.backups)
            text = texts.ADD_STRATEGY_SAVED_BACKUP.format(path=result.path, backups=backups)
        else:
            text = texts.ADD_STRATEGY_SAVED.format(path=result.path)
        self._set_status(text, ok=True)
        self.saved.emit(result)
        self.accept()
        return True

    # -- Interna -----------------------------------------------------------------------

    def _confirm_replace(self, exc: StrategyExistsError) -> bool:
        info = exc.existing[0]
        backup = f"{info.id}_v{info.cls.version}{BACKUP_SUFFIX}"
        answer = QMessageBox.question(
            self,
            texts.ADD_STRATEGY_EXISTS_TITLE,
            texts.ADD_STRATEGY_EXISTS.format(message=exc, backup=backup),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _on_report(self, report: ValidationReport) -> None:
        self._checking = False
        self._report = report
        self.validate_button.setEnabled(True)
        self.report_view.setPlainText(report.format())
        self._set_status(report.summary(), ok=report.ok)
        self.save_button.setEnabled(report.ok and self._checked_code == self.code())

    def _on_code_changed(self) -> None:
        if self._checking:
            return
        if self._report is not None and self._checked_code != self.code():
            self.save_button.setEnabled(False)
            self._set_status(texts.ADD_STRATEGY_CHANGED, ok=None)

    def _set_status(self, text: str, *, ok: bool | None) -> None:
        self.status_label.setText(text)
        color = "" if ok is None else (UP_COLOR if ok else DOWN_COLOR)
        self.status_label.setStyleSheet(f"color: {color};" if color else "")
