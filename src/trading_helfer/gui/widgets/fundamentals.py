"""Kennzahlen-Tab: Tabelle aller Kennzahlen mit Sektor-Median und Verlauf je Kennzahl.

Die Tabelle zeigt für das gewählte Symbol alle Kennzahlen zum Stichtag, daneben den
Median des Sektors und eine farbige Einordnung (grün = besser, rot = schlechter).
Ein Klick auf eine Zeile zeichnet den Verlauf der Kennzahl je Quartal darunter.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any, Final

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, QEvent, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDateEdit,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from trading_helfer.gui import texts
from trading_helfer.gui.widgets.chart import DOWN_COLOR, UP_COLOR, TaskRunner
from trading_helfer.services.fundamentals_service import (
    METRIC_FORMATS,
    Comparison,
    FundamentalsData,
    FundamentalsService,
    MetricFormat,
    format_metric,
)

__all__ = ["COLUMNS", "DEFAULT_METRIC", "FundamentalsWidget"]

COLUMNS: Final[tuple[str, ...]] = (
    texts.FUNDAMENTALS_COL_METRIC,
    texts.FUNDAMENTALS_COL_VALUE,
    texts.FUNDAMENTALS_COL_MEDIAN,
    texts.FUNDAMENTALS_COL_COMPARISON,
)
"""Spaltenüberschriften der Kennzahlentabelle."""

DEFAULT_METRIC: Final = "pe_ratio"
"""Kennzahl, deren Verlauf nach dem ersten Laden gezeigt wird."""

COL_METRIC, COL_VALUE, COL_MEDIAN, COL_COMPARISON = range(4)
_HIGHLIGHT_ALPHA: Final = 70
_COMPARISON_COLORS: Final[dict[Comparison, str]] = {
    Comparison.BETTER: UP_COLOR,
    Comparison.WORSE: DOWN_COLOR,
}
_COMPARISON_TEXTS: Final[dict[Comparison, str]] = {
    Comparison.BETTER: texts.FUNDAMENTALS_BETTER,
    Comparison.WORSE: texts.FUNDAMENTALS_WORSE,
    Comparison.EQUAL: texts.FUNDAMENTALS_EQUAL,
}
_METRIC_ROLE: Final = Qt.ItemDataRole.UserRole

Request = tuple[str, dt.date]


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


def _to_qdate(day: dt.date) -> QDate:
    return QDate(day.year, day.month, day.day)


class FundamentalsWidget(QWidget):
    """Kennzahlen eines Symbols zum wählbaren Stichtag, mit Sektor-Vergleich und Verlauf.

    Args:
        service: Liefert Kennzahlen, Sektor-Mediane und Verlauf.
        runner: Startet das Laden, z. B. :meth:`MainWindow.run_task`; ohne Angabe wird
            sofort im GUI-Thread geladen.
        today: Liefert das heutige Datum (Standard-Stichtag); für Tests austauschbar.
    """

    data_loaded = Signal(object)
    """Wird mit den neuen :class:`FundamentalsData` gesendet, nachdem sie angezeigt sind."""

    load_failed = Signal(str)
    """Wird mit der deutschen Fehlermeldung gesendet, wenn das Laden scheitert."""

    def __init__(
        self,
        service: FundamentalsService,
        runner: TaskRunner | None = None,
        parent: QWidget | None = None,
        today: Callable[[], dt.date] = dt.date.today,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._runner: TaskRunner = runner if runner is not None else _run_inline
        self._today = today
        self._symbol: str | None = None
        self._request: Request | None = None
        self._data: FundamentalsData | None = None
        self._metric = DEFAULT_METRIC

        self._build_toolbar()
        self._build_table()
        self._build_plot()

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.table)
        self.splitter.addWidget(self.graphics)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(self._toolbar)
        layout.addWidget(self.info_label)
        layout.addWidget(self.splitter, 1)

        self.message_label.setText(texts.FUNDAMENTALS_NO_SYMBOL)
        self._apply_palette()

    # -- Aufbau ----------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        self.message_label = QLabel()
        self.message_label.setObjectName("FundamentalsMessage")
        self.info_label = QLabel()
        self.info_label.setObjectName("FundamentalsInfo")
        for label in (self.message_label, self.info_label):
            label.setMinimumWidth(1)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("dd.MM.yyyy")
        self.date_edit.setMinimumDate(QDate(1900, 1, 1))
        self.date_edit.setDate(_to_qdate(self._today()))
        self.date_edit.setToolTip(texts.FUNDAMENTALS_AS_OF_TOOLTIP)
        self.date_edit.dateChanged.connect(lambda _d: self.reload())
        self.today_button = QPushButton(texts.FUNDAMENTALS_TODAY)
        self.today_button.clicked.connect(lambda: self.set_as_of(self._today()))

        self._toolbar = QHBoxLayout()
        self._toolbar.addWidget(self.message_label, 1)
        self._toolbar.addWidget(QLabel(texts.FUNDAMENTALS_AS_OF))
        self._toolbar.addWidget(self.date_edit)
        self._toolbar.addWidget(self.today_button)

    def _build_table(self) -> None:
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_METRIC, QHeaderView.ResizeMode.Stretch)
        for column in (COL_VALUE, COL_MEDIAN, COL_COMPARISON):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.currentCellChanged.connect(self._on_current_cell_changed)

    def _build_plot(self) -> None:
        self.graphics = pg.GraphicsLayoutWidget()
        self.date_axis = pg.DateAxisItem(orientation="bottom")
        self.history_plot: pg.PlotItem = self.graphics.addPlot(
            row=0, col=0, axisItems={"bottom": self.date_axis}
        )
        self.history_plot.showGrid(x=True, y=True, alpha=0.15)
        self.history_plot.setMenuEnabled(False)
        self.history_plot.hideButtons()
        self.history_plot.getAxis("left").setWidth(70)
        self.history_curve = pg.PlotDataItem(
            x=np.zeros(0), y=np.zeros(0), connect="finite", symbol="o", symbolSize=6
        )
        self.history_plot.addItem(self.history_curve)

    def _apply_palette(self) -> None:
        palette = self.palette()
        background = palette.color(QPalette.ColorRole.Base)
        foreground = palette.color(QPalette.ColorRole.Text)
        accent = palette.color(QPalette.ColorRole.Highlight)
        self.graphics.setBackground(background)
        for name in ("left", "bottom"):
            axis = self.history_plot.getAxis(name)
            axis.setPen(foreground)
            axis.setTextPen(foreground)
        self.history_plot.titleLabel.setAttr("color", foreground)
        self.history_curve.setPen(pg.mkPen(accent, width=2))
        self.history_curve.setSymbolBrush(pg.mkBrush(accent))
        self.history_curve.setSymbolPen(pg.mkPen(accent))
        if self._data is not None:
            self._plot_history()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 (Qt-Override)
        """Übernimmt Farben beim Wechsel zwischen Hell- und Dunkelmodus."""
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_palette()
        super().changeEvent(event)

    # -- Öffentliche Schnittstelle ---------------------------------------------------

    @property
    def symbol(self) -> str | None:
        """Angezeigtes oder zuletzt angefordertes Symbol."""
        return self._symbol

    @property
    def as_of(self) -> dt.date:
        """Gewählter Stichtag."""
        qdate = self.date_edit.date()
        return dt.date(qdate.year(), qdate.month(), qdate.day())

    @property
    def data(self) -> FundamentalsData | None:
        """Aktuell angezeigte Kennzahlen."""
        return self._data

    @property
    def selected_metric(self) -> str:
        """Kennzahl, deren Verlauf gezeigt wird."""
        return self._metric

    def load_symbol(self, symbol: str) -> None:
        """Lädt die Kennzahlen von ``symbol`` zum gewählten Stichtag."""
        self._symbol = symbol
        self.reload()

    def set_as_of(self, day: dt.date) -> None:
        """Wechselt den Stichtag und lädt neu."""
        self.date_edit.setDate(_to_qdate(day))

    def reload(self) -> None:
        """Fordert die Kennzahlen für Symbol und Stichtag neu an."""
        if self._symbol is None:
            return
        request: Request = (self._symbol, self.as_of)
        self._request = request
        self.message_label.setText(texts.FUNDAMENTALS_LOADING.format(symbol=self._symbol))
        self._runner(
            self._load,
            *request,
            on_result=self._on_loaded,
            name=texts.FUNDAMENTALS_TASK_NAME.format(symbol=self._symbol),
        )

    def show_metric(self, metric: str) -> None:
        """Wählt die Zeile von ``metric`` und zeichnet ihren Verlauf.

        Raises:
            KeyError: Die Kennzahl steht nicht in der Tabelle.
        """
        for row in range(self.table.rowCount()):
            item = self.table.item(row, COL_METRIC)
            if item is not None and item.data(_METRIC_ROLE) == metric:
                self.table.setCurrentCell(row, COL_METRIC)
                self._metric = metric
                self._plot_history()
                return
        raise KeyError(metric)

    def history_points(self) -> int:
        """Anzahl gezeichneter Punkte im Verlauf (ohne fehlende Werte)."""
        _x, y = self.history_curve.getData()
        if y is None:
            return 0
        return int(np.isfinite(y).sum())

    def cell_text(self, metric: str, column: int) -> str:
        """Text der Zelle in Spalte ``column`` der Zeile von ``metric``.

        Raises:
            KeyError: Die Kennzahl steht nicht in der Tabelle.
        """
        for row in range(self.table.rowCount()):
            item = self.table.item(row, COL_METRIC)
            if item is not None and item.data(_METRIC_ROLE) == metric:
                cell = self.table.item(row, column)
                return cell.text() if cell is not None else ""
        raise KeyError(metric)

    # -- Laden -----------------------------------------------------------------------

    def _load(self, symbol: str, as_of: dt.date) -> tuple[Request, FundamentalsData | str]:
        """Läuft im Hintergrund; Fehler werden als Text zurückgegeben statt geworfen."""
        request: Request = (symbol, as_of)
        try:
            return request, self._service.load(symbol, as_of)
        except Exception as exc:  # jeder Ladefehler wird im Tab angezeigt
            return request, str(exc) or type(exc).__name__

    def _on_loaded(self, result: tuple[Request, FundamentalsData | str]) -> None:
        request, payload = result
        if request != self._request:
            return  # veraltete Antwort
        if isinstance(payload, str):
            message = texts.FUNDAMENTALS_LOAD_FAILED.format(symbol=request[0], message=payload)
            self.message_label.setText(message)
            self.load_failed.emit(message)
            return
        self.set_data(payload)

    # -- Anzeige ---------------------------------------------------------------------

    def set_data(self, data: FundamentalsData) -> None:
        """Zeigt ``data`` in Tabelle und Verlauf."""
        self._data = data
        self._symbol = data.symbol
        if data.has_values:
            self.message_label.setText(texts.FUNDAMENTALS_HEADER.format(symbol=data.symbol))
        else:
            self.message_label.setText(texts.FUNDAMENTALS_NO_DATA.format(symbol=data.symbol))
        sector = (
            texts.FUNDAMENTALS_SECTOR.format(sector=data.sector, count=data.peer_count)
            if data.sector is not None
            else texts.FUNDAMENTALS_NO_SECTOR
        )
        self.info_label.setText(
            texts.FUNDAMENTALS_INFO.format(as_of=f"{data.as_of:%d.%m.%Y}", sector=sector)
        )
        self._fill_table(data)
        self.data_loaded.emit(data)

    def _fill_table(self, data: FundamentalsData) -> None:
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(data.rows))
            selected = 0
            for index, row in enumerate(data.rows):
                if row.metric == self._metric:
                    selected = index
                name = QTableWidgetItem(row.label)
                name.setData(_METRIC_ROLE, row.metric)
                value = QTableWidgetItem(format_metric(row.metric, row.value))
                median = QTableWidgetItem(format_metric(row.metric, row.sector_median))
                verdict = QTableWidgetItem(
                    _COMPARISON_TEXTS[row.comparison] if row.comparison is not None else ""
                )
                for item in (value, median):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                color = _COMPARISON_COLORS.get(row.comparison) if row.comparison else None
                if color is not None:
                    background = QColor(color)
                    background.setAlpha(_HIGHLIGHT_ALPHA)
                    for item in (value, verdict):
                        item.setBackground(QBrush(background))
                for column, item in enumerate((name, value, median, verdict)):
                    self.table.setItem(index, column, item)
            if data.rows:
                self.table.setCurrentCell(selected, COL_METRIC)
                metric_item = self.table.item(selected, COL_METRIC)
                assert metric_item is not None
                self._metric = str(metric_item.data(_METRIC_ROLE))
        finally:
            self.table.blockSignals(False)
        self._plot_history()

    def _on_current_cell_changed(self, row: int, *_args: int) -> None:
        item = self.table.item(row, COL_METRIC) if row >= 0 else None
        if item is None:
            return
        self._metric = str(item.data(_METRIC_ROLE))
        self._plot_history()

    def _plot_history(self) -> None:
        data = self._data
        label = texts.FUNDAMENTALS_HISTORY_TITLE.format(
            metric=_label_of(data, self._metric) if data else self._metric
        )
        empty = np.zeros(0)
        if data is None or data.history.empty or self._metric not in data.history.columns:
            self.history_curve.setData(x=empty, y=empty)
            self.history_plot.setTitle(texts.FUNDAMENTALS_NO_HISTORY)
            return
        history = data.history
        x = pd.to_datetime(history["period_end"]).to_numpy(dtype="datetime64[s]").astype(np.float64)
        y = history[self._metric].to_numpy(dtype=np.float64)
        if METRIC_FORMATS.get(self._metric) is MetricFormat.PERCENT:
            y = y * 100.0
            label += " (%)"
        if not np.isfinite(y).any():
            self.history_curve.setData(x=empty, y=empty)
            self.history_plot.setTitle(texts.FUNDAMENTALS_NO_HISTORY)
            return
        self.history_curve.setData(x=x, y=y)
        self.history_plot.setTitle(label)
        finite = np.isfinite(y)
        low, high = float(np.min(y[finite])), float(np.max(y[finite]))
        pad = (high - low) * 0.05 or max(abs(high) * 0.05, 1.0)
        self.history_plot.setXRange(float(x.min()), float(x.max()), padding=0.02)
        self.history_plot.setYRange(low - pad, high + pad, padding=0)


def _label_of(data: FundamentalsData, metric: str) -> str:
    try:
        return data.row(metric).label
    except KeyError:
        return metric
