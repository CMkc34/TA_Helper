"""Backtest-Tab: Formular, Lauf mit Fortschritt, Ergebnisanzeige und gespeicherte Läufe.

Links steht das Formular (Strategie, aus ``params`` erzeugte Parameterfelder,
Einzelwert oder Index, Zeitrahmen, Zeitraum, Kapital, Kosten) und darunter die Liste
gespeicherter Läufe. Rechts zeigt der Tab das Ergebnis: Kennzahlen-Kacheln, Equity- und
Drawdown-Kurve mit Benchmark, Trade-Tabelle (Doppelklick öffnet den Chart mit Markern),
Monatsrenditen-Heatmap, Vergleich zweier Läufe und das Protokoll.

Die Logik liegt in :class:`~trading_helfer.services.backtest_service.BacktestService`.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, QEvent, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPalette, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from trading_helfer.backtest.results import RunSummary
from trading_helfer.core.markers import Marker
from trading_helfer.core.types import Side, Timeframe
from trading_helfer.gui import texts
from trading_helfer.gui.widgets.chart import DOWN_COLOR, UP_COLOR
from trading_helfer.services.backtest_service import (
    DEFAULT_BENCHMARK,
    INDEX_LABELS,
    METRIC_TILES,
    BacktestOutcome,
    BacktestRequest,
    BacktestService,
    UniverseKind,
    compare_runs,
    format_metric_value,
    metric_value,
    trade_markers,
)
from trading_helfer.strategies.base import Param

__all__ = ["RUN_COLUMNS", "TRADE_TABLE_COLUMNS", "BacktestRunner", "BacktestWidget"]

TRADE_TABLE_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("symbol", texts.BACKTEST_COL_SYMBOL),
    ("side", texts.BACKTEST_COL_SIDE),
    ("entry_date", texts.BACKTEST_COL_ENTRY_DATE),
    ("entry_price", texts.BACKTEST_COL_ENTRY_PRICE),
    ("exit_date", texts.BACKTEST_COL_EXIT_DATE),
    ("exit_price", texts.BACKTEST_COL_EXIT_PRICE),
    ("qty", texts.BACKTEST_COL_QTY),
    ("pnl", texts.BACKTEST_COL_PNL),
    ("pnl_pct", texts.BACKTEST_COL_PNL_PCT),
    ("fees", texts.BACKTEST_COL_FEES),
    ("exit_reason", texts.BACKTEST_COL_REASON),
)
"""Spalten der Trade-Tabelle: Feld der Trade-Liste und Überschrift."""

RUN_COLUMNS: Final[tuple[str, ...]] = (
    texts.BACKTEST_COL_CREATED,
    texts.BACKTEST_COL_STRATEGY,
    texts.BACKTEST_COL_UNIVERSE,
    texts.BACKTEST_COL_TIMEFRAME,
    texts.BACKTEST_COL_RETURN,
)
"""Spalten der Liste gespeicherter Läufe."""

_ROW_ROLE: Final = Qt.ItemDataRole.UserRole
_HEATMAP_FULL_PCT: Final = 10.0  # ab ±10 % volle Farbe
_HEATMAP_MAX_ALPHA: Final = 200
_MAX_SPIN: Final = 1_000_000_000.0
_SIDE_TEXTS: Final = {
    Side.LONG.value: texts.BACKTEST_SIDE_LONG,
    Side.SHORT.value: texts.BACKTEST_SIDE_SHORT,
}


class BacktestRunner(Protocol):
    """Startet eine Aufgabe, z. B. :meth:`MainWindow.run_task`."""

    def __call__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None] | None = ...,
        name: str = ...,
        report_progress: bool = ...,
        on_progress: Callable[[int, str], None] | None = ...,
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
) -> None:
    """Standard-Runner: führt die Aufgabe sofort im GUI-Thread aus."""
    del name
    kwargs: dict[str, Any] = {}
    if report_progress:
        kwargs["progress"] = on_progress if on_progress is not None else (lambda _p, _m: None)
    result = fn(*args, **kwargs)
    if on_result is not None:
        on_result(result)


def _to_qdate(day: dt.date) -> QDate:
    return QDate(day.year, day.month, day.day)


def _from_qdate(qdate: QDate) -> dt.date:
    return dt.date(qdate.year(), qdate.month(), qdate.day())


def _decimals(param: Param) -> int:
    step = param.step if param.step is not None else 0.01
    text = f"{step:.10f}".rstrip("0")
    return max(2, len(text.split(".")[1]) if "." in text else 0)


def _format_cell(column: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NaT:
        return ""
    if column in ("entry_date", "exit_date"):
        return f"{pd.Timestamp(value):%d.%m.%Y}"
    if column == "side":
        return _SIDE_TEXTS.get(str(value), str(value))
    if column in ("entry_price", "exit_price", "pnl", "fees", "qty"):
        return format_metric_value(value, "ratio")
    if column == "pnl_pct":
        return format_metric_value(value, "pct")
    return str(value)


class _Tile(QFrame):
    """Kachel mit Titel und Wert."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        self.title_label = QLabel(title)
        self.value_label = QLabel("–")
        for label in (self.title_label, self.value_label):
            label.setMinimumWidth(1)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        font = self.value_label.font()
        font.setPointSizeF(font.pointSizeF() * 1.4)
        font.setBold(True)
        self.value_label.setFont(font)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, text: str, color: str = "") -> None:
        self.value_label.setText(text)
        self.value_label.setStyleSheet(f"color: {color};" if color else "")


class BacktestWidget(QWidget):
    """Backtest-Tab.

    Args:
        service: Fassade zu Strategien, Engine und Ergebnisspeicher.
        runner: Startet Aufgaben im Hintergrund, z. B. :meth:`MainWindow.run_task`;
            ohne Angabe läuft alles sofort im GUI-Thread.
        today: Liefert das heutige Datum (Standard-Ende des Zeitraums).
    """

    finished = Signal(object)
    """Wird mit dem :class:`BacktestOutcome` gesendet, nachdem es angezeigt ist."""

    failed = Signal(str)
    """Wird mit der deutschen Fehlermeldung gesendet, wenn ein Lauf scheitert."""

    trade_activated = Signal(str, object, object)
    """Doppelklick auf einen Trade: Symbol, Zeitrahmen und Marker aller Trades des Symbols."""

    def __init__(
        self,
        service: BacktestService,
        runner: BacktestRunner | None = None,
        parent: QWidget | None = None,
        today: Callable[[], dt.date] = dt.date.today,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._runner: BacktestRunner = runner if runner is not None else _run_inline
        self._today = today
        self._param_widgets: dict[str, QWidget] = {}
        self._params: dict[str, Param] = {}
        self._outcome: BacktestOutcome | None = None
        self._runs: list[RunSummary] = []
        self._running = False

        form = self._build_form()
        runs = self._build_runs()
        left = QSplitter(Qt.Orientation.Vertical)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form)
        left.addWidget(scroll)
        left.addWidget(runs)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 2)

        results = self._build_results()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(results)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

        # Rollbereich: Der Tab soll die Mindestbreite des Hauptfensters nicht vergrößern;
        # bei schmalem Fenster erscheint stattdessen eine Bildlaufleiste.
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidget(self.splitter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.scroll_area)

        self._fill_strategies()
        self._apply_palette()
        self._runs_loaded_once = False

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 (Qt-Override)
        """Liest die gespeicherten Läufe beim ersten Anzeigen des Tabs."""
        super().showEvent(event)
        if not self._runs_loaded_once:
            self._runs_loaded_once = True
            self.refresh_runs()

    # -- Aufbau: Formular ------------------------------------------------------------

    def _build_form(self) -> QWidget:
        box = QWidget()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox(texts.BACKTEST_FORM_GROUP)
        form = QFormLayout(group)
        self.strategy_combo = QComboBox()
        self.strategy_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.strategy_combo.setMinimumContentsLength(12)
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        form.addRow(texts.BACKTEST_STRATEGY, self.strategy_combo)
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        form.addRow(self.description_label)

        self.symbol_radio = QRadioButton(texts.BACKTEST_UNIVERSE_SYMBOL)
        self.index_radio = QRadioButton(texts.BACKTEST_UNIVERSE_INDEX)
        self.symbol_radio.setChecked(True)
        self.universe_group = QButtonGroup(self)
        self.universe_group.addButton(self.symbol_radio)
        self.universe_group.addButton(self.index_radio)
        self.symbol_radio.toggled.connect(self._on_universe_kind_changed)
        radios = QHBoxLayout()
        radios.addWidget(self.symbol_radio)
        radios.addWidget(self.index_radio)
        form.addRow(texts.BACKTEST_UNIVERSE, radios)
        self.symbol_edit = QLineEdit()
        self.symbol_edit.setPlaceholderText(texts.BACKTEST_SYMBOL_PLACEHOLDER)
        self.index_combo = QComboBox()
        for code, label in INDEX_LABELS.items():
            self.index_combo.addItem(label, code)
        self.index_combo.setToolTip(texts.BACKTEST_UNIVERSE_HINT)
        form.addRow(self.symbol_edit)
        form.addRow(self.index_combo)

        self.timeframe_combo = QComboBox()
        form.addRow(texts.BACKTEST_TIMEFRAME, self.timeframe_combo)
        self.start_edit = self._date_edit(self._service.settings.start_date)
        self.end_edit = self._date_edit(self._today())
        form.addRow(texts.BACKTEST_START, self.start_edit)
        form.addRow(texts.BACKTEST_END, self.end_edit)

        settings = self._service.settings
        self.capital_spin = self._money_spin(settings.default_capital, minimum=1.0)
        self.fee_spin = self._money_spin(settings.fee_per_order)
        self.fee_pct_spin = self._pct_spin(0.0)
        self.slippage_spin = self._pct_spin(settings.slippage_pct)
        form.addRow(texts.BACKTEST_CAPITAL, self.capital_spin)
        form.addRow(texts.BACKTEST_FEE, self.fee_spin)
        form.addRow(texts.BACKTEST_FEE_PCT, self.fee_pct_spin)
        form.addRow(texts.BACKTEST_SLIPPAGE, self.slippage_spin)
        self.benchmark_edit = QLineEdit(DEFAULT_BENCHMARK)
        form.addRow(texts.BACKTEST_BENCHMARK, self.benchmark_edit)
        self.save_checkbox = QCheckBox(texts.BACKTEST_SAVE)
        self.save_checkbox.setChecked(True)
        form.addRow(self.save_checkbox)
        outer.addWidget(group)

        self.params_group = QGroupBox(texts.BACKTEST_PARAMS_GROUP)
        self.params_form = QFormLayout(self.params_group)
        outer.addWidget(self.params_group)

        self.run_button = QPushButton(texts.BACKTEST_RUN)
        self.run_button.clicked.connect(self.start)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel(texts.BACKTEST_IDLE)
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("BacktestStatus")
        outer.addWidget(self.run_button)
        outer.addWidget(self.progress_bar)
        outer.addWidget(self.status_label)
        outer.addStretch(1)
        self._on_universe_kind_changed(True)
        return box

    @staticmethod
    def _date_edit(day: dt.date) -> QDateEdit:
        edit = QDateEdit()
        edit.setCalendarPopup(True)
        edit.setDisplayFormat("dd.MM.yyyy")
        edit.setMinimumDate(QDate(1900, 1, 1))
        edit.setDate(_to_qdate(day))
        return edit

    @staticmethod
    def _money_spin(value: float, minimum: float = 0.0) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, _MAX_SPIN)
        spin.setDecimals(2)
        spin.setSuffix(" $")
        spin.setValue(value)
        return spin

    @staticmethod
    def _pct_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 99.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.01)
        spin.setSuffix(" %")
        spin.setValue(value)
        return spin

    # -- Aufbau: gespeicherte Läufe --------------------------------------------------

    def _build_runs(self) -> QWidget:
        group = QGroupBox(texts.BACKTEST_RUNS_GROUP)
        layout = QVBoxLayout(group)
        self.runs_table = QTableWidget(0, len(RUN_COLUMNS))
        self.runs_table.setHorizontalHeaderLabels(list(RUN_COLUMNS))
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.runs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.runs_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self.runs_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.runs_table.cellDoubleClicked.connect(lambda row, _c: self.load_run(self._run_id(row)))
        layout.addWidget(self.runs_table)

        buttons = QGridLayout()
        self.refresh_button = QPushButton(texts.BACKTEST_RUNS_REFRESH)
        self.refresh_button.clicked.connect(self.refresh_runs)
        self.load_button = QPushButton(texts.BACKTEST_RUNS_LOAD)
        self.load_button.clicked.connect(self._on_load_clicked)
        self.compare_button = QPushButton(texts.BACKTEST_RUNS_COMPARE)
        self.compare_button.clicked.connect(self._on_compare_clicked)
        self.delete_button = QPushButton(texts.BACKTEST_RUNS_DELETE)
        self.delete_button.clicked.connect(self._on_delete_clicked)
        for i, button in enumerate(
            (self.refresh_button, self.load_button, self.compare_button, self.delete_button)
        ):
            buttons.addWidget(button, i // 2, i % 2)
        layout.addLayout(buttons)
        return group

    # -- Aufbau: Ergebnis ------------------------------------------------------------

    def _build_results(self) -> QWidget:
        self.result_label = QLabel()
        self.result_label.setObjectName("BacktestResultHeader")
        self.result_label.setWordWrap(True)
        self.result_tabs = QTabWidget()

        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        tiles = QGridLayout()
        self.tiles: dict[str, _Tile] = {}
        per_row = 4
        for i, (key, title, _fmt) in enumerate(METRIC_TILES):
            tile = _Tile(title)
            tile.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            self.tiles[key] = tile
            tiles.addWidget(tile, i // per_row, i % per_row)
        overview_layout.addLayout(tiles)
        self.graphics = pg.GraphicsLayoutWidget()
        self.equity_plot: pg.PlotItem = self.graphics.addPlot(
            row=0, col=0, axisItems={"bottom": pg.DateAxisItem(orientation="bottom")}
        )
        self.equity_plot.setTitle(texts.BACKTEST_EQUITY_TITLE)
        self.legend = self.equity_plot.addLegend(offset=(10, 10))
        self.equity_curve = pg.PlotDataItem(name=texts.BACKTEST_STRATEGY_CURVE)
        self.benchmark_curve = pg.PlotDataItem(name=DEFAULT_BENCHMARK)
        self.equity_plot.addItem(self.equity_curve)
        self.equity_plot.addItem(self.benchmark_curve)
        self.drawdown_plot: pg.PlotItem = self.graphics.addPlot(
            row=1, col=0, axisItems={"bottom": pg.DateAxisItem(orientation="bottom")}
        )
        self.drawdown_plot.setTitle(texts.BACKTEST_DRAWDOWN_TITLE)
        self.drawdown_plot.setXLink(self.equity_plot)
        self.drawdown_curve = pg.PlotDataItem(fillLevel=0.0)
        self.drawdown_plot.addItem(self.drawdown_curve)
        self.graphics.ci.layout.setRowStretchFactor(0, 3)
        self.graphics.ci.layout.setRowStretchFactor(1, 1)
        for plot in (self.equity_plot, self.drawdown_plot):
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            plot.getAxis("left").setWidth(70)
        overview_layout.addWidget(self.graphics, 1)
        self.result_tabs.addTab(overview, texts.BACKTEST_RESULT_OVERVIEW)

        trades = QWidget()
        trades_layout = QVBoxLayout(trades)
        trades_bar = QHBoxLayout()
        hint = QLabel(texts.BACKTEST_TRADES_HINT)
        hint.setWordWrap(True)
        trades_bar.addWidget(hint, 1)
        self.export_button = QPushButton(texts.BACKTEST_EXPORT)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(lambda: self.export_trades())
        trades_bar.addWidget(self.export_button)
        trades_layout.addLayout(trades_bar)
        self.trades_table = QTableWidget(0, len(TRADE_TABLE_COLUMNS))
        self.trades_table.setHorizontalHeaderLabels([t for _c, t in TRADE_TABLE_COLUMNS])
        self.trades_table.verticalHeader().setVisible(False)
        self.trades_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.trades_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.trades_table.setAlternatingRowColors(True)
        self.trades_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.trades_table.horizontalHeader().setStretchLastSection(True)
        self.trades_table.cellDoubleClicked.connect(lambda row, _c: self.open_trade(row))
        trades_layout.addWidget(self.trades_table)
        self.result_tabs.addTab(trades, texts.BACKTEST_RESULT_TRADES)

        self.monthly_table = QTableWidget(0, 13)
        self.monthly_table.setHorizontalHeaderLabels([*texts.BACKTEST_MONTHS, texts.BACKTEST_YEAR])
        self.monthly_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.monthly_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.result_tabs.addTab(self.monthly_table, texts.BACKTEST_RESULT_MONTHLY)

        compare = QWidget()
        compare_layout = QVBoxLayout(compare)
        self.compare_label = QLabel(texts.BACKTEST_COMPARE_EMPTY)
        self.compare_label.setWordWrap(True)
        self.compare_table = QTableWidget(0, 4)
        self.compare_table.setHorizontalHeaderLabels(
            [texts.BACKTEST_COMPARE_METRIC, "A", "B", texts.BACKTEST_COMPARE_DIFF]
        )
        self.compare_table.verticalHeader().setVisible(False)
        self.compare_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        compare_layout.addWidget(self.compare_label)
        compare_layout.addWidget(self.compare_table)
        self.result_tabs.addTab(compare, texts.BACKTEST_RESULT_COMPARE)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.result_tabs.addTab(self.log_view, texts.BACKTEST_RESULT_LOG)

        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.result_label)
        layout.addWidget(self.result_tabs, 1)
        return box

    def _apply_palette(self) -> None:
        palette = self.palette()
        background = palette.color(QPalette.ColorRole.Base)
        foreground = palette.color(QPalette.ColorRole.Text)
        accent = palette.color(QPalette.ColorRole.Highlight)
        self.graphics.setBackground(background)
        for plot in (self.equity_plot, self.drawdown_plot):
            for name in ("left", "bottom"):
                axis = plot.getAxis(name)
                axis.setPen(foreground)
                axis.setTextPen(foreground)
            plot.titleLabel.setAttr("color", foreground)
        self.equity_curve.setPen(pg.mkPen(accent, width=2))
        bench = QColor(foreground)
        bench.setAlpha(150)
        self.benchmark_curve.setPen(pg.mkPen(bench, width=1, style=Qt.PenStyle.DashLine))
        down = QColor(DOWN_COLOR)
        self.drawdown_curve.setPen(pg.mkPen(down, width=1))
        down.setAlpha(80)
        self.drawdown_curve.setBrush(pg.mkBrush(down))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 (Qt-Override)
        """Übernimmt Farben beim Wechsel zwischen Hell- und Dunkelmodus."""
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_palette()
        super().changeEvent(event)

    # -- Strategien und Parameter ----------------------------------------------------

    def _fill_strategies(self) -> None:
        self.strategy_combo.blockSignals(True)
        self.strategy_combo.clear()
        for info in self._service.strategies():
            self.strategy_combo.addItem(f"{info.name} ({info.id})", info.id)
        self.strategy_combo.blockSignals(False)
        if self.strategy_combo.count():
            self.strategy_combo.setCurrentIndex(0)
            self._on_strategy_changed(0)
        else:
            self.run_button.setEnabled(False)
            self.status_label.setText(texts.BACKTEST_NO_STRATEGY)

    @property
    def strategy_id(self) -> str | None:
        """ID der gewählten Strategie."""
        data = self.strategy_combo.currentData()
        return str(data) if data is not None else None

    def set_strategy(self, strategy_id: str) -> None:
        """Wählt die Strategie ``strategy_id``.

        Raises:
            KeyError: Die Strategie steht nicht in der Auswahl.
        """
        index = self.strategy_combo.findData(strategy_id)
        if index < 0:
            raise KeyError(strategy_id)
        self.strategy_combo.setCurrentIndex(index)

    def _on_strategy_changed(self, _index: int) -> None:
        strategy_id = self.strategy_id
        if strategy_id is None:
            return
        info = self._service.registry.info(strategy_id)
        self.description_label.setText(info.cls.description)
        self._params = self._service.strategy_params(strategy_id)
        self._build_param_widgets()
        current = self.timeframe_combo.currentData()
        self.timeframe_combo.clear()
        for tf in self._service.strategy_timeframes(strategy_id):
            self.timeframe_combo.addItem(tf.value, tf.value)
        index = self.timeframe_combo.findData(current)
        self.timeframe_combo.setCurrentIndex(max(index, 0))

    def _build_param_widgets(self) -> None:
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._param_widgets = {}
        if not self._params:
            self.params_form.addRow(QLabel(texts.BACKTEST_NO_PARAMS))
            return
        for name, param in self._params.items():
            widget = self._param_widget(param)
            widget.setToolTip(param.description)
            widget.setObjectName(f"param_{name}")
            self._param_widgets[name] = widget
            label = QLabel(f"{name}:")
            label.setToolTip(param.description)
            self.params_form.addRow(label, widget)

    @staticmethod
    def _param_widget(param: Param) -> QWidget:
        kind = param.kind
        if kind == "bool":
            box = QCheckBox()
            box.setChecked(bool(param.default))
            return box
        if kind == "choice":
            combo = QComboBox()
            for choice in param.choices or ():
                combo.addItem(str(choice), choice)
            combo.setCurrentIndex(combo.findData(param.default))
            return combo
        if kind == "int":
            spin = QSpinBox()
            spin.setRange(
                int(param.min) if param.min is not None else -1_000_000_000,
                int(param.max) if param.max is not None else 1_000_000_000,
            )
            spin.setSingleStep(int(param.step) if param.step else 1)
            spin.setValue(int(param.default))
            return spin
        dspin = QDoubleSpinBox()
        dspin.setDecimals(_decimals(param))
        dspin.setRange(
            float(param.min) if param.min is not None else -_MAX_SPIN,
            float(param.max) if param.max is not None else _MAX_SPIN,
        )
        dspin.setSingleStep(float(param.step) if param.step else 0.1)
        dspin.setValue(float(param.default))
        return dspin

    def param_widget(self, name: str) -> QWidget:
        """Eingabefeld des Parameters ``name``."""
        return self._param_widgets[name]

    def param_values(self) -> dict[str, Any]:
        """Aktuelle Parameterwerte aus den Eingabefeldern."""
        values: dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            if isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentData()
            elif isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                values[name] = float(widget.value())
        return values

    def set_param(self, name: str, value: Any) -> None:
        """Setzt den Wert des Parameters ``name`` im Formular."""
        widget = self._param_widgets[name]
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            widget.setCurrentIndex(widget.findData(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))

    # -- Formular --------------------------------------------------------------------

    def _on_universe_kind_changed(self, _checked: bool) -> None:
        single = self.symbol_radio.isChecked()
        self.symbol_edit.setVisible(single)
        self.index_combo.setVisible(not single)

    def set_symbol(self, symbol: str) -> None:
        """Wählt Einzelwert und trägt ``symbol`` ein (z. B. aus der Watchlist)."""
        self.symbol_radio.setChecked(True)
        self.symbol_edit.setText(symbol)

    def set_index(self, code: str) -> None:
        """Wählt den Index ``code`` als Universum.

        Raises:
            KeyError: Unbekannter Index.
        """
        index = self.index_combo.findData(code)
        if index < 0:
            raise KeyError(code)
        self.index_radio.setChecked(True)
        self.index_combo.setCurrentIndex(index)

    def set_timeframe(self, timeframe: Timeframe) -> None:
        """Wählt den Zeitrahmen, sofern die Strategie ihn unterstützt.

        Raises:
            KeyError: Die Strategie unterstützt den Zeitrahmen nicht.
        """
        index = self.timeframe_combo.findData(Timeframe(timeframe).value)
        if index < 0:
            raise KeyError(timeframe)
        self.timeframe_combo.setCurrentIndex(index)

    def set_period(self, start: dt.date, end: dt.date) -> None:
        """Setzt den Zeitraum."""
        self.start_edit.setDate(_to_qdate(start))
        self.end_edit.setDate(_to_qdate(end))

    def request(self) -> BacktestRequest:
        """Erzeugt die Backtest-Anfrage aus den Formularfeldern.

        Raises:
            ValueError: Keine Strategie gewählt.
        """
        strategy_id = self.strategy_id
        if strategy_id is None:
            raise ValueError(texts.BACKTEST_NO_STRATEGY)
        if self.symbol_radio.isChecked():
            kind, universe = UniverseKind.SYMBOL, self.symbol_edit.text().strip().upper()
        else:
            kind, universe = UniverseKind.INDEX, str(self.index_combo.currentData())
        return BacktestRequest(
            strategy_id=strategy_id,
            params=self.param_values(),
            universe=universe,
            universe_kind=kind,
            timeframe=Timeframe(self.timeframe_combo.currentData()),
            start=_from_qdate(self.start_edit.date()),
            end=_from_qdate(self.end_edit.date()),
            capital=float(self.capital_spin.value()),
            fee_per_order=float(self.fee_spin.value()),
            fee_pct=float(self.fee_pct_spin.value()),
            slippage_pct=float(self.slippage_spin.value()),
            benchmark=self.benchmark_edit.text().strip().upper(),
            save=self.save_checkbox.isChecked(),
        )

    # -- Lauf ------------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True``, solange ein Lauf aktiv ist."""
        return self._running

    @property
    def outcome(self) -> BacktestOutcome | None:
        """Aktuell angezeigtes Ergebnis."""
        return self._outcome

    def start(self) -> None:
        """Startet den Backtest mit den Formularwerten im Hintergrund."""
        if self._running:
            return
        try:
            request = self.request()
        except ValueError as exc:
            self._fail(str(exc))
            return
        self._running = True
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(texts.BACKTEST_RUNNING)
        self._runner(
            self._run,
            request,
            on_result=self._on_run_finished,
            name=texts.BACKTEST_TASK.format(
                strategy=request.strategy_id, universe=request.universe
            ),
            report_progress=True,
            on_progress=self._on_progress,
        )

    def _run(
        self, request: BacktestRequest, progress: Callable[[int, str], None] | None = None
    ) -> BacktestOutcome | str:
        """Läuft im Hintergrund; Fehler werden als Text zurückgegeben statt geworfen."""
        try:
            return self._service.run(request, progress=progress)
        except Exception as exc:  # jeder Fehler wird im Tab angezeigt
            return str(exc) or type(exc).__name__

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.status_label.setText(texts.BACKTEST_PROGRESS.format(percent=percent, message=message))

    def _on_run_finished(self, payload: BacktestOutcome | str) -> None:
        self._running = False
        self.run_button.setEnabled(True)
        if isinstance(payload, str):
            self.progress_bar.setValue(0)
            self._fail(payload)
            return
        self.progress_bar.setValue(100)
        self.set_outcome(payload)
        if payload.run_id is not None:
            self.refresh_runs()

    def _fail(self, message: str) -> None:
        text = texts.BACKTEST_FAILED.format(message=message)
        self.status_label.setText(text)
        self.failed.emit(text)

    # -- Ergebnisanzeige -------------------------------------------------------------

    def set_outcome(self, outcome: BacktestOutcome) -> None:
        """Zeigt ein Ergebnis in Kacheln, Kurven, Trade-Tabelle, Heatmap und Protokoll."""
        self._outcome = outcome
        result = outcome.result
        header = texts.BACKTEST_DONE.format(
            strategy=result.strategy_id,
            universe=outcome.universe,
            timeframe=result.timeframe.value,
            start=f"{result.start:%d.%m.%Y}",
            end=f"{result.end:%d.%m.%Y}",
        )
        if outcome.run_id is not None:
            header += texts.BACKTEST_SAVED_AS
        self.result_label.setText(header)
        self.status_label.setText(header)
        self._show_tiles(outcome)
        self._show_curves(outcome)
        self._show_trades(outcome)
        self._show_monthly(outcome.monthly)
        self.log_view.setPlainText("\n".join(result.log))
        self.export_button.setEnabled(not result.trades.empty)
        self.finished.emit(outcome)

    def tile_text(self, key: str) -> str:
        """Angezeigter Wert der Kachel ``key``."""
        return self.tiles[key].value_label.text()

    def _show_tiles(self, outcome: BacktestOutcome) -> None:
        for key, _title, fmt in METRIC_TILES:
            value = metric_value(outcome.metrics, key)
            color = ""
            if fmt == "pct" and value is not None and key != "volatility_pct":
                color = UP_COLOR if value > 0 else DOWN_COLOR if value < 0 else ""
            self.tiles[key].set_value(format_metric_value(value, fmt), color)

    def _show_curves(self, outcome: BacktestOutcome) -> None:
        equity = outcome.result.equity
        empty = np.zeros(0)
        if equity.empty:
            for curve in (self.equity_curve, self.benchmark_curve, self.drawdown_curve):
                curve.setData(x=empty, y=empty)
            return
        x = pd.to_datetime(equity["date"]).to_numpy(dtype="datetime64[s]").astype(np.float64)
        self.equity_curve.setData(x=x, y=equity["equity"].to_numpy(dtype=float))
        self.drawdown_curve.setData(x=x, y=equity["drawdown"].to_numpy(dtype=float) * 100.0)
        bench = outcome.benchmark_equity()
        self.legend.removeItem(self.benchmark_curve)
        if bench is None:
            self.benchmark_curve.setData(x=empty, y=empty)
        else:
            self.benchmark_curve.setData(x=x, y=bench.to_numpy(dtype=float))
            self.legend.addItem(self.benchmark_curve, outcome.benchmark_symbol)
        self.equity_plot.enableAutoRange()
        self.drawdown_plot.enableAutoRange()

    def curve_points(self) -> dict[str, int]:
        """Anzahl der Punkte je Kurve (``equity``, ``benchmark``, ``drawdown``)."""
        counts: dict[str, int] = {}
        for key, curve in (
            ("equity", self.equity_curve),
            ("benchmark", self.benchmark_curve),
            ("drawdown", self.drawdown_curve),
        ):
            _x, y = curve.getData()
            counts[key] = 0 if y is None else len(y)
        return counts

    def _show_trades(self, outcome: BacktestOutcome) -> None:
        trades = outcome.result.trades
        self.trades_table.setSortingEnabled(False)
        self.trades_table.setRowCount(len(trades))
        records: list[dict[Any, Any]] = trades.to_dict("records")
        for row, record in enumerate(records):
            for column, (field, _title) in enumerate(TRADE_TABLE_COLUMNS):
                item = QTableWidgetItem(_format_cell(field, record.get(field)))
                item.setData(_ROW_ROLE, row)
                if field in ("entry_price", "exit_price", "qty", "pnl", "pnl_pct", "fees"):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if field in ("pnl", "pnl_pct"):
                    value = record.get(field)
                    if isinstance(value, int | float) and not math.isnan(value) and value != 0:
                        item.setForeground(QBrush(QColor(UP_COLOR if value > 0 else DOWN_COLOR)))
                self.trades_table.setItem(row, column, item)

    def _show_monthly(self, monthly: pd.DataFrame | None) -> None:
        table = self.monthly_table
        if monthly is None or monthly.empty:
            table.setRowCount(0)
            return
        table.setRowCount(len(monthly))
        table.setVerticalHeaderLabels([str(year) for year in monthly.index])
        for row, (_year, values) in enumerate(monthly.iterrows()):
            for column, value in enumerate(values.to_numpy(dtype=float)):
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if np.isfinite(value):
                    item.setText(format_metric_value(value, "pct").replace(" %", ""))
                    color = QColor(UP_COLOR if value >= 0 else DOWN_COLOR)
                    strength = min(abs(value) / _HEATMAP_FULL_PCT, 1.0)
                    color.setAlpha(int(30 + strength * (_HEATMAP_MAX_ALPHA - 30)))
                    item.setBackground(QBrush(color))
                table.setItem(row, column, item)

    def monthly_cell(self, year: int, month: int) -> QTableWidgetItem | None:
        """Zelle der Heatmap für ``year`` und ``month`` (13 = Jahr)."""
        for row in range(self.monthly_table.rowCount()):
            header = self.monthly_table.verticalHeaderItem(row)
            if header is not None and header.text() == str(year):
                return self.monthly_table.item(row, month - 1)
        return None

    # -- Trades und Chart ------------------------------------------------------------

    def open_trade(self, row: int) -> None:
        """Öffnet den Chart für den Trade in Zeile ``row`` mit Markern aller Trades."""
        outcome = self._outcome
        if outcome is None or not 0 <= row < len(outcome.result.trades):
            return
        symbol = str(outcome.result.trades["symbol"].iloc[row])
        markers: list[Marker] = trade_markers(outcome.result.trades, symbol)
        self.trade_activated.emit(symbol, outcome.result.timeframe, markers)

    def export_trades(self, path: str | Path | None = None) -> int | None:
        """Exportiert die Trades als CSV; ohne ``path`` fragt ein Dateidialog.

        Returns:
            Anzahl exportierter Trades oder ``None`` bei Abbruch oder Fehler.
        """
        outcome = self._outcome
        if outcome is None:
            return None
        if path is None:
            suggestion = f"trades_{outcome.result.strategy_id}_{outcome.universe}.csv"
            chosen, _filter = QFileDialog.getSaveFileName(
                self, texts.BACKTEST_EXPORT_DIALOG, suggestion, texts.BACKTEST_EXPORT_FILTER
            )
            if not chosen:
                return None
            path = chosen
        try:
            count = self._service.export_trades(outcome, path)
        except OSError as exc:
            self.status_label.setText(texts.BACKTEST_EXPORT_FAILED.format(message=exc))
            return None
        self.status_label.setText(texts.BACKTEST_EXPORT_DONE.format(count=count, path=path))
        return count

    # -- Gespeicherte Läufe ----------------------------------------------------------

    @property
    def runs(self) -> list[RunSummary]:
        """Angezeigte gespeicherte Läufe."""
        return list(self._runs)

    def refresh_runs(self) -> None:
        """Liest die gespeicherten Läufe neu ein."""
        self._runner(
            self._guard(self._service.list_runs),
            on_result=self._on_runs_loaded,
            name=texts.BACKTEST_RUNS_TASK,
        )

    def _on_runs_loaded(self, payload: list[RunSummary] | str) -> None:
        if isinstance(payload, str):
            self._fail(payload)
            return
        self._runs = list(payload)
        table = self.runs_table
        table.setRowCount(len(self._runs))
        for row, run in enumerate(self._runs):
            ret = run.total_return_pct
            cells = (
                f"{run.created_at:%d.%m.%Y %H:%M}",
                run.strategy_id,
                run.universe,
                run.timeframe.value,
                format_metric_value(ret, "pct"),
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(_ROW_ROLE, run.run_id)
                if column == len(cells) - 1 and ret is not None and ret != 0:
                    item.setForeground(QBrush(QColor(UP_COLOR if ret > 0 else DOWN_COLOR)))
                table.setItem(row, column, item)

    def _run_id(self, row: int) -> str:
        item = self.runs_table.item(row, 0)
        assert item is not None
        return str(item.data(_ROW_ROLE))

    def selected_run_ids(self) -> list[str]:
        """IDs der markierten Läufe in Tabellenreihenfolge."""
        rows = sorted({index.row() for index in self.runs_table.selectedIndexes()})
        return [self._run_id(row) for row in rows]

    def select_runs(self, run_ids: list[str]) -> None:
        """Markiert die Läufe ``run_ids`` in der Liste."""
        self.runs_table.clearSelection()
        mode = self.runs_table.selectionMode()
        self.runs_table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        for row in range(self.runs_table.rowCount()):
            if self._run_id(row) in run_ids:
                self.runs_table.selectRow(row)
        self.runs_table.setSelectionMode(mode)

    def load_run(self, run_id: str) -> None:
        """Lädt einen gespeicherten Lauf und zeigt ihn an."""
        self._runner(
            self._guard(self._service.load_run),
            run_id,
            on_result=self._on_run_loaded,
            name=texts.BACKTEST_LOAD_TASK,
        )

    def _on_run_loaded(self, payload: BacktestOutcome | str) -> None:
        if isinstance(payload, str):
            self._fail(payload)
            return
        self.set_outcome(payload)
        self.result_tabs.setCurrentIndex(0)

    def compare(self, run_a: str, run_b: str) -> None:
        """Vergleicht zwei gespeicherte Läufe in der Vergleichstabelle."""
        self._runner(
            self._guard(lambda a, b: (self._service.load_run(a), self._service.load_run(b))),
            run_a,
            run_b,
            on_result=self._on_compare_loaded,
            name=texts.BACKTEST_COMPARE_TASK,
        )

    def _on_compare_loaded(self, payload: tuple[BacktestOutcome, BacktestOutcome] | str) -> None:
        if isinstance(payload, str):
            self._fail(payload)
            return
        a, b = payload
        self.compare_label.setText(
            texts.BACKTEST_COMPARE_HEADER.format(a=_describe(a), b=_describe(b))
        )
        rows = compare_runs(a.metrics, b.metrics)
        self.compare_table.setRowCount(len(rows))
        for row, cells in enumerate(rows):
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.compare_table.setItem(row, column, item)
        self.result_tabs.setCurrentIndex(3)

    def compare_cell(self, row: int, column: int) -> str:
        """Text einer Zelle der Vergleichstabelle."""
        item = self.compare_table.item(row, column)
        return item.text() if item is not None else ""

    def delete_runs(self, run_ids: list[str]) -> None:
        """Löscht die Läufe ``run_ids`` und aktualisiert die Liste."""

        def delete(ids: list[str]) -> int:
            return sum(self._service.delete_run(rid) for rid in ids)

        self._runner(
            self._guard(delete),
            list(run_ids),
            on_result=lambda _n: self.refresh_runs(),
            name=texts.BACKTEST_RUNS_DELETE,
        )

    def _on_load_clicked(self) -> None:
        ids = self.selected_run_ids()
        if ids:
            self.load_run(ids[0])

    def _on_compare_clicked(self) -> None:
        ids = self.selected_run_ids()
        if len(ids) != 2:
            self.status_label.setText(texts.BACKTEST_COMPARE_SELECT)
            return
        # Ältere Läufe stehen weiter unten; A ist der ältere.
        self.compare(ids[1], ids[0])

    def _on_delete_clicked(self) -> None:
        ids = self.selected_run_ids()
        if ids:
            self.delete_runs(ids)

    @staticmethod
    def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wandelt Ausnahmen in Fehlertexte um, damit der Tab sie anzeigt."""

        def wrapped(*args: Any) -> Any:
            try:
                return fn(*args)
            except Exception as exc:  # jeder Fehler wird im Tab angezeigt
                return str(exc) or type(exc).__name__

        return wrapped


def _describe(outcome: BacktestOutcome) -> str:
    result = outcome.result
    params = ", ".join(f"{k}={v}" for k, v in result.params.items())
    return (
        f"{result.strategy_id} ({params}) · {outcome.universe} · {result.timeframe.value} · "
        f"{result.start:%d.%m.%Y}–{result.end:%d.%m.%Y}"
    )
