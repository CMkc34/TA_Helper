"""Optimierungs-Tab: Parameter-Raster, Walk-Forward, Ergebnis-Heatmap und Monte Carlo.

Links steht das Formular: Strategie, Universum, Zeitrahmen, Zeitraum, Kosten und
Zielgröße, darunter je Parameter ein Bereich (von, bis, Schritt), die Walk-Forward-
Einstellungen und Monte Carlo. „Optimierung starten“ läuft im Hintergrund mit
Fortschritt; „Abbrechen“ setzt nur ein Flag, das nach der laufenden Kombination greift.

Rechts zeigt der Tab die sortierbare Ergebnis-Tabelle, die Heatmap für zwei gewählte
Parameter (übrige fest auf der besten Kombination), die Walk-Forward-Ansicht
(Out-of-Sample-Equity und Parameter je Fenster) und das Monte-Carlo-Histogramm.

Die Logik liegt in :class:`~trading_helfer.services.optimization_service.OptimizationService`.
"""

from __future__ import annotations

import datetime as dt
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, QEvent, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
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
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from trading_helfer.backtest.metrics import MetricsError
from trading_helfer.backtest.monte_carlo import MonteCarloMethod, MonteCarloResult
from trading_helfer.backtest.optimize import (
    MAX_COMBINATIONS,
    METRIC_COLUMNS,
    Objective,
    OptimizationResult,
    ParamRange,
    RangeSpec,
)
from trading_helfer.backtest.portfolio import DEFAULT_MAX_POSITIONS
from trading_helfer.backtest.walkforward import WalkForwardResult
from trading_helfer.core.types import Timeframe
from trading_helfer.gui import texts
from trading_helfer.gui.widgets.backtest import BacktestRunner
from trading_helfer.gui.widgets.chart import DOWN_COLOR, UP_COLOR
from trading_helfer.services.backtest_service import (
    INDEX_LABELS,
    UniverseKind,
    format_metric_value,
)
from trading_helfer.services.optimization_service import (
    OptimizationCancelledError,
    OptimizationMode,
    OptimizationOutcome,
    OptimizationRequest,
    OptimizationService,
)
from trading_helfer.strategies.base import Param

__all__ = ["CANCELLED", "OptimizationWidget", "format_param", "format_params", "histogram_bins"]

CANCELLED: Final = "__cancelled__"
"""Rückgabe des Hintergrundlaufs bei Abbruch."""

_SORT_ROLE: Final = Qt.ItemDataRole.UserRole
_ROW_ROLE: Final = Qt.ItemDataRole.UserRole + 1
_MAX_SPIN: Final = 1_000_000_000.0
_HEATMAP_MAX_ALPHA: Final = 220
_HISTOGRAM_BINS: Final = 30
_METRIC_FORMATS: Final[dict[str, str]] = {
    "total_return_pct": "pct",
    "cagr_pct": "pct",
    "sharpe": "ratio",
    "calmar": "ratio",
    "mar": "ratio",
    "max_drawdown_pct": "pct",
    "num_trades": "int",
}


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


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_param(value: Any) -> str:
    """Parameterwert für die Anzeige (Zahlen deutsch, ganze Zahlen ohne Nachkommastellen)."""
    if isinstance(value, bool | np.bool_):
        return "ja" if value else "nein"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:g}".replace(".", ",")
    return str(value)


def format_params(params: dict[str, Any]) -> str:
    """Parameter als ``name=wert, …``."""
    return ", ".join(f"{name}={format_param(value)}" for name, value in params.items())


def histogram_bins(
    values: np.ndarray, bins: int = _HISTOGRAM_BINS
) -> tuple[np.ndarray, np.ndarray]:
    """Häufigkeiten und Klassengrenzen; bei (fast) gleichen Werten eine einzige Klasse.

    Beim Umordnen der Trades ist das Endkapital in allen Simulationen bis auf
    Rundungsfehler gleich; ``np.histogram`` könnte dann keine 30 Klassen bilden.
    """
    data = np.asarray(values, dtype=float)
    low, high = float(np.min(data)), float(np.max(data))
    if high - low <= 1e-9 * max(1.0, abs(low), abs(high)):
        pad = max(abs(low) * 0.01, 0.5)
        return np.histogram(data, bins=1, range=(low - pad, high + pad))
    return np.histogram(data, bins=bins)


class _SortItem(QTableWidgetItem):
    """Tabellenzelle, die nach einem hinterlegten Wert statt nach dem Text sortiert."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        mine, theirs = self.data(_SORT_ROLE), other.data(_SORT_ROLE)
        if isinstance(mine, int | float) and isinstance(theirs, int | float):
            # Fehlende Werte (NaN) stehen beim Aufsteigend-Sortieren vorn.
            a = -math.inf if math.isnan(mine) else mine
            b = -math.inf if math.isnan(theirs) else theirs
            return a < b
        return str(mine) < str(theirs)


@dataclass
class _RangeRow:
    """Eingabefelder eines Parameters im Raster."""

    param: Param
    check: QCheckBox
    low: QDoubleSpinBox | None
    high: QDoubleSpinBox | None
    step: QDoubleSpinBox | None

    @property
    def numeric(self) -> bool:
        return self.low is not None


class OptimizationWidget(QWidget):
    """Optimierungs-Tab.

    Args:
        service: Fassade zu Strategien, Kursen, Optimierung und Monte Carlo.
        runner: Startet Aufgaben im Hintergrund, z. B. :meth:`MainWindow.run_task`;
            ohne Angabe läuft alles sofort im GUI-Thread.
        today: Liefert das heutige Datum (Standard-Ende des Zeitraums).
    """

    finished = Signal(object)
    """Wird mit dem :class:`OptimizationOutcome` gesendet, nachdem es angezeigt ist."""

    failed = Signal(str)
    """Wird mit der deutschen Fehlermeldung gesendet, wenn ein Lauf scheitert."""

    cancelled = Signal()
    """Wird gesendet, wenn ein Lauf abgebrochen wurde."""

    apply_requested = Signal(object, object)
    """„Im Backtest übernehmen“: :class:`OptimizationRequest` und beste Parameter."""

    def __init__(
        self,
        service: OptimizationService,
        runner: BacktestRunner | None = None,
        parent: QWidget | None = None,
        today: Callable[[], dt.date] = dt.date.today,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._runner: BacktestRunner = runner if runner is not None else _run_inline
        self._today = today
        self._rows: dict[str, _RangeRow] = {}
        self._outcome: OptimizationOutcome | None = None
        self._cancel_event: threading.Event | None = None
        self._histogram: tuple[np.ndarray, np.ndarray] | None = None
        self._monte_carlo: MonteCarloResult | None = None

        form = self._build_form()
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setWidget(form)

        results = self._build_results()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(results)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)

        # Rollbereich: Der Tab soll die Mindestbreite des Hauptfensters nicht vergrößern.
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setWidget(self.splitter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.scroll_area)

        self._fill_strategies()
        self._apply_palette()
        self._update_buttons()

    # -- Aufbau: Formular ------------------------------------------------------------

    def _build_form(self) -> QWidget:
        box = QWidget()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox(texts.OPT_FORM_GROUP)
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
        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setRange(1, 500)
        self.max_positions_spin.setValue(DEFAULT_MAX_POSITIONS)
        self.max_positions_label = QLabel(texts.BACKTEST_MAX_POSITIONS)
        form.addRow(self.max_positions_label, self.max_positions_spin)

        self.timeframe_combo = QComboBox()
        form.addRow(texts.BACKTEST_TIMEFRAME, self.timeframe_combo)
        settings = self._service.settings
        self.start_edit = self._date_edit(settings.start_date)
        self.end_edit = self._date_edit(self._today())
        form.addRow(texts.BACKTEST_START, self.start_edit)
        form.addRow(texts.BACKTEST_END, self.end_edit)
        self.capital_spin = self._spin(settings.default_capital, 1.0, _MAX_SPIN, 2, " $")
        self.fee_spin = self._spin(settings.fee_per_order, 0.0, _MAX_SPIN, 2, " $")
        self.fee_pct_spin = self._spin(0.0, 0.0, 99.0, 3, " %")
        self.slippage_spin = self._spin(settings.slippage_pct, 0.0, 99.0, 3, " %")
        form.addRow(texts.BACKTEST_CAPITAL, self.capital_spin)
        form.addRow(texts.BACKTEST_FEE, self.fee_spin)
        form.addRow(texts.BACKTEST_FEE_PCT, self.fee_pct_spin)
        form.addRow(texts.BACKTEST_SLIPPAGE, self.slippage_spin)
        self.objective_combo = QComboBox()
        for objective in Objective:
            self.objective_combo.addItem(objective.label, objective.value)
        form.addRow(texts.OPT_OBJECTIVE, self.objective_combo)
        outer.addWidget(group)

        self.ranges_group = QGroupBox(texts.OPT_RANGES_GROUP)
        ranges_layout = QVBoxLayout(self.ranges_group)
        hint = QLabel(texts.OPT_RANGES_HINT)
        hint.setWordWrap(True)
        ranges_layout.addWidget(hint)
        self.ranges_grid = QGridLayout()
        ranges_layout.addLayout(self.ranges_grid)
        self.combinations_label = QLabel()
        self.combinations_label.setWordWrap(True)
        ranges_layout.addWidget(self.combinations_label)
        outer.addWidget(self.ranges_group)

        self.wf_group = QGroupBox(texts.OPT_MODE_GROUP)
        wf_form = QFormLayout(self.wf_group)
        self.wf_check = QCheckBox(texts.OPT_WALK_FORWARD)
        self.wf_check.toggled.connect(self._on_mode_changed)
        wf_form.addRow(self.wf_check)
        self.train_spin = QSpinBox()
        self.train_spin.setRange(1, 50)
        self.train_spin.setValue(5)
        self.test_spin = QSpinBox()
        self.test_spin.setRange(1, 20)
        self.test_spin.setValue(1)
        self.anchored_check = QCheckBox(texts.OPT_ANCHORED)
        wf_form.addRow(texts.OPT_TRAIN_YEARS, self.train_spin)
        wf_form.addRow(texts.OPT_TEST_YEARS, self.test_spin)
        wf_form.addRow(self.anchored_check)
        outer.addWidget(self.wf_group)

        mc_group = QGroupBox(texts.OPT_MC_GROUP)
        mc_form = QFormLayout(mc_group)
        self.mc_spin = QSpinBox()
        self.mc_spin.setRange(0, 100_000)
        self.mc_spin.setSingleStep(100)
        self.mc_spin.setValue(1000)
        self.mc_method_combo = QComboBox()
        self.mc_method_combo.addItem(texts.OPT_MC_SHUFFLE, MonteCarloMethod.SHUFFLE.value)
        self.mc_method_combo.addItem(texts.OPT_MC_BOOTSTRAP, MonteCarloMethod.BOOTSTRAP.value)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(42)
        mc_form.addRow(texts.OPT_MC_SIMULATIONS, self.mc_spin)
        mc_form.addRow(texts.OPT_MC_METHOD, self.mc_method_combo)
        mc_form.addRow(texts.OPT_MC_SEED, self.seed_spin)
        outer.addWidget(mc_group)

        buttons = QHBoxLayout()
        self.run_button = QPushButton(texts.OPT_RUN)
        self.run_button.clicked.connect(self.start)
        self.cancel_button = QPushButton(texts.OPT_CANCEL)
        self.cancel_button.clicked.connect(self.cancel)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        outer.addLayout(buttons)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel(texts.OPT_IDLE)
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("OptimizationStatus")
        outer.addWidget(self.progress_bar)
        outer.addWidget(self.status_label)
        outer.addStretch(1)
        self._on_universe_kind_changed(True)
        self._on_mode_changed(False)
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
    def _spin(
        value: float, minimum: float, maximum: float, decimals: int, suffix: str = ""
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(suffix)
        spin.setValue(value)
        return spin

    # -- Aufbau: Ergebnis ------------------------------------------------------------

    def _build_results(self) -> QWidget:
        self.result_label = QLabel()
        self.result_label.setObjectName("OptimizationResultHeader")
        self.result_label.setWordWrap(True)
        best_bar = QHBoxLayout()
        self.best_label = QLabel()
        self.best_label.setWordWrap(True)
        best_bar.addWidget(self.best_label, 1)
        self.apply_button = QPushButton(texts.OPT_APPLY)
        self.apply_button.clicked.connect(self.apply_best)
        best_bar.addWidget(self.apply_button)
        self.result_tabs = QTabWidget()

        table_page = QWidget()
        table_layout = QVBoxLayout(table_page)
        table_hint = QLabel(texts.OPT_TABLE_HINT)
        table_hint.setWordWrap(True)
        table_layout.addWidget(table_hint)
        self.results_table = QTableWidget(0, 0)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        table_layout.addWidget(self.results_table)
        self.result_tabs.addTab(table_page, texts.OPT_RESULT_TABLE)

        heat_page = QWidget()
        heat_layout = QVBoxLayout(heat_page)
        axes = QHBoxLayout()
        self.heat_x_combo = QComboBox()
        self.heat_y_combo = QComboBox()
        self.heat_x_combo.currentIndexChanged.connect(lambda _i: self._show_heatmap())
        self.heat_y_combo.currentIndexChanged.connect(lambda _i: self._show_heatmap())
        axes.addWidget(QLabel(texts.OPT_HEATMAP_X))
        axes.addWidget(self.heat_x_combo, 1)
        axes.addWidget(QLabel(texts.OPT_HEATMAP_Y))
        axes.addWidget(self.heat_y_combo, 1)
        heat_layout.addLayout(axes)
        self.heat_label = QLabel(texts.OPT_HEATMAP_NEED_TWO)
        self.heat_label.setWordWrap(True)
        heat_layout.addWidget(self.heat_label)
        self.heat_table = QTableWidget(0, 0)
        self.heat_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.heat_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        heat_layout.addWidget(self.heat_table, 1)
        self.result_tabs.addTab(heat_page, texts.OPT_RESULT_HEATMAP)

        wf_page = QWidget()
        wf_layout = QVBoxLayout(wf_page)
        self.wf_label = QLabel(texts.OPT_WF_EMPTY)
        self.wf_label.setWordWrap(True)
        wf_layout.addWidget(self.wf_label)
        self.wf_graphics = pg.GraphicsLayoutWidget()
        self.wf_plot: pg.PlotItem = self.wf_graphics.addPlot(
            row=0, col=0, axisItems={"bottom": pg.DateAxisItem(orientation="bottom")}
        )
        self.wf_plot.setTitle(texts.OPT_WF_EQUITY_TITLE)
        self.wf_curve = pg.PlotDataItem()
        self.wf_plot.addItem(self.wf_curve)
        self.wf_table = QTableWidget(0, 0)
        self.wf_table.verticalHeader().setVisible(False)
        self.wf_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.wf_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        wf_split = QSplitter(Qt.Orientation.Vertical)
        wf_split.addWidget(self.wf_graphics)
        wf_split.addWidget(self.wf_table)
        wf_layout.addWidget(wf_split, 1)
        self.result_tabs.addTab(wf_page, texts.OPT_RESULT_WALK_FORWARD)

        mc_page = QWidget()
        mc_layout = QVBoxLayout(mc_page)
        self.mc_label = QLabel(texts.OPT_MC_EMPTY)
        self.mc_label.setWordWrap(True)
        mc_layout.addWidget(self.mc_label)
        mc_bar = QHBoxLayout()
        mc_bar.addWidget(QLabel(texts.OPT_MC_VALUE))
        self.mc_value_combo = QComboBox()
        self.mc_value_combo.addItem(texts.OPT_MC_VALUE_DRAWDOWN, "drawdown")
        self.mc_value_combo.addItem(texts.OPT_MC_VALUE_EQUITY, "equity")
        self.mc_value_combo.setToolTip(texts.OPT_MC_VALUE_HINT)
        self.mc_value_combo.currentIndexChanged.connect(lambda _i: self._show_histogram())
        mc_bar.addWidget(self.mc_value_combo, 1)
        mc_layout.addLayout(mc_bar)
        self.mc_graphics = pg.GraphicsLayoutWidget()
        self.mc_plot: pg.PlotItem = self.mc_graphics.addPlot(row=0, col=0)
        self.mc_bars = pg.BarGraphItem(x0=[], x1=[], height=[], y0=0)
        self.mc_plot.addItem(self.mc_bars)
        self.mc_line = pg.InfiniteLine(angle=90, movable=False)
        self.mc_line.setVisible(False)
        self.mc_plot.addItem(self.mc_line)
        mc_layout.addWidget(self.mc_graphics, 1)
        self.result_tabs.addTab(mc_page, texts.OPT_RESULT_MONTE_CARLO)

        for plot in (self.wf_plot, self.mc_plot):
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            plot.getAxis("left").setWidth(60)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.result_tabs.addTab(self.log_view, texts.OPT_RESULT_LOG)

        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.result_label)
        layout.addLayout(best_bar)
        layout.addWidget(self.result_tabs, 1)
        return box

    def _apply_palette(self) -> None:
        palette = self.palette()
        background = palette.color(QPalette.ColorRole.Base)
        foreground = palette.color(QPalette.ColorRole.Text)
        accent = palette.color(QPalette.ColorRole.Highlight)
        for graphics in (self.wf_graphics, self.mc_graphics):
            graphics.setBackground(background)
        for plot in (self.wf_plot, self.mc_plot):
            for name in ("left", "bottom"):
                axis = plot.getAxis(name)
                axis.setPen(foreground)
                axis.setTextPen(foreground)
            plot.titleLabel.setAttr("color", foreground)
        self.wf_curve.setPen(pg.mkPen(accent, width=2))
        fill = QColor(accent)
        fill.setAlpha(160)
        self.mc_bars.setOpts(brush=pg.mkBrush(fill), pen=pg.mkPen(accent))
        self.mc_line.setPen(pg.mkPen(foreground, width=2, style=Qt.PenStyle.DashLine))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 (Qt-Override)
        """Übernimmt Farben beim Wechsel zwischen Hell- und Dunkelmodus."""
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._apply_palette()
        super().changeEvent(event)

    # -- Strategien und Raster -------------------------------------------------------

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
            self.status_label.setText(texts.BACKTEST_NO_STRATEGY)

    @property
    def strategy_id(self) -> str | None:
        """ID der gewählten Strategie."""
        data = self.strategy_combo.currentData()
        return str(data) if data is not None else None

    def select_strategy(self, strategy_id: str) -> bool:
        """Wählt ``strategy_id``; lädt die Plugins neu, falls sie fehlt.

        Returns:
            ``True``, wenn die Strategie gewählt ist.
        """
        if self.strategy_combo.findData(strategy_id) < 0:
            self._service.reload_strategies()
            self._fill_strategies()
        index = self.strategy_combo.findData(strategy_id)
        if index < 0:
            self._fail(texts.BACKTEST_UNKNOWN_STRATEGY.format(strategy=strategy_id))
            return False
        self.strategy_combo.setCurrentIndex(index)
        return True

    def _on_strategy_changed(self, _index: int) -> None:
        strategy_id = self.strategy_id
        if strategy_id is None:
            return
        self.description_label.setText(self._service.strategy_description(strategy_id))
        self._build_ranges(self._service.strategy_params(strategy_id))
        current = self.timeframe_combo.currentData()
        self.timeframe_combo.clear()
        for tf in self._service.strategy_timeframes(strategy_id):
            self.timeframe_combo.addItem(tf.value, tf.value)
        self.timeframe_combo.setCurrentIndex(max(self.timeframe_combo.findData(current), 0))
        self._update_buttons()

    def _build_ranges(self, params: dict[str, Param]) -> None:
        grid = self.ranges_grid
        while grid.count():
            item = grid.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        self._rows = {}
        if not params:
            grid.addWidget(QLabel(texts.OPT_NO_PARAMS), 0, 0, 1, 4)
            self._update_combinations()
            return
        for column, title in enumerate(
            (texts.OPT_COL_PARAM, texts.OPT_COL_FROM, texts.OPT_COL_TO, texts.OPT_COL_STEP)
        ):
            grid.addWidget(QLabel(f"<b>{title}</b>"), 0, column)
        for row, (name, param) in enumerate(params.items(), start=1):
            check = QCheckBox(name)
            check.setToolTip(param.description)
            check.setObjectName(f"vary_{name}")
            check.toggled.connect(lambda _c: self._update_combinations())
            grid.addWidget(check, row, 0)
            if param.kind in ("int", "float"):
                low, high, step = self._range_spins(param)
                for column, spin in enumerate((low, high, step), start=1):
                    spin.valueChanged.connect(lambda _v: self._update_combinations())
                    grid.addWidget(spin, row, column)
                self._rows[name] = _RangeRow(param, check, low, high, step)
            else:
                grid.addWidget(QLabel(texts.OPT_ALL_VALUES), row, 1, 1, 3)
                self._rows[name] = _RangeRow(param, check, None, None, None)
        self._update_combinations()

    @staticmethod
    def _range_spins(param: Param) -> tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]:
        is_int = param.kind == "int"
        decimals = 0 if is_int else 4
        default = float(param.default)
        low_limit = float(param.min) if param.min is not None else -_MAX_SPIN
        high_limit = float(param.max) if param.max is not None else _MAX_SPIN
        step = float(param.step) if param.step else (1.0 if is_int else 0.1)
        spins = []
        for value, minimum, maximum in (
            (float(param.min) if param.min is not None else default, low_limit, high_limit),
            (float(param.max) if param.max is not None else default, low_limit, high_limit),
            (step, 1.0 if is_int else 10.0**-decimals, _MAX_SPIN),
        ):
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(1.0 if is_int else step)
            spin.setValue(value)
            spin.setMinimumWidth(1)
            spins.append(spin)
        return spins[0], spins[1], spins[2]

    def set_range(
        self,
        name: str,
        low: float | None = None,
        high: float | None = None,
        step: float | None = None,
        vary: bool = True,
    ) -> None:
        """Setzt den Bereich des Parameters ``name`` und ob er variiert wird.

        Raises:
            KeyError: Unbekannter Parameter.
        """
        row = self._rows[name]
        row.check.setChecked(vary)
        for spin, value in ((row.low, low), (row.high, high), (row.step, step)):
            if spin is not None and value is not None:
                spin.setValue(float(value))

    def ranges(self) -> dict[str, RangeSpec]:
        """Bereiche der angehakten Parameter (Zahlen: :class:`ParamRange`, sonst alle Werte)."""
        result: dict[str, RangeSpec] = {}
        for name, row in self._rows.items():
            if not row.check.isChecked():
                continue
            if row.low is None or row.high is None or row.step is None:
                result[name] = None
            else:
                result[name] = ParamRange(row.low.value(), row.high.value(), row.step.value())
        return result

    def _update_combinations(self) -> None:
        strategy_id = self.strategy_id
        if strategy_id is None:
            self.combinations_label.setText("")
            return
        try:
            count = self._service.count_combinations(strategy_id, self.ranges())
        except Exception as exc:  # jeder Eingabefehler wird angezeigt
            self.combinations_label.setText(texts.OPT_COMBINATIONS_INVALID.format(message=exc))
            self.combinations_label.setStyleSheet(f"color: {DOWN_COLOR};")
            return
        text = format_metric_value(count, "int")
        if count > MAX_COMBINATIONS:
            self.combinations_label.setText(texts.OPT_COMBINATIONS_WARN.format(count=text))
            self.combinations_label.setStyleSheet(f"color: {DOWN_COLOR};")
        else:
            self.combinations_label.setText(texts.OPT_COMBINATIONS.format(count=text))
            self.combinations_label.setStyleSheet("")

    # -- Formular --------------------------------------------------------------------

    def _on_universe_kind_changed(self, _checked: bool) -> None:
        single = self.symbol_radio.isChecked()
        self.symbol_edit.setVisible(single)
        self.index_combo.setVisible(not single)
        self.max_positions_spin.setVisible(not single)
        self.max_positions_label.setVisible(not single)

    def _on_mode_changed(self, checked: bool) -> None:
        for widget in (self.train_spin, self.test_spin, self.anchored_check):
            widget.setEnabled(checked)

    def set_symbol(self, symbol: str) -> None:
        """Wählt Einzelwert und trägt ``symbol`` ein."""
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

    def set_period(self, start: dt.date, end: dt.date) -> None:
        """Setzt den Zeitraum."""
        self.start_edit.setDate(_to_qdate(start))
        self.end_edit.setDate(_to_qdate(end))

    def set_objective(self, objective: Objective) -> None:
        """Wählt die Zielgröße."""
        self.objective_combo.setCurrentIndex(self.objective_combo.findData(objective.value))

    def request(self) -> OptimizationRequest:
        """Erzeugt die Anfrage aus den Formularfeldern.

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
        mode = OptimizationMode.WALK_FORWARD if self.wf_check.isChecked() else OptimizationMode.GRID
        return OptimizationRequest(
            strategy_id=strategy_id,
            ranges=self.ranges(),
            universe=universe,
            universe_kind=kind,
            timeframe=Timeframe(self.timeframe_combo.currentData()),
            start=_from_qdate(self.start_edit.date()),
            end=_from_qdate(self.end_edit.date()),
            objective=Objective(self.objective_combo.currentData()),
            mode=mode,
            train_years=int(self.train_spin.value()),
            test_years=int(self.test_spin.value()),
            anchored=self.anchored_check.isChecked(),
            capital=float(self.capital_spin.value()),
            fee_per_order=float(self.fee_spin.value()),
            fee_pct=float(self.fee_pct_spin.value()),
            slippage_pct=float(self.slippage_spin.value()),
            max_positions=int(self.max_positions_spin.value()),
            mc_simulations=int(self.mc_spin.value()),
            mc_method=MonteCarloMethod(self.mc_method_combo.currentData()),
            seed=int(self.seed_spin.value()),
        )

    # -- Lauf und Abbruch ------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True``, solange ein Lauf aktiv ist."""
        return self._cancel_event is not None

    @property
    def outcome(self) -> OptimizationOutcome | None:
        """Aktuell angezeigtes Ergebnis."""
        return self._outcome

    def _update_buttons(self) -> None:
        running = self.is_running
        self.run_button.setEnabled(not running and self.strategy_id is not None)
        cancelling = self._cancel_event is not None and self._cancel_event.is_set()
        self.cancel_button.setEnabled(running and not cancelling)
        best = self._outcome is not None and self._best_params() is not None
        self.apply_button.setEnabled(best and not running)

    def start(self) -> None:
        """Startet die Optimierung mit den Formularwerten im Hintergrund."""
        if self.is_running:
            return
        try:
            request = self.request()
        except ValueError as exc:
            self._fail(str(exc))
            return
        event = threading.Event()
        self._cancel_event = event
        self.progress_bar.setValue(0)
        self.status_label.setText(texts.OPT_RUNNING)
        self._update_buttons()
        self._runner(
            self._run,
            request,
            event,
            on_result=self._on_run_finished,
            name=texts.OPT_TASK.format(strategy=request.strategy_id, universe=request.universe),
            report_progress=True,
            on_progress=self._on_progress,
        )

    def cancel(self) -> None:
        """Bricht den laufenden Lauf nach der aktuellen Kombination ab."""
        if self._cancel_event is None or self._cancel_event.is_set():
            return
        self._cancel_event.set()
        self.status_label.setText(texts.OPT_CANCELLING)
        self._update_buttons()

    def _run(
        self,
        request: OptimizationRequest,
        event: threading.Event,
        progress: Callable[[int, str], None] | None = None,
    ) -> OptimizationOutcome | str:
        """Läuft im Hintergrund; Fehler werden als Text zurückgegeben statt geworfen."""
        try:
            return self._service.run(request, progress=progress, should_cancel=event.is_set)
        except OptimizationCancelledError:
            return CANCELLED
        except Exception as exc:  # jeder Fehler wird im Tab angezeigt
            return str(exc) or type(exc).__name__

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        if self._cancel_event is not None and self._cancel_event.is_set():
            return
        self.status_label.setText(texts.BACKTEST_PROGRESS.format(percent=percent, message=message))

    def _on_run_finished(self, payload: OptimizationOutcome | str) -> None:
        self._cancel_event = None
        if payload == CANCELLED:
            self.progress_bar.setValue(0)
            self.status_label.setText(texts.OPT_CANCELLED)
            self._update_buttons()
            self.cancelled.emit()
            return
        if isinstance(payload, str):
            self.progress_bar.setValue(0)
            self._fail(payload)
            self._update_buttons()
            return
        self.progress_bar.setValue(100)
        self.set_outcome(payload)
        self._update_buttons()

    def _fail(self, message: str) -> None:
        text = texts.OPT_FAILED.format(message=message)
        self.status_label.setText(text)
        self.failed.emit(text)

    # -- Ergebnisanzeige -------------------------------------------------------------

    def set_outcome(self, outcome: OptimizationOutcome) -> None:
        """Zeigt Tabelle, Heatmap, Walk-Forward, Monte Carlo und Protokoll."""
        self._outcome = outcome
        request = outcome.request
        result = outcome.optimization
        wf = outcome.walk_forward
        if wf is not None:
            header = texts.OPT_DONE_WF.format(
                strategy=request.strategy_id,
                universe=request.universe,
                timeframe=request.timeframe.value,
                windows=len(wf.windows),
            )
        else:
            header = texts.OPT_DONE.format(
                strategy=request.strategy_id,
                universe=request.universe,
                timeframe=request.timeframe.value,
                count=format_metric_value(len(result.table), "int"),
                objective=result.objective.label,
            )
        self.result_label.setText(header)
        self.status_label.setText(header)
        self.best_label.setText(self._best_text(result))
        self._show_table(result)
        self._fill_heatmap_axes(result)
        self._show_walk_forward(wf)
        self._show_monte_carlo(outcome.monte_carlo)
        self.log_view.setPlainText("\n".join(outcome.log))
        self._update_buttons()
        self.finished.emit(outcome)

    @staticmethod
    def _best_text(result: OptimizationResult) -> str:
        params = result.best_params
        if params is None:
            return texts.OPT_BEST_NONE
        robustness = result.best_robustness
        return texts.OPT_BEST.format(
            params=format_params(params),
            value=format_metric_value(result.best_value, "ratio"),
            robustness=format_metric_value(None if robustness is None else robustness * 100, "pct"),
        )

    def _best_params(self) -> dict[str, Any] | None:
        outcome = self._outcome
        if outcome is None:
            return None
        if outcome.walk_forward is not None and outcome.walk_forward.windows:
            return dict(outcome.walk_forward.windows[-1].params)
        return outcome.optimization.best_params

    def apply_best(self) -> None:
        """Sendet die besten Parameter (Walk-Forward: des letzten Fensters) an den Backtest."""
        params = self._best_params()
        if self._outcome is None or params is None:
            return
        self.apply_requested.emit(self._outcome.request, params)

    def _table_columns(self, result: OptimizationResult) -> list[tuple[str, str]]:
        columns = [(name, name) for name in result.param_names]
        columns.append(("objective", f"{texts.OPT_COL_OBJECTIVE} ({result.objective.label})"))
        columns += [(c, texts.OPT_METRIC_TITLES.get(c, c)) for c in METRIC_COLUMNS]
        columns.append(("robustness", texts.OPT_COL_ROBUSTNESS))
        columns.append(("error", texts.OPT_COL_ERROR))
        return columns

    def _show_table(self, result: OptimizationResult) -> None:
        table = self.results_table
        columns = self._table_columns(result)
        table.setSortingEnabled(False)
        table.clear()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels([title for _key, title in columns])
        frame = result.table
        table.setRowCount(len(frame))
        bold = QFont(table.font())
        bold.setBold(True)
        records: list[dict[Any, Any]] = frame.to_dict("records")
        for row, record in enumerate(records):
            for column, (key, _title) in enumerate(columns):
                value = record[key]
                item = _SortItem(self._cell_text(key, value, result))
                item.setData(_ROW_ROLE, row)
                if key in result.grid:
                    sort_value: Any = (
                        float(value)
                        if isinstance(value, int | float) and not isinstance(value, bool)
                        else str(value)
                    )
                elif key == "error":
                    sort_value = str(value or "")
                else:
                    sort_value = math.nan if _is_missing(value) else float(value)
                item.setData(_SORT_ROLE, sort_value)
                if key != "error":
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if row == result.best_index:
                    item.setFont(bold)
                table.setItem(row, column, item)
        table.setSortingEnabled(True)

    @staticmethod
    def _cell_text(key: str, value: Any, result: OptimizationResult) -> str:
        if key in result.grid:
            return format_param(value)
        if key == "error":
            return str(value or "")
        if _is_missing(value):
            return "–"
        if key == "robustness":
            return format_metric_value(float(value) * 100, "pct")
        if key == "objective":
            return format_metric_value(value, "ratio")
        return format_metric_value(value, _METRIC_FORMATS.get(key, "ratio"))

    def table_column(self, key: str) -> list[str]:
        """Texte der Spalte ``key`` in angezeigter Reihenfolge (z. B. nach Sortieren)."""
        outcome = self._outcome
        if outcome is None:
            return []
        keys = [k for k, _t in self._table_columns(outcome.optimization)]
        column = keys.index(key)
        texts_: list[str] = []
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, column)
            texts_.append(item.text() if item is not None else "")
        return texts_

    def sort_by(self, key: str, descending: bool = True) -> None:
        """Sortiert die Ergebnis-Tabelle nach der Spalte ``key``."""
        outcome = self._outcome
        if outcome is None:
            return
        keys = [k for k, _t in self._table_columns(outcome.optimization)]
        order = Qt.SortOrder.DescendingOrder if descending else Qt.SortOrder.AscendingOrder
        self.results_table.sortItems(keys.index(key), order)

    # -- Heatmap ---------------------------------------------------------------------

    def _fill_heatmap_axes(self, result: OptimizationResult) -> None:
        varied = [name for name, values in result.grid.items() if len(values) > 1]
        names = varied if len(varied) >= 2 else list(result.grid)
        for combo in (self.heat_x_combo, self.heat_y_combo):
            combo.blockSignals(True)
            combo.clear()
            for name in names:
                combo.addItem(name, name)
            combo.blockSignals(False)
        if len(names) >= 2:
            self.heat_x_combo.setCurrentIndex(0)
            self.heat_y_combo.setCurrentIndex(1)
        self._show_heatmap()

    def set_heatmap_axes(self, x: str, y: str) -> None:
        """Wählt die Parameter der Heatmap-Spalten (``x``) und -Zeilen (``y``).

        Raises:
            KeyError: Parameter steht nicht zur Wahl.
        """
        for combo, name in ((self.heat_x_combo, x), (self.heat_y_combo, y)):
            index = combo.findData(name)
            if index < 0:
                raise KeyError(name)
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
        self._show_heatmap()

    def heatmap_matrix(self) -> pd.DataFrame | None:
        """Matrix der aktuell gezeigten Heatmap oder ``None``."""
        outcome = self._outcome
        x, y = self.heat_x_combo.currentData(), self.heat_y_combo.currentData()
        if outcome is None or x is None or y is None or x == y:
            return None
        return outcome.optimization.heatmap(str(x), str(y))

    def _show_heatmap(self) -> None:
        table = self.heat_table
        matrix = self.heatmap_matrix()
        if matrix is None:
            table.setRowCount(0)
            table.setColumnCount(0)
            self.heat_label.setText(texts.OPT_HEATMAP_NEED_TWO)
            return
        self.heat_label.setText(texts.OPT_HEATMAP_FIXED)
        values = matrix.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        scale = float(np.max(np.abs(finite))) if finite.size else 0.0
        table.clear()
        table.setRowCount(len(matrix.index))
        table.setColumnCount(len(matrix.columns))
        x, y = self.heat_x_combo.currentData(), self.heat_y_combo.currentData()
        table.setHorizontalHeaderLabels([f"{x}={format_param(v)}" for v in matrix.columns])
        table.setVerticalHeaderLabels([f"{y}={format_param(v)}" for v in matrix.index])
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if np.isfinite(value):
                    item.setText(format_metric_value(value, "ratio"))
                    item.setBackground(QBrush(self.heat_color(value, scale)))
                else:
                    item.setText("–")
                table.setItem(row, column, item)

    @staticmethod
    def heat_color(value: float, scale: float) -> QColor:
        """Farbe einer Heatmap-Zelle: grün positiv, rot negativ, kräftiger je größer."""
        color = QColor(UP_COLOR if value >= 0 else DOWN_COLOR)
        strength = min(abs(value) / scale, 1.0) if scale > 0 else 0.0
        color.setAlpha(int(30 + strength * (_HEATMAP_MAX_ALPHA - 30)))
        return color

    def heatmap_cell(self, row: int, column: int) -> QTableWidgetItem | None:
        """Zelle der Heatmap."""
        return self.heat_table.item(row, column)

    # -- Walk-Forward ----------------------------------------------------------------

    def _show_walk_forward(self, wf: WalkForwardResult | None) -> None:
        empty = np.zeros(0)
        table = self.wf_table
        if wf is None:
            self.wf_label.setText(texts.OPT_WF_EMPTY)
            self.wf_curve.setData(x=empty, y=empty)
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        try:
            metrics = wf.metrics
            self.wf_label.setText(
                texts.OPT_WF_SUMMARY.format(
                    equity=format_metric_value(wf.final_equity, "money"),
                    cagr=format_metric_value(metrics.cagr_pct, "pct"),
                    drawdown=format_metric_value(metrics.max_drawdown_pct, "pct"),
                    sharpe=format_metric_value(metrics.sharpe, "ratio"),
                )
            )
        except MetricsError:
            self.wf_label.setText(
                texts.OPT_WF_SUMMARY.format(
                    equity=format_metric_value(wf.final_equity, "money"),
                    cagr="–",
                    drawdown="–",
                    sharpe="–",
                )
            )
        if wf.equity.empty:
            self.wf_curve.setData(x=empty, y=empty)
        else:
            x = pd.to_datetime(wf.equity["date"]).to_numpy(dtype="datetime64[s]")
            self.wf_curve.setData(
                x=x.astype(np.float64), y=wf.equity["equity"].to_numpy(dtype=float)
            )
            self.wf_plot.enableAutoRange()
        names = list(wf.windows[0].params) if wf.windows else []
        headers = [
            texts.OPT_COL_WINDOW,
            texts.OPT_COL_TRAIN,
            texts.OPT_COL_TEST,
            *names,
            texts.OPT_COL_TRAIN_VALUE,
            texts.OPT_COL_TEST_VALUE,
            texts.OPT_COL_ROBUSTNESS,
        ]
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(wf.windows))
        for row, window in enumerate(wf.windows):
            span = window.span
            robustness = None if window.robustness is None else window.robustness * 100
            cells = [
                str(span.index),
                f"{span.train_start:%d.%m.%Y} – {span.train_end:%d.%m.%Y}",
                f"{span.test_start:%d.%m.%Y} – {span.test_end:%d.%m.%Y}",
                *(format_param(window.params.get(n)) for n in names),
                format_metric_value(window.train_value, "ratio"),
                format_metric_value(window.test_value, "ratio"),
                format_metric_value(robustness, "pct"),
            ]
            for column, text in enumerate(cells):
                table.setItem(row, column, QTableWidgetItem(text))

    def walk_forward_points(self) -> int:
        """Anzahl der Punkte der Out-of-Sample-Kurve."""
        _x, y = self.wf_curve.getData()
        return 0 if y is None else len(y)

    # -- Monte Carlo -----------------------------------------------------------------

    def _show_monte_carlo(self, mc: MonteCarloResult | None) -> None:
        self._monte_carlo = mc
        if mc is None or mc.final_equity.size == 0:
            self.mc_label.setText(texts.OPT_MC_EMPTY)
        else:
            equity = mc.final_equity_percentiles
            drawdown = mc.max_drawdown_percentiles
            self.mc_label.setText(
                texts.OPT_MC_SUMMARY.format(
                    n=format_metric_value(mc.n_simulations, "int"),
                    trades=format_metric_value(mc.n_trades, "int"),
                    p5=format_metric_value(equity.p5, "money"),
                    p50=format_metric_value(equity.p50, "money"),
                    p95=format_metric_value(equity.p95, "money"),
                    dd=format_metric_value(drawdown.p5, "pct"),
                    loss=format_metric_value(mc.loss_probability_pct, "pct"),
                )
            )
            # Umordnen ändert das Endkapital nicht, nur den Weg dorthin: dann zeigt das
            # Histogramm den Drawdown, beim Ziehen mit Zurücklegen das Endkapital.
            value = "drawdown" if mc.method is MonteCarloMethod.SHUFFLE else "equity"
            self.mc_value_combo.blockSignals(True)
            self.mc_value_combo.setCurrentIndex(self.mc_value_combo.findData(value))
            self.mc_value_combo.blockSignals(False)
        self._show_histogram()

    def set_histogram_value(self, value: str) -> None:
        """Wählt die Verteilung des Histogramms: ``"drawdown"`` oder ``"equity"``.

        Raises:
            KeyError: Unbekannter Wert.
        """
        index = self.mc_value_combo.findData(value)
        if index < 0:
            raise KeyError(value)
        self.mc_value_combo.setCurrentIndex(index)

    def _show_histogram(self) -> None:
        mc = self._monte_carlo
        if mc is None or mc.final_equity.size == 0:
            self._histogram = None
            self.mc_bars.setOpts(x0=[], x1=[], height=[])
            self.mc_line.setVisible(False)
            return
        if self.mc_value_combo.currentData() == "equity":
            values, original = mc.final_equity, mc.original_final_equity
            self.mc_plot.setTitle(texts.OPT_MC_TITLE_EQUITY)
        else:
            values, original = mc.max_drawdown_pct, mc.original_max_drawdown_pct
            self.mc_plot.setTitle(texts.OPT_MC_TITLE_DRAWDOWN)
        counts, edges = histogram_bins(values)
        self._histogram = (counts, edges)
        self.mc_bars.setOpts(x0=edges[:-1], x1=edges[1:], height=counts)
        self.mc_line.setValue(original)
        self.mc_line.setVisible(True)
        self.mc_plot.enableAutoRange()

    def histogram(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Häufigkeiten und Klassengrenzen des Monte-Carlo-Histogramms oder ``None``."""
        return self._histogram
