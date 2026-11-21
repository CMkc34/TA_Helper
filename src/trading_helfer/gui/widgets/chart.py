"""Kurschart auf pyqtgraph: Kerzen, Volumen, Zeitrahmen, Fadenkreuz, log/linear,
Indikatoren (im Kurs-Chart oder in eigenen Panels) und Marker für Signale und Trades.

Die x-Achse zählt Kerzen (0, 1, 2 …) statt Kalendertage; dadurch entstehen keine
Lücken für Wochenenden und Feiertage. :class:`DateAxis` beschriftet die Positionen
mit dem Datum der Kerze. Kerzen und Volumen werden je Farbe als ein Grafik-Element
gezeichnet (vektorisiert), damit auch 25 Jahre Tagesdaten schnell erscheinen.

Indikatoren kommen aus der Registry (über :class:`IndicatorService`); der Dialog erzeugt
ihre Parameterfelder aus den Metadaten, daher braucht kein Indikator eigenen GUI-Code.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from trading_helfer.core.markers import Marker, MarkerKind
from trading_helfer.core.types import Timeframe
from trading_helfer.gui import texts
from trading_helfer.gui.widgets.indicator_dialog import IndicatorDialog
from trading_helfer.indicators.registry import IndicatorSpec
from trading_helfer.services.chart_service import ChartData, ChartService
from trading_helfer.services.indicator_service import (
    IndicatorConfig,
    IndicatorError,
    IndicatorService,
    TemplateError,
    marker_indices,
)

__all__ = [
    "DEFAULT_VISIBLE_BARS",
    "MARKER_COLORS",
    "TIMEFRAMES",
    "CandleArrays",
    "ChartWidget",
    "DateAxis",
    "TaskRunner",
    "format_price",
    "format_volume",
]

TIMEFRAMES: Final[tuple[Timeframe, ...]] = (Timeframe.D1, Timeframe.D3, Timeframe.W1, Timeframe.MN1)
"""Zeitrahmen in der Reihenfolge der Umschalter."""

DEFAULT_VISIBLE_BARS: Final = 250
"""Anzahl Kerzen, die nach dem Laden sichtbar sind (die jüngsten)."""

UP_COLOR: Final = "#26a69a"
DOWN_COLOR: Final = "#ef5350"
PARTIAL_ALPHA: Final = 90
"""Deckkraft (0–255) der unvollständigen letzten Kerze."""

BODY_WIDTH: Final = 0.7
LINE_WIDTH: Final = 1.5
MARKER_COLORS: Final[dict[MarkerKind, str]] = {
    MarkerKind.BUY: "#2e7d32",
    MarkerKind.SELL: "#c62828",
    MarkerKind.EXIT: "#ef6c00",
    MarkerKind.SIGNAL: "#1565c0",
}
"""Standardfarbe je Marker-Art."""
_MARKER_SYMBOLS: Final[dict[MarkerKind, str]] = {
    MarkerKind.BUY: "t1",  # Dreieck nach oben
    MarkerKind.SELL: "t",  # Dreieck nach unten
    MarkerKind.EXIT: "x",
    MarkerKind.SIGNAL: "d",
}
_MARKER_SIZE: Final = 12
_MARKER_OFFSET: Final = 0.02
"""Abstand der Marker zu Tief bzw. Hoch als Anteil der sichtbaren Preisspanne."""
_STRETCH_PRICE: Final = 8
_STRETCH_VOLUME: Final = 2
_STRETCH_PANE: Final = 3
_Y_PADDING: Final = 0.05
_RIGHT_MARGIN_BARS: Final = 3
_MIN_VISIBLE_BARS: Final = 5
_TIMEFRAME_BUTTON_WIDTH: Final = 44
_BAR_DAYS: Final[dict[Timeframe, float]] = {
    Timeframe.D1: 1.4,  # Kalendertage je Kerze (inkl. Wochenenden)
    Timeframe.D3: 4.2,
    Timeframe.W1: 7.0,
    Timeframe.MN1: 30.4,
}

FloatArray = npt.NDArray[np.float64]


class TaskRunner(Protocol):
    """Führt ``fn(*args)`` aus (z. B. im Thread-Pool) und ruft ``on_result`` im GUI-Thread."""

    def __call__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None] | None = ...,
        name: str = ...,
    ) -> Any:
        """Startet die Aufgabe."""
        ...


def _run_inline(
    fn: Callable[..., Any],
    *args: Any,
    on_result: Callable[[Any], None] | None = None,
    name: str = "",
) -> None:
    """Standard-Runner: führt die Aufgabe sofort im aufrufenden Thread aus."""
    del name
    result = fn(*args)
    if on_result is not None:
        on_result(result)


def format_price(value: float) -> str:
    """Formatiert einen Kurs deutsch mit zwei Nachkommastellen (``1.234,50``)."""
    return f"{value:,.2f}".replace(",", "\0").replace(".", ",").replace("\0", ".")


def format_volume(value: float) -> str:
    """Formatiert ein Volumen deutsch mit Tausenderpunkten (``1.234.567``)."""
    return f"{round(value):,}".replace(",", ".")


@dataclass(frozen=True)
class CandleArrays:
    """Kerzen als NumPy-Arrays (eine Zeile je Kerze, Position = Index)."""

    dates: npt.NDArray[np.datetime64]
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray
    volume: FloatArray
    partial: npt.NDArray[np.bool_]

    @classmethod
    def from_chart_data(cls, data: ChartData) -> CandleArrays:
        """Übernimmt die Spalten aus ``data.bars``."""
        bars = data.bars
        return cls(
            dates=bars["date"].to_numpy(dtype="datetime64[D]"),
            open=bars["open"].to_numpy(dtype=np.float64),
            high=bars["high"].to_numpy(dtype=np.float64),
            low=bars["low"].to_numpy(dtype=np.float64),
            close=bars["close"].to_numpy(dtype=np.float64),
            volume=bars["volume"].to_numpy(dtype=np.float64),
            partial=bars["is_partial"].to_numpy(dtype=np.bool_),
        )

    def __len__(self) -> int:
        """Anzahl der Kerzen."""
        return len(self.close)


class DateAxis(pg.AxisItem):  # type: ignore[misc]
    """x-Achse, die Kerzenpositionen mit dem Datum der Kerze beschriftet."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(orientation="bottom", **kwargs)
        self._dates: npt.NDArray[np.datetime64] = np.array([], dtype="datetime64[D]")
        self._bar_days = _BAR_DAYS[Timeframe.D1]

    def set_dates(self, dates: npt.NDArray[np.datetime64], timeframe: Timeframe) -> None:
        """Setzt die Datumswerte je Kerzenposition."""
        self._dates = dates
        self._bar_days = _BAR_DAYS[timeframe]
        self.picture = None
        self.update()

    def label_for(self, value: float, spacing: float) -> str:
        """Beschriftung für Position ``value`` bei Tick-Abstand ``spacing`` (in Kerzen)."""
        index = round(value)
        if abs(value - index) > 1e-6 or not 0 <= index < len(self._dates):
            return ""
        day = self._dates[index].astype(object)
        span_days = spacing * self._bar_days
        if span_days >= 300:
            return f"{day:%Y}"
        if span_days >= 25:
            return f"{day:%m.%Y}"
        return f"{day:%d.%m.%y}"

    def tickStrings(  # noqa: N802 (pyqtgraph-Override)
        self, values: list[float], scale: float, spacing: float
    ) -> list[str]:
        """Beschriftet jede Tick-Position mit dem Datum der Kerze."""
        del scale
        return [self.label_for(value, spacing) for value in values]


class _CandleLayer:
    """Kerzenkörper, Dochte und Volumenbalken einer Farbe."""

    def __init__(self, price_plot: pg.PlotItem, volume_plot: pg.PlotItem, color: QColor) -> None:
        pen = pg.mkPen(color, width=1)
        brush = pg.mkBrush(color)
        empty = np.zeros(0)
        self.wicks = pg.PlotCurveItem(x=empty, y=empty, pen=pen, connect="pairs")
        self.bodies = pg.BarGraphItem(
            x=empty, y0=empty, height=empty, width=BODY_WIDTH, pen=pen, brush=brush
        )
        self.volume = pg.BarGraphItem(
            x=empty, y0=empty, height=empty, width=BODY_WIDTH, pen=pen, brush=brush
        )
        price_plot.addItem(self.wicks)
        price_plot.addItem(self.bodies)
        volume_plot.addItem(self.volume)
        self.count = 0

    def set_data(
        self,
        x: FloatArray,
        open_: FloatArray,
        high: FloatArray,
        low: FloatArray,
        close: FloatArray,
        volume: FloatArray,
    ) -> None:
        self.count = len(x)
        self.wicks.setData(
            x=np.repeat(x, 2), y=np.column_stack((low, high)).ravel(), connect="pairs"
        )
        bottom = np.minimum(open_, close)
        self.bodies.setOpts(x=x, y0=bottom, height=np.maximum(open_, close) - bottom)
        self.volume.setOpts(x=x, y0=np.zeros_like(volume), height=volume)


@dataclass
class _ActiveIndicator:
    """Ein Indikator im Chart mit seinen Grafik-Elementen und berechneten Werten."""

    key: int
    config: IndicatorConfig
    spec: IndicatorSpec
    plot: pg.PlotItem
    label: str
    items: dict[str, Any] = field(default_factory=dict)
    values: dict[str, FloatArray] = field(default_factory=dict)
    error: str = ""


class ChartWidget(QWidget):
    """Kerzenchart mit Volumen-Panel, Zeitrahmen-Umschalter, Fadenkreuz, Indikatoren
    und Markern.

    Args:
        service: Liefert die Kerzen (lädt, bereinigt und rechnet Zeitrahmen um).
        runner: Startet das Laden, z. B. :meth:`MainWindow.run_task` für den
            Thread-Pool; ohne Angabe wird sofort im GUI-Thread geladen.
        indicator_service: Liefert Indikatoren und Vorlagen; Standard ist ein
            :class:`IndicatorService` ohne gespeicherte Vorlagen.
    """

    data_loaded = Signal(object)
    """Wird mit den neuen :class:`ChartData` gesendet, nachdem sie gezeichnet sind."""

    load_failed = Signal(str)
    """Wird mit der deutschen Fehlermeldung gesendet, wenn das Laden scheitert."""

    indicators_changed = Signal()
    """Wird gesendet, wenn Indikatoren hinzugefügt, geändert oder entfernt wurden."""

    def __init__(
        self,
        service: ChartService,
        runner: TaskRunner | None = None,
        parent: QWidget | None = None,
        indicator_service: IndicatorService | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._indicator_service = (
            indicator_service if indicator_service is not None else IndicatorService()
        )
        self._indicators: dict[int, _ActiveIndicator] = {}
        self._next_key = 1
        self._panes: list[pg.PlotItem] = []
        self._pane_axes: dict[pg.PlotItem, DateAxis] = {}
        self._pane_lines: dict[pg.PlotItem, pg.InfiniteLine] = {}
        self._markers: list[Marker] = []
        self._marker_points: list[tuple[int, float, Marker]] = []
        self._runner: TaskRunner = runner if runner is not None else _run_inline
        self._symbol: str | None = None
        self._timeframe = Timeframe.D1
        self._request: tuple[str, Timeframe, bool] | None = None
        self._data: ChartData | None = None
        self._arrays: CandleArrays | None = None
        self._hover_index: int | None = None

        self._build_toolbar()
        self._build_plots()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(self._toolbar)
        layout.addWidget(self.info_label)
        layout.addWidget(self.graphics, 1)

        self.message_label.setText(texts.CHART_NO_SYMBOL)
        self._apply_palette()

    # -- Aufbau ----------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        self.message_label = QLabel()
        self.message_label.setObjectName("ChartMessage")
        self.info_label = QLabel()
        self.info_label.setObjectName("ChartInfo")
        # Lange Texte dürfen das Fenster nicht verbreitern (sie werden abgeschnitten).
        for label in (self.message_label, self.info_label):
            label.setMinimumWidth(1)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        self.timeframe_group = QButtonGroup(self)
        self.timeframe_group.setExclusive(True)
        self.timeframe_buttons: dict[Timeframe, QPushButton] = {}
        self._toolbar = QHBoxLayout()
        self._toolbar.addWidget(self.message_label, 1)
        for timeframe in TIMEFRAMES:
            button = QPushButton(timeframe.value)
            button.setCheckable(True)
            button.setMaximumWidth(_TIMEFRAME_BUTTON_WIDTH)
            button.setToolTip(texts.CHART_TIMEFRAME_TOOLTIP.format(timeframe=timeframe.value))
            button.clicked.connect(lambda _checked=False, tf=timeframe: self.set_timeframe(tf))
            self.timeframe_group.addButton(button)
            self.timeframe_buttons[timeframe] = button
            self._toolbar.addWidget(button)
        self.timeframe_buttons[self._timeframe].setChecked(True)

        self.log_checkbox = QCheckBox(texts.CHART_LOG)
        self.log_checkbox.toggled.connect(self._on_log_toggled)
        self.adjusted_checkbox = QCheckBox(texts.CHART_ADJUSTED)
        self.adjusted_checkbox.setToolTip(texts.CHART_ADJUSTED_TOOLTIP)
        self.adjusted_checkbox.setChecked(True)
        self.adjusted_checkbox.toggled.connect(lambda _checked: self.reload())
        self._toolbar.addSpacing(12)
        self._toolbar.addWidget(self.log_checkbox)
        self._toolbar.addWidget(self.adjusted_checkbox)

        self.add_indicator_button = QPushButton(texts.INDICATOR_ADD_BUTTON)
        self.add_indicator_button.setToolTip(texts.INDICATOR_ADD_TOOLTIP)
        self.add_indicator_button.clicked.connect(lambda _checked=False: self.open_add_dialog())
        self.indicator_menu = QMenu(self)
        self.indicator_menu.aboutToShow.connect(self._rebuild_indicator_menu)
        self.indicator_menu_button = QToolButton()
        self.indicator_menu_button.setText(texts.INDICATOR_MENU_BUTTON)
        self.indicator_menu_button.setMenu(self.indicator_menu)
        self.indicator_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._toolbar.addSpacing(12)
        self._toolbar.addWidget(self.add_indicator_button)
        self._toolbar.addWidget(self.indicator_menu_button)

    def _build_plots(self) -> None:
        self.graphics = pg.GraphicsLayoutWidget()
        self.date_axis = DateAxis()
        self.price_plot: pg.PlotItem = self.graphics.addPlot(row=0, col=0)
        self.volume_plot: pg.PlotItem = self.graphics.addPlot(
            row=1, col=0, axisItems={"bottom": self.date_axis}
        )
        layout = self.graphics.ci.layout
        layout.setRowStretchFactor(0, _STRETCH_PRICE)
        layout.setRowStretchFactor(1, _STRETCH_VOLUME)
        layout.setVerticalSpacing(0)

        self.price_plot.hideAxis("bottom")
        self.volume_plot.setXLink(self.price_plot)
        for plot in (self.price_plot, self.volume_plot):
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setMenuEnabled(False)
            plot.hideButtons()
            plot.setMouseEnabled(x=True, y=False)
            plot.getAxis("left").setWidth(70)
            plot.vb.disableAutoRange()
        self.volume_plot.getAxis("left").enableAutoSIPrefix(True)

        self.price_vb: pg.ViewBox = self.price_plot.vb
        self.price_vb.sigXRangeChanged.connect(lambda *_a: self._autoscale_y())

        self._layers: dict[tuple[bool, bool], _CandleLayer] = {}
        for up, color in ((True, UP_COLOR), (False, DOWN_COLOR)):
            for partial in (False, True):
                qcolor = QColor(color)
                if partial:
                    qcolor.setAlpha(PARTIAL_ALPHA)
                self._layers[(up, partial)] = _CandleLayer(
                    self.price_plot, self.volume_plot, qcolor
                )

        crosshair_pen = pg.mkPen((128, 128, 128), width=1, style=pg.QtCore.Qt.PenStyle.DashLine)
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=crosshair_pen)
        self.volume_v_line = pg.InfiniteLine(angle=90, movable=False, pen=crosshair_pen)
        for line in (self.v_line, self.h_line):
            self.price_plot.addItem(line, ignoreBounds=True)
        self.volume_plot.addItem(self.volume_v_line, ignoreBounds=True)
        self._crosshair_pen = crosshair_pen
        self._set_crosshair_visible(False)
        self.graphics.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self.marker_item = pg.ScatterPlotItem(size=_MARKER_SIZE, pxMode=True)
        self.marker_item.setZValue(20)
        self.price_plot.addItem(self.marker_item)

    def _apply_palette(self) -> None:
        palette = self.palette()
        background = palette.color(QPalette.ColorRole.Base)
        foreground = palette.color(QPalette.ColorRole.Text)
        self.graphics.setBackground(background)
        for plot in (self.price_plot, self.volume_plot, *self._panes):
            for name in ("left", "bottom"):
                axis = plot.getAxis(name)
                axis.setPen(foreground)
                axis.setTextPen(foreground)

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
    def timeframe(self) -> Timeframe:
        """Gewählter Zeitrahmen."""
        return self._timeframe

    @property
    def adjusted(self) -> bool:
        """``True``, wenn bereinigte Kurse gezeigt werden."""
        return self.adjusted_checkbox.isChecked()

    @property
    def log_scale(self) -> bool:
        """``True`` bei logarithmischer Preisachse."""
        return self.log_checkbox.isChecked()

    @property
    def data(self) -> ChartData | None:
        """Aktuell gezeichnete Kerzen."""
        return self._data

    def candle_count(self) -> int:
        """Anzahl gezeichneter Kerzen (alle Farben, inklusive unvollständiger)."""
        return sum(layer.count for layer in self._layers.values())

    def layer_counts(self) -> dict[tuple[bool, bool], int]:
        """Anzahl Kerzen je ``(steigend, unvollständig)``."""
        return {key: layer.count for key, layer in self._layers.items()}

    def load_symbol(self, symbol: str) -> None:
        """Lädt ``symbol`` im gewählten Zeitrahmen."""
        self._symbol = symbol
        self.reload()

    def set_timeframe(self, timeframe: Timeframe) -> None:
        """Wechselt den Zeitrahmen und lädt die Kerzen neu."""
        timeframe = Timeframe(timeframe)
        self.timeframe_buttons[timeframe].setChecked(True)
        if timeframe == self._timeframe:
            return
        self._timeframe = timeframe
        self.reload()

    def set_log_scale(self, enabled: bool) -> None:
        """Schaltet die Preisachse logarithmisch oder linear."""
        self.log_checkbox.setChecked(enabled)

    def set_adjusted(self, enabled: bool) -> None:
        """Zeigt bereinigte (``True``) oder rohe Kurse."""
        self.adjusted_checkbox.setChecked(enabled)

    def reload(self) -> None:
        """Fordert die Kerzen für Symbol, Zeitrahmen und Bereinigung neu an."""
        if self._symbol is None:
            return
        request = (self._symbol, self._timeframe, self.adjusted)
        self._request = request
        self.message_label.setText(texts.CHART_LOADING.format(symbol=self._symbol))
        self._runner(
            self._load,
            *request,
            on_result=self._on_loaded,
            name=texts.CHART_TASK_NAME.format(symbol=self._symbol),
        )

    def _load(
        self, symbol: str, timeframe: Timeframe, adjusted: bool
    ) -> tuple[tuple[str, Timeframe, bool], ChartData | str]:
        """Läuft im Hintergrund; Fehler werden als Text zurückgegeben statt geworfen."""
        request = (symbol, timeframe, adjusted)
        try:
            return request, self._service.load(symbol, timeframe, adjusted)
        except Exception as exc:  # jeder Ladefehler wird im Chart angezeigt
            return request, str(exc) or type(exc).__name__

    def _on_loaded(self, result: tuple[tuple[str, Timeframe, bool], ChartData | str]) -> None:
        request, payload = result
        if request != self._request:
            return  # veraltete Antwort, inzwischen wurde etwas anderes angefordert
        if isinstance(payload, str):
            message = texts.CHART_LOAD_FAILED.format(symbol=request[0], message=payload)
            self.message_label.setText(message)
            self.load_failed.emit(message)
            return
        self.set_data(payload)

    # -- Zeichnen --------------------------------------------------------------------

    def set_data(self, data: ChartData) -> None:
        """Zeichnet ``data`` und zeigt die jüngsten Kerzen."""
        previous = self._data
        self._data = data
        self._symbol = data.symbol
        if data.timeframe != self._timeframe:
            self._timeframe = data.timeframe
            self.timeframe_buttons[data.timeframe].setChecked(True)
        self._arrays = CandleArrays.from_chart_data(data)
        self._compute_indicators()
        self._draw()

        keep_range = (
            previous is not None
            and previous.symbol == data.symbol
            and previous.timeframe == data.timeframe
            and len(previous) == len(data)
        )
        self._update_limits()
        if keep_range:
            self._autoscale_y()
        else:
            self.show_last(DEFAULT_VISIBLE_BARS)

        if data.is_empty:
            self.message_label.setText(texts.CHART_NO_DATA.format(symbol=data.symbol))
        else:
            mode = texts.CHART_MODE_ADJUSTED if data.adjusted else texts.CHART_MODE_RAW
            self.message_label.setText(
                texts.CHART_HEADER.format(
                    symbol=data.symbol, timeframe=data.timeframe.value, mode=mode
                )
            )
        self.show_bar_info(None)
        self.data_loaded.emit(data)

    def _price_values(self, values: FloatArray) -> FloatArray:
        return np.log10(values) if self.log_scale else values

    def _draw(self) -> None:
        arrays = self._arrays
        if arrays is None:
            return
        x = np.arange(len(arrays), dtype=np.float64)
        up = arrays.close >= arrays.open
        open_ = self._price_values(arrays.open)
        high = self._price_values(arrays.high)
        low = self._price_values(arrays.low)
        close = self._price_values(arrays.close)
        for (is_up, is_partial), layer in self._layers.items():
            mask = (up == is_up) & (arrays.partial == is_partial)
            layer.set_data(
                x[mask], open_[mask], high[mask], low[mask], close[mask], arrays.volume[mask]
            )
        self.price_plot.getAxis("left").setLogMode(False, self.log_scale)
        if self._data is not None:
            self.date_axis.set_dates(arrays.dates, self._data.timeframe)
            for axis in self._pane_axes.values():
                axis.set_dates(arrays.dates, self._data.timeframe)
        self._draw_indicators()
        self._draw_markers()

    def _update_limits(self) -> None:
        count = len(self._arrays) if self._arrays is not None else 0
        upper = max(count, 1) - 1 + _RIGHT_MARGIN_BARS + 0.5
        for plot in (self.price_plot, self.volume_plot, *self._panes):
            plot.vb.setLimits(
                xMin=-1.0,
                xMax=upper,
                minXRange=_MIN_VISIBLE_BARS,
                maxXRange=upper + 1.0,
            )

    def show_last(self, bars: int) -> None:
        """Zeigt die jüngsten ``bars`` Kerzen (alle, wenn es weniger gibt)."""
        count = len(self._arrays) if self._arrays is not None else 0
        visible = max(min(bars, count), _MIN_VISIBLE_BARS)
        end = count - 0.5 + _RIGHT_MARGIN_BARS
        self.price_vb.setXRange(end - visible - _RIGHT_MARGIN_BARS, end, padding=0)
        self._autoscale_y()

    def visible_range(self) -> tuple[int, int]:
        """Erste und letzte (ausschließlich) sichtbare Kerzenposition."""
        count = len(self._arrays) if self._arrays is not None else 0
        (x0, x1), _ = self.price_vb.viewRange()
        first = min(max(math.floor(x0 + 0.5), 0), count)
        last = min(max(math.ceil(x1 - 0.5) + 1, first), count)
        return first, last

    def _autoscale_y(self) -> None:
        arrays = self._arrays
        if arrays is None or len(arrays) == 0:
            return
        first, last = self.visible_range()
        if first >= last:
            return
        low = float(self._price_values(arrays.low[first:last]).min())
        high = float(self._price_values(arrays.high[first:last]).max())
        for ind in self._indicators.values():
            if ind.plot is self.price_plot:
                low, high = _extend_range(
                    (self._price_values(v[first:last]) for v in ind.values.values()), low, high
                )
        pad = (high - low) * _Y_PADDING or abs(high) * _Y_PADDING or 1.0
        self.price_vb.setYRange(low - pad, high + pad, padding=0)
        volume_max = float(arrays.volume[first:last].max()) or 1.0
        self.volume_plot.vb.setYRange(0.0, volume_max * (1 + _Y_PADDING), padding=0)
        for pane in self._panes:
            values = [
                v[first:last]
                for ind in self._indicators.values()
                if ind.plot is pane
                for v in ind.values.values()
            ]
            p_low, p_high = _extend_range(values, math.inf, -math.inf)
            if math.isinf(p_low):
                continue
            p_pad = (p_high - p_low) * _Y_PADDING or abs(p_high) * _Y_PADDING or 1.0
            pane.vb.setYRange(p_low - p_pad, p_high + p_pad, padding=0)
        self._place_markers()

    def _on_log_toggled(self, _checked: bool) -> None:
        self._draw()
        self._autoscale_y()
        self.show_bar_info(self._hover_index)

    # -- Fadenkreuz ------------------------------------------------------------------

    def _set_crosshair_visible(self, visible: bool) -> None:
        for line in (self.v_line, self.h_line, self.volume_v_line, *self._pane_lines.values()):
            line.setVisible(visible)

    def _on_mouse_moved(self, pos: QPointF) -> None:
        in_price = self.price_plot.sceneBoundingRect().contains(pos)
        in_lower = any(
            plot.sceneBoundingRect().contains(pos) for plot in (self.volume_plot, *self._panes)
        )
        if not (in_price or in_lower) or self._arrays is None or len(self._arrays) == 0:
            self._set_crosshair_visible(False)
            self.show_bar_info(None)
            return
        point = self.price_vb.mapSceneToView(pos)
        self.move_crosshair(point.x(), point.y() if in_price else None)

    def move_crosshair(self, x: float, y: float | None = None) -> None:
        """Setzt das Fadenkreuz auf Position ``x`` (Kerze) und optional Preis ``y``."""
        count = len(self._arrays) if self._arrays is not None else 0
        if count == 0:
            return
        index = min(max(round(x), 0), count - 1)
        self.v_line.setPos(index)
        self.volume_v_line.setPos(index)
        self.v_line.setVisible(True)
        self.volume_v_line.setVisible(True)
        for line in self._pane_lines.values():
            line.setPos(index)
            line.setVisible(True)
        if y is None:
            self.h_line.setVisible(False)
        else:
            self.h_line.setPos(y)
            self.h_line.setVisible(True)
        self.show_bar_info(index)

    def show_bar_info(self, index: int | None) -> None:
        """Zeigt Datum und OHLCV der Kerze ``index`` (``None`` = jüngste Kerze)."""
        self._hover_index = index
        arrays = self._arrays
        if arrays is None or len(arrays) == 0:
            self.info_label.setText("")
            return
        i = len(arrays) - 1 if index is None else index
        day = arrays.dates[i].astype(object)
        self.info_label.setText(
            texts.CHART_OHLCV.format(
                date=f"{day:%d.%m.%Y}",
                partial=texts.CHART_PARTIAL if arrays.partial[i] else "",
                open=format_price(arrays.open[i]),
                high=format_price(arrays.high[i]),
                low=format_price(arrays.low[i]),
                close=format_price(arrays.close[i]),
                volume=format_volume(arrays.volume[i]),
            )
            + self._indicator_info(i)
        )

    def _indicator_info(self, index: int) -> str:
        parts: list[str] = []
        for ind in self._indicators.values():
            values = [
                format_price(float(v[index]))
                for v in ind.values.values()
                if index < len(v) and np.isfinite(v[index])
            ]
            if values:
                parts.append(f"{ind.label} {' / '.join(values)}")
        return "".join(f"   {part}" for part in parts)

    # -- Indikatoren -----------------------------------------------------------------

    @property
    def indicator_service(self) -> IndicatorService:
        """Fassade für Indikatoren und Vorlagen."""
        return self._indicator_service

    def indicators(self) -> list[tuple[int, IndicatorConfig]]:
        """Indikatoren im Chart als ``(Schlüssel, Konfiguration)`` in Reihenfolge."""
        return [(key, ind.config) for key, ind in self._indicators.items()]

    def indicator_label(self, key: int) -> str:
        """Anzeigename des Indikators ``key``, z. B. ``SMA(20)``."""
        return self._indicators[key].label

    def indicator_error(self, key: int) -> str:
        """Letzte Fehlermeldung bei der Berechnung (leer, wenn sie gelang)."""
        return self._indicators[key].error

    def indicator_plot(self, key: int) -> pg.PlotItem:
        """Panel, in dem der Indikator ``key`` gezeichnet wird."""
        return self._indicators[key].plot

    def indicator_items(self, key: int) -> dict[str, Any]:
        """Grafik-Elemente des Indikators ``key`` je Ausgabe."""
        return dict(self._indicators[key].items)

    def indicator_values(self, key: int) -> dict[str, FloatArray]:
        """Berechnete Werte des Indikators ``key`` je gezeichneter Ausgabe."""
        return dict(self._indicators[key].values)

    def line_count(self, key: int | None = None) -> int:
        """Anzahl gezeichneter Linien (eines Indikators oder aller)."""
        if key is not None:
            return len(self._indicators[key].items)
        return sum(len(ind.items) for ind in self._indicators.values())

    def pane_count(self) -> int:
        """Anzahl eigener Indikator-Panels (ohne Kurs- und Volumen-Panel)."""
        return len(self._panes)

    def panes(self) -> list[pg.PlotItem]:
        """Eigene Indikator-Panels von oben nach unten."""
        return list(self._panes)

    def add_indicator(self, config: IndicatorConfig | str) -> int:
        """Fügt einen Indikator hinzu und gibt seinen Schlüssel zurück.

        Args:
            config: Konfiguration oder nur der Name (dann mit Standardwerten).

        Raises:
            IndicatorError: Unbekannter Indikator oder ungültige Parameter.
        """
        if isinstance(config, str):
            config = IndicatorConfig(name=config)
        config = self._indicator_service.normalize(config, self.line_count())
        key = self._next_key
        self._next_key += 1
        spec = self._indicator_service.spec(config.name)
        plot = self.price_plot if spec.pane == "price" else self._create_pane()
        ind = _ActiveIndicator(
            key=key,
            config=config,
            spec=spec,
            plot=plot,
            label=self._indicator_service.label(config),
        )
        self._indicators[key] = ind
        self._create_items(ind)
        self._compute(ind)
        self._draw_indicator(ind)
        self._redraw_after_indicator_change()
        return key

    def update_indicator(self, key: int, config: IndicatorConfig) -> None:
        """Ändert Parameter und Farben des Indikators ``key`` (der Indikator bleibt gleich).

        Raises:
            KeyError: Unbekannter Schlüssel.
            IndicatorError: Anderer Indikatorname oder ungültige Parameter.
        """
        ind = self._indicators[key]
        if config.name != ind.config.name:
            raise IndicatorError(
                f"Beim Bearbeiten kann der Indikator nicht gewechselt werden ({config.name!r})."
            )
        ind.config = self._indicator_service.normalize(config)
        ind.label = self._indicator_service.label(ind.config)
        self._remove_items(ind)
        self._create_items(ind)
        self._compute(ind)
        self._draw_indicator(ind)
        self._redraw_after_indicator_change()

    def remove_indicator(self, key: int) -> None:
        """Entfernt den Indikator ``key``; ein leeres eigenes Panel verschwindet mit."""
        ind = self._indicators.pop(key, None)
        if ind is None:
            return
        self._remove_items(ind)
        if ind.plot is not self.price_plot:
            self._remove_pane(ind.plot)
        self._redraw_after_indicator_change()

    def clear_indicators(self) -> None:
        """Entfernt alle Indikatoren."""
        for key in list(self._indicators):
            self.remove_indicator(key)

    def apply_template(self, name: str) -> None:
        """Ersetzt die Indikatoren im Chart durch die der Vorlage ``name``.

        Raises:
            TemplateError: Vorlage gibt es nicht.
        """
        configs = self._indicator_service.template(name)
        self.clear_indicators()
        for config in configs:
            self.add_indicator(config)

    def save_template(self, name: str) -> str:
        """Speichert die aktuellen Indikatoren als Vorlage ``name``.

        Returns:
            Bereinigter Name der Vorlage.

        Raises:
            TemplateError: Name ungültig oder keine Indikatoren im Chart.
        """
        return self._indicator_service.save_template(
            name, [ind.config for ind in self._indicators.values()]
        )

    def _redraw_after_indicator_change(self) -> None:
        self._update_limits()
        self._autoscale_y()
        self.show_bar_info(self._hover_index)
        self.indicators_changed.emit()

    def _create_pane(self) -> pg.PlotItem:
        axis = DateAxis()
        pane = pg.PlotItem(axisItems={"bottom": axis})
        pane.showGrid(x=True, y=True, alpha=0.15)
        pane.setMenuEnabled(False)
        pane.hideButtons()
        pane.setMouseEnabled(x=True, y=False)
        pane.getAxis("left").setWidth(70)
        pane.vb.disableAutoRange()
        pane.setXLink(self.price_plot)
        line = pg.InfiniteLine(angle=90, movable=False, pen=self._crosshair_pen)
        line.setVisible(self.v_line.isVisible())
        pane.addItem(line, ignoreBounds=True)
        if self._arrays is not None and self._data is not None:
            axis.set_dates(self._arrays.dates, self._data.timeframe)
        self._pane_axes[pane] = axis
        self._pane_lines[pane] = line
        self._panes.append(pane)
        self._relayout()
        self._apply_palette()
        return pane

    def _remove_pane(self, pane: pg.PlotItem) -> None:
        if pane not in self._panes:
            return
        self.graphics.ci.removeItem(pane)
        self._panes.remove(pane)
        del self._pane_axes[pane]
        del self._pane_lines[pane]
        self._relayout()

    def _relayout(self) -> None:
        """Ordnet die eigenen Panels unter dem Volumen an; nur das unterste zeigt Daten."""
        ci = self.graphics.ci
        for pane in self._panes:
            if pane in ci.items:
                ci.removeItem(pane)
        layout = ci.layout
        for row in range(2, layout.rowCount()):
            layout.setRowStretchFactor(row, 0)
        for i, pane in enumerate(self._panes):
            ci.addItem(pane, row=2 + i, col=0)
            layout.setRowStretchFactor(2 + i, _STRETCH_PANE)
        lower = [self.volume_plot, *self._panes]
        for plot in lower[:-1]:
            plot.hideAxis("bottom")
        lower[-1].showAxis("bottom")

    def _create_items(self, ind: _ActiveIndicator) -> None:
        for output in ind.spec.visible_outputs():
            color = QColor(ind.config.colors.get(output, "#808080"))
            style = ind.spec.style(output)
            item: Any
            if style == "histogram":
                item = pg.BarGraphItem(
                    x=np.zeros(0),
                    y0=np.zeros(0),
                    height=np.zeros(0),
                    width=BODY_WIDTH,
                    pen=pg.mkPen(color),
                    brush=pg.mkBrush(color),
                )
            elif style == "dots":
                item = pg.ScatterPlotItem(
                    size=4, pen=pg.mkPen(None), brush=pg.mkBrush(color), pxMode=True
                )
            else:
                item = pg.PlotCurveItem(pen=pg.mkPen(color, width=LINE_WIDTH), connect="finite")
            item.setZValue(10)
            ind.plot.addItem(item)
            ind.items[output] = item

    @staticmethod
    def _remove_items(ind: _ActiveIndicator) -> None:
        for item in ind.items.values():
            ind.plot.removeItem(item)
        ind.items.clear()

    def _compute_indicators(self) -> None:
        for ind in self._indicators.values():
            self._compute(ind)

    def _compute(self, ind: _ActiveIndicator) -> None:
        """Berechnet die Werte; Fehler werden am Indikator vermerkt und angezeigt."""
        ind.values = {}
        ind.error = ""
        if self._data is None or self._data.is_empty:
            return
        try:
            frame = self._indicator_service.compute(ind.config, self._data.bars)
        except IndicatorError as exc:
            ind.error = str(exc)
            self.message_label.setText(
                texts.INDICATOR_COMPUTE_FAILED.format(label=ind.label, message=exc)
            )
            return
        for output in ind.spec.visible_outputs():
            ind.values[output] = frame[output].to_numpy(dtype=np.float64, na_value=np.nan)

    def _draw_indicators(self) -> None:
        for ind in self._indicators.values():
            self._draw_indicator(ind)

    def _draw_indicator(self, ind: _ActiveIndicator) -> None:
        for output, item in ind.items.items():
            values = ind.values.get(output, np.zeros(0))
            if ind.plot is self.price_plot:
                with np.errstate(invalid="ignore", divide="ignore"):
                    values = self._price_values(values)
            x = np.arange(len(values), dtype=np.float64)
            if isinstance(item, pg.BarGraphItem):
                heights = np.nan_to_num(values, nan=0.0)
                item.setOpts(x=x, y0=np.zeros_like(heights), height=heights)
            elif isinstance(item, pg.ScatterPlotItem):
                finite = np.isfinite(values)
                item.setData(x=x[finite], y=values[finite])
            else:
                item.setData(x=x, y=values, connect="finite")

    # -- Marker ----------------------------------------------------------------------

    def add_markers(self, markers: Iterable[Marker]) -> None:
        """Zeigt Marker für Signale und Trades im Kurs-Chart.

        Die Marker bleiben beim Wechsel von Zeitrahmen oder Symbol erhalten, bis
        :meth:`clear_markers` sie entfernt; jeder steht auf der Kerze, deren Zeitraum
        sein Datum enthält.
        """
        self._markers.extend(markers)
        self._draw_markers()

    def clear_markers(self) -> None:
        """Entfernt alle Marker."""
        self._markers.clear()
        self._draw_markers()

    def markers(self) -> list[Marker]:
        """Alle Marker (auch die, die gerade keiner Kerze zugeordnet sind)."""
        return list(self._markers)

    def marker_points(self) -> list[tuple[int, float, Marker]]:
        """Gezeichnete Marker als ``(Kerzenposition, Preis, Marker)``."""
        return list(self._marker_points)

    def _draw_markers(self) -> None:
        arrays = self._arrays
        self._marker_points = []
        if arrays is not None and len(arrays) and self._markers:
            for index, marker in marker_indices(self._markers, arrays.dates):
                if marker.price is not None:
                    price = marker.price
                elif marker.kind is MarkerKind.BUY:
                    price = float(arrays.low[index])
                elif marker.kind in (MarkerKind.SELL, MarkerKind.EXIT):
                    price = float(arrays.high[index])
                else:
                    price = float(arrays.close[index])
                self._marker_points.append((index, price, marker))
        self._place_markers()

    def _place_markers(self) -> None:
        """Setzt die Marker; Abstand zur Kerze hängt von der sichtbaren Preisspanne ab."""
        if not self._marker_points:
            self.marker_item.setData([])
            return
        (_x0, _x1), (y0, y1) = self.price_vb.viewRange()
        offset = (y1 - y0) * _MARKER_OFFSET
        spots: list[dict[str, Any]] = []
        for index, price, marker in self._marker_points:
            y = float(self._price_values(np.array([price]))[0])
            if marker.price is None and marker.kind is MarkerKind.BUY:
                y -= offset
            elif marker.price is None and marker.kind in (MarkerKind.SELL, MarkerKind.EXIT):
                y += offset
            color = QColor(marker.color or MARKER_COLORS[marker.kind])
            spots.append(
                {
                    "pos": (float(index), y),
                    "symbol": _MARKER_SYMBOLS[marker.kind],
                    "brush": pg.mkBrush(color),
                    "pen": pg.mkPen(color.darker(130)),
                    "data": marker.text,
                }
            )
        self.marker_item.setData(spots)

    # -- Dialoge und Menü ------------------------------------------------------------

    def open_add_dialog(self) -> int | None:
        """Zeigt den Dialog „Indikator hinzufügen“ und fügt die Auswahl hinzu."""
        dialog = IndicatorDialog(self._indicator_service, None, self.line_count(), self)
        return self._run_indicator_dialog(dialog, None)

    def open_edit_dialog(self, key: int) -> int | None:
        """Zeigt den Dialog zum Bearbeiten des Indikators ``key``."""
        dialog = IndicatorDialog(self._indicator_service, self._indicators[key].config, 0, self)
        return self._run_indicator_dialog(dialog, key)

    def _run_indicator_dialog(self, dialog: IndicatorDialog, key: int | None) -> int | None:
        while dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                if key is None:
                    return self.add_indicator(dialog.config())
                self.update_indicator(key, dialog.config())
                return key
            except IndicatorError as exc:
                QMessageBox.warning(self, texts.INDICATOR_ERROR_TITLE, str(exc))
        return None

    def save_template_dialog(self) -> str | None:
        """Fragt nach einem Namen und speichert die Indikatoren als Vorlage."""
        name, ok = QInputDialog.getText(self, texts.TEMPLATE_SAVE_TITLE, texts.TEMPLATE_SAVE_PROMPT)
        if not ok:
            return None
        try:
            return self.save_template(name)
        except TemplateError as exc:
            QMessageBox.warning(self, texts.TEMPLATE_ERROR_TITLE, str(exc))
            return None

    def _rebuild_indicator_menu(self) -> None:
        menu = self.indicator_menu
        menu.clear()
        if not self._indicators:
            menu.addAction(texts.INDICATOR_NONE).setEnabled(False)
        for key, ind in self._indicators.items():
            sub = menu.addMenu(ind.label)
            sub.addAction(texts.INDICATOR_EDIT).triggered.connect(
                lambda _c=False, k=key: self.open_edit_dialog(k)
            )
            sub.addAction(texts.INDICATOR_REMOVE).triggered.connect(
                lambda _c=False, k=key: self.remove_indicator(k)
            )
        remove_all = menu.addAction(texts.INDICATOR_REMOVE_ALL)
        remove_all.setEnabled(bool(self._indicators))
        remove_all.triggered.connect(lambda _c=False: self.clear_indicators())
        menu.addSeparator()
        save = menu.addAction(texts.TEMPLATE_SAVE)
        save.setEnabled(bool(self._indicators))
        save.triggered.connect(lambda _c=False: self.save_template_dialog())
        names = self._indicator_service.template_names()
        load_menu = menu.addMenu(texts.TEMPLATE_LOAD)
        delete_menu = menu.addMenu(texts.TEMPLATE_DELETE)
        for name in names:
            load_menu.addAction(name).triggered.connect(
                lambda _c=False, n=name: self.apply_template(n)
            )
            delete_menu.addAction(name).triggered.connect(
                lambda _c=False, n=name: self._indicator_service.delete_template(n)
            )
        if not names:
            load_menu.addAction(texts.TEMPLATE_NONE).setEnabled(False)
            delete_menu.addAction(texts.TEMPLATE_NONE).setEnabled(False)
        menu.addSeparator()
        clear = menu.addAction(texts.MARKER_CLEAR)
        clear.setEnabled(bool(self._markers))
        clear.triggered.connect(lambda _c=False: self.clear_markers())


def _extend_range(arrays: Iterable[FloatArray], low: float, high: float) -> tuple[float, float]:
    """Erweitert ``[low, high]`` um die endlichen Werte aus ``arrays``."""
    for values in arrays:
        finite = values[np.isfinite(values)]
        if finite.size:
            low = min(low, float(finite.min()))
            high = max(high, float(finite.max()))
    return low, high
