"""Kurschart auf pyqtgraph: Kerzen, Volumen, Zeitrahmen, Fadenkreuz, log/linear.

Die x-Achse zählt Kerzen (0, 1, 2 …) statt Kalendertage; dadurch entstehen keine
Lücken für Wochenenden und Feiertage. :class:`DateAxis` beschriftet die Positionen
mit dem Datum der Kerze. Kerzen und Volumen werden je Farbe als ein Grafik-Element
gezeichnet (vektorisiert), damit auch 25 Jahre Tagesdaten schnell erscheinen.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from trading_helfer.core.types import Timeframe
from trading_helfer.gui import texts
from trading_helfer.services.chart_service import ChartData, ChartService

__all__ = [
    "DEFAULT_VISIBLE_BARS",
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


class ChartWidget(QWidget):
    """Kerzenchart mit Volumen-Panel, Zeitrahmen-Umschalter und Fadenkreuz.

    Args:
        service: Liefert die Kerzen (lädt, bereinigt und rechnet Zeitrahmen um).
        runner: Startet das Laden, z. B. :meth:`MainWindow.run_task` für den
            Thread-Pool; ohne Angabe wird sofort im GUI-Thread geladen.
    """

    data_loaded = Signal(object)
    """Wird mit den neuen :class:`ChartData` gesendet, nachdem sie gezeichnet sind."""

    load_failed = Signal(str)
    """Wird mit der deutschen Fehlermeldung gesendet, wenn das Laden scheitert."""

    def __init__(
        self,
        service: ChartService,
        runner: TaskRunner | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
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

    def _build_plots(self) -> None:
        self.graphics = pg.GraphicsLayoutWidget()
        self.date_axis = DateAxis()
        self.price_plot: pg.PlotItem = self.graphics.addPlot(row=0, col=0)
        self.volume_plot: pg.PlotItem = self.graphics.addPlot(
            row=1, col=0, axisItems={"bottom": self.date_axis}
        )
        layout = self.graphics.ci.layout
        layout.setRowStretchFactor(0, 4)
        layout.setRowStretchFactor(1, 1)
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
        self._set_crosshair_visible(False)
        self.graphics.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def _apply_palette(self) -> None:
        palette = self.palette()
        background = palette.color(QPalette.ColorRole.Base)
        foreground = palette.color(QPalette.ColorRole.Text)
        self.graphics.setBackground(background)
        for plot in (self.price_plot, self.volume_plot):
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

    def _update_limits(self) -> None:
        count = len(self._arrays) if self._arrays is not None else 0
        upper = max(count, 1) - 1 + _RIGHT_MARGIN_BARS + 0.5
        for plot in (self.price_plot, self.volume_plot):
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
        pad = (high - low) * _Y_PADDING or abs(high) * _Y_PADDING or 1.0
        self.price_vb.setYRange(low - pad, high + pad, padding=0)
        volume_max = float(arrays.volume[first:last].max()) or 1.0
        self.volume_plot.vb.setYRange(0.0, volume_max * (1 + _Y_PADDING), padding=0)

    def _on_log_toggled(self, _checked: bool) -> None:
        self._draw()
        self._autoscale_y()
        self.show_bar_info(self._hover_index)

    # -- Fadenkreuz ------------------------------------------------------------------

    def _set_crosshair_visible(self, visible: bool) -> None:
        for line in (self.v_line, self.h_line, self.volume_v_line):
            line.setVisible(visible)

    def _on_mouse_moved(self, pos: QPointF) -> None:
        in_price = self.price_plot.sceneBoundingRect().contains(pos)
        in_volume = self.volume_plot.sceneBoundingRect().contains(pos)
        if not (in_price or in_volume) or self._arrays is None or len(self._arrays) == 0:
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
        )
