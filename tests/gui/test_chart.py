"""Chart-Widget: Kerzen je Zeitrahmen, Umschalter, Fadenkreuz, log/linear, Watchlist."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.gui import texts
from trading_helfer.gui.main_window import MainWindow
from trading_helfer.gui.theme import apply_theme
from trading_helfer.gui.widgets.chart import (
    DEFAULT_VISIBLE_BARS,
    TIMEFRAMES,
    ChartWidget,
    DateAxis,
    format_price,
    format_volume,
)
from trading_helfer.services.app_service import AppService
from trading_helfer.services.chart_service import ChartData, ChartService

FIXTURE = Path(__file__).parents[1] / "fixtures" / "chart" / "SYNTH_daily.csv"
EXPECTED_COUNTS = {
    Timeframe.D1: 475,
    Timeframe.D3: 159,
    Timeframe.W1: 99,
    Timeframe.MN1: 23,
}


@pytest.fixture
def daily() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, parse_dates=["date"])


class RecordingService(ChartService):
    """ChartService, der jede Anfrage mitschreibt."""

    def __init__(self, daily: pd.DataFrame) -> None:
        super().__init__(lambda symbol: daily if symbol == "SYNTH" else daily.head(0))
        self.requests: list[tuple[str, Timeframe, bool]] = []

    def load(self, symbol: str, timeframe: Timeframe, adjusted: bool = True) -> ChartData:
        self.requests.append((symbol, Timeframe(timeframe), adjusted))
        return super().load(symbol, timeframe, adjusted)


@pytest.fixture
def chart_service(daily: pd.DataFrame) -> RecordingService:
    return RecordingService(daily)


@pytest.fixture
def chart(qtbot: QtBot, chart_service: RecordingService) -> ChartWidget:
    widget = ChartWidget(chart_service)
    qtbot.addWidget(widget)
    widget.resize(1000, 600)
    return widget


def test_empty_chart_shows_hint(chart: ChartWidget) -> None:
    assert chart.candle_count() == 0
    assert chart.message_label.text() == texts.CHART_NO_SYMBOL
    assert chart.timeframe is Timeframe.D1
    assert chart.timeframe_buttons[Timeframe.D1].isChecked()
    assert [b.text() for b in chart.timeframe_buttons.values()] == ["1D", "3D", "1W", "1M"]
    chart.reload()  # ohne Symbol passiert nichts
    chart.move_crosshair(3.0)
    chart.show_bar_info(None)
    assert chart.info_label.text() == ""


@pytest.mark.parametrize("timeframe", TIMEFRAMES)
def test_candle_count_per_timeframe(chart: ChartWidget, timeframe: Timeframe) -> None:
    chart.set_timeframe(timeframe)
    chart.load_symbol("SYNTH")
    assert chart.candle_count() == EXPECTED_COUNTS[timeframe]
    partial = sum(n for (_up, is_partial), n in chart.layer_counts().items() if is_partial)
    assert partial == (0 if timeframe is Timeframe.D1 else 1)
    assert chart.data is not None
    assert chart.data.timeframe is timeframe
    assert chart.message_label.text() == f"SYNTH · {timeframe.value} · bereinigt"


def test_up_and_down_candles_are_split(chart: ChartWidget) -> None:
    chart.load_symbol("SYNTH")
    data = chart.data
    assert data is not None
    up = int((data.bars["close"] >= data.bars["open"]).sum())
    counts = chart.layer_counts()
    assert counts[(True, False)] + counts[(True, True)] == up
    assert counts[(False, False)] + counts[(False, True)] == len(data) - up
    body = chart._layers[(True, False)].bodies
    assert len(body.opts["x"]) == counts[(True, False)]


def test_switching_timeframe_triggers_reload(
    chart: ChartWidget, chart_service: RecordingService, qtbot: QtBot
) -> None:
    chart.load_symbol("SYNTH")
    assert chart_service.requests == [("SYNTH", Timeframe.D1, True)]
    with qtbot.waitSignal(chart.data_loaded) as blocker:
        qtbot.mouseClick(chart.timeframe_buttons[Timeframe.W1], Qt.MouseButton.LeftButton)
    assert blocker.args[0].timeframe is Timeframe.W1
    assert chart_service.requests[-1] == ("SYNTH", Timeframe.W1, True)
    assert chart.candle_count() == EXPECTED_COUNTS[Timeframe.W1]
    # Gleicher Zeitrahmen noch einmal: keine neue Anfrage.
    chart.set_timeframe(Timeframe.W1)
    assert len(chart_service.requests) == 2


def test_adjusted_toggle_reloads_raw_prices(
    chart: ChartWidget, chart_service: RecordingService
) -> None:
    chart.load_symbol("SYNTH")
    adjusted_first = chart.data.bars["close"].iloc[0]  # type: ignore[union-attr]
    chart.set_adjusted(False)
    assert chart_service.requests[-1] == ("SYNTH", Timeframe.D1, False)
    assert not chart.adjusted
    raw_first = chart.data.bars["close"].iloc[0]  # type: ignore[union-attr]
    assert raw_first / adjusted_first == pytest.approx(4 / 0.99)
    assert chart.message_label.text().endswith("roh")


def test_log_scale_redraws_without_reload(
    chart: ChartWidget, chart_service: RecordingService
) -> None:
    chart.load_symbol("SYNTH")
    linear_y = chart.price_vb.viewRange()[1]
    chart.set_log_scale(True)
    assert chart.log_scale
    assert len(chart_service.requests) == 1
    log_y = chart.price_vb.viewRange()[1]
    assert log_y[1] < 3 < linear_y[1]  # log10 der Kurse statt Kurse
    wicks = chart._layers[(True, False)].wicks
    assert wicks.yData.max() < 3
    assert chart.price_plot.getAxis("left").logMode
    chart.set_log_scale(False)
    assert chart.price_vb.viewRange()[1] == pytest.approx(linear_y)


def test_initial_range_shows_latest_bars_and_autoscale(chart: ChartWidget) -> None:
    chart.load_symbol("SYNTH")
    first, last = chart.visible_range()
    assert last == EXPECTED_COUNTS[Timeframe.D1]
    assert last - first == DEFAULT_VISIBLE_BARS
    data = chart.data
    assert data is not None
    visible = data.bars.iloc[first:last]
    y0, y1 = chart.price_vb.viewRange()[1]
    assert y0 < visible["low"].min() < visible["high"].max() < y1
    assert y1 - y0 < (visible["high"].max() - visible["low"].min()) * 1.2
    # Verschieben nach links skaliert die y-Achse neu.
    chart.price_vb.setXRange(0, 50, padding=0)
    early = data.bars.iloc[0:51]
    y0, y1 = chart.price_vb.viewRange()[1]
    assert y0 < early["low"].min()
    assert early["high"].max() < y1


def test_reload_keeps_view_range(chart: ChartWidget) -> None:
    chart.load_symbol("SYNTH")
    chart.price_vb.setXRange(100, 200, padding=0)
    chart.set_adjusted(False)
    first, _last = chart.visible_range()
    assert 95 <= first <= 105


def test_small_dataset_shows_all(chart: ChartWidget, daily: pd.DataFrame) -> None:
    chart._service.set_loader(lambda _s: daily.head(20))
    chart.load_symbol("SYNTH")
    assert chart.visible_range() == (0, 20)


def test_crosshair_shows_ohlcv(chart: ChartWidget) -> None:
    chart.load_symbol("SYNTH")
    data = chart.data
    assert data is not None
    row = data.bars.iloc[10]
    chart.move_crosshair(10.3, 50.0)
    text = chart.info_label.text()
    assert text.startswith(f"{row['date']:%d.%m.%Y}")
    assert f"Schluss {format_price(row['close'])}" in text
    assert f"Volumen {format_volume(row['volume'])}" in text
    assert chart.v_line.isVisible()
    assert chart.h_line.isVisible()
    assert chart.v_line.value() == 10
    chart.move_crosshair(9999.0)
    assert not chart.h_line.isVisible()
    assert chart.v_line.value() == len(data) - 1


def test_crosshair_marks_partial_candle(chart: ChartWidget) -> None:
    chart.set_timeframe(Timeframe.W1)
    chart.load_symbol("SYNTH")
    assert texts.CHART_PARTIAL in chart.info_label.text()  # jüngste Kerze ist partiell
    chart.move_crosshair(0)
    assert texts.CHART_PARTIAL not in chart.info_label.text()


def test_mouse_move_drives_crosshair(chart: ChartWidget, qtbot: QtBot) -> None:
    chart.show()
    qtbot.waitExposed(chart)
    chart.load_symbol("SYNTH")
    center = chart.price_plot.sceneBoundingRect().center()
    chart._on_mouse_moved(center)
    assert chart.v_line.isVisible()
    assert chart.h_line.isVisible()
    volume_center = chart.volume_plot.sceneBoundingRect().center()
    chart._on_mouse_moved(volume_center)
    assert chart.v_line.isVisible()
    assert not chart.h_line.isVisible()
    chart._on_mouse_moved(QPointF(-100, -100))
    assert not chart.v_line.isVisible()


def test_wheel_zoom_and_pan_move_x_only(chart: ChartWidget) -> None:
    chart.load_symbol("SYNTH")
    (x0, x1), _ = chart.price_vb.viewRange()
    chart.price_vb.scaleBy(x=0.5, y=None, center=QPointF(x1 - 10, 0))
    (z0, z1), _ = chart.price_vb.viewRange()
    assert z1 - z0 == pytest.approx((x1 - x0) * 0.5)
    chart.price_vb.translateBy(x=-30)
    (p0, _p1), _ = chart.price_vb.viewRange()
    assert p0 == pytest.approx(z0 - 30)
    assert chart.volume_plot.vb.viewRange()[0] == pytest.approx(chart.price_vb.viewRange()[0])
    assert chart.price_vb.state["mouseEnabled"] == [True, False]


def test_date_axis_labels_have_no_weekend_gaps(chart: ChartWidget) -> None:
    chart.load_symbol("SYNTH")
    axis = chart.date_axis
    # Erster Tag ist Di 2023-01-03 (Mo Feiertag); Position 3 ist Freitag 2023-01-06,
    # Position 4 direkt danach Montag 2023-01-09.
    assert axis.tickStrings([3.0, 4.0], 1.0, 1.0) == ["06.01.23", "09.01.23"]
    assert axis.tickStrings([5.5, -1.0, 9999.0], 1.0, 1.0) == ["", "", ""]
    assert axis.label_for(5.0, 50.0) == "01.2023"
    assert axis.label_for(5.0, 250.0) == "2023"


def test_date_axis_standalone() -> None:
    axis = DateAxis()
    axis.set_dates(np.array(["2024-01-31", "2024-02-29"], dtype="datetime64[D]"), Timeframe.MN1)
    assert axis.tickStrings([0.0, 1.0], 1.0, 1.0) == ["01.2024", "02.2024"]
    assert axis.label_for(1.0, 12.0) == "2024"


def test_unknown_symbol_shows_no_data(chart: ChartWidget) -> None:
    chart.load_symbol("LEER")
    assert chart.candle_count() == 0
    assert chart.message_label.text() == texts.CHART_NO_DATA.format(symbol="LEER")
    assert chart.info_label.text() == ""


def test_load_error_is_shown(qtbot: QtBot) -> None:
    def broken(_symbol: str) -> pd.DataFrame:
        raise OSError("Datenbank gesperrt")

    widget = ChartWidget(ChartService(broken))
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.load_failed) as blocker:
        widget.load_symbol("AAPL")
    assert "Datenbank gesperrt" in blocker.args[0]
    assert widget.message_label.text() == blocker.args[0]
    assert widget.candle_count() == 0


def test_stale_results_are_ignored(qtbot: QtBot, chart_service: RecordingService) -> None:
    pending: list[Callable[[], None]] = []

    def deferred(
        fn: Callable[..., Any],
        *args: Any,
        on_result: Callable[[Any], None] | None = None,
        name: str = "",
    ) -> None:
        assert name == "Chart SYNTH"
        assert on_result is not None
        pending.append(lambda: on_result(fn(*args)))

    widget = ChartWidget(chart_service, runner=deferred)
    qtbot.addWidget(widget)
    widget.load_symbol("SYNTH")
    widget.set_timeframe(Timeframe.MN1)
    assert widget.message_label.text() == texts.CHART_LOADING.format(symbol="SYNTH")
    pending[1]()
    pending[0]()  # veraltete 1D-Antwort kommt später an
    assert widget.candle_count() == EXPECTED_COUNTS[Timeframe.MN1]
    assert widget.timeframe is Timeframe.MN1


def test_set_data_directly_selects_timeframe(
    chart: ChartWidget, chart_service: RecordingService
) -> None:
    chart.set_data(chart_service.load("SYNTH", Timeframe.D3))
    assert chart.timeframe is Timeframe.D3
    assert chart.timeframe_buttons[Timeframe.D3].isChecked()
    assert chart.symbol == "SYNTH"


def test_palette_follows_theme(chart: ChartWidget, qtbot: QtBot) -> None:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    apply_theme(app, "dark")
    try:
        qtbot.waitUntil(lambda: chart.graphics.backgroundBrush().color().name() == "#1e1e1e")
        QApplication.sendEvent(chart, QEvent(QEvent.Type.PaletteChange))
        assert chart.graphics.backgroundBrush().color().name() == "#1e1e1e"
    finally:
        apply_theme(app, "light")


def test_formatting() -> None:
    assert format_price(1234.5) == "1.234,50"
    assert format_volume(1234567.4) == "1.234.567"


def test_watchlist_click_loads_symbol(
    qtbot: QtBot, tmp_path: Path, daily: pd.DataFrame, dialogs: list[tuple[str, str]]
) -> None:
    service = AppService(Settings(data_dir=tmp_path / "data"), tmp_path / "state.json")
    win = MainWindow(service, chart_service=ChartService(lambda _s: daily))
    qtbot.addWidget(win)
    assert win.tab_widget("chart") is win.chart
    win.tabs.setCurrentIndex(2)
    win.watchlist.add_symbol("SYNTH")
    item = win.watchlist.symbol_list.item(0)
    with qtbot.waitSignal(win.chart.data_loaded, timeout=10_000):
        win.watchlist.symbol_list.itemClicked.emit(item)
    assert win.tabs.currentWidget() is win.chart
    assert win.chart.symbol == "SYNTH"
    assert win.chart.candle_count() == EXPECTED_COUNTS[Timeframe.D1]
    qtbot.waitUntil(lambda: "SYNTH" in win.statusBar().currentMessage())
    assert not dialogs


def _synthetic_daily(rows: int) -> pd.DataFrame:
    """Synthetischer Trend mit Sägezahn über ``rows`` Handelstage (Mo–Fr)."""
    dates = pd.bdate_range("2000-01-03", periods=rows)
    steps = np.arange(rows, dtype=np.float64)
    close = 20 + steps * 0.05 + 3 * ((steps % 40) / 40)
    open_ = close * 0.995
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": close * 1.01,
            "low": open_ * 0.99,
            "close": close,
            "volume": (1_000_000 + (steps % 7) * 10_000).astype("int64"),
            "adj_close": close,
        }
    )


@pytest.mark.slow
def test_25_years_draw_fast_and_scroll_smoothly(qtbot: QtBot) -> None:
    daily = _synthetic_daily(6500)
    service = ChartService(lambda _s: daily)
    service.load("BIG", Timeframe.D1)  # Laden zählt nicht zum Zeichnen
    widget = ChartWidget(service)
    qtbot.addWidget(widget)
    widget.resize(1400, 800)
    widget.show()
    qtbot.waitExposed(widget)

    data = service.load("BIG", Timeframe.D1)
    widget.show_last(6500)
    start = time.perf_counter()
    widget.set_data(data)
    widget.show_last(len(data))  # alle 25 Jahre sichtbar
    widget.graphics.grab()  # erzwingt das vollständige Zeichnen
    draw_seconds = time.perf_counter() - start
    assert widget.candle_count() == 6500
    assert draw_seconds < 0.5, f"Zeichnen dauerte {draw_seconds * 1000:.0f} ms"

    # Flüssig scrollen: Verschieben plus Neuzeichnen kostet je Bild höchstens 16 ms mehr
    # als ein leerer Chart gleicher Größe (die Grundkosten des Offscreen-Renderers
    # hängen von der Maschine ab und zählen nicht).
    empty = ChartWidget(ChartService())
    qtbot.addWidget(empty)
    empty.resize(1400, 800)
    empty.show()
    qtbot.waitExposed(empty)
    widget.show_last(DEFAULT_VISIBLE_BARS)

    def median_frame_ms(step: Callable[[], None]) -> float:
        frames: list[float] = []
        for _ in range(30):
            frame_start = time.perf_counter()
            step()
            frames.append(time.perf_counter() - frame_start)
        frames.sort()
        return frames[len(frames) // 2] * 1000

    def scroll() -> None:
        widget.price_vb.translateBy(x=-10)
        widget.graphics.viewport().repaint()

    baseline_ms = median_frame_ms(lambda: empty.graphics.viewport().repaint())
    scroll_ms = median_frame_ms(scroll)
    assert widget.visible_range()[0] < len(data) - DEFAULT_VISIBLE_BARS - 250
    assert scroll_ms - baseline_ms < 16, (
        f"Scrollen: {scroll_ms:.1f} ms je Bild, leerer Chart {baseline_ms:.1f} ms"
    )
