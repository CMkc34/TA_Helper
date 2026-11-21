"""Indikatoren, Vorlagen und Marker im Chart-Widget."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QColorDialog, QDialog, QInputDialog
from pytestqt.qtbot import QtBot

from trading_helfer.core.markers import Marker, MarkerKind
from trading_helfer.core.types import Timeframe
from trading_helfer.gui import texts
from trading_helfer.gui.main_window import MainWindow
from trading_helfer.gui.widgets.chart import MARKER_COLORS, ChartWidget
from trading_helfer.gui.widgets.indicator_dialog import ColorButton, IndicatorDialog
from trading_helfer.indicators import registry
from trading_helfer.indicators.registry import IndicatorError, indicator
from trading_helfer.services.app_service import AppService
from trading_helfer.services.chart_service import ChartService
from trading_helfer.services.indicator_service import (
    TEMPLATES_FILE_NAME,
    IndicatorConfig,
    IndicatorService,
    TemplateError,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "chart" / "SYNTH_daily.csv"
SPECS = registry.list_all()


@pytest.fixture
def daily() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, parse_dates=["date"])


@pytest.fixture
def indicator_service(tmp_path: Path) -> IndicatorService:
    return IndicatorService(tmp_path / TEMPLATES_FILE_NAME)


@pytest.fixture
def chart(qtbot: QtBot, daily: pd.DataFrame, indicator_service: IndicatorService) -> ChartWidget:
    service = ChartService(lambda symbol: daily if symbol == "SYNTH" else daily.head(0))
    widget = ChartWidget(service, indicator_service=indicator_service)
    qtbot.addWidget(widget)
    widget.resize(1000, 700)
    return widget


@pytest.fixture
def loaded(chart: ChartWidget) -> ChartWidget:
    chart.load_symbol("SYNTH")
    return chart


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_every_registered_indicator_adds_expected_lines(
    loaded: ChartWidget, spec: registry.IndicatorSpec
) -> None:
    key = loaded.add_indicator(spec.name)
    expected = len(spec.visible_outputs())
    assert loaded.line_count(key) == expected
    assert loaded.indicator_error(key) == ""
    count = len(loaded.data)  # type: ignore[arg-type]
    for output, values in loaded.indicator_values(key).items():
        assert len(values) == count
        assert np.isfinite(values).any(), output
    plot = loaded.indicator_plot(key)
    if spec.pane == "price":
        assert plot is loaded.price_plot
        assert loaded.pane_count() == 0
    else:
        assert loaded.panes() == [plot]
    values = loaded.indicator_values(key)
    for output, item in loaded.indicator_items(key).items():
        assert item in plot.items
        finite = np.isfinite(values[output])
        if isinstance(item, pg.BarGraphItem):
            drawn = np.asarray(item.opts["height"])[finite]
        elif isinstance(item, pg.ScatterPlotItem):
            drawn = np.array([p.pos().y() for p in item.points()])
        else:
            drawn = item.getData()[1][finite]
        np.testing.assert_allclose(drawn, values[output][finite])
    loaded.remove_indicator(key)
    assert loaded.line_count() == 0
    assert loaded.pane_count() == 0


def test_dialog_fields_generated_for_every_indicator(
    qtbot: QtBot, indicator_service: IndicatorService
) -> None:
    dialog = IndicatorDialog(indicator_service)
    qtbot.addWidget(dialog)
    assert dialog.indicator_combo.count() == len(SPECS)
    for spec in SPECS:
        dialog.select_indicator(spec.name)
        assert dialog.indicator_name == spec.name
        assert list(dialog.param_widgets) == list(spec.params)
        assert list(dialog.color_buttons) == list(spec.visible_outputs())
        assert dialog.config() == indicator_service.default_config(spec.name)


def test_new_indicator_without_gui_code_appears(
    loaded: ChartWidget, qtbot: QtBot, indicator_service: IndicatorService
) -> None:
    @indicator(
        "_demo",
        params={"window": 5, "factor": 1.5, "source": "close", "mode": "a"},
        pane="separate",
        outputs=["fast", "slow", "flag"],
        styles={"slow": "histogram", "flag": "hidden"},
    )
    def demo(
        df: pd.DataFrame,
        window: int = 5,
        factor: float = 1.5,
        source: str = "close",
        mode: str = "",
    ) -> pd.DataFrame:
        """Demo-Indikator für Tests."""
        base = df[source].rolling(window).mean()
        return pd.DataFrame({"fast": base * factor, "slow": base, "flag": 1.0}, index=df.index)

    try:
        dialog = IndicatorDialog(indicator_service)
        qtbot.addWidget(dialog)
        dialog.select_indicator("_demo")
        assert dialog.hint_label.text() == "Demo-Indikator für Tests."
        dialog.param_widgets["window"].setValue(7)  # type: ignore[attr-defined]
        dialog.param_widgets["factor"].setValue(2.0)  # type: ignore[attr-defined]
        dialog.param_widgets["source"].setCurrentText("high")  # type: ignore[attr-defined]
        dialog.param_widgets["mode"].setText("b")  # type: ignore[attr-defined]
        dialog.color_buttons["fast"].set_color("#00ff00")
        config = dialog.config()
        assert config.params == {"window": 7, "factor": 2.0, "source": "high", "mode": "b"}
        key = loaded.add_indicator(config)
        assert loaded.line_count(key) == 2
        items = loaded.indicator_items(key)
        assert isinstance(items["slow"], pg.BarGraphItem)
        assert isinstance(items["fast"], pg.PlotCurveItem)
        assert items["fast"].opts["pen"].color().name() == "#00ff00"
        assert loaded.indicator_label(key) == "_DEMO(7, 2, high, b)"
    finally:
        registry._unregister("_demo")


def test_separate_panes_layout_and_date_axis(loaded: ChartWidget) -> None:
    rsi = loaded.add_indicator("rsi")
    loaded.add_indicator("macd")
    sma = loaded.add_indicator("sma")
    assert loaded.pane_count() == 2
    rsi_pane, macd_pane = loaded.panes()
    layout = loaded.graphics.ci
    assert layout.getItem(2, 0) is rsi_pane
    assert layout.getItem(3, 0) is macd_pane
    assert not loaded.volume_plot.getAxis("bottom").isVisible()
    assert not rsi_pane.getAxis("bottom").isVisible()
    assert macd_pane.getAxis("bottom").isVisible()
    assert loaded.indicator_plot(sma) is loaded.price_plot
    # Panels folgen dem Zoom des Kurs-Charts.
    loaded.show_last(50)
    assert macd_pane.vb.viewRange()[0] == pytest.approx(loaded.price_vb.viewRange()[0])
    y0, y1 = rsi_pane.vb.viewRange()[1]
    assert 0 <= y0 < y1 <= 105

    loaded.remove_indicator(rsi)
    assert loaded.panes() == [macd_pane]
    assert layout.getItem(2, 0) is macd_pane
    assert macd_pane.getAxis("bottom").isVisible()
    loaded.clear_indicators()
    assert loaded.indicators() == []
    assert loaded.volume_plot.getAxis("bottom").isVisible()
    loaded.remove_indicator(999)  # unbekannt: nichts passiert


def test_update_indicator_changes_params_and_color(loaded: ChartWidget) -> None:
    key = loaded.add_indicator("sma")
    before = loaded.indicator_values(key)["sma"].copy()
    config = loaded.indicators()[0][1].model_copy(
        update={"params": {"period": 5}, "colors": {"sma": "#ff0000"}}
    )
    loaded.update_indicator(key, config)
    assert loaded.indicator_label(key) == "SMA(5, close)"
    assert np.isnan(before[5])
    assert not np.isnan(loaded.indicator_values(key)["sma"][5])
    pen = loaded.indicator_items(key)["sma"].opts["pen"]
    assert pen.color().name() == "#ff0000"
    with pytest.raises(IndicatorError, match="nicht gewechselt"):
        loaded.update_indicator(key, IndicatorConfig(name="ema"))


def test_indicators_survive_timeframe_and_log(loaded: ChartWidget) -> None:
    key = loaded.add_indicator("ema")
    loaded.set_timeframe(Timeframe.W1)
    values = loaded.indicator_values(key)["ema"]
    assert len(values) == len(loaded.data)  # type: ignore[arg-type]
    curve = loaded.indicator_items(key)["ema"]
    _x, linear_y = curve.getData()
    loaded.set_log_scale(True)
    _x, log_y = curve.getData()
    finite = np.isfinite(values)
    np.testing.assert_allclose(log_y[finite], np.log10(values[finite]))
    np.testing.assert_allclose(linear_y[finite], values[finite])


def test_indicator_before_data_and_empty_symbol(chart: ChartWidget) -> None:
    key = chart.add_indicator("sma")
    assert chart.indicator_values(key) == {}
    chart.load_symbol("SYNTH")
    assert len(chart.indicator_values(key)["sma"]) > 0
    chart.load_symbol("LEER")
    assert chart.indicator_values(key) == {}


def test_compute_error_is_shown(loaded: ChartWidget) -> None:
    @indicator("_fails", params={"n": 1}, pane="price")
    def fails(df: pd.DataFrame, n: int = 1) -> pd.Series:
        raise ValueError("geht nicht")

    try:
        key = loaded.add_indicator("_fails")
        assert "geht nicht" in loaded.indicator_error(key)
        assert "_FAILS(1)" in loaded.message_label.text()
        assert loaded.indicator_values(key) == {}
    finally:
        registry._unregister("_fails")


def test_crosshair_shows_indicator_values(loaded: ChartWidget) -> None:
    loaded.add_indicator("sma")
    loaded.add_indicator("rsi")
    loaded.move_crosshair(100.0, None)
    text = loaded.info_label.text()
    assert "SMA(20, close) " in text
    assert "RSI(14, close) " in text
    pane_line = next(iter(loaded._pane_lines.values()))
    assert pane_line.isVisible()
    assert pane_line.value() == 100
    loaded.move_crosshair(0.0, None)  # Anlaufphase: keine Werte
    assert "SMA" not in loaded.info_label.text()
    pane = loaded.panes()[0]
    center = pane.sceneBoundingRect().center()
    loaded._on_mouse_moved(QPointF(center))
    assert pane_line.isVisible()
    loaded._on_mouse_moved(QPointF(-50, -50))
    assert not pane_line.isVisible()


# -- Vorlagen ------------------------------------------------------------------------


def test_templates_save_and_apply(loaded: ChartWidget) -> None:
    with pytest.raises(TemplateError):
        loaded.save_template("leer")
    loaded.add_indicator(IndicatorConfig(name="sma", params={"period": 10}))
    loaded.add_indicator("macd")
    assert loaded.save_template("Trend 1W") == "Trend 1W"
    loaded.clear_indicators()
    loaded.add_indicator("rsi")
    loaded.apply_template("Trend 1W")
    assert [c.name for _k, c in loaded.indicators()] == ["sma", "macd"]
    assert loaded.indicators()[0][1].params["period"] == 10
    assert loaded.pane_count() == 1


def test_templates_survive_restart(
    qtbot: QtBot, service: AppService, dialogs: list[tuple[str, str]], daily: pd.DataFrame
) -> None:
    chart_service = ChartService(lambda _s: daily)
    first = MainWindow(service, chart_service=chart_service)
    qtbot.addWidget(first)
    first.chart.load_symbol("SYNTH")
    first.chart.add_indicator("bbands")
    first.chart.add_indicator("rsi")
    first.chart.save_template("Trend 1W")
    expected = [c for _k, c in first.chart.indicators()]
    first.close()
    assert (service.state_path.parent / TEMPLATES_FILE_NAME).exists()

    restarted = AppService(service.settings, service.state_path)
    second = MainWindow(restarted, chart_service=chart_service)
    qtbot.addWidget(second)
    assert second.indicator_service.template_names() == ["Trend 1W"]
    second.chart.load_symbol("SYNTH")
    second.chart.apply_template("Trend 1W")
    assert [c for _k, c in second.chart.indicators()] == expected
    assert second.chart.line_count() == 4


def test_save_template_dialog(loaded: ChartWidget, monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        "trading_helfer.gui.widgets.chart.QMessageBox.warning",
        lambda _p, _t, text: warnings.append(text),
    )
    answers = iter([("", False), ("  ", True), ("Mein Setup", True)])
    monkeypatch.setattr(QInputDialog, "getText", lambda *_a, **_k: next(answers))
    assert loaded.save_template_dialog() is None
    loaded.add_indicator("sma")
    assert loaded.save_template_dialog() is None
    assert warnings
    assert "Namen" in warnings[0]
    assert loaded.save_template_dialog() == "Mein Setup"
    assert loaded.indicator_service.template_names() == ["Mein Setup"]


# -- Dialog und Menü ----------------------------------------------------------------


def _accept_later(chart: ChartWidget, edit: object = None) -> None:
    def accept() -> None:
        dialog = chart.findChild(IndicatorDialog)
        assert dialog is not None
        if callable(edit):
            edit(dialog)
        dialog.accept()

    QTimer.singleShot(0, accept)


def test_add_dialog_adds_indicator(loaded: ChartWidget) -> None:
    def choose(dialog: IndicatorDialog) -> None:
        dialog.select_indicator("rsi")
        dialog.param_widgets["period"].setValue(9)  # type: ignore[attr-defined]

    _accept_later(loaded, choose)
    key = loaded.open_add_dialog()
    assert key is not None
    assert loaded.indicator_label(key) == "RSI(9, close)"


def test_add_dialog_cancel(loaded: ChartWidget) -> None:
    def cancel() -> None:
        dialog = loaded.findChild(IndicatorDialog)
        assert dialog is not None
        dialog.reject()

    QTimer.singleShot(0, cancel)
    assert loaded.open_add_dialog() is None
    assert loaded.indicators() == []


def test_edit_dialog_keeps_indicator(loaded: ChartWidget) -> None:
    key = loaded.add_indicator("bbands")

    def edit(dialog: IndicatorDialog) -> None:
        assert not dialog.indicator_combo.isEnabled()
        assert dialog.windowTitle() == texts.INDICATOR_EDIT_TITLE
        dialog.param_widgets["num_std"].setValue(2.5)  # type: ignore[attr-defined]

    _accept_later(loaded, edit)
    assert loaded.open_edit_dialog(key) == key
    assert loaded.indicators()[0][1].params["num_std"] == 2.5


def test_dialog_error_is_shown_and_dialog_reopens(
    loaded: ChartWidget, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        "trading_helfer.gui.widgets.chart.QMessageBox.warning",
        lambda _p, _t, text: warnings.append(text),
    )
    calls = iter([QDialog.DialogCode.Accepted, QDialog.DialogCode.Rejected])
    monkeypatch.setattr(IndicatorDialog, "exec", lambda _self: next(calls))
    monkeypatch.setattr(
        IndicatorDialog, "config", lambda _self: IndicatorConfig(name="sma", params={"x": 1})
    )
    assert loaded.open_add_dialog() is None
    assert warnings
    assert "Unbekannte" in warnings[0]


def test_color_button_uses_color_dialog(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    button = ColorButton("#112233")
    qtbot.addWidget(button)
    assert button.color == "#112233"
    monkeypatch.setattr(QColorDialog, "getColor", lambda *_a: QColor("#abcdef"))
    button.click()
    assert button.color == "#abcdef"
    monkeypatch.setattr(QColorDialog, "getColor", lambda *_a: QColor())
    button.click()
    assert button.color == "#abcdef"


def test_indicator_menu_entries(loaded: ChartWidget, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded._rebuild_indicator_menu()
    titles = [a.text() for a in loaded.indicator_menu.actions()]
    assert texts.INDICATOR_NONE in titles

    key = loaded.add_indicator("sma")
    loaded.save_template("Vorlage A")
    loaded.add_markers([Marker(date(2024, 1, 2), MarkerKind.BUY)])
    loaded._rebuild_indicator_menu()
    actions = {a.text(): a for a in loaded.indicator_menu.actions()}
    sub = actions[loaded.indicator_label(key)].menu()
    sub_actions = {a.text(): a for a in sub.actions()}
    edited: list[int] = []
    monkeypatch.setattr(loaded, "open_edit_dialog", edited.append)
    sub_actions[texts.INDICATOR_EDIT].trigger()
    assert edited == [key]

    load_menu = actions[texts.TEMPLATE_LOAD].menu()
    load_menu.actions()[0].trigger()
    assert [c.name for _k, c in loaded.indicators()] == ["sma"]

    new_key = loaded.indicators()[0][0]
    loaded._rebuild_indicator_menu()
    actions = {a.text(): a for a in loaded.indicator_menu.actions()}
    sub = actions[loaded.indicator_label(new_key)].menu()
    {a.text(): a for a in sub.actions()}[texts.INDICATOR_REMOVE].trigger()
    assert loaded.indicators() == []

    loaded.add_indicator("ema")
    loaded._rebuild_indicator_menu()
    actions = {a.text(): a for a in loaded.indicator_menu.actions()}
    actions[texts.INDICATOR_REMOVE_ALL].trigger()
    assert loaded.indicators() == []
    saved: list[bool] = []
    monkeypatch.setattr(loaded, "save_template_dialog", lambda: saved.append(True))
    actions[texts.TEMPLATE_SAVE].trigger()
    assert saved == [True]
    actions[texts.TEMPLATE_DELETE].menu().actions()[0].trigger()
    assert loaded.indicator_service.template_names() == []
    actions[texts.MARKER_CLEAR].trigger()
    assert loaded.markers() == []
    loaded._rebuild_indicator_menu()
    actions = {a.text(): a for a in loaded.indicator_menu.actions()}
    assert actions[texts.TEMPLATE_LOAD].menu().actions()[0].text() == texts.TEMPLATE_NONE


def test_add_button_opens_dialog(loaded: ChartWidget, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[bool] = []
    monkeypatch.setattr(loaded, "open_add_dialog", lambda: opened.append(True))
    loaded.add_indicator_button.click()
    assert opened == [True]


def test_palette_change_colors_panes(loaded: ChartWidget) -> None:
    loaded.add_indicator("rsi")
    app = QApplication.instance()
    assert app is not None
    loaded._apply_palette()
    pane = loaded.panes()[0]
    assert pane.getAxis("left").textPen().color() == loaded.palette().color(
        loaded.palette().ColorRole.Text
    )


# -- Marker --------------------------------------------------------------------------


def test_markers_at_correct_dates(loaded: ChartWidget) -> None:
    bars = loaded.data.bars  # type: ignore[union-attr]
    d10 = bars["date"].iloc[10].date()
    d20 = bars["date"].iloc[20].date()
    d30 = bars["date"].iloc[30].date()
    loaded.add_markers(
        [
            Marker(d10, MarkerKind.BUY, text="Kauf"),
            Marker(d20, MarkerKind.SELL),
            Marker(d30, MarkerKind.SIGNAL, price=123.0, color="#000000"),
            Marker(date(1900, 1, 1), MarkerKind.EXIT),  # außerhalb: nicht gezeichnet
        ]
    )
    points = loaded.marker_points()
    assert [(i, m.kind) for i, _p, m in points] == [
        (10, MarkerKind.BUY),
        (20, MarkerKind.SELL),
        (30, MarkerKind.SIGNAL),
    ]
    assert points[0][1] == pytest.approx(bars["low"].iloc[10])
    assert points[1][1] == pytest.approx(bars["high"].iloc[20])
    assert points[2][1] == 123.0
    spots = loaded.marker_item.points()
    assert [round(p.pos().x()) for p in spots] == [10, 20, 30]
    assert spots[0].pos().y() < bars["low"].iloc[10]  # unter dem Tief
    assert spots[1].pos().y() > bars["high"].iloc[20]  # über dem Hoch
    assert spots[0].brush().color().name() == MARKER_COLORS[MarkerKind.BUY]
    assert spots[0].data() == "Kauf"
    assert len(loaded.markers()) == 4


def test_markers_follow_timeframe(loaded: ChartWidget) -> None:
    daily_bars = loaded.data.bars  # type: ignore[union-attr]
    day = daily_bars["date"].iloc[100].date()
    loaded.add_markers([Marker(day, MarkerKind.EXIT)])
    loaded.set_timeframe(Timeframe.W1)
    weekly = loaded.data.bars  # type: ignore[union-attr]
    ((index, price, _m),) = loaded.marker_points()
    assert weekly["date"].iloc[index].date() >= day
    assert index == 0 or weekly["date"].iloc[index - 1].date() < day
    assert price == pytest.approx(weekly["high"].iloc[index])
    loaded.set_log_scale(True)
    (spot,) = loaded.marker_item.points()
    assert spot.pos().y() > np.log10(weekly["high"].iloc[index])
    loaded.clear_markers()
    assert loaded.marker_points() == []
    assert len(loaded.marker_item.points()) == 0


def test_markers_before_data(chart: ChartWidget) -> None:
    chart.add_markers([Marker(date(2024, 1, 2))])
    assert chart.marker_points() == []
