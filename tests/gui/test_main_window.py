"""Hauptfenster: Tabs, Menüs, Theme, Layout-Persistenz und Start."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QLabel
from pytestqt.qtbot import QtBot

from trading_helfer.core.config import Settings
from trading_helfer.gui import texts
from trading_helfer.gui.application import get_application, install_exception_hook, run_gui
from trading_helfer.gui.main_window import TABS, MainWindow
from trading_helfer.gui.theme import build_palette
from trading_helfer.services.app_service import AppService


def test_window_has_all_tabs(window: MainWindow) -> None:
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert titles == [
        "Chart",
        "Kennzahlen",
        "Backtest",
        "Optimierung",
        "Scanner",
        "Strategien",
        "Daten",
    ]
    assert window.tab_keys() == [key for key, _ in TABS]
    assert window.windowTitle() == texts.APP_TITLE
    assert window.statusBar().currentMessage() == texts.STATUS_READY


def test_menus(window: MainWindow) -> None:
    menus = [action.text() for action in window.menuBar().actions()]
    assert menus == [texts.MENU_FILE, texts.MENU_VIEW, texts.MENU_HELP]


def test_set_view_replaces_placeholder(window: MainWindow, qtbot: QtBot) -> None:
    assert isinstance(window.tab_widget("backtest"), QLabel)
    view = QLabel("echter Backtest")
    window.set_view("backtest", view)
    assert window.tab_widget("backtest") is view
    assert window.tabs.tabText(2) == "Backtest"


def test_about_dialog(window: MainWindow, dialogs: list[tuple[str, str]]) -> None:
    window.about_action.trigger()
    assert dialogs
    assert dialogs[0][0] == "about"


def test_theme_switch_is_saved(window: MainWindow, service: AppService) -> None:
    window.theme_actions["dark"].trigger()
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    assert app.property("theme") == "dark"
    assert app.palette().window().color() == build_palette("dark").window().color()
    assert service.theme == "dark"
    assert window.theme_actions["dark"].isChecked()


def test_layout_survives_restart(
    qtbot: QtBot, tmp_path: Path, state_path: Path, dialogs: list[tuple[str, str]]
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    first = MainWindow(AppService(settings, state_path))
    qtbot.addWidget(first)
    first.set_theme("dark")
    first.watchlist.add_symbol("AAPL")
    first.resize(700, 500)
    first.move(30, 40)
    first.show()
    qtbot.waitExposed(first)
    saved_size = first.size()
    saved_pos = first.geometry().topLeft()
    first.close()

    second = MainWindow(AppService(settings, state_path))
    qtbot.addWidget(second)
    # Die Plattform darf die Wunschgröße an Mindestgrößen anpassen; verglichen wird
    # deshalb mit der tatsächlich gespeicherten Größe.
    assert second.size() == saved_size
    assert second.geometry().topLeft() == saved_pos
    assert saved_size.height() == 500
    assert second.service.theme == "dark"
    assert second.theme_actions["dark"].isChecked()
    assert second.watchlist.visible_symbols() == ["AAPL"]
    second.set_theme("light")


def test_invalid_geometry_uses_default_size(
    qtbot: QtBot, service: AppService, dialogs: list[tuple[str, str]]
) -> None:
    service.set_window_layout("AAAA", None)
    win = MainWindow(service)
    qtbot.addWidget(win)
    assert (win.width(), win.height()) == (1280, 800)


def test_window_starts_fast(qtbot: QtBot, service: AppService) -> None:
    start = time.perf_counter()
    win = MainWindow(service)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    assert time.perf_counter() - start < 3.0


def test_symbol_click_shows_status(window: MainWindow, qtbot: QtBot) -> None:
    window.watchlist.add_symbol("MSFT")
    with qtbot.waitSignal(window.watchlist.symbol_selected) as blocker:
        item = window.watchlist.symbol_list.item(0)
        window.watchlist.symbol_list.itemClicked.emit(item)
    assert blocker.args == ["MSFT"]
    assert "MSFT" in window.statusBar().currentMessage()


def test_run_gui_without_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dialogs: list[tuple[str, str]]
) -> None:
    monkeypatch.setattr("trading_helfer.gui.application.setup_logging", lambda _s: tmp_path)
    hook = sys.excepthook
    config = tmp_path / "config.toml"
    config.write_text(f'data_dir = "{(tmp_path / "d").as_posix()}"\n', encoding="utf-8")
    code = run_gui(config_path=config, state_path=tmp_path / "s.json", exec_loop=False)
    assert code == 0
    assert sys.excepthook is hook
    assert (tmp_path / "s.json").exists()
    QApplication.instance().setPalette(build_palette("light"))  # type: ignore[union-attr]


def test_run_gui_with_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dialogs: list[tuple[str, str]]
) -> None:
    monkeypatch.setattr("trading_helfer.gui.application.setup_logging", lambda _s: tmp_path)
    monkeypatch.setattr(QApplication, "exec", lambda _self: 5)
    config = tmp_path / "config.toml"
    config.write_text(f'data_dir = "{(tmp_path / "d").as_posix()}"\n', encoding="utf-8")
    assert run_gui(config_path=config, state_path=tmp_path / "s.json") == 5


def test_run_gui_with_invalid_config(tmp_path: Path, dialogs: list[tuple[str, str]]) -> None:
    config = tmp_path / "config.toml"
    config.write_text("default_capital = -5\n", encoding="utf-8")
    assert run_gui(config_path=config, state_path=tmp_path / "s.json", exec_loop=False) == 1
    assert dialogs[0][0] == "critical"
    assert "default_capital" in dialogs[0][1]


def test_get_application_reuses_instance(qapp: QApplication) -> None:
    assert get_application() is qapp


def test_exception_hook_logs_and_shows_dialog(
    dialogs: list[tuple[str, str]], caplog: pytest.LogCaptureFixture
) -> None:
    previous = install_exception_hook()
    try:
        try:
            raise ValueError("kaputt")
        except ValueError as exc:
            sys.excepthook(type(exc), exc, exc.__traceback__)
    finally:
        sys.excepthook = previous
    assert "kaputt" in caplog.text
    assert dialogs[0][0] == "critical"


def test_exception_hook_passes_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[type[BaseException]] = []
    monkeypatch.setattr(sys, "excepthook", lambda t, _e, _tb: seen.append(t))
    install_exception_hook()
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert seen == [KeyboardInterrupt]
