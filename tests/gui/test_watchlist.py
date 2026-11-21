"""Watchlist: hinzufügen, entfernen, suchen, Listen wechseln."""

from __future__ import annotations

from PySide6.QtCore import Qt
from pytestqt.qtbot import QtBot

from trading_helfer.gui.main_window import MainWindow
from trading_helfer.services.app_service import DEFAULT_WATCHLISTS, AppService


def test_add_and_remove(window: MainWindow, qtbot: QtBot, service: AppService) -> None:
    wl = window.watchlist
    qtbot.keyClicks(wl.symbol_edit, "aapl")
    qtbot.mouseClick(wl.add_button, Qt.MouseButton.LeftButton)
    wl.symbol_edit.setText("msft")
    wl.symbol_edit.returnPressed.emit()
    assert wl.visible_symbols() == ["AAPL", "MSFT"]
    assert wl.symbol_edit.text() == ""
    assert service.symbols("Eigene") == ["AAPL", "MSFT"]

    wl.symbol_list.setCurrentRow(0)
    qtbot.mouseClick(wl.remove_button, Qt.MouseButton.LeftButton)
    assert wl.visible_symbols() == ["MSFT"]
    assert service.symbols("Eigene") == ["MSFT"]


def test_remove_without_selection_does_nothing(window: MainWindow) -> None:
    window.watchlist.add_symbol("AAPL")
    window.watchlist.symbol_list.setCurrentItem(None)
    window.watchlist.remove_selected()
    assert window.watchlist.visible_symbols() == ["AAPL"]


def test_invalid_and_duplicate_symbols_warn(
    window: MainWindow, dialogs: list[tuple[str, str]]
) -> None:
    wl = window.watchlist
    assert wl.add_symbol("AAPL")
    assert not wl.add_symbol("aapl")
    assert not wl.add_symbol("")
    assert [kind for kind, _ in dialogs] == ["warning", "warning"]
    assert wl.visible_symbols() == ["AAPL"]


def test_search_filters(window: MainWindow) -> None:
    wl = window.watchlist
    for symbol in ("AAPL", "AMZN", "MSFT"):
        wl.add_symbol(symbol)
    wl.symbol_list.setCurrentRow(2)
    wl.search_edit.setText("ms")
    assert wl.visible_symbols() == ["MSFT"]
    assert wl.symbol_list.currentItem().text() == "MSFT"
    wl.search_edit.clear()
    assert wl.visible_symbols() == ["AAPL", "AMZN", "MSFT"]


def test_lists_per_index(window: MainWindow, service: AppService) -> None:
    wl = window.watchlist
    names = [wl.list_selector.itemText(i) for i in range(wl.list_selector.count())]
    assert names == list(DEFAULT_WATCHLISTS)
    wl.add_symbol("AAPL")
    wl.list_selector.setCurrentText("Nasdaq-100")
    assert wl.current_list == "Nasdaq-100"
    assert wl.visible_symbols() == []
    wl.add_symbol("NVDA")
    assert service.symbols("Nasdaq-100") == ["NVDA"]
    assert service.active_watchlist == "Nasdaq-100"
