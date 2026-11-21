"""Watchlist links im Hauptfenster: Listen je Index, Suche, Hinzufügen und Entfernen."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from trading_helfer.gui import texts
from trading_helfer.services.app_service import AppService, WatchlistError

__all__ = ["WatchlistWidget"]


class WatchlistWidget(QWidget):
    """Zeigt die Symbole einer Watchlist; Änderungen speichert der :class:`AppService`."""

    symbol_selected = Signal(str)
    """Wird beim Anklicken eines Symbols gesendet."""

    def __init__(self, service: AppService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._service = service

        self.list_selector = QComboBox()
        self.list_selector.addItems(service.watchlist_names())
        self.list_selector.setCurrentText(service.active_watchlist)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(texts.WATCHLIST_SEARCH)
        self.search_edit.setClearButtonEnabled(True)

        self.symbol_list = QListWidget()

        self.symbol_edit = QLineEdit()
        self.symbol_edit.setPlaceholderText(texts.WATCHLIST_NEW_SYMBOL)
        self.add_button = QPushButton(texts.WATCHLIST_ADD)
        self.remove_button = QPushButton(texts.WATCHLIST_REMOVE)

        add_row = QHBoxLayout()
        add_row.addWidget(self.symbol_edit, 1)
        add_row.addWidget(self.add_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.list_selector)
        layout.addWidget(self.search_edit)
        layout.addWidget(self.symbol_list, 1)
        layout.addLayout(add_row)
        layout.addWidget(self.remove_button)

        self.list_selector.currentTextChanged.connect(self._on_list_changed)
        self.search_edit.textChanged.connect(lambda _text: self.refresh())
        self.symbol_edit.returnPressed.connect(self.add_from_input)
        self.add_button.clicked.connect(self.add_from_input)
        self.remove_button.clicked.connect(self.remove_selected)
        self.symbol_list.itemClicked.connect(self._on_item_clicked)
        self.symbol_list.itemActivated.connect(self._on_item_clicked)

        self.refresh()

    @property
    def current_list(self) -> str:
        """Name der angezeigten Watchlist."""
        return self.list_selector.currentText()

    def visible_symbols(self) -> list[str]:
        """Symbole, die nach Suche in der Liste stehen."""
        return [self.symbol_list.item(i).text() for i in range(self.symbol_list.count())]

    def refresh(self) -> None:
        """Lädt die Liste neu und wendet den Suchfilter an."""
        selected = self._selected_symbol()
        self.symbol_list.clear()
        for symbol in self._service.search_symbols(self.current_list, self.search_edit.text()):
            item = QListWidgetItem(symbol)
            self.symbol_list.addItem(item)
            if symbol == selected:
                item.setSelected(True)
                self.symbol_list.setCurrentItem(item)

    def add_symbol(self, symbol: str) -> bool:
        """Fügt ``symbol`` zur angezeigten Liste hinzu; Fehler erscheinen als Hinweis."""
        try:
            clean = self._service.add_symbol(self.current_list, symbol)
        except WatchlistError as exc:
            QMessageBox.warning(self, texts.WATCHLIST_ERROR_TITLE, str(exc))
            return False
        self.search_edit.clear()
        self.refresh()
        matches = self.symbol_list.findItems(clean, Qt.MatchFlag.MatchExactly)
        if matches:
            self.symbol_list.setCurrentItem(matches[0])
        return True

    def add_from_input(self) -> None:
        """Übernimmt das Symbol aus dem Eingabefeld."""
        if self.add_symbol(self.symbol_edit.text()):
            self.symbol_edit.clear()

    def remove_selected(self) -> None:
        """Entfernt das ausgewählte Symbol aus der angezeigten Liste."""
        symbol = self._selected_symbol()
        if symbol is None:
            return
        self._service.remove_symbol(self.current_list, symbol)
        self.refresh()

    def _selected_symbol(self) -> str | None:
        item = self.symbol_list.currentItem()
        return item.text() if item is not None else None

    def _on_list_changed(self, name: str) -> None:
        if name:
            self._service.set_active_watchlist(name)
            self.refresh()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self.symbol_selected.emit(item.text())
