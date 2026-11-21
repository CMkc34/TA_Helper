"""Fassade zwischen GUI und Logik: Einstellungen, Watchlists und GUI-Zustand.

:class:`AppService` ist die einzige Brücke, über die die GUI auf Logik zugreift.
Das Modul importiert kein Qt; der GUI-Zustand (Fenstergeometrie, Theme,
Watchlists) liegt als JSON-Datei neben der ``config.toml``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, Field, ValidationError

from trading_helfer.core.config import Settings, default_config_path

__all__ = [
    "DEFAULT_WATCHLISTS",
    "GUI_STATE_FILE_NAME",
    "AppService",
    "GuiState",
    "ThemeName",
    "WatchlistError",
    "default_gui_state_path",
    "normalize_symbol",
]

logger = logging.getLogger(__name__)

GUI_STATE_FILE_NAME: Final = "gui_state.json"

DEFAULT_WATCHLISTS: Final = ("Eigene", "S&P 500", "Nasdaq-100")
"""Watchlists, die immer vorhanden sind (eigene Liste und je Index eine)."""

ThemeName = Literal["light", "dark"]

_SYMBOL_PATTERN: Final = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")


class WatchlistError(ValueError):
    """Ungültige Watchlist-Aktion; die Meldung ist deutsch."""


def default_gui_state_path() -> Path:
    """Gibt den Standardpfad der GUI-Zustandsdatei zurück (neben der ``config.toml``)."""
    return default_config_path().parent / GUI_STATE_FILE_NAME


def normalize_symbol(symbol: str) -> str:
    """Bereinigt ein Tickersymbol (Leerzeichen entfernt, Großbuchstaben) und prüft es.

    Raises:
        WatchlistError: wenn das Symbol leer oder ungültig ist.
    """
    text = symbol.strip().upper()
    if not text:
        raise WatchlistError("Bitte ein Symbol eingeben.")
    if not _SYMBOL_PATTERN.match(text):
        raise WatchlistError(f"„{text}“ ist kein gültiges Symbol.")
    return text


def _default_watchlists() -> dict[str, list[str]]:
    return {name: [] for name in DEFAULT_WATCHLISTS}


class GuiState(BaseModel):
    """Zustand der Oberfläche, der einen Neustart überdauert."""

    theme: ThemeName = "light"
    window_geometry: str | None = Field(default=None, description="Base64 von saveGeometry()")
    window_state: str | None = Field(default=None, description="Base64 von saveState()")
    watchlists: dict[str, list[str]] = Field(default_factory=_default_watchlists)
    active_watchlist: str = DEFAULT_WATCHLISTS[0]


class AppService:
    """Einzige Schnittstelle der GUI zur Logik.

    Args:
        settings: Geprüfte Einstellungen der App.
        state_path: Datei für den GUI-Zustand; Standard ist :func:`default_gui_state_path`.
    """

    def __init__(self, settings: Settings, state_path: Path | None = None) -> None:
        self._settings = settings
        self._state_path = state_path if state_path is not None else default_gui_state_path()
        self._state = self._load_state()

    @property
    def settings(self) -> Settings:
        """Die geladenen Einstellungen."""
        return self._settings

    @property
    def state_path(self) -> Path:
        """Pfad der GUI-Zustandsdatei."""
        return self._state_path

    # -- Zustand laden und speichern -------------------------------------------------

    def _load_state(self) -> GuiState:
        if not self._state_path.exists():
            return self._complete(GuiState())
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            state = GuiState.model_validate(data)
        except (OSError, ValueError, ValidationError) as exc:
            logger.warning(
                "GUI-Zustand %s nicht lesbar, Standard wird genutzt: %s", self._state_path, exc
            )
            state = GuiState()
        return self._complete(state)

    @staticmethod
    def _complete(state: GuiState) -> GuiState:
        for name in DEFAULT_WATCHLISTS:
            state.watchlists.setdefault(name, [])
        if state.active_watchlist not in state.watchlists:
            state.active_watchlist = DEFAULT_WATCHLISTS[0]
        return state

    def save_state(self) -> None:
        """Schreibt den GUI-Zustand auf die Platte; Fehler werden nur geloggt."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(self._state.model_dump_json(indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("GUI-Zustand %s kann nicht gespeichert werden: %s", self._state_path, exc)

    # -- Theme und Fenster -----------------------------------------------------------

    @property
    def theme(self) -> ThemeName:
        """Gewähltes Farbschema (``light`` oder ``dark``)."""
        return self._state.theme

    def set_theme(self, theme: ThemeName) -> None:
        """Setzt das Farbschema und speichert den Zustand."""
        self._state.theme = theme
        self.save_state()

    def window_layout(self) -> tuple[str | None, str | None]:
        """Gibt die gespeicherte Fenstergeometrie und den Fensterzustand (Base64) zurück."""
        return self._state.window_geometry, self._state.window_state

    def set_window_layout(self, geometry: str | None, state: str | None) -> None:
        """Speichert Fenstergeometrie und Fensterzustand (Base64)."""
        self._state.window_geometry = geometry
        self._state.window_state = state
        self.save_state()

    # -- Watchlists ------------------------------------------------------------------

    def watchlist_names(self) -> list[str]:
        """Namen aller Watchlists in fester Reihenfolge."""
        return list(self._state.watchlists)

    @property
    def active_watchlist(self) -> str:
        """Zuletzt gewählte Watchlist."""
        return self._state.active_watchlist

    def set_active_watchlist(self, name: str) -> None:
        """Merkt sich die gewählte Watchlist.

        Raises:
            WatchlistError: wenn es die Watchlist nicht gibt.
        """
        self._require_list(name)
        self._state.active_watchlist = name
        self.save_state()

    def symbols(self, name: str) -> list[str]:
        """Symbole der Watchlist ``name`` (Kopie)."""
        return list(self._require_list(name))

    def add_symbol(self, name: str, symbol: str) -> str:
        """Fügt ``symbol`` zur Watchlist ``name`` hinzu und gibt das bereinigte Symbol zurück.

        Raises:
            WatchlistError: bei ungültigem Symbol, unbekannter Liste oder Duplikat.
        """
        items = self._require_list(name)
        clean = normalize_symbol(symbol)
        if clean in items:
            raise WatchlistError(f"„{clean}“ ist bereits in der Liste „{name}“.")
        items.append(clean)
        self.save_state()
        return clean

    def remove_symbol(self, name: str, symbol: str) -> None:
        """Entfernt ``symbol`` aus der Watchlist ``name``; fehlende Symbole werden ignoriert."""
        items = self._require_list(name)
        clean = symbol.strip().upper()
        if clean in items:
            items.remove(clean)
            self.save_state()

    def search_symbols(self, name: str, query: str) -> list[str]:
        """Symbole der Watchlist ``name``, die ``query`` enthalten (ohne Groß-/Kleinschreibung)."""
        needle = query.strip().upper()
        return [s for s in self._require_list(name) if needle in s]

    def _require_list(self, name: str) -> list[str]:
        try:
            return self._state.watchlists[name]
        except KeyError:
            raise WatchlistError(f"Watchlist „{name}“ gibt es nicht.") from None
