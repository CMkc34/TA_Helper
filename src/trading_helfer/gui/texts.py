"""Alle deutschen Texte der Oberfläche an einer Stelle (später übersetzbar)."""

from __future__ import annotations

from typing import Final

APP_TITLE: Final = "Trading Helfer"

# Menüs
MENU_FILE: Final = "&Datei"
MENU_VIEW: Final = "&Ansicht"
MENU_HELP: Final = "&Hilfe"
ACTION_QUIT: Final = "&Beenden"
ACTION_THEME_LIGHT: Final = "&Hell"
ACTION_THEME_DARK: Final = "&Dunkel"
ACTION_TOGGLE_WATCHLIST: Final = "&Watchlist anzeigen"
ACTION_ABOUT: Final = "Ü&ber Trading Helfer"
MENU_THEME: Final = "&Farbschema"

ABOUT_TITLE: Final = "Über Trading Helfer"
ABOUT_TEXT: Final = (
    "<b>Trading Helfer {version}</b><br>"
    "Technische Analyse, Kennzahlen und Backtesting für US-Aktien.<br>"
    "Die App erzeugt nur Signale und ist keine Anlageberatung."
)

# Tabs
TAB_CHART: Final = "Chart"
TAB_FUNDAMENTALS: Final = "Kennzahlen"
TAB_BACKTEST: Final = "Backtest"
TAB_OPTIMIZATION: Final = "Optimierung"
TAB_SCANNER: Final = "Scanner"
TAB_STRATEGIES: Final = "Strategien"
TAB_DATA: Final = "Daten"
PLACEHOLDER: Final = "„{name}“ folgt in einer späteren Version."

# Watchlist
WATCHLIST_TITLE: Final = "Watchlist"
WATCHLIST_SEARCH: Final = "Suchen …"
WATCHLIST_NEW_SYMBOL: Final = "Symbol, z. B. AAPL"
WATCHLIST_ADD: Final = "Hinzufügen"
WATCHLIST_REMOVE: Final = "Entfernen"
WATCHLIST_ERROR_TITLE: Final = "Watchlist"

# Statusleiste und Fehler
STATUS_READY: Final = "Bereit"
STATUS_RUNNING: Final = "{name} läuft …"
STATUS_PROGRESS: Final = "{name}: {percent} % {message}"
STATUS_DONE: Final = "{name} abgeschlossen"
STATUS_FAILED: Final = "{name} fehlgeschlagen"
STATUS_SYMBOL: Final = "Ausgewählt: {symbol}"
ERROR_TITLE: Final = "Fehler"
ERROR_TASK: Final = "Die Aufgabe „{name}“ ist fehlgeschlagen:\n\n{message}"
ERROR_UNEXPECTED: Final = "Unerwarteter Fehler:\n\n{message}\n\nDetails stehen im Log."
ERROR_CONFIG_TITLE: Final = "Einstellungen ungültig"
TASK_DEFAULT_NAME: Final = "Aufgabe"

# Chart
CHART_TIMEFRAME_TOOLTIP: Final = "Zeitrahmen {timeframe}"
CHART_LOG: Final = "Logarithmisch"
CHART_ADJUSTED: Final = "Bereinigt"
CHART_ADJUSTED_TOOLTIP: Final = "Kurse um Splits und Dividenden bereinigt (sonst roh)"
CHART_NO_SYMBOL: Final = "Symbol in der Watchlist anklicken, um den Chart zu laden."
CHART_LOADING: Final = "{symbol} wird geladen …"
CHART_NO_DATA: Final = "Keine Kursdaten für {symbol} vorhanden."
CHART_LOAD_FAILED: Final = "{symbol} konnte nicht geladen werden: {message}"
CHART_TASK_NAME: Final = "Chart {symbol}"
CHART_HEADER: Final = "{symbol} · {timeframe} · {mode}"
CHART_MODE_ADJUSTED: Final = "bereinigt"
CHART_MODE_RAW: Final = "roh"
CHART_PARTIAL: Final = " (unvollständig)"
CHART_OHLCV: Final = (
    "{date}{partial}   Eröffnung {open}   Hoch {high}   Tief {low}   "
    "Schluss {close}   Volumen {volume}"
)
