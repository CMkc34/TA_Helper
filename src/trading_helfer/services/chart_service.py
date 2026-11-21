"""Fassade für den Chart: Tageskurse laden, bereinigen und in Zeitrahmen umrechnen.

Das Modul importiert kein Qt. Woher die Tageskurse kommen, bestimmt ein Loader
(``symbol -> DataFrame`` im Schema aus :mod:`trading_helfer.core.bars`). Die App nutzt
:func:`store_loader` (DuckDB-Kursspeicher); ohne Angabe liefert :func:`empty_loader`
leere Kursreihen.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from trading_helfer.core.bars import (
    ADJ_CLOSE_COLUMN,
    BAR_COLUMNS,
    PRICE_COLUMNS,
    empty_bars,
    validate_bars,
)
from trading_helfer.core.types import Timeframe
from trading_helfer.data.resample import PARTIAL_COLUMN, ResampleCache
from trading_helfer.data.storage import MarketStore

__all__ = [
    "CHART_COLUMNS",
    "ChartData",
    "ChartService",
    "DailyBarsLoader",
    "adjust_prices",
    "empty_loader",
    "store_loader",
]

DailyBarsLoader = Callable[[str], pd.DataFrame]
"""Liefert die Tageskurse eines Symbols (leer, wenn keine vorhanden sind)."""

CHART_COLUMNS: Final[tuple[str, ...]] = (*BAR_COLUMNS, PARTIAL_COLUMN)
"""Spalten der Kerzen, die der Chart erhält."""

_DAILY_CACHE_SIZE: Final = 16


def empty_loader(symbol: str) -> pd.DataFrame:
    """Standard-Loader ohne Kursspeicher: gibt für jedes Symbol eine leere Kursreihe zurück."""
    del symbol
    return empty_bars()


def store_loader(db_path: Path) -> DailyBarsLoader:
    """Loader, der rohe Tageskurse samt ``adj_close`` aus dem Kursspeicher liest.

    Jeder Aufruf öffnet die Datenbank nur lesend und schließt sie wieder, damit der
    Loader in Hintergrund-Threads laufen kann. Fehlt die Datenbankdatei, ist das
    Ergebnis leer.

    Args:
        db_path: Pfad der DuckDB-Datei (``Settings.database_path``).
    """

    def load(symbol: str) -> pd.DataFrame:
        if not db_path.exists():
            return empty_bars()
        with MarketStore(db_path, read_only=True) as store:
            return store.load_bars(symbol, adjusted=False)

    return load


def adjust_prices(daily: pd.DataFrame) -> pd.DataFrame:
    """Rechnet Open, High, Low und Close mit ``adj_factor = adj_close / close`` um.

    Ohne Spalte ``adj_close`` bleiben die Kurse unverändert. Fehlende ``adj_close``-Werte
    zählen als Faktor 1.

    Args:
        daily: Geprüfte Tageskurse.

    Returns:
        Neues DataFrame mit bereinigten Kursen; ``adj_close`` entspricht danach ``close``.
    """
    result = daily.copy()
    if ADJ_CLOSE_COLUMN not in result.columns:
        return result
    factor = (result[ADJ_CLOSE_COLUMN] / result["close"]).fillna(1.0)
    for column in PRICE_COLUMNS:
        result[column] = result[column] * factor
    return result


@dataclass(frozen=True)
class ChartData:
    """Kerzen eines Symbols für den Chart.

    Attributes:
        symbol: Tickersymbol.
        timeframe: Zeitrahmen der Kerzen.
        adjusted: ``True``, wenn die Kurse bereinigt sind.
        bars: Spalten :data:`CHART_COLUMNS`, nach Datum sortiert, ``RangeIndex``.
    """

    symbol: str
    timeframe: Timeframe
    adjusted: bool
    bars: pd.DataFrame

    def __len__(self) -> int:
        """Anzahl der Kerzen."""
        return len(self.bars)

    @property
    def is_empty(self) -> bool:
        """``True``, wenn es keine Kerzen gibt."""
        return self.bars.empty


class ChartService:
    """Liefert die Kerzen für den Chart; rechnet Zeitrahmen nur bei Bedarf neu.

    Args:
        loader: Quelle der Tageskurse; Standard ist :func:`empty_loader`.
    """

    def __init__(self, loader: DailyBarsLoader | None = None) -> None:
        self._loader: DailyBarsLoader = loader if loader is not None else empty_loader
        self._daily: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._resampled = ResampleCache(maxsize=64)

    def set_loader(self, loader: DailyBarsLoader) -> None:
        """Wechselt die Quelle der Tageskurse und leert alle Zwischenspeicher."""
        self._loader = loader
        self.invalidate()

    def invalidate(self, symbol: str | None = None) -> None:
        """Verwirft zwischengespeicherte Tageskurse (eines Symbols oder aller)."""
        if symbol is None:
            self._daily.clear()
        else:
            self._daily.pop(symbol, None)
        # Geänderte Kurse können dieselben Daten und Zeilenzahl haben; daher alles verwerfen.
        self._resampled.clear()

    def daily_bars(self, symbol: str) -> pd.DataFrame:
        """Geprüfte Tageskurse von ``symbol``; nach dem ersten Laden aus dem Speicher.

        Raises:
            BarsValidationError: der Loader liefert Daten, die das Schema verletzen.
        """
        cached = self._daily.get(symbol)
        if cached is None:
            cached = validate_bars(self._loader(symbol))
            self._daily[symbol] = cached
            if len(self._daily) > _DAILY_CACHE_SIZE:
                self._daily.popitem(last=False)
        else:
            self._daily.move_to_end(symbol)
        return cached

    def load(self, symbol: str, timeframe: Timeframe, adjusted: bool = True) -> ChartData:
        """Kerzen von ``symbol`` im Zeitrahmen ``timeframe``, bereinigt oder roh.

        Args:
            symbol: Tickersymbol.
            timeframe: Ziel-Zeitrahmen.
            adjusted: ``True`` (Standard) für bereinigte, ``False`` für rohe Kurse.

        Returns:
            Kerzen mit den Spalten :data:`CHART_COLUMNS`.
        """
        timeframe = Timeframe(timeframe)
        daily = self.daily_bars(symbol)
        prices = adjust_prices(daily) if adjusted else daily
        key = f"{symbol}|{'adj' if adjusted else 'raw'}"
        bars = self._resampled.get(key, prices, timeframe)
        return ChartData(
            symbol=symbol,
            timeframe=timeframe,
            adjusted=adjusted,
            bars=bars.loc[:, list(CHART_COLUMNS)],
        )
