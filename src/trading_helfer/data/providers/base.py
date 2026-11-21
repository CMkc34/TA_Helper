"""Schnittstellen der Datenquellen (Provider) und gemeinsame Hilfen.

Die Protokolle :class:`PriceProvider`, :class:`FundamentalsProvider`,
:class:`InstrumentProvider` und :class:`OptionChainProvider` beschreiben, was eine
Datenquelle liefern muss (siehe ``docs/SPEC.md``, Abschnitt „Module im Detail ›
Daten-Provider“). Fehler einer Quelle (Netzwerk, Rate-Limit, unerwartete Antwort) meldet
jeder Provider einheitlich als :class:`ProviderError`. Ein unbekanntes Symbol ist kein
Fehler, sondern ergibt ein leeres Ergebnis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

import pandas as pd

__all__ = [
    "ACTION_COLUMNS",
    "FUNDAMENTALS_COLUMNS",
    "OPTION_CHAIN_COLUMNS",
    "OPTION_TYPES",
    "ActionType",
    "FundamentalsProvider",
    "InstrumentInfo",
    "InstrumentProvider",
    "OptionChainProvider",
    "PriceProvider",
    "ProviderError",
    "empty_actions",
    "empty_option_chain",
]

ACTION_COLUMNS: Final[tuple[str, ...]] = ("date", "type", "value")
"""Spalten der Corporate Actions: Tag, :class:`ActionType`, Wert."""

FUNDAMENTALS_COLUMNS: Final[tuple[str, ...]] = (
    "period_end",
    "period_type",
    "metric",
    "value",
    "reported_date",
)
"""Spalten der Fundamentaldaten im Langformat."""

OPTION_CHAIN_COLUMNS: Final[tuple[str, ...]] = (
    "expiry",
    "strike",
    "type",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "iv",
    "underlying_price",
)
"""Spalten einer Optionskette: Verfall, Basispreis, ``CALL``/``PUT``, Kurse, Umsatz,
offene Kontrakte, implizite Volatilität (Dezimalzahl, 0.25 = 25 %) und Kurs des
Basiswerts zum Abrufzeitpunkt."""

OPTION_TYPES: Final = frozenset({"CALL", "PUT"})
"""Erlaubte Werte der Spalte ``type`` einer Optionskette."""


class ActionType(StrEnum):
    """Art einer Corporate Action (Wert in ``corporate_actions.type``)."""

    SPLIT = "SPLIT"
    """Aktiensplit; ``value`` ist das Verhältnis (4:1 → ``4.0``, 1:10 → ``0.1``)."""
    DIVIDEND = "DIVIDEND"
    """Bardividende; ``value`` ist der Betrag je Aktie in Handelswährung."""


class ProviderError(Exception):
    """Eine Datenquelle konnte eine Anfrage nicht beantworten.

    Attributes:
        provider: Name des Providers (z. B. ``"yfinance"``).
        symbol: Betroffenes Symbol oder ``None``.
        reason: Grund auf Deutsch.
    """

    def __init__(self, provider: str, symbol: str | None, reason: str) -> None:
        prefix = f"{provider}: {symbol}: " if symbol else f"{provider}: "
        super().__init__(prefix + reason)
        self.provider = provider
        self.symbol = symbol
        self.reason = reason


@dataclass(frozen=True, slots=True)
class InstrumentInfo:
    """Stammdaten eines Wertpapiers (Tabelle ``instruments``)."""

    symbol: str
    """Symbol in App-Schreibweise (z. B. ``BRK.B``)."""
    name: str | None = None
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None
    source: str = ""
    """Name des Providers, der die Daten geliefert hat."""


@runtime_checkable
class PriceProvider(Protocol):
    """Quelle für Tageskurse und Corporate Actions."""

    name: str

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Tageskurse von ``start`` bis einschließlich ``end``.

        Returns:
            Spalten ``date, open, high, low, close, volume, adj_close`` (rohe Kurse plus
            bereinigter Schlusskurs, ``date`` ohne Zeitzone); leer, wenn nichts da ist.

        Raises:
            ProviderError: Die Quelle war nicht erreichbar oder antwortete fehlerhaft.
        """
        ...

    def fetch_actions(self, symbol: str) -> pd.DataFrame:
        """Alle Splits und Dividenden des Symbols.

        Returns:
            Spalten ``date, type, value`` (siehe :class:`ActionType`); leer, wenn nichts
            da ist.

        Raises:
            ProviderError: Die Quelle war nicht erreichbar oder antwortete fehlerhaft.
        """
        ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    """Quelle für Fundamentaldaten."""

    def fetch_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Fundamentaldaten im Langformat.

        Returns:
            Spalten ``period_end, period_type, metric, value, reported_date``.
        """
        ...


@runtime_checkable
class InstrumentProvider(Protocol):
    """Quelle für Stammdaten."""

    name: str

    def fetch_instrument(self, symbol: str) -> InstrumentInfo | None:
        """Stammdaten des Symbols oder ``None``, wenn die Quelle es nicht kennt.

        Raises:
            ProviderError: Die Quelle war nicht erreichbar oder antwortete fehlerhaft.
        """
        ...


@runtime_checkable
class OptionChainProvider(Protocol):
    """Quelle für aktuelle Optionsketten (ohne Historie)."""

    name: str

    def fetch_option_chain(self, symbol: str, max_expiries: int | None = None) -> pd.DataFrame:
        """Aktuelle Optionskette des Symbols über alle (oder die nächsten) Verfallstage.

        Args:
            symbol: Symbol in App-Schreibweise.
            max_expiries: Nur die nächsten ``n`` Verfallstage; ``None`` = alle.

        Returns:
            Spalten :data:`OPTION_CHAIN_COLUMNS`; leer, wenn es keine Optionen gibt.

        Raises:
            ProviderError: Die Quelle war nicht erreichbar oder antwortete fehlerhaft.
        """
        ...


def empty_option_chain() -> pd.DataFrame:
    """Leere Optionskette mit den richtigen Datentypen."""
    return pd.DataFrame(
        {
            "expiry": pd.Series(dtype="datetime64[ns]"),
            "strike": pd.Series(dtype="float64"),
            "type": pd.Series(dtype="object"),
            "bid": pd.Series(dtype="float64"),
            "ask": pd.Series(dtype="float64"),
            "last": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="Int64"),
            "open_interest": pd.Series(dtype="Int64"),
            "iv": pd.Series(dtype="float64"),
            "underlying_price": pd.Series(dtype="float64"),
        },
        index=pd.RangeIndex(0),
    )


def empty_actions() -> pd.DataFrame:
    """Leere Tabelle der Corporate Actions mit den richtigen Datentypen."""
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "type": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        },
        index=pd.RangeIndex(0),
    )
