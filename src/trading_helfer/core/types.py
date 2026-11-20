"""Gemeinsame Kern-Datentypen: Zeitrahmen, Positionsrichtung, Order- und Signaltypen.

Alle Typen sind ``StrEnum``s. Ihre Werte sind stabil, weil sie in der Datenbank
(``backtest_runs.timeframe``, ``backtest_trades.side``, ``signals.signal``), in
Einstellungen und in Strategie-Plugins gespeichert werden.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["OrderType", "Side", "SignalType", "Timeframe"]


class Timeframe(StrEnum):
    """Zeitrahmen einer Kerze.

    Gespeichert wird nur ``D1``; ``D3``, ``W1`` und ``MN1`` werden daraus berechnet
    (Regeln siehe ``docs/SPEC.md``, Abschnitt „Zeitrahmen-Regeln“).
    """

    D1 = "1D"
    """Ein Handelstag."""
    D3 = "3D"
    """Drei aufeinanderfolgende Handelstage ab dem festen Anker 2000-01-03."""
    W1 = "1W"
    """Eine Kalenderwoche (Montag bis Freitag)."""
    MN1 = "1M"
    """Ein Kalendermonat."""


class Side(StrEnum):
    """Richtung einer Position."""

    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(StrEnum):
    """Order-Typen der Backtest-Engine."""

    MARKET = "MARKET"
    STOP = "STOP"
    LIMIT = "LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class SignalType(StrEnum):
    """Signalarten von Strategien und Scanner."""

    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"
