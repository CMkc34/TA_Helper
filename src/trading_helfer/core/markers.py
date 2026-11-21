"""Marker für Signale und Trades im Chart (ohne Qt, auch für Backtest und Scanner)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

__all__ = ["Marker", "MarkerKind"]


class MarkerKind(StrEnum):
    """Art eines Markers; bestimmt Symbol, Farbe und Lage an der Kerze."""

    BUY = "BUY"
    """Kauf oder Einstieg Long: Dreieck nach oben unter der Kerze."""
    SELL = "SELL"
    """Verkauf oder Einstieg Short: Dreieck nach unten über der Kerze."""
    EXIT = "EXIT"
    """Ausstieg: Kreuz über der Kerze."""
    SIGNAL = "SIGNAL"
    """Neutrales Signal: Raute am Schlusskurs."""


@dataclass(frozen=True)
class Marker:
    """Ein Marker an einem Datum.

    Der Chart setzt den Marker auf die Kerze, deren Zeitraum ``date`` enthält (bei 1W
    die Woche, bei 1M den Monat); Marker außerhalb der Kerzen erscheinen nicht.

    Attributes:
        date: Handelstag des Ereignisses.
        kind: Art des Markers.
        price: Kurs, an dem der Marker steht; ``None`` setzt ihn je nach Art unter das
            Tief, über das Hoch oder auf den Schlusskurs der Kerze.
        text: Kurzer Hinweistext (z. B. „Kauf 10 Stk.“).
        color: Farbe als ``#RRGGBB``; leer bedeutet Standardfarbe der Art.
    """

    date: date
    kind: MarkerKind = MarkerKind.SIGNAL
    price: float | None = None
    text: str = ""
    color: str = ""
