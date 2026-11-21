"""Vorlage für eine eigene Strategie (siehe docs/strategy_guide.md).

Datei kopieren, ohne führenden Unterstrich speichern (z. B. ``plugins/meine_strategie.py``),
alle Stellen mit ``ANPASSEN`` ändern und im Strategien-Tab prüfen lassen. Dateien, deren
Name mit ``_`` beginnt, lädt die App nicht.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from trading_helfer.core.types import OrderType, Timeframe
from trading_helfer.indicators.trend import ema
from trading_helfer.indicators.volatility import atr
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    Params,
    PlotSpec,
    Sizing,
    Strategy,
)


class MeineStrategie(Strategy):  # ANPASSEN: Klassenname
    """ANPASSEN: Idee der Strategie in ein bis zwei Sätzen."""

    # Pflichtfelder ---------------------------------------------------------
    id = "meine_strategie"  # ANPASSEN: eindeutig, nur a-z, 0-9 und _
    name = "Meine Strategie"  # ANPASSEN: Anzeigename auf Deutsch
    version = "1.0.0"  # SemVer; bei jeder Regeländerung erhöhen
    description = (  # ANPASSEN: Einstieg, Ausstieg, Zeitrahmen, Long/Short
        "Kauf, wenn der Close über die EMA steigt; Ausstieg über einen Trailing Stop "
        "im Abstand von 3 × ATR(14). Für 3-Tages- und Wochenkerzen. Nur Long."
    )
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D3, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        # Param(Standard, Minimum, Maximum, Schrittweite, Beschreibung)
        "ema_period": Param(30, 5, 100, 5, "Periode der Trend-EMA"),
        "atr_mult": Param(3.0, 1.0, 6.0, 0.5, "Abstand des Trailing Stops in ATR(14)"),
    }

    def __init__(self) -> None:
        """Legt die Strategie mit Standardparametern an."""
        self._p = self.make_params()

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """Alle Indikatoren vektorisiert; Zeile t darf nur Kerzen bis t verwenden."""
        self._p = p  # Parameter für on_bar merken
        return pd.DataFrame(
            {
                "ema": ema(bars, p.ema_period),
                "atr": atr(bars, 14),
            },
            index=bars.index,
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Entscheidung auf dem Close der Kerze t; Ausführung am Open von t+1."""
        position = ctx.position
        if position is None:
            # Einstieg: nur wenn der Wert am Tag t zum Universum gehört.
            if ctx.is_member and ctx.crossed_above("close", "ema"):
                return [ctx.buy(sizing=Sizing.percent_equity(100), tag="Close über EMA")]
            return []
        if position.entry_date == ctx.date:
            # Kerze, auf der der Einstieg ausgeführt wurde: Schutz einmalig setzen.
            return ctx.exit(
                type=OrderType.TRAILING_STOP,
                trail_atr=float(self._p.atr_mult),
                tag="ATR-Trailing-Stop",
            )
        return []

    def plot(self) -> list[PlotSpec]:
        """Linien im Chart; jede Spalte muss aus indicators() stammen."""
        return [
            PlotSpec("ema", label="EMA", color="#1f77b4"),
            PlotSpec("atr", pane="separate", label="ATR(14)", color="#7f7f7f"),
        ]
