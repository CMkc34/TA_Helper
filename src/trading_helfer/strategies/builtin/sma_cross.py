"""Beispielstrategie: SMA-Kreuzung (Trendfolge auf Wochenkerzen)."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from trading_helfer.core.types import Timeframe
from trading_helfer.indicators.trend import sma
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    Params,
    PlotSpec,
    Sizing,
    Strategy,
)

__all__ = ["SmaCross"]


class SmaCross(Strategy):
    """Kauft, wenn die schnelle SMA die langsame von unten kreuzt, und verkauft umgekehrt.

    Standard: SMA 10 über SMA 40 auf 1W. Nur Long, volles Kapital je Trade.
    """

    id = "trend_sma_cross"
    name = "SMA-Kreuzung"
    version = "1.0.0"
    description = (
        "Trendfolge: Kauf, wenn die schnelle SMA (Standard 10) die langsame SMA "
        "(Standard 40) von unten kreuzt; Verkauf, wenn sie wieder darunter kreuzt. "
        "Gedacht für Wochenkerzen (1W). Nur Long, Order zum Open der Folgekerze."
    )
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1, Timeframe.D3, Timeframe.MN1]
    params: ClassVar[dict[str, Param]] = {
        "fast": Param(10, 2, 50, 1, "Periode der schnellen SMA"),
        "slow": Param(40, 10, 200, 5, "Periode der langsamen SMA"),
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """Schnelle und langsame SMA auf den Schlusskursen."""
        return pd.DataFrame(
            {"sma_fast": sma(bars, p.fast), "sma_slow": sma(bars, p.slow)}, index=bars.index
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Einstieg bei Kreuzung nach oben, Ausstieg bei Kreuzung nach unten."""
        if ctx.position is None:
            if ctx.crossed_above("sma_fast", "sma_slow"):
                return [ctx.buy(sizing=Sizing.percent_equity(100), tag="SMA-Kreuzung")]
            return []
        if ctx.crossed_below("sma_fast", "sma_slow"):
            return ctx.exit(tag="SMA-Kreuzung abwärts")
        return []

    def plot(self) -> list[PlotSpec]:
        """Beide SMA im Kurs-Chart."""
        return [
            PlotSpec("sma_fast", label="SMA schnell", color="#1f77b4"),
            PlotSpec("sma_slow", label="SMA langsam", color="#ff7f0e"),
        ]
