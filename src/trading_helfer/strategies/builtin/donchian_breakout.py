"""Beispielstrategie: Donchian-Ausbruch mit ATR-Trailing-Stop (Wochenkerzen)."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from trading_helfer.core.types import OrderType, Timeframe
from trading_helfer.indicators.volatility import atr, donchian
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    Params,
    PlotSpec,
    Sizing,
    Strategy,
)

__all__ = ["DonchianBreakout"]


class DonchianBreakout(Strategy):
    """Kauft den Ausbruch über das Hoch der letzten ``period`` Kerzen.

    Der Ausstieg erfolgt nur über einen Trailing Stop im Abstand von ``atr_mult``
    ATR(14) unter dem Close; er wird auf der Kerze gesetzt, auf der der Einstieg
    ausgeführt wurde, und danach von der Engine nach jedem Close nachgezogen.
    """

    id = "trend_donchian_breakout"
    name = "Donchian-Ausbruch"
    version = "1.0.0"
    description = (
        "Trendfolge: Kauf, wenn der Close über das höchste Hoch der vorherigen 20 Kerzen "
        "steigt (Donchian-Kanal). Ausstieg über einen Trailing Stop im Abstand von "
        "3 × ATR(14) unter dem Close, der nur nachgezogen wird. Gedacht für Wochenkerzen "
        "(1W). Nur Long."
    )
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1, Timeframe.D3, Timeframe.MN1]
    params: ClassVar[dict[str, Param]] = {
        "period": Param(20, 5, 100, 1, "Kerzen für das Ausbruchshoch (Donchian-Kanal)"),
        "atr_mult": Param(3.0, 1.0, 6.0, 0.5, "Abstand des Trailing Stops in ATR(14)"),
    }

    def __init__(self) -> None:
        """Legt die Strategie mit Standardparametern an."""
        self._p = self.make_params()

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """Donchian-Kanal der Vorkerzen und ATR(14) (nur zur Anzeige)."""
        self._p = p
        channel = donchian(bars, p.period).shift(1)
        return pd.DataFrame(
            {
                "dc_upper": channel["upper"],
                "dc_lower": channel["lower"],
                "atr": atr(bars, 14),
            },
            index=bars.index,
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Einstieg beim Ausbruch, Trailing Stop direkt nach der Ausführung."""
        position = ctx.position
        if position is None:
            if ctx.value("close") > ctx.value("dc_upper"):
                return [ctx.buy(sizing=Sizing.percent_equity(100), tag="Donchian-Ausbruch")]
            return []
        if position.entry_date == ctx.date:
            return ctx.exit(
                type=OrderType.TRAILING_STOP,
                trail_atr=float(self._p.atr_mult),
                tag="ATR-Trailing-Stop",
            )
        return []

    def plot(self) -> list[PlotSpec]:
        """Donchian-Kanal im Kurs-Chart, ATR in eigenem Panel."""
        return [
            PlotSpec("dc_upper", label="Donchian oben", color="#2ca02c"),
            PlotSpec("dc_lower", label="Donchian unten", color="#d62728"),
            PlotSpec("atr", pane="separate", label="ATR(14)", color="#7f7f7f"),
        ]
