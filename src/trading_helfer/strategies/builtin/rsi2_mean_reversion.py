"""Beispielstrategie: RSI(2)-Mean-Reversion im Aufwärtstrend (3-Tages-Kerzen)."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from trading_helfer.core.types import Timeframe
from trading_helfer.indicators.momentum import rsi
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

__all__ = ["Rsi2MeanReversion"]


class Rsi2MeanReversion(Strategy):
    """Kauft kurze Rücksetzer (RSI(2) < 10), solange der Close über der SMA 200 liegt.

    Ausstieg, sobald der RSI(2) über 70 steigt. Die Schwellen stehen als Spalten
    ``entry_level`` und ``exit_level`` in den Indikatoren und erscheinen im RSI-Panel.
    """

    id = "mr_rsi2_trend"
    name = "RSI(2) im Aufwärtstrend"
    version = "1.0.0"
    description = (
        "Mean-Reversion: Kauf, wenn der RSI(2) unter 10 fällt und der Close über der "
        "SMA 200 liegt (Aufwärtstrend). Verkauf, sobald der RSI(2) über 70 steigt. "
        "Gedacht für 3-Tages-Kerzen (3D). Nur Long."
    )
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D3, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        "rsi_period": Param(2, 2, 14, 1, "Periode des RSI"),
        "entry_below": Param(10.0, 1.0, 40.0, 1.0, "Kauf, wenn der RSI unter diesem Wert liegt"),
        "exit_above": Param(70.0, 50.0, 95.0, 5.0, "Verkauf, wenn der RSI über diesem Wert liegt"),
        "trend_period": Param(200, 50, 300, 10, "Periode der Trend-SMA"),
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """RSI, Trend-SMA und die beiden Schwellen als konstante Spalten."""
        return pd.DataFrame(
            {
                "rsi": rsi(bars, p.rsi_period),
                "trend_sma": sma(bars, p.trend_period),
                "entry_level": float(p.entry_below),
                "exit_level": float(p.exit_above),
            },
            index=bars.index,
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Einstieg bei überverkauftem RSI im Aufwärtstrend, Ausstieg bei hohem RSI."""
        value = ctx.value
        if ctx.position is None:
            if value("close") > value("trend_sma") and value("rsi") < value("entry_level"):
                return [ctx.buy(sizing=Sizing.percent_equity(100), tag="RSI überverkauft")]
            return []
        if value("rsi") > value("exit_level"):
            return ctx.exit(tag="RSI über Ausstiegsschwelle")
        return []

    def plot(self) -> list[PlotSpec]:
        """Trend-SMA im Kurs-Chart, RSI mit Schwellen in eigenem Panel."""
        return [
            PlotSpec("trend_sma", label="SMA Trend", color="#ff7f0e"),
            PlotSpec("rsi", pane="separate", label="RSI", color="#9467bd"),
            PlotSpec("entry_level", pane="separate", label="Kaufschwelle", color="#2ca02c"),
            PlotSpec("exit_level", pane="separate", label="Verkaufsschwelle", color="#d62728"),
        ]
