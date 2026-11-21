"""SMA-Kreuzung: kauft, wenn die schnelle SMA die langsame von unten kreuzt."""

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


class SmaCross(Strategy):
    id = "example_sma_cross"
    name = "SMA-Kreuzung"
    version = "1.0.0"
    description = "Kauft, wenn die schnelle SMA die langsame von unten kreuzt."
    timeframes = [Timeframe.D3, Timeframe.W1]
    params = {
        "fast": Param(10, 2, 50, 1, "Periode der schnellen SMA"),
        "slow": Param(30, 10, 200, 5, "Periode der langsamen SMA"),
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        return pd.DataFrame({"fast": sma(bars, p.fast), "slow": sma(bars, p.slow)})

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and ctx.crossed_above("fast", "slow"):
            return [ctx.buy(sizing=Sizing.percent_equity(100), tag="SMA-Kreuzung")]
        if ctx.position is not None and ctx.crossed_below("fast", "slow"):
            return ctx.exit(tag="SMA-Kreuzung abwärts")
        return []

    def plot(self) -> list[PlotSpec]:
        return [PlotSpec("fast"), PlotSpec("slow")]
