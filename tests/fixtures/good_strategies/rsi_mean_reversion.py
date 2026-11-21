"""RSI-Mean-Reversion im Aufwärtstrend."""

import math

import pandas as pd

from trading_helfer.core.types import Timeframe
from trading_helfer.indicators.momentum import rsi
from trading_helfer.indicators.trend import sma
from trading_helfer.strategies.base import BarContext, Order, Param, Params, Sizing, Strategy


class RsiMeanReversion(Strategy):
    id = "example_rsi_mean_reversion"
    name = "RSI-Mean-Reversion im Aufwärtstrend"
    version = "0.3.1"
    description = "Kauft überverkaufte Kurse, solange der Kurs über der langen SMA liegt."
    timeframes = [Timeframe.D3, Timeframe.W1]
    params = {
        "rsi_period": Param(14, 2, 30, 1, "RSI-Periode"),
        "trend": Param(50, 20, 200, 10, "Periode der Trend-SMA"),
        "buy_below": Param(35.0, 5.0, 50.0, 5.0, "Kaufen unter diesem RSI"),
        "sell_above": Param(60.0, 50.0, 95.0, 5.0, "Verkaufen über diesem RSI"),
    }

    def __init__(self) -> None:
        super().__init__()
        self.p = self.make_params()

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        self.p = p
        return pd.DataFrame({"rsi": rsi(bars, p.rsi_period), "trend": sma(bars, p.trend)})

    def on_bar(self, ctx: BarContext) -> list[Order]:
        value = ctx.ind["rsi"]
        if math.isnan(value):
            return []
        if ctx.position is None:
            if value < self.p.buy_below and ctx.bars.get("close") > ctx.ind["trend"]:
                return [ctx.buy(sizing=Sizing.percent_equity(50), tag="RSI überverkauft")]
            return []
        if value > self.p.sell_above:
            return ctx.exit(tag="RSI erholt")
        return []
