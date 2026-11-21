"""Donchian-Ausbruch: kauft neue Hochs, verkauft neue Tiefs."""

import numpy as np
import pandas as pd

from trading_helfer.core.types import Timeframe
from trading_helfer.strategies.base import BarContext, Order, Param, Params, Sizing, Strategy


class DonchianBreakout(Strategy):
    id = "example_donchian_breakout"
    name = "Donchian-Ausbruch"
    version = "1.2.0"
    description = "Kauft beim Ausbruch über das Hoch der letzten n Kerzen."
    timeframes = [Timeframe.W1]
    params = {
        "entry": Param(20, 5, 100, 5, "Kerzen für das Ausbruchshoch"),
        "exit": Param(10, 3, 50, 1, "Kerzen für das Ausstiegstief"),
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        upper = bars["high"].rolling(p.entry).max().shift(1)
        lower = bars["low"].rolling(p.exit).min().shift(1)
        return pd.DataFrame({"upper": upper, "lower": lower})

    def on_bar(self, ctx: BarContext) -> list[Order]:
        close = ctx.bars.get("close")
        if ctx.position is None:
            if not np.isnan(close) and close > ctx.ind["upper"]:
                return [ctx.buy(sizing=Sizing.percent_equity(95), tag="Ausbruch")]
            return []
        if close < ctx.ind["lower"]:
            return ctx.exit(tag="Ausstiegstief")
        return []
