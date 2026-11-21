"""Ungültig: kauft und verkauft zufällig, ohne festen Seed (nicht reproduzierbar)."""

import numpy as np
import pandas as pd

from trading_helfer.core.types import Timeframe
from trading_helfer.strategies.base import BarContext, Order, Params, Sizing, Strategy


class CoinFlip(Strategy):
    id = "bad_unseeded_random"
    name = "Münzwurf"
    version = "1.0.0"
    description = "Würfelt jede Kerze neu."
    timeframes = [Timeframe.W1]

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        rng = np.random.default_rng()
        return pd.DataFrame({"coin": rng.random(len(bars))})

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and ctx.ind["coin"] > 0.7:
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        if ctx.position is not None and ctx.ind["coin"] < 0.3:
            return ctx.exit()
        return []
