"""Ungültig: kauft, wenn der Schlusskurs von morgen höher ist (Blick in die Zukunft)."""

import pandas as pd

from trading_helfer.core.types import Timeframe
from trading_helfer.strategies.base import BarContext, Order, Params, Sizing, Strategy


class Clairvoyant(Strategy):
    id = "bad_lookahead"
    name = "Hellseher"
    version = "1.0.0"
    description = "Nutzt den Schlusskurs der nächsten Kerze."
    timeframes = [Timeframe.W1]

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        tomorrow = bars["close"].shift(-1).fillna(bars["close"])
        return pd.DataFrame({"tomorrow": tomorrow})

    def on_bar(self, ctx: BarContext) -> list[Order]:
        rising = ctx.ind["tomorrow"] > ctx.bars.get("close")
        if ctx.position is None and rising:
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        if ctx.position is not None and not rising:
            return ctx.exit()
        return []
