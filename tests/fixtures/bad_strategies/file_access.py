"""Ungültig: schreibt Signale in eine Datei (verbotener Aufruf open)."""

import pandas as pd

from trading_helfer.core.types import Timeframe
from trading_helfer.strategies.base import BarContext, Order, Params, Sizing, Strategy


class FileWriter(Strategy):
    id = "bad_file_access"
    name = "Schreibt Dateien"
    version = "1.0.0"
    description = "Protokolliert jedes Signal in einer Datei."
    timeframes = [Timeframe.W1]

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        return pd.DataFrame({"sma": bars["close"].rolling(10).mean()})

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and ctx.crossed_above("close", "sma"):
            with open("signale.txt", "a", encoding="utf-8") as log:
                log.write(f"{ctx.date} Kauf\n")
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        return []
