"""Ungültig: wartet in on_bar auf ein Ereignis, das nie eintritt (Endlosschleife)."""

from trading_helfer.core.types import Timeframe
from trading_helfer.strategies.base import BarContext, Order, Strategy


class Forever(Strategy):
    id = "bad_endless_loop"
    name = "Endlosschleife"
    version = "1.0.0"
    description = "Hängt in einer Schleife fest."
    timeframes = [Timeframe.W1]

    def on_bar(self, ctx: BarContext) -> list[Order]:
        count = 0
        while count >= 0:
            count += 1
        return []
