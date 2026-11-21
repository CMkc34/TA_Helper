"""Ungültig: on_bar ohne Kontext-Parameter (falsche Signatur)."""

from trading_helfer.core.types import Timeframe
from trading_helfer.strategies.base import Order, Strategy


class NoContext(Strategy):
    id = "bad_wrong_signature"
    name = "Falsche Signatur"
    version = "1.0.0"
    description = "on_bar erwartet keinen Kontext."
    timeframes = [Timeframe.W1]

    def on_bar(self) -> list[Order]:  # type: ignore[override]
        return []
