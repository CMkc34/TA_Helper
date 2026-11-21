"""Kostenmodell des simulierten Brokers: Gebühren, Slippage und Leihgebühr.

Alle Prozentwerte werden wie in den Einstellungen in Prozent angegeben
(``0.05`` = 0,05 %). Slippage wirkt immer gegen den Trader: Käufe werden teurer,
Verkäufe billiger ausgeführt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from trading_helfer.strategies.base import OrderSide

if TYPE_CHECKING:
    from trading_helfer.core.config import Settings

__all__ = ["DAYS_PER_YEAR", "CostModel", "is_buy"]

DAYS_PER_YEAR = 365
"""Kalendertage je Jahr für die anteilige Leihgebühr."""


def is_buy(side: OrderSide) -> bool:
    """``True`` für Orders, die Stücke kaufen (``BUY``, ``BUY_TO_COVER``)."""
    return side in (OrderSide.BUY, OrderSide.BUY_TO_COVER)


def _check_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} darf nicht negativ sein, erhalten: {value}.")


@dataclass(frozen=True)
class CostModel:
    """Kosten je Order und für Short-Positionen.

    Attributes:
        fee_per_order: Fixe Gebühr je ausgeführter Order in Kontowährung.
        fee_pct: Prozentuale Gebühr auf den Orderwert.
        slippage_pct: Slippage in Prozent des Kurses, gegen den Trader.
        borrow_fee_pct: Leihgebühr für Short-Positionen in Prozent p. a. auf den
            Positionswert; wird täglich anteilig (Kalendertage / 365) berechnet.
    """

    fee_per_order: float = 1.0
    fee_pct: float = 0.0
    slippage_pct: float = 0.05
    borrow_fee_pct: float = 0.0

    def __post_init__(self) -> None:
        _check_non_negative("fee_per_order", self.fee_per_order)
        _check_non_negative("fee_pct", self.fee_pct)
        _check_non_negative("slippage_pct", self.slippage_pct)
        _check_non_negative("borrow_fee_pct", self.borrow_fee_pct)
        if self.slippage_pct >= 100:
            raise ValueError(f"slippage_pct muss unter 100 liegen, erhalten: {self.slippage_pct}.")

    @classmethod
    def from_settings(
        cls, settings: Settings, *, fee_pct: float = 0.0, borrow_fee_pct: float = 0.0
    ) -> CostModel:
        """Kostenmodell aus den Einstellungen (Gebühr je Order und Slippage).

        Args:
            settings: Geprüfte Einstellungen.
            fee_pct: Prozentuale Gebühr (nicht Teil der Einstellungen).
            borrow_fee_pct: Leihgebühr p. a. in Prozent (nicht Teil der Einstellungen).
        """
        return cls(
            fee_per_order=settings.fee_per_order,
            fee_pct=fee_pct,
            slippage_pct=settings.slippage_pct,
            borrow_fee_pct=borrow_fee_pct,
        )

    def fee(self, notional: float) -> float:
        """Gebühr für eine Order mit Orderwert ``notional`` (fix plus Prozent)."""
        return self.fee_per_order + abs(notional) * self.fee_pct / 100.0

    def fill_price(self, price: float, side: OrderSide) -> float:
        """Ausführungskurs nach Slippage gegen den Trader.

        Args:
            price: Kurs vor Slippage (z. B. Open oder Stop-Kurs).
            side: Richtung der Order.

        Returns:
            ``price · (1 + s)`` für Käufe, ``price · (1 − s)`` für Verkäufe.
        """
        s = self.slippage_pct / 100.0
        return price * (1.0 + s) if is_buy(side) else price * (1.0 - s)

    def borrow_fee(self, notional: float, days: float) -> float:
        """Leihgebühr für eine Short-Position mit Wert ``notional`` über ``days`` Kalendertage."""
        if days <= 0:
            return 0.0
        return abs(notional) * self.borrow_fee_pct / 100.0 * days / DAYS_PER_YEAR

    def max_qty_for_budget(self, budget: float, price: float) -> float:
        """Größte ungerundete Stückzahl, deren Orderwert plus Gebühren ``budget`` einhält.

        Args:
            budget: Verfügbarer Betrag inklusive Gebühren.
            price: Ausführungskurs je Stück (nach Slippage).

        Returns:
            Stückzahl ≥ 0.
        """
        if price <= 0:
            return 0.0
        available = budget - self.fee_per_order
        if available <= 0:
            return 0.0
        return available / (price * (1.0 + self.fee_pct / 100.0))
