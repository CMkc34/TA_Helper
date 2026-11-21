"""Simulierter Broker: führt Orders kerzenweise aus und verwaltet Kapital und Positionen.

Ablauf (siehe Spezifikation, Backtest-Engine): Die Strategie gibt auf Kerze ``t``
Orders ab (:meth:`SimBroker.submit`); die Engine ruft danach für Kerze ``t+1``
:meth:`SimBroker.process_bar` auf. Dort gilt:

- **Market**: Ausführung zum Open von ``t+1``.
- **Stop**: Kauf-Stop wird ausgelöst, wenn das High den Stop erreicht, Preis =
  ``max(Stop, Open)``; Verkaufs-Stop, wenn das Low den Stop erreicht, Preis =
  ``min(Stop, Open)``. Ein Gap über den Stop wird also zum Open ausgeführt.
- **Limit**: Kauf-Limit, wenn das Low das Limit erreicht, Preis = ``min(Limit, Open)``;
  Verkaufs-Limit, wenn das High das Limit erreicht, Preis = ``max(Limit, Open)``.
- **Trailing Stop**: wie Stop; der Stop wird nach jedem Close nachgezogen (in % vom
  Close oder in ATR-Vielfachen) und bewegt sich nur in Richtung des Traders.
- Reihenfolge innerhalb einer Kerze: Market, dann Stops (inkl. Trailing), dann Limits.
  Erreicht eine Kerze Stop und Ziel einer Position, gilt der Stop (konservativ); die
  übrigen Ausstiegs-Orders werden gestrichen, sobald die Position geschlossen ist.
- Slippage wirkt gegen den Trader bei Market- und Stop-Ausführungen; Limit-Orders
  werden höchstens zum Limitkurs ausgeführt.
- Gebühr je Order fix plus Prozent; Short-Positionen zahlen die Leihgebühr täglich
  anteilig (Kalendertage seit der letzten Kerze).

Stop-, Limit- und Trailing-Orders bleiben offen, bis sie ausgeführt oder mit
:meth:`SimBroker.cancel` gestrichen werden. Market-Orders gelten nur für die nächste
Kerze.

Kapitalregeln: Eine eröffnende Order wird verworfen und protokolliert, wenn Orderwert
plus Gebühr die Kaufkraft übersteigt. Short-Positionen sind voll gedeckt: Der Erlös und
noch einmal derselbe Betrag sind als Sicherheit gebunden (keine Hebelwirkung).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date as Date  # noqa: N812
from typing import TYPE_CHECKING

from trading_helfer.backtest.costs import CostModel, is_buy
from trading_helfer.core.types import OrderType, Side
from trading_helfer.strategies.base import Order, OrderSide, Position, SizingMethod

if TYPE_CHECKING:
    from trading_helfer.core.config import Settings

__all__ = [
    "QTY_DECIMALS",
    "Bar",
    "Fill",
    "PendingOrder",
    "Rejection",
    "SimBroker",
    "Trade",
    "round_qty",
]

logger = logging.getLogger(__name__)

QTY_DECIMALS = 4
"""Nachkommastellen der Stückzahl bei erlaubten Bruchstücken."""

_EPS = 1e-9


def round_qty(qty: float, *, allow_fractional: bool = True) -> float:
    """Rundet eine Stückzahl ab: auf 4 Nachkommastellen oder auf ganze Stücke.

    Args:
        qty: Ungerundete Stückzahl.
        allow_fractional: Bruchstücke erlaubt (Standard).

    Returns:
        Abgerundete Stückzahl, nie negativ.
    """
    if not math.isfinite(qty) or qty <= 0:
        return 0.0
    if not allow_fractional:
        return float(math.floor(qty + _EPS))
    scale = float(10**QTY_DECIMALS)
    return math.floor(qty * scale + _EPS) / scale


@dataclass(frozen=True)
class Bar:
    """Eine Kerze, wie der Broker sie für die Ausführung braucht.

    Attributes:
        date: Datum der Kerze.
        open: Eröffnungskurs.
        high: Höchstkurs.
        low: Tiefstkurs.
        close: Schlusskurs.
    """

    date: Date
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Fill:
    """Ausgeführte Order.

    Attributes:
        id: Laufende Nummer der Order (Reihenfolge der Abgabe).
        date: Datum der Ausführungskerze.
        symbol: Wertpapier.
        side: Richtung der Order.
        type: Order-Typ.
        qty: Ausgeführte Stückzahl.
        price: Ausführungskurs inklusive Slippage.
        fee: Gebühr der Order.
        tag: Freier Text der Order.
    """

    id: int
    date: Date
    symbol: str
    side: OrderSide
    type: OrderType
    qty: float
    price: float
    fee: float
    tag: str = ""


@dataclass(frozen=True)
class Rejection:
    """Verworfene Order mit Begründung (Text auf Deutsch).

    Attributes:
        id: Laufende Nummer der Order.
        date: Datum, an dem die Order verworfen wurde.
        order: Die verworfene Order.
        reason: Begründung.
    """

    id: int
    date: Date
    order: Order
    reason: str


@dataclass(frozen=True)
class Trade:
    """Abgeschlossener Trade (ganz oder teilweise geschlossene Position).

    Attributes:
        symbol: Wertpapier.
        side: Long oder Short.
        entry_date: Datum der ersten Einstiegs-Ausführung.
        entry_price: Durchschnittlicher Einstiegskurs inklusive Slippage.
        exit_date: Datum der Ausstiegs-Ausführung.
        exit_price: Ausstiegskurs inklusive Slippage.
        qty: Geschlossene Stückzahl.
        pnl: Gewinn oder Verlust nach allen Kosten.
        pnl_pct: ``pnl`` in Prozent des Einstiegswerts.
        fees: Anteilige Gebühren und Leihgebühren dieses Trades.
        exit_reason: Tag der Ausstiegs-Order oder ihr Order-Typ.
    """

    symbol: str
    side: Side
    entry_date: Date
    entry_price: float
    exit_date: Date
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    fees: float
    exit_reason: str


@dataclass
class PendingOrder:
    """Offene Order im Broker.

    Attributes:
        id: Laufende Nummer (Reihenfolge der Abgabe).
        order: Die Order der Strategie.
        submitted: Datum der Kerze, auf der die Order abgegeben wurde.
        stop: Aktueller Stop-Kurs (Stop und Trailing Stop), sonst ``None``.
        volatility: Volatilität des Werts in Prozent p. a. für ``VOLATILITY_TARGET``.
    """

    id: int
    order: Order
    submitted: Date
    stop: float | None = None
    volatility: float | None = None

    @property
    def symbol(self) -> str:
        """Wertpapier der Order."""
        return self.order.symbol


@dataclass
class _OpenPosition:
    symbol: str
    side: Side
    qty: float
    entry_price: float
    entry_date: Date
    fees: float = 0.0
    last_accrual: Date | None = None

    @property
    def sign(self) -> float:
        return 1.0 if self.side is Side.LONG else -1.0

    def to_position(self) -> Position:
        return Position(
            self.symbol, self.side, float(self.qty), float(self.entry_price), self.entry_date
        )


@dataclass
class _Ledger:
    fills: list[Fill] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)


_TYPE_RANK = {
    OrderType.MARKET: 0,
    OrderType.STOP: 1,
    OrderType.TRAILING_STOP: 1,
    OrderType.LIMIT: 2,
}


class SimBroker:
    """Simulierter Broker für Backtests (ein Konto, mehrere Werte).

    Beispiel::

        broker = SimBroker(1000.0, CostModel(fee_per_order=1.0))
        broker.submit(Order("AAA", OrderSide.BUY, qty=5), date=d0)
        fills = broker.process_bar("AAA", Bar(d1, 100.0, 105.0, 99.0, 104.0))
    """

    def __init__(
        self,
        capital: float = 1000.0,
        costs: CostModel | None = None,
        *,
        allow_fractional: bool = True,
    ) -> None:
        """Legt ein Konto mit Startkapital an.

        Args:
            capital: Startkapital in Kontowährung (> 0).
            costs: Kostenmodell; ``None`` = Standardkosten.
            allow_fractional: Bruchstücke erlaubt (4 Nachkommastellen), sonst ganze Stücke.

        Raises:
            ValueError: Startkapital nicht positiv.
        """
        if not math.isfinite(capital) or capital <= 0:
            raise ValueError(f"Startkapital muss größer als 0 sein, erhalten: {capital}.")
        self.initial_capital = float(capital)
        self.costs = costs if costs is not None else CostModel()
        self.allow_fractional = allow_fractional
        self._cash = float(capital)
        self._positions: dict[str, _OpenPosition] = {}
        self._pending: list[PendingOrder] = []
        self._last_price: dict[str, float] = {}
        self._ledger = _Ledger()
        self._next_id = 1

    @classmethod
    def from_settings(cls, settings: Settings, costs: CostModel | None = None) -> SimBroker:
        """Broker mit Startkapital, Bruchstück-Regel und Kosten aus den Einstellungen."""
        return cls(
            settings.default_capital,
            costs if costs is not None else CostModel.from_settings(settings),
            allow_fractional=settings.allow_fractional,
        )

    # -- Zustand ------------------------------------------------------------

    @property
    def cash(self) -> float:
        """Freies Kapital (inklusive Erlöse aus Short-Verkäufen)."""
        return self._cash

    @property
    def fills(self) -> list[Fill]:
        """Alle Ausführungen in zeitlicher Reihenfolge (Kopie)."""
        return list(self._ledger.fills)

    @property
    def rejections(self) -> list[Rejection]:
        """Alle verworfenen Orders mit Begründung (Kopie)."""
        return list(self._ledger.rejections)

    @property
    def trades(self) -> list[Trade]:
        """Alle abgeschlossenen Trades (Kopie)."""
        return list(self._ledger.trades)

    @property
    def positions(self) -> dict[str, Position]:
        """Offene Positionen je Symbol."""
        return {s: p.to_position() for s, p in self._positions.items()}

    def position(self, symbol: str) -> Position | None:
        """Offene Position im Symbol oder ``None``."""
        pos = self._positions.get(symbol)
        return pos.to_position() if pos is not None else None

    def open_orders(self, symbol: str | None = None) -> list[PendingOrder]:
        """Offene Orders, optional nur für ein Symbol, in Reihenfolge der Abgabe."""
        return [p for p in self._pending if symbol is None or p.symbol == symbol]

    def last_price(self, symbol: str) -> float | None:
        """Zuletzt bekannter Kurs des Symbols (Open bei Ausführung, sonst Close)."""
        return self._last_price.get(symbol)

    def _market_value(self, pos: _OpenPosition) -> float:
        return pos.qty * self._last_price.get(pos.symbol, pos.entry_price)

    def equity(self) -> float:
        """Gesamtkapital: Cash plus Long-Werte minus Short-Verbindlichkeiten."""
        return self._cash + sum(p.sign * self._market_value(p) for p in self._positions.values())

    def buying_power(self) -> float:
        """Für neue Positionen verfügbares Kapital.

        Cash abzüglich der doppelten Short-Werte (Erlös plus 100 % Sicherheitsleistung).
        """
        shorts = sum(
            self._market_value(p) for p in self._positions.values() if p.side is Side.SHORT
        )
        return self._cash - 2.0 * shorts

    # -- Orders abgeben -----------------------------------------------------

    def submit(
        self,
        order: Order,
        *,
        date: Date,
        close: float | None = None,
        atr: float | None = None,
        volatility: float | None = None,
    ) -> PendingOrder | None:
        """Nimmt eine Order zur Ausführung ab der nächsten Kerze an.

        Args:
            order: Order der Strategie.
            date: Datum der Kerze ``t``, auf der die Order abgegeben wird.
            close: Close von ``t``; nötig für Trailing Stops ohne ``stop_price``.
            atr: ATR bei ``t``; nötig für Trailing Stops mit ``trail_atr`` ohne ``stop_price``.
            volatility: Volatilität des Werts in Prozent p. a.; nötig für
                ``VOLATILITY_TARGET``.

        Returns:
            Die offene Order oder ``None``, wenn sie sofort verworfen wurde (siehe
            :attr:`rejections`).
        """
        pending = PendingOrder(self._next_id, order, date, volatility=volatility)
        self._next_id += 1
        if order.type is OrderType.STOP:
            pending.stop = order.stop_price
        elif order.type is OrderType.TRAILING_STOP:
            if order.stop_price is not None:
                pending.stop = order.stop_price
            else:
                start = self._trail_level(order, close, atr)
                if start is None:
                    self._reject(
                        pending,
                        date,
                        "Trailing Stop ohne Startwert: Close"
                        + (" und ATR" if order.trail_atr is not None else "")
                        + " fehlen.",
                    )
                    return None
                pending.stop = start
        self._pending.append(pending)
        return pending

    def cancel(self, symbol: str | None = None) -> int:
        """Streicht offene Orders (alle oder nur die eines Symbols).

        Returns:
            Anzahl gestrichener Orders.
        """
        before = len(self._pending)
        self._pending = [p for p in self._pending if symbol is not None and p.symbol != symbol]
        return before - len(self._pending)

    def force_close(self, symbol: str, day: Date, price: float, reason: str) -> Fill | None:
        """Schließt eine Position sofort zum angegebenen Kurs (z. B. Delisting).

        Ohne Slippage, mit der normalen Gebühr. Alle offenen Orders des Symbols werden
        gestrichen.

        Args:
            symbol: Wertpapier.
            day: Datum der Schließung.
            price: Ausführungskurs (z. B. letzter Close).
            reason: Text für ``exit_reason`` und ``tag``.

        Returns:
            Die Ausführung oder ``None``, wenn keine Position offen war.
        """
        self.cancel(symbol)
        pos = self._positions.get(symbol)
        if pos is None:
            return None
        side = OrderSide.SELL if pos.side is Side.LONG else OrderSide.BUY_TO_COVER
        pending = PendingOrder(self._next_id, Order(symbol, side, tag=reason), day)
        self._next_id += 1
        self._last_price[symbol] = price
        return self._close(pending, pos, day, price)

    # -- Kerze verarbeiten --------------------------------------------------

    def process_bar(self, symbol: str, bar: Bar, *, atr: float | None = None) -> list[Fill]:
        """Führt die offenen Orders eines Symbols auf einer Kerze aus.

        Reihenfolge: Leihgebühr seit der letzten Kerze buchen, Orders ausführen (Market,
        Stops, Limits), danach Kurs auf Close setzen und Trailing Stops nachziehen.

        Args:
            symbol: Wertpapier.
            bar: Kerze ``t+1``.
            atr: ATR zum Close dieser Kerze für Trailing Stops in ATR-Vielfachen.

        Returns:
            Ausführungen auf dieser Kerze.
        """
        self._accrue_borrow(symbol, bar.date)
        self._last_price[symbol] = bar.open
        fills: list[Fill] = []
        candidates = sorted(
            (p for p in self._pending if p.symbol == symbol),
            key=lambda p: (_TYPE_RANK[p.order.type], p.id),
        )
        for pending in candidates:
            if pending not in self._pending:
                continue  # durch geschlossene Position gestrichen
            fill = self._try_execute(pending, bar)
            if fill is not None:
                fills.append(fill)
        self._last_price[symbol] = bar.close
        self._update_trailing(symbol, bar.close, atr)
        return fills

    def _accrue_borrow(self, symbol: str, day: Date) -> None:
        pos = self._positions.get(symbol)
        if pos is None or pos.side is not Side.SHORT:
            return
        if pos.last_accrual is not None:
            fee = self.costs.borrow_fee(self._market_value(pos), (day - pos.last_accrual).days)
            self._cash -= fee
            pos.fees += fee
        pos.last_accrual = day

    def _try_execute(self, pending: PendingOrder, bar: Bar) -> Fill | None:
        order = pending.order
        pos = self._positions.get(order.symbol)
        if not order.side.is_entry and (pos is None or pos.side is not order.side.position_side):
            if order.type is OrderType.MARKET:
                self._remove(pending)
                self._reject(pending, bar.date, "Keine passende Position zum Schließen offen.")
            return None  # Ausstiegs-Order wartet auf eine Position
        raw = self._trigger_price(pending, bar)
        if raw is None:
            return None
        self._remove(pending)
        price = raw if order.type is OrderType.LIMIT else self.costs.fill_price(raw, order.side)
        if order.side.is_entry:
            return self._open(pending, bar.date, price)
        assert pos is not None
        return self._close(pending, pos, bar.date, price)

    def _trigger_price(self, pending: PendingOrder, bar: Bar) -> float | None:
        order = pending.order
        buy = is_buy(order.side)
        if order.type is OrderType.MARKET:
            return bar.open
        if order.type is OrderType.LIMIT:
            limit = order.limit_price
            assert limit is not None
            if buy:
                return min(limit, bar.open) if bar.low <= limit else None
            return max(limit, bar.open) if bar.high >= limit else None
        stop = pending.stop
        assert stop is not None
        if buy:
            return max(stop, bar.open) if bar.high >= stop else None
        return min(stop, bar.open) if bar.low <= stop else None

    # -- Eröffnen und Schließen ---------------------------------------------

    def _open(self, pending: PendingOrder, day: Date, price: float) -> Fill | None:
        order = pending.order
        side = order.side.position_side
        pos = self._positions.get(order.symbol)
        if pos is not None and pos.side is not side:
            self._reject(pending, day, "Gegenposition im selben Wert ist offen.")
            return None
        if order.qty is not None:
            qty = round_qty(order.qty, allow_fractional=self.allow_fractional)
        else:
            sized = self._size(pending, price)
            if isinstance(sized, str):
                self._reject(pending, day, sized)
                return None
            qty = sized
        if qty <= 0:
            self._reject(pending, day, "Stückzahl nach dem Abrunden ist 0.")
            return None
        notional = qty * price
        fee = self.costs.fee(notional)
        available = self.buying_power()
        if notional + fee > available + _EPS:
            self._reject(
                pending,
                day,
                f"Kapital reicht nicht: benötigt {notional + fee:.2f}, verfügbar {available:.2f}.",
            )
            return None
        self._cash += -(notional + fee) if side is Side.LONG else notional - fee
        if pos is None:
            self._positions[order.symbol] = _OpenPosition(
                order.symbol, side, qty, price, day, fee, day if side is Side.SHORT else None
            )
        else:
            total = pos.qty + qty
            pos.entry_price = (pos.entry_price * pos.qty + price * qty) / total
            pos.qty = total
            pos.fees += fee
        return self._record(pending, day, qty, price, fee)

    def _close(self, pending: PendingOrder, pos: _OpenPosition, day: Date, price: float) -> Fill:
        order = pending.order
        qty = pos.qty if order.qty is None else min(order.qty, pos.qty)
        notional = qty * price
        fee = self.costs.fee(notional)
        self._cash += notional - fee if pos.side is Side.LONG else -(notional + fee)
        share = qty / pos.qty
        entry_fees = pos.fees * share
        gross = pos.sign * (price - pos.entry_price) * qty
        fees = entry_fees + fee
        pnl = gross - fees
        entry_value = pos.entry_price * qty
        self._ledger.trades.append(
            Trade(
                symbol=pos.symbol,
                side=pos.side,
                entry_date=pos.entry_date,
                entry_price=pos.entry_price,
                exit_date=day,
                exit_price=price,
                qty=qty,
                pnl=pnl,
                pnl_pct=pnl / entry_value * 100.0 if entry_value > 0 else 0.0,
                fees=fees,
                exit_reason=order.tag or order.type.value,
            )
        )
        remaining = pos.qty - qty
        if remaining <= _EPS:
            del self._positions[pos.symbol]
            self._drop_exit_orders(pos.symbol, pos.side)
        else:
            pos.qty = remaining
            pos.fees -= entry_fees
        return self._record(pending, day, qty, price, fee)

    def _size(self, pending: PendingOrder, price: float) -> float | str:
        sizing = pending.order.sizing
        assert sizing is not None
        equity = self.equity()
        method = sizing.method
        if method is SizingMethod.FIXED_AMOUNT:
            raw = self.costs.max_qty_for_budget(sizing.value, price)
        elif method is SizingMethod.PERCENT_EQUITY:
            raw = self.costs.max_qty_for_budget(equity * sizing.value / 100.0, price)
        elif method is SizingMethod.RISK:
            stop = self._protective_stop(pending.order)
            if stop is None:
                return "Risiko je Trade braucht einen offenen Schutz-Stop."
            distance = price - stop if pending.order.side is OrderSide.BUY else stop - price
            if distance <= 0:
                return f"Schutz-Stop {stop:.4f} liegt nicht auf der Verlustseite des Einstiegs."
            raw = equity * sizing.value / 100.0 / distance
        else:
            vol = pending.volatility
            if vol is None or not math.isfinite(vol) or vol <= 0:
                return "Volatilitäts-Ziel braucht die Volatilität des Werts."
            raw = equity * sizing.value / vol / price
        return round_qty(raw, allow_fractional=self.allow_fractional)

    def _protective_stop(self, entry: Order) -> float | None:
        exit_side = OrderSide.SELL if entry.side is OrderSide.BUY else OrderSide.BUY_TO_COVER
        for p in self._pending:
            if p.symbol == entry.symbol and p.order.side is exit_side and p.stop is not None:
                return p.stop
        return None

    # -- Hilfen -------------------------------------------------------------

    @staticmethod
    def _trail_level(order: Order, close: float | None, atr: float | None) -> float | None:
        if close is None or not math.isfinite(close):
            return None
        buy = is_buy(order.side)
        if order.trail_pct is not None:
            factor = order.trail_pct / 100.0
            return close * (1.0 + factor) if buy else close * (1.0 - factor)
        assert order.trail_atr is not None
        if atr is None or not math.isfinite(atr) or atr <= 0:
            return None
        offset = order.trail_atr * atr
        return close + offset if buy else close - offset

    def _update_trailing(self, symbol: str, close: float, atr: float | None) -> None:
        for p in self._pending:
            if p.symbol != symbol or p.order.type is not OrderType.TRAILING_STOP:
                continue
            level = self._trail_level(p.order, close, atr)
            if level is None or p.stop is None:
                continue
            p.stop = min(p.stop, level) if is_buy(p.order.side) else max(p.stop, level)

    def _drop_exit_orders(self, symbol: str, side: Side) -> None:
        self._pending = [
            p
            for p in self._pending
            if not (
                p.symbol == symbol
                and not p.order.side.is_entry
                and p.order.side.position_side is side
            )
        ]

    def _remove(self, pending: PendingOrder) -> None:
        self._pending.remove(pending)

    def _record(
        self, pending: PendingOrder, day: Date, qty: float, price: float, fee: float
    ) -> Fill:
        o = pending.order
        fill = Fill(pending.id, day, o.symbol, o.side, o.type, qty, price, fee, o.tag)
        self._ledger.fills.append(fill)
        return fill

    def _reject(self, pending: PendingOrder, day: Date, reason: str) -> None:
        order = pending.order
        self._ledger.rejections.append(Rejection(pending.id, day, order, reason))
        logger.info(
            "Order %d verworfen (%s %s %s, %s): %s",
            pending.id,
            day.isoformat(),
            order.symbol,
            order.side.value,
            order.type.value,
            reason,
        )
