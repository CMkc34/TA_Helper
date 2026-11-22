"""Simulierter Broker: Order-Typen mit handgerechneten Fällen, Kapital, Sizing, Kosten."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pytest

from trading_helfer.backtest.broker_sim import Bar, SimBroker, round_qty
from trading_helfer.backtest.costs import CostModel
from trading_helfer.core.config import Settings
from trading_helfer.core.types import OrderType, Side
from trading_helfer.strategies.base import Order, OrderSide, Sizing

D0 = date(2024, 1, 5)
SYM = "AAA"
NO_COSTS = CostModel(fee_per_order=0.0, fee_pct=0.0, slippage_pct=0.0)


def _day(n: int) -> date:
    return D0 + timedelta(days=n)


def _bar(n: int, o: float, h: float, low: float, c: float) -> Bar:
    return Bar(_day(n), o, h, low, c)


def _broker(costs: CostModel = NO_COSTS, capital: float = 1000.0, **kw: bool) -> SimBroker:
    return SimBroker(capital, costs, **kw)


def _long(broker: SimBroker, qty: float = 10.0, price: float = 100.0) -> None:
    broker.submit(Order(SYM, OrderSide.BUY, qty=qty), date=_day(0))
    broker.process_bar(SYM, _bar(1, price, price, price, price))
    assert broker.position(SYM) is not None


# -- Market ---------------------------------------------------------------


def test_market_round_trip_with_costs() -> None:
    broker = _broker(CostModel(fee_per_order=1.0, slippage_pct=0.5))
    broker.submit(Order(SYM, OrderSide.BUY, qty=5), date=_day(0))
    (fill,) = broker.process_bar(SYM, _bar(1, 100, 104, 99, 103))
    assert fill.price == pytest.approx(100.5)
    assert fill.fee == pytest.approx(1.0)
    assert broker.cash == pytest.approx(1000 - 502.5 - 1)
    assert broker.equity() == pytest.approx(496.5 + 5 * 103)

    broker.submit(Order(SYM, OrderSide.SELL), date=_day(1))
    (fill,) = broker.process_bar(SYM, _bar(2, 110, 111, 108, 109))
    assert fill.price == pytest.approx(109.45)
    assert broker.cash == pytest.approx(1042.75)
    assert broker.position(SYM) is None
    (trade,) = broker.trades
    assert trade.side is Side.LONG
    assert trade.entry_price == pytest.approx(100.5)
    assert trade.exit_price == pytest.approx(109.45)
    assert trade.fees == pytest.approx(2.0)
    assert trade.pnl == pytest.approx(42.75)
    assert trade.pnl_pct == pytest.approx(42.75 / 502.5 * 100)
    assert trade.exit_reason == "MARKET"
    assert broker.open_orders() == []


def test_market_order_only_valid_for_next_bar_and_market_exit_without_position() -> None:
    broker = _broker()
    broker.submit(Order(SYM, OrderSide.SELL, tag="Ausstieg"), date=_day(0))
    assert broker.process_bar(SYM, _bar(1, 100, 100, 100, 100)) == []
    (rej,) = broker.rejections
    assert "Keine passende Position" in rej.reason
    assert broker.open_orders() == []


# -- Stop -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("bar", "expected"),
    [
        (_bar(1, 102, 106, 101, 105), 105.0),  # Stop intraday erreicht
        (_bar(1, 108, 110, 107, 109), 108.0),  # Gap über den Stop: Open
        (_bar(1, 102, 104, 101, 103), None),  # nicht erreicht
    ],
)
def test_buy_stop(bar: Bar, expected: float | None) -> None:
    broker = _broker()
    broker.submit(Order(SYM, OrderSide.BUY, OrderType.STOP, qty=1, stop_price=105.0), date=_day(0))
    fills = broker.process_bar(SYM, bar)
    if expected is None:
        assert fills == []
        assert len(broker.open_orders(SYM)) == 1
    else:
        assert fills[0].price == pytest.approx(expected)
        assert broker.open_orders() == []


@pytest.mark.parametrize(
    ("bar", "expected"),
    [
        (_bar(2, 98, 99, 94, 96), 95.0),
        (_bar(2, 90, 91, 88, 89), 90.0),  # Gap unter den Stop: Open
    ],
)
def test_sell_stop(bar: Bar, expected: float) -> None:
    broker = _broker()
    _long(broker)
    broker.submit(Order(SYM, OrderSide.SELL, OrderType.STOP, stop_price=95.0), date=_day(1))
    (fill,) = broker.process_bar(SYM, bar)
    assert fill.price == pytest.approx(expected)
    assert broker.trades[0].pnl == pytest.approx((expected - 100) * 10)


def test_stop_with_slippage() -> None:
    broker = _broker(CostModel(fee_per_order=0.0, slippage_pct=1.0))
    broker.submit(Order(SYM, OrderSide.BUY, OrderType.STOP, qty=1, stop_price=105.0), date=_day(0))
    (fill,) = broker.process_bar(SYM, _bar(1, 102, 106, 101, 105))
    assert fill.price == pytest.approx(106.05)


# -- Limit ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("bar", "expected"),
    [
        (_bar(1, 97, 98, 94, 96), 95.0),
        (_bar(1, 93, 94, 92, 93), 93.0),  # Gap unter das Limit: Open (besser)
        (_bar(1, 97, 98, 96, 97), None),
    ],
)
def test_buy_limit(bar: Bar, expected: float | None) -> None:
    broker = _broker(CostModel(fee_per_order=0.0, slippage_pct=1.0))  # Limit ohne Slippage
    broker.submit(Order(SYM, OrderSide.BUY, OrderType.LIMIT, qty=1, limit_price=95.0), date=_day(0))
    fills = broker.process_bar(SYM, bar)
    if expected is None:
        assert fills == []
    else:
        assert fills[0].price == pytest.approx(expected)


@pytest.mark.parametrize(
    ("bar", "expected"),
    [
        (_bar(2, 108, 111, 107, 109), 110.0),
        (_bar(2, 112, 113, 111, 112), 112.0),  # Gap über das Ziel: Open
    ],
)
def test_sell_limit(bar: Bar, expected: float) -> None:
    broker = _broker()
    _long(broker)
    broker.submit(
        Order(SYM, OrderSide.SELL, OrderType.LIMIT, limit_price=110.0, tag="Ziel"), date=_day(1)
    )
    (fill,) = broker.process_bar(SYM, bar)
    assert fill.price == pytest.approx(expected)
    assert broker.trades[0].exit_reason == "Ziel"


def test_stop_and_target_same_bar_stop_first() -> None:
    broker = _broker()
    _long(broker)
    broker.submit(
        Order(SYM, OrderSide.SELL, OrderType.LIMIT, limit_price=110.0, tag="Ziel"), date=_day(1)
    )
    broker.submit(
        Order(SYM, OrderSide.SELL, OrderType.STOP, stop_price=95.0, tag="Stop"), date=_day(1)
    )
    fills = broker.process_bar(SYM, _bar(2, 100, 111, 94, 105))
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(95.0)
    assert broker.trades[0].exit_reason == "Stop"
    assert broker.open_orders() == []  # Ziel gestrichen
    assert broker.position(SYM) is None


def test_exit_stop_waits_for_position() -> None:
    broker = _broker()
    broker.submit(Order(SYM, OrderSide.SELL, OrderType.STOP, stop_price=95.0), date=_day(0))
    broker.submit(Order(SYM, OrderSide.BUY, qty=1), date=_day(0))
    fills = broker.process_bar(SYM, _bar(1, 100, 101, 99, 100))
    assert [f.side for f in fills] == [OrderSide.BUY]
    assert len(broker.open_orders(SYM)) == 1
    (fill,) = broker.process_bar(SYM, _bar(2, 96, 97, 93, 94))
    assert fill.price == pytest.approx(95.0)


# -- Trailing Stop --------------------------------------------------------


def test_trailing_stop_pct_long() -> None:
    broker = _broker()
    _long(broker)
    pending = broker.submit(
        Order(SYM, OrderSide.SELL, OrderType.TRAILING_STOP, trail_pct=10.0), date=_day(1), close=100
    )
    assert pending is not None
    assert pending.stop == pytest.approx(90.0)
    assert broker.process_bar(SYM, _bar(2, 101, 121, 100, 120)) == []
    assert pending.stop == pytest.approx(108.0)
    assert broker.process_bar(SYM, _bar(3, 118, 119, 109, 110)) == []
    assert pending.stop == pytest.approx(108.0)  # zieht nur nach oben nach
    (fill,) = broker.process_bar(SYM, _bar(4, 109, 110, 107, 108))
    assert fill.price == pytest.approx(108.0)
    assert fill.type is OrderType.TRAILING_STOP


def test_trailing_stop_atr_and_explicit_start() -> None:
    broker = _broker()
    _long(broker)
    pending = broker.submit(
        Order(SYM, OrderSide.SELL, OrderType.TRAILING_STOP, trail_atr=3.0),
        date=_day(1),
        close=100,
        atr=2.0,
    )
    assert pending is not None
    assert pending.stop == pytest.approx(94.0)
    broker.process_bar(SYM, _bar(2, 101, 111, 100, 110), atr=2.0)
    assert pending.stop == pytest.approx(104.0)
    broker.process_bar(SYM, _bar(3, 110, 112, 109, 111))  # ohne ATR: unverändert
    assert pending.stop == pytest.approx(104.0)

    other = broker.submit(
        Order(SYM, OrderSide.SELL, OrderType.TRAILING_STOP, stop_price=80.0, trail_pct=5.0),
        date=_day(3),
    )
    assert other is not None
    assert other.stop == pytest.approx(80.0)


def test_trailing_stop_short_moves_down() -> None:
    broker = _broker()
    broker.submit(Order(SYM, OrderSide.SELL_SHORT, qty=1), date=_day(0))
    broker.process_bar(SYM, _bar(1, 100, 100, 100, 100))
    pending = broker.submit(
        Order(SYM, OrderSide.BUY_TO_COVER, OrderType.TRAILING_STOP, trail_pct=10.0),
        date=_day(1),
        close=100,
    )
    assert pending is not None
    assert pending.stop == pytest.approx(110.0)
    broker.process_bar(SYM, _bar(2, 95, 96, 79, 80))
    assert pending.stop == pytest.approx(88.0)
    broker.process_bar(SYM, _bar(3, 81, 90, 80, 85))
    assert broker.position(SYM) is None
    assert broker.trades[0].exit_price == pytest.approx(88.0)


@pytest.mark.parametrize(
    ("order", "close", "atr"),
    [
        (Order(SYM, OrderSide.SELL, OrderType.TRAILING_STOP, trail_pct=5.0), None, None),
        (Order(SYM, OrderSide.SELL, OrderType.TRAILING_STOP, trail_atr=2.0), 100.0, None),
    ],
)
def test_trailing_without_start_rejected(order: Order, close: float | None, atr: None) -> None:
    broker = _broker()
    assert broker.submit(order, date=_day(0), close=close, atr=atr) is None
    assert "Trailing Stop ohne Startwert" in broker.rejections[0].reason


# -- Kapital --------------------------------------------------------------


def test_no_buy_without_capital(caplog: pytest.LogCaptureFixture) -> None:
    broker = _broker(CostModel(fee_per_order=1.0, slippage_pct=0.0))
    broker.submit(Order(SYM, OrderSide.BUY, qty=10), date=_day(0))
    with caplog.at_level(logging.INFO, logger="trading_helfer.backtest.broker_sim"):
        assert broker.process_bar(SYM, _bar(1, 100, 101, 99, 100)) == []
    assert broker.cash == 1000.0
    assert broker.position(SYM) is None
    (rej,) = broker.rejections
    assert "Kapital reicht nicht" in rej.reason
    assert "verworfen" in caplog.text


def test_short_collateral_and_borrow_fee() -> None:
    broker = _broker(CostModel(fee_per_order=1.0, slippage_pct=0.0, borrow_fee_pct=36.5))
    broker.submit(Order(SYM, OrderSide.SELL_SHORT, qty=5), date=_day(0))
    broker.process_bar(SYM, _bar(1, 100, 100, 100, 100))
    assert broker.cash == pytest.approx(1499.0)
    assert broker.buying_power() == pytest.approx(498.0)  # 1 USD Gebührenreserve
    assert broker.equity() == pytest.approx(999.0)
    pos = broker.position(SYM)
    assert pos is not None
    assert pos.side is Side.SHORT
    broker.submit(Order(SYM, OrderSide.BUY_TO_COVER), date=_day(1))
    broker.process_bar(SYM, _bar(11, 90, 90, 90, 90))  # 10 Tage Leihgebühr = 5,00
    assert broker.cash == pytest.approx(1499.0 - 5.0 - 451.0)
    (trade,) = broker.trades
    assert trade.pnl == pytest.approx(50.0 - 1.0 - 5.0 - 1.0)
    assert trade.fees == pytest.approx(7.0)


def test_opposite_position_rejected() -> None:
    broker = _broker()
    _long(broker, qty=1)
    broker.submit(Order(SYM, OrderSide.SELL_SHORT, qty=1), date=_day(1))
    broker.process_bar(SYM, _bar(2, 100, 100, 100, 100))
    assert "Gegenposition" in broker.rejections[0].reason


def test_add_on_and_partial_exit() -> None:
    broker = _broker(CostModel(fee_per_order=1.0, slippage_pct=0.0))
    _long(broker, qty=2, price=100)
    broker.submit(Order(SYM, OrderSide.BUY, qty=2), date=_day(1))
    broker.process_bar(SYM, _bar(2, 110, 110, 110, 110))
    pos = broker.position(SYM)
    assert pos is not None
    assert pos.qty == pytest.approx(4)
    assert pos.entry_price == pytest.approx(105.0)
    broker.submit(Order(SYM, OrderSide.SELL, qty=1), date=_day(2))
    broker.process_bar(SYM, _bar(3, 120, 120, 120, 120))
    trade = broker.trades[0]
    assert trade.qty == pytest.approx(1)
    assert trade.pnl == pytest.approx(15.0 - 0.5 - 1.0)
    broker.submit(Order(SYM, OrderSide.SELL, qty=10), date=_day(3))  # mehr als offen
    broker.process_bar(SYM, _bar(4, 120, 120, 120, 120))
    assert broker.trades[1].qty == pytest.approx(3)
    assert broker.trades[1].fees == pytest.approx(1.5 + 1.0)
    assert broker.positions == {}


# -- Sizing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("sizing", "expected"),
    [
        (Sizing.fixed_amount(500), 4.99),
        (Sizing.percent_equity(100), 9.98),  # 1 USD bleibt als Reserve für den Ausstieg
        (Sizing.percent_equity(50), 4.99),
    ],
)
def test_sizing_budget_includes_fee(sizing: Sizing, expected: float) -> None:
    broker = _broker(CostModel(fee_per_order=1.0, slippage_pct=0.0))
    broker.submit(Order(SYM, OrderSide.BUY, sizing=sizing), date=_day(0))
    (fill,) = broker.process_bar(SYM, _bar(1, 100, 100, 100, 100))
    assert fill.qty == pytest.approx(expected)
    assert broker.cash >= 0


def test_exit_fee_reserve_keeps_cash_non_negative() -> None:
    broker = _broker(CostModel(fee_per_order=2.0, slippage_pct=0.0))
    broker.submit(Order(SYM, OrderSide.BUY, sizing=Sizing.percent_equity(100)), date=_day(0))
    broker.process_bar(SYM, _bar(1, 100, 100, 100, 100))
    assert broker.exit_fee_reserve() == pytest.approx(2.0)
    assert broker.cash >= 2.0 - 1e-9
    broker.submit(Order(SYM, OrderSide.SELL), date=_day(1))
    broker.process_bar(SYM, _bar(2, 0.01, 0.01, 0.01, 0.01))  # Wert fast null
    assert broker.positions == {}
    assert broker.cash >= 0
    assert broker.equity() >= 0


def test_explicit_qty_rejected_when_only_reserve_missing() -> None:
    broker = _broker(CostModel(fee_per_order=1.0, slippage_pct=0.0))
    broker.submit(Order(SYM, OrderSide.BUY, qty=9.99), date=_day(0))
    assert broker.process_bar(SYM, _bar(1, 100, 100, 100, 100)) == []
    assert "Gebührenreserve" in broker.rejections[0].reason


def test_add_on_needs_no_new_reserve() -> None:
    broker = _broker(CostModel(fee_per_order=1.0, slippage_pct=0.0))
    _long(broker, qty=5, price=100)  # Cash 499, Reserve 1
    broker.submit(Order(SYM, OrderSide.BUY, qty=4.97), date=_day(1))
    (fill,) = broker.process_bar(SYM, _bar(2, 100, 100, 100, 100))
    assert fill.qty == pytest.approx(4.97)
    assert broker.cash == pytest.approx(1.0)


def test_sizing_whole_shares() -> None:
    broker = _broker(CostModel(fee_per_order=1.0, slippage_pct=0.0), allow_fractional=False)
    broker.submit(Order(SYM, OrderSide.BUY, sizing=Sizing.percent_equity(100)), date=_day(0))
    (fill,) = broker.process_bar(SYM, _bar(1, 100, 100, 100, 100))
    assert fill.qty == 9.0


def test_sizing_risk_per_trade() -> None:
    broker = _broker()
    broker.submit(Order(SYM, OrderSide.SELL, OrderType.STOP, stop_price=95.0), date=_day(0))
    broker.submit(Order(SYM, OrderSide.BUY, sizing=Sizing.risk(1)), date=_day(0))
    (fill,) = broker.process_bar(SYM, _bar(1, 100, 100, 100, 100))
    assert fill.qty == pytest.approx(2.0)  # 10 USD Risiko / 5 USD Abstand


def test_sizing_risk_short() -> None:
    broker = _broker()
    broker.submit(
        Order(SYM, OrderSide.BUY_TO_COVER, OrderType.STOP, stop_price=104.0), date=_day(0)
    )
    broker.submit(Order(SYM, OrderSide.SELL_SHORT, sizing=Sizing.risk(2)), date=_day(0))
    (fill,) = broker.process_bar(SYM, _bar(1, 100, 100, 100, 100))
    assert fill.qty == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("stop", "reason"),
    [(None, "Schutz-Stop"), (101.0, "Verlustseite")],
)
def test_sizing_risk_needs_valid_stop(stop: float | None, reason: str) -> None:
    broker = _broker()
    if stop is not None:
        broker.submit(Order(SYM, OrderSide.SELL, OrderType.STOP, stop_price=stop), date=_day(0))
    broker.submit(Order(SYM, OrderSide.BUY, sizing=Sizing.risk(1)), date=_day(0))
    broker.process_bar(SYM, _bar(1, 100, 100, 100, 100))
    assert reason in broker.rejections[0].reason


def test_sizing_volatility_target() -> None:
    broker = _broker()
    broker.submit(
        Order(SYM, OrderSide.BUY, sizing=Sizing.volatility_target(10)),
        date=_day(0),
        volatility=20.0,
    )
    (fill,) = broker.process_bar(SYM, _bar(1, 100, 100, 100, 100))
    assert fill.qty == pytest.approx(5.0)
    broker.submit(Order("BBB", OrderSide.BUY, sizing=Sizing.volatility_target(10)), date=_day(1))
    broker.process_bar("BBB", _bar(2, 50, 50, 50, 50))
    assert "Volatilität" in broker.rejections[0].reason


def test_qty_rounds_to_zero_rejected() -> None:
    broker = _broker()
    broker.submit(Order(SYM, OrderSide.BUY, qty=0.00001), date=_day(0))
    broker.process_bar(SYM, _bar(1, 100, 100, 100, 100))
    assert "Stückzahl" in broker.rejections[0].reason


@pytest.mark.parametrize(
    ("qty", "fractional", "expected"),
    [
        (1.23456789, True, 1.2345),
        (0.1 + 0.2, True, 0.3),
        (9.99, False, 9.0),
        (-1.0, True, 0.0),
        (float("nan"), True, 0.0),
    ],
)
def test_round_qty(qty: float, fractional: bool, expected: float) -> None:
    assert round_qty(qty, allow_fractional=fractional) == pytest.approx(expected)


# -- Verwaltung -----------------------------------------------------------


def test_cancel_and_open_orders() -> None:
    broker = _broker()
    broker.submit(Order(SYM, OrderSide.BUY, OrderType.LIMIT, qty=1, limit_price=90), date=D0)
    broker.submit(Order("BBB", OrderSide.BUY, OrderType.LIMIT, qty=1, limit_price=90), date=D0)
    assert [p.symbol for p in broker.open_orders()] == [SYM, "BBB"]
    assert broker.cancel(SYM) == 1
    assert broker.cancel() == 1
    assert broker.open_orders() == []


def test_from_settings_and_state() -> None:
    broker = SimBroker.from_settings(Settings(default_capital=5000, allow_fractional=False))
    assert broker.cash == 5000.0
    assert broker.allow_fractional is False
    assert broker.costs.fee_per_order == 1.0
    assert broker.last_price(SYM) is None
    assert broker.fills == []
    assert SimBroker().costs == CostModel()


def test_invalid_capital() -> None:
    with pytest.raises(ValueError, match="Startkapital"):
        SimBroker(0.0)


def test_deterministic() -> None:
    def run() -> list[object]:
        broker = _broker(CostModel(fee_per_order=1.0, slippage_pct=0.1))
        broker.submit(Order(SYM, OrderSide.BUY, sizing=Sizing.percent_equity(50)), date=_day(0))
        broker.process_bar(SYM, _bar(1, 100, 105, 99, 104))
        broker.submit(Order(SYM, OrderSide.SELL), date=_day(1))
        broker.process_bar(SYM, _bar(2, 106, 107, 103, 104))
        return [*broker.fills, *broker.trades, broker.cash]

    assert run() == run()
