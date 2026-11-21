"""Kostenmodell: Gebühren, Slippage, Leihgebühr und Budget-Rechnung."""

from __future__ import annotations

import pytest

from trading_helfer.backtest.costs import CostModel, is_buy
from trading_helfer.core.config import Settings
from trading_helfer.strategies.base import OrderSide


def test_fee_fixed_plus_percent() -> None:
    costs = CostModel(fee_per_order=1.0, fee_pct=0.1)
    assert costs.fee(1000.0) == pytest.approx(2.0)
    assert costs.fee(-1000.0) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        (OrderSide.BUY, 100.5),
        (OrderSide.BUY_TO_COVER, 100.5),
        (OrderSide.SELL, 99.5),
        (OrderSide.SELL_SHORT, 99.5),
    ],
)
def test_slippage_against_trader(side: OrderSide, expected: float) -> None:
    assert CostModel(slippage_pct=0.5).fill_price(100.0, side) == pytest.approx(expected)


def test_is_buy() -> None:
    assert is_buy(OrderSide.BUY)
    assert not is_buy(OrderSide.SELL)


def test_borrow_fee_pro_rata_per_day() -> None:
    costs = CostModel(borrow_fee_pct=5.0)
    assert costs.borrow_fee(10_000.0, 73) == pytest.approx(100.0)
    assert costs.borrow_fee(10_000.0, 0) == 0.0


def test_max_qty_for_budget() -> None:
    costs = CostModel(fee_per_order=1.0, fee_pct=0.0)
    assert costs.max_qty_for_budget(1001.0, 100.0) == pytest.approx(10.0)
    assert costs.max_qty_for_budget(0.5, 100.0) == 0.0
    assert costs.max_qty_for_budget(100.0, 0.0) == 0.0
    pct = CostModel(fee_per_order=0.0, fee_pct=1.0)
    assert pct.max_qty_for_budget(101.0, 10.0) == pytest.approx(10.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fee_per_order": -1.0},
        {"fee_pct": -0.1},
        {"slippage_pct": float("nan")},
        {"borrow_fee_pct": -2.0},
        {"slippage_pct": 100.0},
    ],
)
def test_invalid_costs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"slippage_pct|darf nicht negativ"):
        CostModel(**kwargs)


def test_from_settings() -> None:
    settings = Settings(fee_per_order=2.0, slippage_pct=0.1)
    costs = CostModel.from_settings(settings, fee_pct=0.2, borrow_fee_pct=3.0)
    assert costs == CostModel(2.0, 0.2, 0.1, 3.0)
