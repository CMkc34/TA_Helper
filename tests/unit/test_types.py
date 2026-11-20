"""Kern-Datentypen."""

from __future__ import annotations

from enum import StrEnum

import pytest

from trading_helfer.core.types import OrderType, Side, SignalType, Timeframe


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (Timeframe.D1, "1D"),
        (Timeframe.D3, "3D"),
        (Timeframe.W1, "1W"),
        (Timeframe.MN1, "1M"),
    ],
)
def test_timeframe_values(member: Timeframe, value: str) -> None:
    assert member.value == value
    assert str(member) == value
    assert Timeframe(value) is member


def test_timeframe_order_and_completeness() -> None:
    assert list(Timeframe) == [Timeframe.D1, Timeframe.D3, Timeframe.W1, Timeframe.MN1]


def test_side_values() -> None:
    assert [s.value for s in Side] == ["LONG", "SHORT"]


def test_order_type_values() -> None:
    assert [o.value for o in OrderType] == ["MARKET", "STOP", "LIMIT", "TRAILING_STOP"]


def test_signal_type_values() -> None:
    assert [s.value for s in SignalType] == ["BUY", "SELL", "EXIT"]


@pytest.mark.parametrize("enum_cls", [Timeframe, Side, OrderType, SignalType])
def test_all_are_str_enums(enum_cls: type[StrEnum]) -> None:
    assert issubclass(enum_cls, StrEnum)
    for member in enum_cls:
        assert isinstance(member, str)
        assert member == member.value


def test_unknown_value_raises() -> None:
    with pytest.raises(ValueError, match="2W"):
        Timeframe("2W")
