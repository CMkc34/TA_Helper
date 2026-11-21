"""Strategie-Basisklasse: Parameter, Orders, Schreibschutz und Hilfsfunktionen."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trading_helfer.core.types import OrderType, Side, Timeframe
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    OrderError,
    OrderSide,
    Param,
    ParamError,
    Params,
    PlotSpec,
    Position,
    PreparedSeries,
    Sizing,
    SizingMethod,
    Strategy,
)


def _bars(close: list[float]) -> pd.DataFrame:
    n = len(close)
    c = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": c - 0.5,
            "high": c + 1.0,
            "low": c - 1.0,
            "close": c,
            "volume": np.arange(1, n + 1, dtype="int64") * 100,
        }
    )


class SmaCross(Strategy):
    id = "test_sma_cross"
    name = "SMA-Kreuzung (Test)"
    version = "1.0.0"
    description = "Testet die Basisklasse."
    timeframes = [Timeframe.D3, Timeframe.W1]  # noqa: RUF012
    params = {  # noqa: RUF012
        "fast": Param(2, 1, 10, 1, "schnell"),
        "slow": Param(4, 2, 50, 1, "langsam"),
        "risk": Param(1.0, 0.1, 5.0, 0.1, "Risiko in %"),
        "short": Param(False, description="Short erlaubt"),
        "mode": Param("close", description="Quelle", choices=("close", "open")),
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        src = bars[p.mode]
        return pd.DataFrame(
            {"fast": src.rolling(p.fast).mean(), "slow": src.rolling(p.slow).mean()},
            index=bars.index,
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and ctx.is_member and ctx.crossed_above("fast", "slow"):
            return [ctx.buy(sizing=Sizing.percent_equity(100), tag="Kreuzung")]
        if ctx.position is not None and ctx.crossed_below("fast", "slow"):
            return ctx.exit(tag="Gegenkreuzung")
        return []

    def plot(self) -> list[PlotSpec]:
        return [PlotSpec("fast"), PlotSpec("slow", color="#ff0000")]


def _ctx(
    close: list[float],
    t: int,
    ind: pd.DataFrame | None = None,
    position: Position | None = None,
) -> BarContext:
    series = PreparedSeries(_bars(close), ind)
    return BarContext.at(
        series, t, symbol="TEST", timeframe=Timeframe.W1, position=position, cash=1000.0
    )


# -- Param / Params -----------------------------------------------------------


def test_param_kinds() -> None:
    assert Param(3, 1, 5).kind == "int"
    assert Param(1.5, 0.0, 2.0).kind == "float"
    assert Param(True).kind == "bool"
    assert Param("a", choices=("a", "b")).kind == "choice"
    with pytest.raises(ParamError):
        _ = Param("text").kind
    with pytest.raises(ParamError):
        Param("text")  # Typ nicht ableitbar


@pytest.mark.parametrize(
    "kwargs",
    [
        {"default": 1, "min": 5, "max": 2},
        {"default": 1, "min": 0, "max": 2, "step": 0},
        {"default": 10, "min": 0, "max": 5},
        {"default": True, "min": 0},
        {"default": "x", "choices": ()},
        {"default": "x", "choices": ("a", "b")},
    ],
)
def test_param_definition_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ParamError):
        Param(**kwargs)  # type: ignore[arg-type]


def test_params_defaults_and_access() -> None:
    p = SmaCross.make_params()
    assert p.fast == 2
    assert p["slow"] == 4
    assert p.risk == 1.0
    assert p.short is False
    assert p.mode == "close"
    assert len(p) == 5
    assert list(p) == ["fast", "slow", "risk", "short", "mode"]
    assert p.to_dict() == {"fast": 2, "slow": 4, "risk": 1.0, "short": False, "mode": "close"}
    assert p == SmaCross.make_params({})
    assert p == p.to_dict()
    assert p != 3
    assert hash(p) == hash(SmaCross.make_params())
    assert "fast=2" in repr(p)
    with pytest.raises(AttributeError):
        _ = p.unknown


def test_params_normalise_types() -> None:
    p = SmaCross.make_params({"fast": 3.0, "risk": 2, "short": np.bool_(True)})
    assert p.fast == 3
    assert isinstance(p.fast, int)
    assert isinstance(p.risk, float)
    assert p.short is True


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"fast": 0}, "Minimum"),
        ({"fast": 11}, "Maximum"),
        ({"risk": 5.01}, "Maximum"),
        ({"risk": 0.0}, "Minimum"),
        ({"fast": 2.5}, "ganze Zahl"),
        ({"fast": "3"}, "Zahl"),
        ({"fast": True}, "Zahl"),
        ({"risk": float("nan")}, "endliche"),
        ({"short": 1}, "wahr/falsch"),
        ({"mode": "high"}, "nicht erlaubt"),
        ({"unknown": 1}, "Unbekannte"),
    ],
)
def test_params_out_of_range_rejected(values: dict[str, object], message: str) -> None:
    with pytest.raises(ParamError, match=message):
        SmaCross.make_params(values)


def test_params_are_read_only() -> None:
    p = SmaCross.make_params()
    with pytest.raises(AttributeError):
        p.fast = 5
    with pytest.raises(TypeError):
        p["fast"] = 5  # type: ignore[index]


# -- Orders --------------------------------------------------------------------


def test_order_valid_variants() -> None:
    o = Order("AAPL", OrderSide.BUY, qty=1.5)
    assert o.type is OrderType.MARKET
    assert o.side.is_entry
    assert o.side.position_side is Side.LONG
    assert OrderSide.BUY_TO_COVER.position_side is Side.SHORT
    assert not OrderSide.SELL.is_entry
    assert Order("AAPL", "SELL").qty is None  # type: ignore[arg-type]  # ganze Position
    assert Order("AAPL", OrderSide.SELL, OrderType.STOP, stop_price=90.0).stop_price == 90.0
    assert Order("AAPL", OrderSide.BUY, OrderType.LIMIT, qty=1, limit_price=95.0).limit_price
    assert Order("AAPL", OrderSide.SELL, OrderType.TRAILING_STOP, trail_pct=5.0).trail_pct == 5.0
    assert Order("AAPL", OrderSide.SELL, OrderType.TRAILING_STOP, trail_atr=2.0).trail_atr == 2.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"symbol": "", "side": OrderSide.BUY, "qty": 1},
        {"symbol": "A", "side": OrderSide.BUY},
        {"symbol": "A", "side": OrderSide.BUY, "qty": 1, "sizing": Sizing.percent_equity(10)},
        {"symbol": "A", "side": OrderSide.BUY, "qty": 0},
        {"symbol": "A", "side": OrderSide.BUY, "qty": float("inf")},
        {"symbol": "A", "side": OrderSide.SELL, "type": OrderType.STOP},
        {"symbol": "A", "side": OrderSide.SELL, "type": OrderType.STOP, "stop_price": -1.0},
        {
            "symbol": "A",
            "side": OrderSide.SELL,
            "type": OrderType.STOP,
            "stop_price": 1.0,
            "limit_price": 2.0,
        },
        {"symbol": "A", "side": OrderSide.SELL, "type": OrderType.LIMIT},
        {"symbol": "A", "side": OrderSide.SELL, "stop_price": 1.0},
        {"symbol": "A", "side": OrderSide.SELL, "trail_pct": 1.0},
        {"symbol": "A", "side": OrderSide.SELL, "type": OrderType.TRAILING_STOP},
        {
            "symbol": "A",
            "side": OrderSide.SELL,
            "type": OrderType.TRAILING_STOP,
            "trail_pct": 1.0,
            "trail_atr": 2.0,
        },
        {
            "symbol": "A",
            "side": OrderSide.SELL,
            "type": OrderType.TRAILING_STOP,
            "trail_pct": 1.0,
            "limit_price": 2.0,
        },
    ],
)
def test_order_invalid(kwargs: dict[str, object]) -> None:
    with pytest.raises(OrderError):
        Order(**kwargs)  # type: ignore[arg-type]


def test_sizing() -> None:
    assert Sizing.fixed_amount(500).method is SizingMethod.FIXED_AMOUNT
    assert Sizing.percent_equity(10).value == 10.0
    assert Sizing.risk(1).method is SizingMethod.RISK
    assert Sizing.volatility_target(15).method is SizingMethod.VOLATILITY_TARGET
    with pytest.raises(OrderError):
        Sizing.percent_equity(101)
    with pytest.raises(OrderError):
        Sizing.fixed_amount(0)


# -- Schreibschutz und kein Lookahead -----------------------------------------


def test_bars_contain_no_row_after_t() -> None:
    close = [10.0, 11.0, 12.0, 13.0, 14.0]
    for t in range(len(close)):
        ctx = _ctx(close, t)
        assert len(ctx.bars) == t + 1
        assert ctx.bars.close.tolist() == close[: t + 1]
        assert ctx.bars.get("close") == close[t]
        assert ctx.date == date(2024, 1, 1 + t)
        assert ctx.bars.dates[-1] == np.datetime64(f"2024-01-0{1 + t}")
    with pytest.raises(IndexError):
        _ctx(close, 5)
    with pytest.raises(IndexError):
        _ctx(close, 1).bars.get("close", ago=-1)
    with pytest.raises(IndexError):
        _ctx(close, 1).bars.get("close", ago=2)


def test_bars_are_read_only() -> None:
    bars = _bars([10.0, 11.0, 12.0])
    series = PreparedSeries(bars)
    ctx = BarContext.at(series, 1, symbol="TEST", timeframe=Timeframe.D3)
    with pytest.raises(ValueError, match="read-only"):
        ctx.bars.close[0] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        ctx.bars["open"][-1] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        ctx.bars.dates[0] = np.datetime64("2000-01-01")
    with pytest.raises(TypeError):
        ctx.bars["close"] = np.zeros(2)  # type: ignore[index]
    with pytest.raises(AttributeError):
        ctx.bars.close = np.zeros(2)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        ctx.symbol = "X"  # type: ignore[misc]
    # Kopie ist veränderbar, ändert aber nichts an Kontext oder Ausgangsdaten.
    frame = ctx.bars.to_frame()
    frame.loc[0, "close"] = 99.0
    assert ctx.bars.close[0] == 10.0
    assert bars.loc[0, "close"] == 10.0
    assert list(frame.columns) == ["date", "open", "high", "low", "close", "volume"]
    # Änderungen an den Ausgangsdaten wirken nicht in die vorbereitete Reihe.
    bars.loc[0, "close"] = 50.0
    assert series.bars_at(0).close[0] == 10.0


def test_bars_view_accessors() -> None:
    ctx = _ctx([10.0, 11.0], 1)
    assert ctx.bars.open.tolist() == [9.5, 10.5]
    assert ctx.bars.high.tolist() == [11.0, 12.0]
    assert ctx.bars.low.tolist() == [9.0, 10.0]
    assert ctx.bars.volume.tolist() == [100, 200]
    assert ctx.bars["date"].tolist() == ctx.bars.dates.tolist()
    assert "close" in ctx.bars
    assert ctx.bars.columns == ["open", "high", "low", "close", "volume"]
    with pytest.raises(KeyError):
        ctx.bars.column("missing")


def test_indicator_view() -> None:
    ind = pd.DataFrame({"sma": [np.nan, 1.0, 2.0, 3.0]})
    ctx = _ctx([1.0, 2.0, 3.0, 4.0], 2, ind)
    assert ctx.ind["sma"] == 2.0
    assert ctx.ind.sma == 2.0
    assert ctx.ind.get("sma", ago=1) == 1.0
    assert ctx.ind.column("sma").tolist()[1:] == [1.0, 2.0]
    assert len(ctx.ind) == 3
    assert ctx.ind.row() == {"sma": 2.0}
    with pytest.raises(ValueError, match="read-only"):
        ctx.ind.column("sma")[0] = 5.0
    with pytest.raises(AttributeError):
        _ = ctx.ind.missing
    with pytest.raises(AttributeError):
        _ = ctx.ind._private
    with pytest.raises(TypeError):
        ctx.ind["sma"] = 1.0  # type: ignore[index]
    empty = _ctx([1.0], 0)
    assert empty.ind.row() == {}
    assert len(empty.ind) == 0


def test_prepared_series_checks() -> None:
    with pytest.raises(ValueError, match="date"):
        PreparedSeries(pd.DataFrame({"close": [1.0]}))
    with pytest.raises(ValueError, match="Zeilen"):
        PreparedSeries(_bars([1.0, 2.0]), pd.DataFrame({"x": [1.0]}))


# -- Hilfsfunktionen -----------------------------------------------------------


def test_crossed_above_and_below() -> None:
    # fast kreuzt slow bei t=2 nach oben und bei t=4 nach unten.
    ind = pd.DataFrame(
        {
            "fast": [np.nan, 1.0, 3.0, 4.0, 2.0],
            "slow": [2.0, 2.0, 2.0, 3.0, 3.0],
        }
    )
    close = [1.0, 2.0, 3.0, 4.0, 5.0]
    above = [_ctx(close, t, ind).crossed_above("fast", "slow") for t in range(5)]
    below = [_ctx(close, t, ind).crossed_below("fast", "slow") for t in range(5)]
    assert above == [False, False, True, False, False]
    assert below == [False, False, False, False, True]


def test_crossing_touch_counts_once() -> None:
    # Berührung bei t=1 (gleich), Kreuzung erst bei t=2.
    ind = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 2.0, 2.0]})
    close = [1.0, 1.0, 1.0]
    assert [_ctx(close, t, ind).crossed_above("a", "b") for t in range(3)] == [
        False,
        False,
        True,
    ]


def test_crossing_with_constants_and_prices() -> None:
    ind = pd.DataFrame({"rsi": [25.0, 35.0, 28.0], "sma": [11.0, 11.0, 11.0]})
    close = [10.0, 12.0, 10.5]
    assert _ctx(close, 1, ind).crossed_above("rsi", 30)
    assert _ctx(close, 2, ind).crossed_below("rsi", 30.0)
    assert _ctx(close, 1, ind).crossed_above("close", "sma")
    assert _ctx(close, 2, ind).crossed_below("close", "sma")
    assert _ctx(close, 1, ind).value(5) == 5.0
    with pytest.raises(TypeError):
        _ctx(close, 1, ind).value(True)
    with pytest.raises(KeyError):
        _ctx(close, 1, ind).crossed_above("missing", 1.0)


def test_order_helpers() -> None:
    ctx = _ctx([10.0, 11.0], 1)
    buy = ctx.buy(2.0, tag="Einstieg")
    assert buy == Order("TEST", OrderSide.BUY, qty=2.0, tag="Einstieg")
    stop_buy = ctx.buy(sizing=Sizing.risk(1), type=OrderType.STOP, stop_price=12.0)
    assert stop_buy.type is OrderType.STOP
    assert stop_buy.sizing == Sizing.risk(1)
    short = ctx.sell_short(sizing=Sizing.fixed_amount(500))
    assert short.side is OrderSide.SELL_SHORT
    assert ctx.exit() == []

    long_pos = Position("TEST", Side.LONG, 2.0, 10.0, date(2024, 1, 1))
    long_ctx = _ctx([10.0, 11.0], 1, position=long_pos)
    assert long_ctx.exit() == [Order("TEST", OrderSide.SELL)]
    trail = long_ctx.exit(type=OrderType.TRAILING_STOP, trail_atr=3.0)[0]
    assert trail.trail_atr == 3.0

    short_pos = Position("TEST", Side.SHORT, 1.0, 10.0, date(2024, 1, 1))
    short_ctx = _ctx([10.0, 11.0], 1, position=short_pos)
    assert short_ctx.exit(1.0)[0].side is OrderSide.BUY_TO_COVER


# -- Strategy ------------------------------------------------------------------


def test_strategy_end_to_end() -> None:
    """Durchlauf wie in der Engine: Einstieg nach Aufwärts-, Ausstieg nach Abwärtskreuzung."""
    close = [10.0, 9.0, 8.0, 7.0, 8.0, 10.0, 12.0, 13.0, 11.0, 9.0, 7.0]
    strategy = SmaCross()
    p = SmaCross.make_params()
    series = strategy.prepare(_bars(close), p)
    position: Position | None = None
    events: list[tuple[int, OrderSide, str]] = []
    for t in range(series.length):
        ctx = BarContext.at(
            series, t, symbol="TEST", timeframe=Timeframe.W1, position=position, equity=1000.0
        )
        for order in strategy.on_bar(ctx):
            events.append((t, order.side, order.tag))
            if order.side is OrderSide.BUY:
                position = Position("TEST", Side.LONG, 1.0, close[t], ctx.date)
            else:
                position = None
    assert events == [(5, OrderSide.BUY, "Kreuzung"), (9, OrderSide.SELL, "Gegenkreuzung")]
    assert [s.column for s in strategy.plot()] == ["fast", "slow"]


class Minimal(Strategy):
    id = "minimal"
    name = "Minimal"
    version = "0.1.0"
    description = "Nur on_bar."
    timeframes = [Timeframe.W1]  # noqa: RUF012

    def on_bar(self, ctx: BarContext) -> list[Order]:
        return []


def test_strategy_defaults() -> None:
    s = Minimal()
    bars = _bars([1.0, 2.0])
    assert s.indicators(bars, s.make_params()).shape == (2, 0)
    assert s.plot() == []
    assert len(s.make_params()) == 0
    assert s.prepare(bars, s.make_params()).ind_at(1).columns == []


def test_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]
