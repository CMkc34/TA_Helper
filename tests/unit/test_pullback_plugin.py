"""Tests des Kandidaten R-04 „Pullback im Trend“ (``plugins/pullback_trend.py``)."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.engine import run_backtest
from trading_helfer.backtest.portfolio import run_portfolio_backtest
from trading_helfer.core.config import Settings
from trading_helfer.core.types import OrderType, Side, Timeframe
from trading_helfer.data.resample import PARTIAL_COLUMN, resample
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    OrderSide,
    Position,
    PreparedSeries,
    Strategy,
)
from trading_helfer.strategies.registry import StrategyRegistry, check_fields
from trading_helfer.strategies.validator import validate

ROOT = Path(__file__).parents[2]
PLUGINS = ROOT / "plugins"
PLUGIN_FILE = PLUGINS / "pullback_trend.py"
FIXTURE = ROOT / "tests" / "fixtures" / "strategies" / "SYNTH_long_daily.csv"


@pytest.fixture(scope="module")
def registry() -> StrategyRegistry:
    reg = StrategyRegistry(PLUGINS)
    reg.load()
    return reg


@pytest.fixture(scope="module")
def cls(registry: StrategyRegistry) -> type[Strategy]:
    return registry.get("pullback_trend")


@pytest.fixture(scope="module")
def module(cls: type[Strategy]) -> ModuleType:
    import sys

    return sys.modules[cls.__module__]


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, parse_dates=["date"])


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


# -- Hilfen ---------------------------------------------------------------------


def make_bars(closes: list[float], start: str = "2001-01-02") -> pd.DataFrame:
    """Kerzen mit Open = Vor-Close und schmaler Spanne (synthetisch)."""
    close = np.asarray(closes, dtype=float)
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(close), freq="3B"),
            "open": opens,
            "high": np.maximum(opens, close) + 0.5,
            "low": np.minimum(opens, close) - 0.5,
            "close": close,
            "volume": 1000,
        }
    )


def uptrend_with_dip(n_trend: int = 80, dip: int = 3) -> list[float]:
    """Stetiger Anstieg, danach ``dip`` tiefere Closes (über der Trend-SMA)."""
    closes = [100.0 + i for i in range(n_trend)]
    for k in range(1, dip + 1):
        closes.append(closes[n_trend - 1] - 1.5 * k)
    return closes


def context(
    strategy: Strategy,
    bars: pd.DataFrame,
    t: int,
    params: dict[str, Any] | None = None,
    position: Position | None = None,
    is_member: bool = True,
) -> BarContext:
    p = strategy.make_params(params)
    ind = strategy.indicators(bars, p)
    series = PreparedSeries(bars, ind)
    return BarContext.at(
        series,
        t,
        symbol="SYN",
        timeframe=Timeframe.D3,
        position=position,
        equity=1000.0,
        cash=1000.0,
        is_member=is_member,
    )


def long_at(bars: pd.DataFrame, t: int, price: float | None = None) -> Position:
    day = pd.Timestamp(bars["date"].iloc[t]).date()
    entry = float(bars["open"].iloc[t]) if price is None else price
    return Position("SYN", Side.LONG, 1.0, entry, day)


def _frame(daily: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    frame = resample(daily, tf)
    return frame.loc[~frame[PARTIAL_COLUMN]].drop(columns=[PARTIAL_COLUMN]).reset_index(drop=True)


# -- Laden und Metadaten ------------------------------------------------------------


def test_plugin_loads_and_passes_validator(registry: StrategyRegistry) -> None:
    assert not registry.errors
    assert not registry.info("pullback_trend").is_builtin
    report = validate(PLUGIN_FILE.read_text(encoding="utf-8"), filename=PLUGIN_FILE.name)
    assert report.ok, report


def test_metadata_matches_issue(cls: type[Strategy]) -> None:
    assert check_fields(cls) == []
    assert cls.timeframes[0] is Timeframe.D3
    assert cls.make_params().to_dict() == {
        "trend_period": 67,
        "trigger": "rsi",
        "rsi_entry": 10.0,
        "down_closes": 3,
        "rsi_exit": 70.0,
        "max_bars": 10,
        "atr_mult": 2.0,
    }
    for name, param in cls.params.items():
        assert param.description, name


def test_neighborhood_varies_three_params(cls: type[Strategy]) -> None:
    from trading_helfer.research.study import neighborhood_grid

    grid = neighborhood_grid(cls, cls.make_params())
    varied = {k for k, v in grid.items() if len(v) > 1}
    assert varied == {"trend_period", "rsi_entry", "atr_mult"}
    assert grid["trend_period"] == [57, 67, 77]
    assert grid["atr_mult"] == [1.5, 2.0, 2.5]


def test_plot_columns_exist(cls: type[Strategy], daily: pd.DataFrame) -> None:
    strategy = cls()
    frame = _frame(daily, Timeframe.D3)
    ind = strategy.indicators(frame, strategy.make_params())
    assert {s.column for s in strategy.plot()} <= set(ind.columns)
    assert len(ind) == len(frame)


def test_down_close_streak(module: ModuleType) -> None:
    close = pd.Series([5.0, 4.0, 3.0, 3.0, 2.0, 1.0, 0.5, 2.0])
    streak = module.down_close_streak(close)
    assert streak.tolist() == [0, 1, 2, 0, 1, 2, 3, 0]


# -- Regeln auf Kerze t --------------------------------------------------------------


def test_entry_on_rsi_pullback_in_uptrend(cls: type[Strategy]) -> None:
    strategy = cls()
    bars = make_bars(uptrend_with_dip())
    t = len(bars) - 1
    ctx = context(strategy, bars, t)
    assert ctx.value("close") > ctx.value("trend_sma")
    assert ctx.value("rsi2") < 10
    orders = strategy.on_bar(ctx)
    assert len(orders) == 1
    order = orders[0]
    assert order.side is OrderSide.BUY
    assert order.type is OrderType.MARKET
    assert order.sizing is not None
    assert order.sizing.value == pytest.approx(10.0)
    assert order.strength == pytest.approx(100.0 - ctx.value("rsi2"))


def test_no_entry_without_pullback_trend_or_membership(cls: type[Strategy]) -> None:
    strategy = cls()
    rising = make_bars([100.0 + i for i in range(80)])
    assert strategy.on_bar(context(strategy, rising, 79)) == []
    # Rücksetzer, aber unter der Trend-SMA (Abwärtstrend).
    falling = make_bars([200.0 - i for i in range(80)])
    assert strategy.on_bar(context(strategy, falling, 79)) == []
    # Rücksetzer im Aufwärtstrend, aber nicht im Universum.
    dip = make_bars(uptrend_with_dip())
    assert strategy.on_bar(context(strategy, dip, len(dip) - 1, is_member=False)) == []
    # Vor dem Anlauf der Trend-SMA (NaN) kein Einstieg.
    short = make_bars(uptrend_with_dip(n_trend=30))
    assert strategy.on_bar(context(strategy, short, len(short) - 1)) == []


def test_trigger_down_closes(cls: type[Strategy]) -> None:
    strategy = cls()
    two = make_bars(uptrend_with_dip(dip=2))
    three = make_bars(uptrend_with_dip(dip=3))
    params = {"trigger": "closes", "rsi_entry": 2.0}
    assert strategy.on_bar(context(strategy, two, len(two) - 1, params)) == []
    orders = strategy.on_bar(context(strategy, three, len(three) - 1, params))
    assert [o.side for o in orders] == [OrderSide.BUY]


def test_stop_set_on_entry_bar(cls: type[Strategy]) -> None:
    strategy = cls()
    bars = make_bars(uptrend_with_dip(dip=4))
    t = len(bars) - 1
    position = long_at(bars, t, price=170.0)
    ctx = context(strategy, bars, t, position=position)
    orders = strategy.on_bar(ctx)
    assert len(orders) == 1
    stop = orders[0]
    assert stop.type is OrderType.STOP
    assert stop.side is OrderSide.SELL
    assert stop.stop_price == pytest.approx(170.0 - 2.0 * ctx.value("atr"))


def test_exit_on_recovery(cls: type[Strategy]) -> None:
    strategy = cls()
    closes = [*uptrend_with_dip(), 190.0]
    bars = make_bars(closes)
    t = len(bars) - 1
    entry = long_at(bars, t - 1)
    ctx = context(strategy, bars, t, position=entry)
    assert ctx.value("rsi2") > 70
    orders = strategy.on_bar(ctx)
    assert [(o.type, o.tag) for o in orders] == [(OrderType.MARKET, "Erholung (RSI(2))")]


def test_time_stop_after_max_bars(cls: type[Strategy], module: ModuleType) -> None:
    strategy = cls()
    base = uptrend_with_dip()
    flat = [base[-1] - 0.1 * (k % 2) for k in range(12)]  # seitwärts, keine Erholung
    bars = make_bars([*base, *flat])
    entry_t = len(base)
    position = long_at(bars, entry_t)
    params = {"rsi_exit": 95.0}
    for held in range(1, 12):
        t = entry_t + held
        ctx = context(strategy, bars, t, params, position=position)
        assert module.bars_held(ctx) == held
        orders = strategy.on_bar(ctx)
        if held < 10:
            assert orders == [], held
        else:
            assert [o.tag for o in orders] == ["Zeit-Stop"]
    assert module.bars_held(context(strategy, bars, entry_t)) == 0


def test_stop_skipped_when_atr_missing(cls: type[Strategy]) -> None:
    strategy = cls()
    bars = make_bars([100.0 + i for i in range(10)])
    position = long_at(bars, 9)
    orders: list[Order] = strategy.on_bar(context(strategy, bars, 9, position=position))
    assert all(o.type is not OrderType.STOP for o in orders)


# -- Läufe ------------------------------------------------------------------------------


def test_backtest_d3_on_fixture(
    cls: type[Strategy], daily: pd.DataFrame, settings: Settings
) -> None:
    result = run_backtest(
        cls, None, "SYNTH", Timeframe.D3, "1995-01-01", "2009-12-31", settings, bars=daily
    )
    trades = result.trades
    assert len(trades) >= 10
    reasons = set(trades["exit_reason"])
    assert reasons <= {"Erholung (RSI(2))", "Zeit-Stop", "ATR-Stop"}
    assert "Erholung (RSI(2))" in reasons
    assert not (result.orders["status"] == "REJECTED").any()
    # Höchstens eine Position zu 10 % des Kapitals.
    first = trades.iloc[0]
    assert first["qty"] * first["entry_price"] <= 0.1 * settings.default_capital + 1e-6
    again = run_backtest(
        cls, None, "SYNTH", Timeframe.D3, "1995-01-01", "2009-12-31", settings, bars=daily
    )
    assert again.fingerprint() == result.fingerprint()


def test_stop_exits_below_entry(
    cls: type[Strategy], daily: pd.DataFrame, settings: Settings
) -> None:
    result = run_backtest(
        cls,
        {"atr_mult": 1.0, "rsi_exit": 95.0, "max_bars": 30},
        "SYNTH",
        Timeframe.D3,
        "1995-01-01",
        "2009-12-31",
        settings,
        bars=daily,
    )
    stops = result.trades.loc[result.trades["exit_reason"] == "ATR-Stop"]
    assert not stops.empty
    assert (stops["exit_price"] < stops["entry_price"]).all()


def test_portfolio_limits_positions(
    cls: type[Strategy], daily: pd.DataFrame, settings: Settings
) -> None:
    bars: dict[str, pd.DataFrame] = {}
    for k, sym in enumerate(["AAA", "BBB", "CCC"]):
        frame = daily.copy()
        frame[["open", "high", "low", "close"]] *= 1.0 + 0.1 * k
        bars[sym] = frame
    result = run_portfolio_backtest(
        cls,
        None,
        list(bars),
        Timeframe.D3,
        "1995-01-01",
        "2009-12-31",
        settings,
        bars=bars,
        max_positions=2,
        workers=1,
    )
    trades = result.trades
    assert len(trades) > 0
    days = pd.date_range("1996-01-01", "2009-12-31", freq="7D")
    for day in days:
        ts = pd.Timestamp(day)
        open_ = (pd.to_datetime(trades["entry_date"]) <= ts) & (
            pd.to_datetime(trades["exit_date"]) > ts
        )
        assert int(open_.sum()) <= 2
    assert trades["symbol"].nunique() > 1


def test_study_runs_on_d3(cls: type[Strategy], daily: pd.DataFrame, settings: Settings) -> None:
    from trading_helfer.research.study import StudyConfig, run_study

    bars = {"SYNTH": daily, "SPY": daily, "QQQ": daily}
    study = run_study(
        cls,
        {},
        "D3",
        settings,
        symbol="SYNTH",
        bars=bars,
        config=StudyConfig(workers=1, mc_simulations=50, end=pd.Timestamp("2009-12-31").date()),
    )
    assert study.timeframe is Timeframe.D3
    assert study.robustness.source == "Parameter-Nachbarschaft"
    assert len(study.criteria) == 5
