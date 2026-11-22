"""Tests für R-03: Querschnitts-Rang in den Engines und die Momentum-Rotation."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import StrategyError, run_backtest
from trading_helfer.backtest.portfolio import run_portfolio_backtest
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.data.resample import PARTIAL_COLUMN, resample
from trading_helfer.data.universe import IndexChange, IndexHistory
from trading_helfer.indicators.relative_strength import percentile_rank, rs_rank
from trading_helfer.strategies.base import (
    POSITION_SUFFIX,
    RANK_SUFFIX,
    BarContext,
    Order,
    Param,
    Params,
    PreparedSeries,
    Strategy,
)
from trading_helfer.strategies.builtin import MomentumRotation
from trading_helfer.strategies.builtin.momentum_rotation import (
    MARKET_ABOVE_COLUMN,
    MOMENTUM_COLUMN,
)
from trading_helfer.strategies.registry import check_fields

NO_COSTS = CostModel(fee_per_order=0.0, slippage_pct=0.0)
START, END = "2020-01-01", "2023-12-31"
DAYS = 900


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


def _bars(close: np.ndarray, start: str = "2019-01-01") -> pd.DataFrame:
    """Tageskurse an Werktagen; Open = Vortages-Close, High/Low ±1 %."""
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(close)),
            "open": opens,
            "high": np.maximum(opens, close) * 1.01,
            "low": np.minimum(opens, close) * 0.99,
            "close": close,
            "volume": 1000,
        }
    )


def _growth(rate: float, n: int = DAYS) -> np.ndarray:
    """Exponentieller Trend mit fester Tagesrendite plus kleiner Welle."""
    t = np.arange(n)
    return 100.0 * np.exp(rate * t) * (1.0 + 0.01 * np.sin(t / 7.0))


def _universe() -> dict[str, pd.DataFrame]:
    """Sechs Werte mit klar getrennter Stärke: AAA am stärksten, FFF am schwächsten."""
    rates = {"AAA": 0.0020, "BBB": 0.0015, "CCC": 0.0010, "DDD": 0.0, "EEE": -0.0005}
    bars = {s: _bars(_growth(r)) for s, r in rates.items()}
    # FFF: zuerst der stärkste Wert, ab Tag 450 Absturz.
    t = np.arange(DAYS)
    fff = np.where(
        t < 450, 100.0 * np.exp(0.003 * t), 100.0 * np.exp(0.003 * 449 - 0.004 * (t - 449))
    )
    bars["FFF"] = _bars(fff)
    return bars


def _weekly(daily: pd.DataFrame) -> pd.DataFrame:
    frame = resample(daily, Timeframe.W1)
    return frame.loc[~frame[PARTIAL_COLUMN]].drop(columns=[PARTIAL_COLUMN]).reset_index(drop=True)


# -- percentile_rank / PreparedSeries -----------------------------------------------


def test_percentile_rank_linear_ties_and_nan() -> None:
    values = pd.Series({"A": 1.0, "B": 3.0, "C": 3.0, "D": np.nan, "E": 0.0})
    ranks = percentile_rank(values)
    assert list(ranks.index) == list(values.index)
    assert ranks["E"] == 0.0
    assert ranks["A"] == pytest.approx(100 / 3)
    assert ranks["B"] == ranks["C"] == pytest.approx(250 / 3)
    assert math.isnan(ranks["D"])


def test_percentile_rank_single_and_empty() -> None:
    single = percentile_rank(pd.Series({"A": 5.0, "B": np.nan}))
    assert single["A"] == 100.0
    assert math.isnan(single["B"])
    assert percentile_rank(pd.Series(dtype="float64")).empty


def test_with_indicators_adds_columns_without_touching_bars() -> None:
    bars = _bars(np.arange(1.0, 6.0))
    series = PreparedSeries(bars, pd.DataFrame({"a": np.arange(5.0)}))
    extended = series.with_indicators({"b": np.arange(5.0) * 2})
    view = extended.ind_at(2)
    assert view.columns == ["a", "b"]
    assert view["b"] == 4.0
    assert "b" not in series.ind_at(2)
    with pytest.raises(ValueError, match="Kursdaten"):
        series.with_indicators({"c": np.arange(3.0)})
    with pytest.raises(ValueError, match=r"read-only|not writeable"):
        extended.ind_at(4).column("b")[0] = 1.0


# -- Querschnitts-Rang in der Portfolio-Engine ----------------------------------------


class RankProbe(Strategy):
    """Zeichnet Rang und Platz je Wert und Termin auf; handelt nie."""

    id = "test_rank_probe"
    name = "Rang-Sonde"
    version = "1.0.0"
    description = "Zeichnet den Querschnitts-Rang auf."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {"lookback": Param(8, 1, 52, 1, "Kerzen")}
    rank_columns: ClassVar[tuple[str, ...]] = ("ret",)
    seen: ClassVar[dict[tuple[str, date], tuple[float, float, float]]] = {}

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        close = bars["close"]
        return pd.DataFrame({"ret": close / close.shift(int(p.lookback)) - 1.0})

    def on_bar(self, ctx: BarContext) -> list[Order]:
        RankProbe.seen[(ctx.symbol, ctx.date)] = (
            float(ctx.ind["ret"]),
            float(ctx.ind["ret" + RANK_SUFFIX]),
            float(ctx.ind["ret" + POSITION_SUFFIX]),
        )
        return []


def test_portfolio_rank_matches_rs_rank(settings: Settings) -> None:
    """Der Engine-Rang entspricht dem RS-Rang aus X-03 (gleiche Rendite, gleicher Stichtag)."""
    RankProbe.seen = {}
    bars = _universe()
    run_portfolio_backtest(
        RankProbe, None, list(bars), Timeframe.W1, START, END, settings, bars=bars, costs=NO_COSTS
    )
    weekly = {s: _weekly(b) for s, b in bars.items()}
    days = sorted({d for _, d in RankProbe.seen})
    assert len(days) > 50
    for day in days[::10]:
        expected = rs_rank(weekly, day, 8)
        for symbol, value in expected.items():
            ret, rank, _ = RankProbe.seen[(symbol, day)]
            assert rank == pytest.approx(value), (symbol, day)
            assert math.isfinite(ret)
        places = sorted((RankProbe.seen[(s, day)][2], s) for s in expected.index)
        assert [s for _, s in places] == list(expected.index)  # Platz 1 = höchster Rang


def test_portfolio_rank_counts_only_members(settings: Settings) -> None:
    RankProbe.seen = {}
    bars = _universe()
    joined = date(2021, 1, 4)
    history = IndexHistory(
        "TEST",
        frozenset(bars),
        (IndexChange(joined, frozenset({"AAA"}), frozenset()),),
    )
    run_portfolio_backtest(
        RankProbe, None, history, Timeframe.W1, START, END, settings, bars=bars, costs=NO_COSTS
    )
    days = sorted({d for _, d in RankProbe.seen})
    assert any(d < joined for d in days)
    assert any(d >= joined for d in days)
    # AAA wird vor der Aufnahme nicht aufgerufen (kein Mitglied, keine Position).
    assert not [d for (s, d) in RankProbe.seen if s == "AAA" and d < joined]
    for day in days:
        members = sorted(bars) if day >= joined else sorted(set(bars) - {"AAA"})
        rets = pd.Series({s: RankProbe.seen[(s, day)][0] for s in members}).dropna()
        if rets.empty:
            continue
        expected = percentile_rank(rets)
        order = sorted(rets.index, key=lambda s: (-rets[s], s))
        for place, symbol in enumerate(order, start=1):
            _, rank, got_place = RankProbe.seen[(symbol, day)]
            assert rank == pytest.approx(expected[symbol])
            assert got_place == place


class MissingRank(RankProbe):
    id = "test_missing_rank"
    rank_columns: ClassVar[tuple[str, ...]] = ("nope",)


def test_missing_rank_column_raises(settings: Settings) -> None:
    bars = _universe()
    with pytest.raises(StrategyError, match="nope"):
        run_portfolio_backtest(
            MissingRank, None, list(bars), Timeframe.W1, START, END, settings, bars=bars
        )
    with pytest.raises(StrategyError, match="nope"):
        run_backtest(MissingRank, None, "AAA", Timeframe.W1, START, END, settings, bars=bars["AAA"])


def test_single_backtest_rank_is_100_and_place_1(settings: Settings) -> None:
    RankProbe.seen = {}
    bars = _universe()
    run_backtest(RankProbe, None, "AAA", Timeframe.W1, START, END, settings, bars=bars["AAA"])
    values = list(RankProbe.seen.values())
    assert values
    assert all(rank == 100.0 and place == 1.0 for _, rank, place in values)


# -- Strategie ---------------------------------------------------------------------


def test_metadata_and_defaults() -> None:
    assert check_fields(MomentumRotation) == []
    assert MomentumRotation.timeframes == [Timeframe.W1]
    assert MomentumRotation.rank_columns == (MOMENTUM_COLUMN,)
    assert MomentumRotation.market_symbol == "SPY"
    assert MomentumRotation.make_params().to_dict() == {
        "lookback": 26,
        "skip": 1,
        "top_n": 10,
        "exit_rank": 30.0,
        "market_sma": 40,
        "market_filter": True,
    }


def test_momentum_skips_last_bar() -> None:
    frame = _weekly(_universe()["AAA"])
    ind = MomentumRotation().indicators(frame, MomentumRotation.make_params())
    close = frame["close"]
    expected = close.shift(1) / close.shift(27) - 1.0
    pd.testing.assert_series_equal(ind[MOMENTUM_COLUMN], expected, check_names=False)
    assert ind["strength"].equals(ind[MOMENTUM_COLUMN])
    assert ind[MOMENTUM_COLUMN].iloc[:27].isna().all()


def test_market_indicators_flag_above_sma() -> None:
    close = np.concatenate([np.full(10, 100.0), np.full(5, 120.0), np.full(5, 80.0)])
    market = pd.DataFrame({"close": close, "open": close, "high": close, "low": close})
    p = MomentumRotation.make_params({"market_sma": 10})
    flag = MomentumRotation().market_indicators(market, p)["above_sma"]
    assert flag.iloc[:9].isna().all()
    assert flag.iloc[9] == 0.0  # Close = SMA
    assert flag.iloc[10] == 1.0
    assert flag.iloc[-1] == 0.0


def _rotation(
    settings: Settings,
    bars: dict[str, pd.DataFrame],
    end: str = END,
    **params: object,
) -> pd.DataFrame:
    values = {"lookback": 8, "top_n": 2, **params}
    result = run_portfolio_backtest(
        MomentumRotation,
        values,
        list(bars),
        Timeframe.W1,
        START,
        end,
        settings,
        max_positions=2,
        bars=bars,
        costs=NO_COSTS,
    )
    return result.orders


def test_rotation_buys_top_and_sells_on_weak_rank(settings: Settings) -> None:
    orders = _rotation(settings, _universe())
    filled = orders[orders["status"] == "FILLED"]
    bought = set(filled.loc[filled["side"] == "BUY", "symbol"])
    # Nur Werte, die je unter den besten zwei standen.
    assert bought <= {"AAA", "BBB", "FFF"}
    assert {"FFF", "AAA"} <= bought
    # FFF stürzt ab und wird wegen des schwachen RS-Rangs verkauft.
    fff_exits = filled[(filled["symbol"] == "FFF") & (filled["side"] == "SELL")]
    assert len(fff_exits) >= 1
    assert (fff_exits["tag"] == "RS-Rang zu schwach").all()
    assert "DDD" not in bought
    assert "EEE" not in bought


def test_rotation_sells_when_removed_from_index(settings: Settings) -> None:
    bars = _universe()
    removed = date(2021, 6, 1)
    history = IndexHistory(
        "TEST",
        frozenset(bars) - {"AAA"},
        (IndexChange(removed, frozenset(), frozenset({"AAA"})),),
    )
    result = run_portfolio_backtest(
        MomentumRotation,
        {"lookback": 8, "top_n": 2},
        history,
        Timeframe.W1,
        START,
        END,
        settings,
        max_positions=2,
        bars=bars,
        costs=NO_COSTS,
    )
    aaa = result.trades[result.trades["symbol"] == "AAA"]
    assert len(aaa) >= 1
    assert pd.Timestamp(aaa["exit_date"].max()).date() <= date(2021, 6, 30)
    assert not any(p.symbol == "AAA" for p in result.open_positions)


def test_rotation_no_lookahead_and_deterministic(settings: Settings) -> None:
    bars = _universe()
    full = run_portfolio_backtest(
        MomentumRotation,
        {"lookback": 8, "top_n": 2},
        list(bars),
        Timeframe.W1,
        START,
        END,
        settings,
        max_positions=2,
        bars=bars,
    )
    again = run_portfolio_backtest(
        MomentumRotation,
        {"lookback": 8, "top_n": 2},
        list(bars),
        Timeframe.W1,
        START,
        END,
        settings,
        max_positions=2,
        bars=bars,
    )
    assert full.fingerprint() == again.fingerprint()
    cut = run_portfolio_backtest(
        MomentumRotation,
        {"lookback": 8, "top_n": 2},
        list(bars),
        Timeframe.W1,
        START,
        "2021-12-31",
        settings,
        max_positions=2,
        bars=bars,
    )
    assert len(cut.trades) > 0
    pd.testing.assert_frame_equal(cut.trades, full.trades.iloc[: len(cut.trades)])


def test_market_filter_blocks_entries_only(settings: Settings) -> None:
    """Mit Marktspalte 0 keine Käufe; ohne Marktspalte gilt der Filter als erfüllt."""
    strategy = MomentumRotation()
    p = MomentumRotation.make_params({"top_n": 2})
    strategy.indicators(_weekly(_universe()["AAA"]), p)
    frame = _weekly(_universe()["AAA"])
    t = 40
    base = {
        MOMENTUM_COLUMN: np.full(len(frame), 0.5),
        MOMENTUM_COLUMN + RANK_SUFFIX: np.full(len(frame), 90.0),
        MOMENTUM_COLUMN + POSITION_SUFFIX: np.full(len(frame), 1.0),
    }

    def ctx(market: float | None) -> BarContext:
        cols = dict(base)
        if market is not None:
            cols[MARKET_ABOVE_COLUMN] = np.full(len(frame), market)
        series = PreparedSeries(frame, pd.DataFrame(cols))
        return BarContext.at(series, t, symbol="AAA", timeframe=Timeframe.W1, equity=1000.0)

    assert strategy.on_bar(ctx(None))
    assert strategy.on_bar(ctx(1.0))
    assert strategy.on_bar(ctx(0.0)) == []
    assert strategy.on_bar(ctx(np.nan)) == []
    off = MomentumRotation()
    off.indicators(frame, MomentumRotation.make_params({"top_n": 2, "market_filter": False}))
    assert off.on_bar(ctx(0.0))


def test_entry_needs_top_place_and_rank() -> None:
    strategy = MomentumRotation()
    frame = _weekly(_universe()["AAA"])
    strategy.indicators(frame, MomentumRotation.make_params({"top_n": 3, "exit_rank": 40.0}))

    def orders(rank: float, place: float) -> list[Order]:
        cols = {
            MOMENTUM_COLUMN: np.full(len(frame), 0.2),
            MOMENTUM_COLUMN + RANK_SUFFIX: np.full(len(frame), rank),
            MOMENTUM_COLUMN + POSITION_SUFFIX: np.full(len(frame), place),
        }
        series = PreparedSeries(frame, pd.DataFrame(cols))
        ctx = BarContext.at(series, 30, symbol="AAA", timeframe=Timeframe.W1, equity=1000.0)
        return strategy.on_bar(ctx)

    buy = orders(95.0, 3.0)
    assert len(buy) == 1
    assert buy[0].sizing is not None
    assert buy[0].sizing.value == pytest.approx(100 / 3)
    assert buy[0].strength == pytest.approx(0.2)
    assert orders(95.0, 4.0) == []
    assert orders(30.0, 1.0) == []
    assert orders(np.nan, np.nan) == []


def test_market_filter_in_portfolio_run(settings: Settings) -> None:
    """Liegt SPY unter seiner SMA, kauft die Rotation nicht; danach wieder."""
    bars = _universe()
    t = np.arange(DAYS)
    # SPY fällt bis Tag 600 und steigt danach stark.
    spy = np.where(t < 600, 200.0 * np.exp(-0.001 * t), 200.0 * np.exp(-0.6 + 0.004 * (t - 600)))
    with_market = {**bars, "SPY": _bars(spy)}
    result = run_portfolio_backtest(
        MomentumRotation,
        {"lookback": 8, "top_n": 2, "market_sma": 10},
        list(bars),
        Timeframe.W1,
        START,
        END,
        settings,
        max_positions=2,
        bars=with_market,
        costs=NO_COSTS,
    )
    buys = result.orders[(result.orders["side"] == "BUY") & (result.orders["status"] == "FILLED")]
    assert len(buys) > 0
    turn = pd.Timestamp(_bars(spy)["date"].iloc[600])
    assert (pd.to_datetime(buys["submitted"]) > turn).all()
    off = run_portfolio_backtest(
        MomentumRotation,
        {"lookback": 8, "top_n": 2, "market_sma": 10, "market_filter": False},
        list(bars),
        Timeframe.W1,
        START,
        END,
        settings,
        max_positions=2,
        bars=with_market,
        costs=NO_COSTS,
    )
    first_off = pd.to_datetime(off.orders.loc[off.orders["side"] == "BUY", "submitted"]).min()
    assert first_off < turn
