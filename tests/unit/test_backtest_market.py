"""Tests des Marktfilters (``Strategy.market_symbol``, ``backtest/market.py``)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import BacktestResult, StrategyError, run_backtest
from trading_helfer.backtest.market import (
    MarketError,
    attach_market,
    load_market_daily,
    market_table,
)
from trading_helfer.backtest.portfolio import run_portfolio_backtest
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.data.storage import MarketStore
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    Params,
    Sizing,
    Strategy,
)

NO_COSTS = CostModel(fee_per_order=0.0, slippage_pct=0.0)


def _bars(close: np.ndarray | list[float], start: str = "2020-01-02") -> pd.DataFrame:
    """Tageskurse an Werktagen; Open = Vortages-Close, High/Low ±1 %."""
    close_arr = np.asarray(close, dtype=float)
    opens = np.concatenate([[close_arr[0]], close_arr[:-1]])
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(close_arr)),
            "open": opens,
            "high": np.maximum(opens, close_arr) * 1.01,
            "low": np.minimum(opens, close_arr) * 0.99,
            "close": close_arr,
            "volume": 1000,
        }
    )


def _market(n: int = 80, turn: int = 30) -> pd.DataFrame:
    """Markt fällt bis Tag ``turn``, steigt danach."""
    down = 200.0 - 2.0 * np.arange(turn)
    up = down[-1] + 2.0 * np.arange(1, n - turn + 1)
    return _bars(np.concatenate([down, up]))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


class MarketFiltered(Strategy):
    """Kauft nur, wenn der Markt über seiner SMA liegt; ohne Marktspalten immer."""

    id = "test_market_filtered"
    name = "Mit Marktfilter"
    version = "1.0.0"
    description = "Kauft und hält, solange der Markt über der SMA liegt."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        "sma": Param(5, 2, 50, 1, "Periode der Markt-SMA"),
    }
    market_symbol: ClassVar[str | None] = "MKT"

    def market_indicators(self, market: pd.DataFrame, p: Params) -> pd.DataFrame:
        return pd.DataFrame({"sma": market["close"].rolling(int(p.sma)).mean()})

    def on_bar(self, ctx: BarContext) -> list[Order]:
        risk_on = "market_sma" not in ctx.ind or ctx.value("market_close") > ctx.value("market_sma")
        if ctx.position is None:
            return [ctx.buy(sizing=Sizing.percent_equity(30), tag="Ein")] if risk_on else []
        return [] if risk_on else ctx.exit(tag="Markt")


class BrokenMarket(MarketFiltered):
    id = "test_broken_market"
    MODE: ClassVar[str] = "raise"

    def market_indicators(self, market: pd.DataFrame, p: Params) -> pd.DataFrame:
        if self.MODE == "raise":
            raise RuntimeError("kaputt")
        if self.MODE == "type":
            return market["close"]  # type: ignore[return-value]
        return pd.DataFrame({"sma": [1.0]})


class NoMarket(MarketFiltered):
    id = "test_no_market"
    market_symbol: ClassVar[str | None] = None


# -- market_table / attach_market ------------------------------------------------


def test_market_table_prefixes_columns_and_keeps_close() -> None:
    strategy = MarketFiltered()
    table = market_table(strategy, strategy.make_params(), _market(), Timeframe.D1)
    assert table is not None
    assert list(table.columns) == ["date", "market_close", "market_sma"]
    assert table["market_sma"].isna().sum() == 4
    assert table["market_close"].iloc[0] == 200.0


def test_market_table_drops_partial_last_bar() -> None:
    strategy = MarketFiltered()
    daily = _bars(np.linspace(100, 120, 23), start="2020-01-06")  # Mo, 4 Wochen + 3 Tage
    table = market_table(strategy, strategy.make_params(), daily, Timeframe.W1)
    assert table is not None
    assert len(table) == 4


def test_market_table_none_and_empty() -> None:
    strategy = MarketFiltered()
    p = strategy.make_params()
    assert market_table(strategy, p, None, Timeframe.D1) is None
    assert market_table(strategy, p, _bars([1.0, 2.0], "2020-01-06"), Timeframe.W1) is None


def test_market_table_default_indicators_only_close() -> None:
    class Plain(NoMarket):
        market_symbol: ClassVar[str | None] = "MKT"

        def market_indicators(self, market: pd.DataFrame, p: Params) -> pd.DataFrame:
            return Strategy.market_indicators(self, market, p)

    strategy = Plain()
    table = market_table(strategy, strategy.make_params(), _market(), Timeframe.D1)
    assert table is not None
    assert list(table.columns) == ["date", "market_close"]


def test_market_table_ignores_close_column_from_indicators() -> None:
    class CloseToo(MarketFiltered):
        def market_indicators(self, market: pd.DataFrame, p: Params) -> pd.DataFrame:
            return pd.DataFrame({"close": market["close"] * 0, "sma": market["close"]})

    strategy = CloseToo()
    table = market_table(strategy, strategy.make_params(), _market(), Timeframe.D1)
    assert table is not None
    assert table["market_close"].iloc[0] == 200.0


@pytest.mark.parametrize(
    ("mode", "text"),
    [("raise", "kaputt"), ("type", "muss ein DataFrame"), ("len", "Zeilen")],
)
def test_market_table_errors(mode: str, text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BrokenMarket, "MODE", mode)
    strategy = BrokenMarket()
    with pytest.raises(MarketError, match=text):
        market_table(strategy, strategy.make_params(), _market(), Timeframe.D1)


def test_attach_market_uses_last_market_bar_until_t() -> None:
    table = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-03", "2020-01-10"]),
            "market_close": [1.0, 2.0],
        }
    )
    dates = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-09", "2020-01-10"]))
    ind = pd.DataFrame({"x": [1, 2, 3, 4]}, index=[5, 6, 7, 8])
    result = attach_market(ind, dates, table)
    assert list(result.index) == [0, 1, 2, 3]
    assert np.isnan(result["market_close"].iloc[0])
    assert result["market_close"].tolist()[1:] == [1.0, 1.0, 2.0]
    assert attach_market(ind, dates, None)["x"].tolist() == [1, 2, 3, 4]


# -- load_market_daily -------------------------------------------------------------


def test_load_market_daily_sources(settings: Settings) -> None:
    log: list[str] = []
    end = date(2020, 2, 3)
    market = _market()
    assert load_market_daily(NoMarket(), settings, end, market=market, bars=None, log=log) is None

    given = load_market_daily(MarketFiltered(), settings, end, market=market, bars=None, log=log)
    assert given is not None
    assert pd.Timestamp(given["date"].iloc[-1]).date() == end

    from_bars = load_market_daily(
        MarketFiltered(), settings, end, market=None, bars={"MKT": market}, log=log
    )
    assert from_bars is not None
    assert len(from_bars) == len(given)
    assert log == []

    missing = load_market_daily(MarketFiltered(), settings, end, market=None, bars={}, log=log)
    assert missing is None
    assert log == ["Keine Kurse für den Markt MKT: Marktfilter ist ausgeschaltet."]

    log.clear()
    early = load_market_daily(
        MarketFiltered(), settings, date(2019, 1, 1), market=market, bars=None, log=log
    )
    assert early is None
    assert "bis 2019-01-01" in log[0]


def test_load_market_daily_from_store(settings: Settings) -> None:
    with MarketStore(settings.database_path) as store:
        store.upsert_bars("MKT", _market(), source="test")
    log: list[str] = []
    found = load_market_daily(
        MarketFiltered(), settings, date(2030, 1, 1), market=None, bars={}, log=log
    )
    assert found is not None
    assert len(found) == 80
    absent = load_market_daily(
        type("Other", (MarketFiltered,), {"market_symbol": "ZZZ"})(),
        settings,
        date(2030, 1, 1),
        market=None,
        bars=None,
        log=log,
    )
    assert absent is None
    assert log == ["Keine Kurse für den Markt ZZZ: Marktfilter ist ausgeschaltet."]


# -- Engines ---------------------------------------------------------------------


def _first_entry(result: BacktestResult) -> date:
    days = [pd.Timestamp(d).date() for d in result.trades["entry_date"]]
    days += [p.entry_date for p in result.open_positions]
    return min(days)


def test_run_backtest_waits_for_market_filter(settings: Settings) -> None:
    stock = _bars(100 + 0.5 * np.arange(80))
    market = _market()
    filtered = run_backtest(
        MarketFiltered,
        None,
        "AAA",
        Timeframe.D1,
        "2020-01-01",
        "2020-12-31",
        settings,
        bars=stock,
        costs=NO_COSTS,
        market=market,
    )
    plain = run_backtest(
        NoMarket,
        None,
        "AAA",
        Timeframe.D1,
        "2020-01-01",
        "2020-12-31",
        settings,
        bars=stock,
        costs=NO_COSTS,
    )
    turn_day = pd.Timestamp(market["date"].iloc[30]).date()
    assert _first_entry(filtered) > turn_day
    assert _first_entry(plain) < turn_day
    assert "market_sma" in filtered.last_indicators


def test_run_backtest_without_market_data_logs(settings: Settings) -> None:
    result = run_backtest(
        MarketFiltered,
        None,
        "AAA",
        Timeframe.D1,
        "2020-01-01",
        "2020-12-31",
        settings,
        bars=_bars(100 + 0.5 * np.arange(40)),
        costs=NO_COSTS,
    )
    assert "Keine Kurse für den Markt MKT: Marktfilter ist ausgeschaltet." in result.log
    assert "market_close" not in result.last_indicators


def test_run_backtest_on_market_symbol_uses_own_bars(settings: Settings) -> None:
    result = run_backtest(
        MarketFiltered,
        None,
        "MKT",
        Timeframe.D1,
        "2020-01-01",
        "2020-12-31",
        settings,
        bars=_market(),
        costs=NO_COSTS,
    )
    assert "market_sma" in result.last_indicators
    assert not any("Marktfilter" in line for line in result.log)


def test_run_backtest_market_error_is_strategy_error(settings: Settings) -> None:
    with pytest.raises(StrategyError, match="kaputt"):
        run_backtest(
            BrokenMarket,
            None,
            "AAA",
            Timeframe.D1,
            "2020-01-01",
            "2020-12-31",
            settings,
            bars=_bars(100 + np.arange(20.0)),
            market=_market(),
        )


def test_market_columns_do_not_look_ahead(settings: Settings) -> None:
    stock = _bars(100 + 0.5 * np.arange(80))
    market = _market()
    changed = market.copy()
    changed.loc[60:, "close"] = 1.0  # Crash nach Tag 60
    base = {
        "strategy": MarketFiltered,
        "params": None,
        "symbol": "AAA",
        "timeframe": Timeframe.D1,
        "start": "2020-01-01",
        "settings": settings,
        "bars": stock,
        "costs": NO_COSTS,
    }
    cut = pd.Timestamp(market["date"].iloc[59]).date()
    a = run_backtest(**base, end=cut, market=market)  # type: ignore[arg-type]
    b = run_backtest(**base, end=cut, market=changed)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(a.equity, b.equity)
    full = run_backtest(**base, end="2020-12-31", market=changed)  # type: ignore[arg-type]
    assert "Markt" in set(full.trades["exit_reason"])


def test_portfolio_uses_market_from_bars_and_argument(settings: Settings) -> None:
    bars = {
        "AAA": _bars(100 + 0.5 * np.arange(80)),
        "BBB": _bars(50 + 0.2 * np.arange(80)),
        "MKT": _market(),
    }
    turn_day = pd.Timestamp(bars["MKT"]["date"].iloc[30]).date()
    from_bars = run_portfolio_backtest(
        MarketFiltered,
        None,
        ["AAA", "BBB"],
        Timeframe.D1,
        "2020-01-01",
        "2020-12-31",
        settings,
        bars=bars,
        costs=NO_COSTS,
        workers=1,
    )
    assert _first_entry(from_bars) > turn_day
    explicit = run_portfolio_backtest(
        MarketFiltered,
        None,
        ["AAA", "BBB"],
        Timeframe.D1,
        "2020-01-01",
        "2020-12-31",
        settings,
        bars={k: v for k, v in bars.items() if k != "MKT"},
        costs=NO_COSTS,
        workers=1,
        market=bars["MKT"],
    )
    pd.testing.assert_frame_equal(from_bars.trades, explicit.trades)


def test_portfolio_market_error_is_strategy_error(settings: Settings) -> None:
    with pytest.raises(StrategyError, match="kaputt"):
        run_portfolio_backtest(
            BrokenMarket,
            None,
            ["AAA"],
            Timeframe.D1,
            "2020-01-01",
            "2020-12-31",
            settings,
            bars={"AAA": _bars(100 + np.arange(20.0)), "MKT": _market()},
            workers=1,
        )


def test_load_market_daily_without_database_path() -> None:
    log: list[str] = []
    found = load_market_daily(
        MarketFiltered(),
        Settings.model_construct(),
        date(2030, 1, 1),
        market=None,
        bars=None,
        log=log,
    )
    assert found is None
    assert log == ["Keine Kurse für den Markt MKT: Marktfilter ist ausgeschaltet."]
