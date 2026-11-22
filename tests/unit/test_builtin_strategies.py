"""Tests der mitgelieferten Beispielstrategien (B-07)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.engine import BacktestResult, run_backtest
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.data.resample import PARTIAL_COLUMN, resample
from trading_helfer.strategies.base import Strategy
from trading_helfer.strategies.builtin import (
    BUILTIN_STRATEGIES,
    DonchianBreakout,
    MomentumRotation,
    Rsi2MeanReversion,
    SmaCross,
)
from trading_helfer.strategies.registry import StrategyRegistry, check_fields

FIXTURE = Path(__file__).parents[1] / "fixtures" / "strategies" / "SYNTH_long_daily.csv"
START, END = "1995-01-01", "2009-12-31"

# Regressionswerte: Anzahl abgeschlossener Trades auf der Fixture mit Standardparametern.
EXPECTED_TRADES: dict[tuple[type[Strategy], Timeframe], int] = {
    (SmaCross, Timeframe.W1): 11,
    (SmaCross, Timeframe.D3): 21,
    (SmaCross, Timeframe.MN1): 2,
    (DonchianBreakout, Timeframe.W1): 10,
    (DonchianBreakout, Timeframe.D3): 16,
    (DonchianBreakout, Timeframe.MN1): 1,
    (Rsi2MeanReversion, Timeframe.D3): 31,
    (Rsi2MeanReversion, Timeframe.W1): 18,
    # Einzelwert: RS-Rang immer 100, also Kauf und Halten bis zum Ende (offene Position).
    (MomentumRotation, Timeframe.W1): 0,
}

# Strategien, die erst im Portfolio-Backtest (Querschnitts-Rang) Trades schließen;
# ihre Regeln und der Lookahead-Test stehen in eigenen Testdateien.
PORTFOLIO_ONLY: frozenset[type[Strategy]] = frozenset({MomentumRotation})
SINGLE_SYMBOL = tuple(c for c in BUILTIN_STRATEGIES if c not in PORTFOLIO_ONLY)

# Zeitrahmen, für den die Strategie laut Issue gedacht ist.
MAIN_TIMEFRAME: dict[type[Strategy], Timeframe] = {
    SmaCross: Timeframe.W1,
    DonchianBreakout: Timeframe.W1,
    Rsi2MeanReversion: Timeframe.D3,
    MomentumRotation: Timeframe.W1,
}


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, parse_dates=["date"])


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


def _run(
    cls: type[Strategy],
    tf: Timeframe,
    daily: pd.DataFrame,
    settings: Settings,
    end: str = END,
) -> BacktestResult:
    return run_backtest(cls, None, "SYNTH", tf, START, end, settings, bars=daily)


def _frame(daily: pd.DataFrame, tf: Timeframe) -> pd.DataFrame:
    frame = resample(daily, tf)
    return frame.loc[~frame[PARTIAL_COLUMN]].drop(columns=[PARTIAL_COLUMN]).reset_index(drop=True)


# -- Metadaten ---------------------------------------------------------------


def test_registry_loads_all_builtins() -> None:
    registry = StrategyRegistry(None)
    report = registry.load()
    assert report.ok
    assert set(registry.ids()) == {
        "trend_sma_cross",
        "trend_donchian_breakout",
        "mr_rsi2_trend",
        "momentum_rotation",
    }
    for cls in BUILTIN_STRATEGIES:
        assert registry.info(cls.id).is_builtin


@pytest.mark.parametrize("cls", BUILTIN_STRATEGIES, ids=lambda c: c.id)
def test_metadata_complete(cls: type[Strategy]) -> None:
    assert check_fields(cls) == []
    assert cls.description
    assert cls.name
    assert cls.timeframes[0] is MAIN_TIMEFRAME[cls]
    for name, param in cls.params.items():
        assert param.description, name
        if param.kind in ("int", "float"):
            assert param.min is not None
            assert param.max is not None
            # Ohne step bleibt ein Parameter im Robustheitsraster der Studie fest.
            assert param.step is not None or cls in PORTFOLIO_ONLY
        param.validate(param.default, name)


def test_default_params_match_issue() -> None:
    assert SmaCross.make_params().to_dict() == {"fast": 10, "slow": 40}
    assert DonchianBreakout.make_params().to_dict() == {"period": 20, "atr_mult": 3.0}
    assert Rsi2MeanReversion.make_params().to_dict() == {
        "rsi_period": 2,
        "entry_below": 10.0,
        "exit_above": 70.0,
        "trend_period": 200,
    }


@pytest.mark.parametrize("cls", BUILTIN_STRATEGIES, ids=lambda c: c.id)
def test_plot_columns_exist(cls: type[Strategy], daily: pd.DataFrame) -> None:
    strategy = cls()
    frame = _frame(daily, MAIN_TIMEFRAME[cls])
    ind = strategy.indicators(frame, cls.make_params())
    specs = strategy.plot()
    assert specs
    assert {s.column for s in specs} <= set(ind.columns)
    assert len(ind) == len(frame)


# -- Läufe auf der Fixture -------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "tf"), list(EXPECTED_TRADES), ids=lambda v: getattr(v, "id", getattr(v, "value", v))
)
def test_runs_without_errors_and_trade_count(
    cls: type[Strategy], tf: Timeframe, daily: pd.DataFrame, settings: Settings
) -> None:
    result = _run(cls, tf, daily, settings)
    assert len(result.trades) == EXPECTED_TRADES[(cls, tf)]
    assert not (result.orders["status"] == "REJECTED").any()
    assert not any("verworfen" in line for line in result.log)
    assert result.final_equity > 0
    assert result.warmup_end is not None


def test_every_supported_timeframe_is_covered() -> None:
    for cls in BUILTIN_STRATEGIES:
        for tf in cls.timeframes:
            assert (cls, tf) in EXPECTED_TRADES


@pytest.mark.parametrize("cls", BUILTIN_STRATEGIES, ids=lambda c: c.id)
def test_deterministic(cls: type[Strategy], daily: pd.DataFrame, settings: Settings) -> None:
    tf = MAIN_TIMEFRAME[cls]
    assert (
        _run(cls, tf, daily, settings).fingerprint() == _run(cls, tf, daily, settings).fingerprint()
    )


@pytest.mark.parametrize("cls", SINGLE_SYMBOL, ids=lambda c: c.id)
def test_no_lookahead(cls: type[Strategy], daily: pd.DataFrame, settings: Settings) -> None:
    """Trades bis zu einem Stichtag ändern sich nicht, wenn spätere Daten fehlen."""
    tf = MAIN_TIMEFRAME[cls]
    full = _run(cls, tf, daily, settings)
    cut = _run(cls, tf, daily, settings, end="2003-06-30")
    assert len(cut.trades) > 0
    pd.testing.assert_frame_equal(cut.trades, full.trades.iloc[: len(cut.trades)])


# -- Regeln der einzelnen Strategien ---------------------------------------------


def _entries(result: BacktestResult) -> pd.DataFrame:
    orders = result.orders
    return orders.loc[(orders["side"] == "BUY") & (orders["status"] == "FILLED")]


def test_sma_cross_rules(daily: pd.DataFrame, settings: Settings) -> None:
    result = _run(SmaCross, Timeframe.W1, daily, settings)
    frame = _frame(daily, Timeframe.W1)
    ind = SmaCross().indicators(frame, SmaCross.make_params())
    diff = (ind["sma_fast"] - ind["sma_slow"]).to_numpy()
    dates = pd.DatetimeIndex(frame["date"])
    for submitted in _entries(result)["submitted"]:
        t = dates.get_loc(submitted)
        assert diff[t - 1] <= 0 < diff[t]
    for exit_date in result.trades["exit_date"]:
        t = dates.get_loc(exit_date)
        assert diff[t - 1] < 0 <= diff[t - 2]
    assert set(result.trades["exit_reason"]) == {"SMA-Kreuzung abwärts"}


def test_donchian_rules(daily: pd.DataFrame, settings: Settings) -> None:
    result = _run(DonchianBreakout, Timeframe.W1, daily, settings)
    frame = _frame(daily, Timeframe.W1)
    ind = DonchianBreakout().indicators(frame, DonchianBreakout.make_params())
    dates = pd.DatetimeIndex(frame["date"])
    entries = _entries(result)
    for submitted in entries["submitted"]:
        t = dates.get_loc(submitted)
        assert frame["close"].iloc[t] > ind["dc_upper"].iloc[t]
        assert ind["dc_upper"].iloc[t] == frame["high"].iloc[t - 20 : t].max()
    stops = result.orders.loc[result.orders["type"] == "TRAILING_STOP"]
    # Je Einstieg genau ein Trailing Stop, abgegeben auf der Kerze der Ausführung.
    assert list(stops["submitted"]) == list(entries["fill_date"])
    assert set(result.trades["exit_reason"]) == {"ATR-Trailing-Stop"}
    assert (stops["status"].iloc[:-1] == "FILLED").all()


def test_donchian_atr_mult_changes_exits(daily: pd.DataFrame, settings: Settings) -> None:
    tight = run_backtest(
        DonchianBreakout, {"atr_mult": 1.0}, "SYNTH", Timeframe.W1, START, END, settings, bars=daily
    )
    wide = _run(DonchianBreakout, Timeframe.W1, daily, settings)
    assert len(tight.trades) > len(wide.trades)
    assert tight.trades["exit_reason"].eq("ATR-Trailing-Stop").all()


def test_rsi2_rules(daily: pd.DataFrame, settings: Settings) -> None:
    result = _run(Rsi2MeanReversion, Timeframe.D3, daily, settings)
    frame = _frame(daily, Timeframe.D3)
    ind = Rsi2MeanReversion().indicators(frame, Rsi2MeanReversion.make_params())
    dates = pd.DatetimeIndex(frame["date"])
    for submitted in _entries(result)["submitted"]:
        t = dates.get_loc(submitted)
        assert ind["rsi"].iloc[t] < 10
        assert frame["close"].iloc[t] > ind["trend_sma"].iloc[t]
    sells = result.orders.loc[
        (result.orders["side"] == "SELL") & (result.orders["status"] == "FILLED")
    ]
    for submitted in sells["submitted"]:
        assert ind["rsi"].iloc[dates.get_loc(submitted)] > 70
    assert np.all(ind["entry_level"] == 10.0)
    assert np.all(ind["exit_level"] == 70.0)


def test_rsi2_custom_thresholds(daily: pd.DataFrame, settings: Settings) -> None:
    strict = run_backtest(
        Rsi2MeanReversion,
        {"entry_below": 5.0},
        "SYNTH",
        Timeframe.D3,
        START,
        END,
        settings,
        bars=daily,
    )
    default = _run(Rsi2MeanReversion, Timeframe.D3, daily, settings)
    assert 0 < len(strict.trades) < len(default.trades)
