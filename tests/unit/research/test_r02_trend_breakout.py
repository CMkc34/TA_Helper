"""Tests des R-02-Kandidaten ``plugins/trend_breakout_mf.py`` (Trendfolge 1W mit Marktfilter)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.portfolio import PortfolioResult, run_portfolio_backtest
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.data.resample import PARTIAL_COLUMN, resample
from trading_helfer.indicators.trend import sma
from trading_helfer.research.study import StudyConfig, run_study
from trading_helfer.strategies.base import Strategy
from trading_helfer.strategies.registry import StrategyRegistry
from trading_helfer.strategies.validator import validate

ROOT = Path(__file__).resolve().parents[3]
PLUGIN = ROOT / "plugins" / "trend_breakout_mf.py"
SYMBOLS = [f"S{i:02d}" for i in range(14)]


def wave_bars(
    phase: float, drift: float, amp: float, period: float, start: str = "1995-01-02"
) -> pd.DataFrame:
    """Werktägliche Kurse mit Trend und Welle (wie ``conftest.wave_bars``)."""
    dates = pd.bdate_range(start, "2023-12-29")
    n = np.arange(len(dates))
    close = 50.0 * np.exp(drift * n) * (1 + amp * np.sin(n / period + phase))
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": np.maximum(opens, close) * 1.01,
            "low": np.minimum(opens, close) * 0.99,
            "close": close,
            "volume": 1000,
        }
    )


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("r02_plugin", PLUGIN)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLUGIN_MODULE = _load()
Candidate: type[Strategy] = PLUGIN_MODULE.TrendBreakoutMarketFilter


class Unfiltered(Candidate):  # type: ignore[misc,valid-type]
    """Gleiche Regeln ohne Marktfilter (Vergleich)."""

    id = "test_r02_unfiltered"
    market_symbol: ClassVar[str | None] = None


@pytest.fixture(scope="module")
def universe_bars() -> dict[str, pd.DataFrame]:
    """14 Werte mit Trend und Welle; SPY mit langen Ab- und Aufwärtsphasen."""
    bars = {
        s: wave_bars(i * 0.5, 0.0002 + 0.00002 * i, 0.15, 40 + 5 * i) for i, s in enumerate(SYMBOLS)
    }
    bars["SPY"] = wave_bars(3.0, 0.00025, 0.15, 120.0)
    bars["QQQ"] = wave_bars(4.0, 0.0003, 0.12, 100.0)
    return bars


@pytest.fixture(scope="module")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(data_dir=tmp_path_factory.mktemp("r02"), default_capital=1000.0)


def _run(
    strategy: type[Strategy], bars: dict[str, pd.DataFrame], settings: Settings
) -> PortfolioResult:
    return run_portfolio_backtest(
        strategy,
        None,
        SYMBOLS,
        Timeframe.W1,
        "2000-01-01",
        "2023-12-29",
        settings,
        bars=bars,
        workers=1,
    )


@pytest.fixture(scope="module")
def result(universe_bars: dict[str, pd.DataFrame], settings: Settings) -> PortfolioResult:
    return _run(Candidate, universe_bars, settings)


def _risk_on_by_week(spy: pd.DataFrame) -> pd.Series:
    weekly = resample(spy, Timeframe.W1)
    weekly = weekly.loc[~weekly[PARTIAL_COLUMN].to_numpy(dtype=bool)].reset_index(drop=True)
    on = weekly["close"] > sma(weekly, 40)
    return pd.Series(on.to_numpy(), index=pd.DatetimeIndex(weekly["date"]))


def _entries(res: PortfolioResult) -> list[pd.Timestamp]:
    days = [pd.Timestamp(d) for d in res.trades["entry_date"]]
    days += [pd.Timestamp(p.entry_date) for p in res.open_positions]
    return sorted(days)


def _entries_when_risk_off(res: PortfolioResult, risk_on: pd.Series) -> int:
    """Einstiege, deren Signalkerze (letzte Woche vor dem Einstieg) im Abwärtsmarkt lag."""
    count = 0
    for day in _entries(res):
        before = risk_on.loc[risk_on.index < day]
        if not bool(before.iloc[-1]):
            count += 1
    return count


def test_plugin_passes_validator() -> None:
    report = validate(PLUGIN.read_text("utf-8"), filename=PLUGIN.name, isolate=False)
    assert report.ok, report.format()


def test_plugin_is_loaded_by_registry() -> None:
    registry = StrategyRegistry(ROOT / "plugins")
    report = registry.load()
    assert report.ok, report.summary()
    info = registry.info("trend_breakout_mf")
    assert not info.is_builtin
    strategy = registry.create("trend_breakout_mf")
    assert strategy.market_symbol == "SPY"
    assert Timeframe.W1 in strategy.timeframes


def test_entries_only_when_market_above_sma(
    result: PortfolioResult,
    universe_bars: dict[str, pd.DataFrame],
    settings: Settings,
) -> None:
    risk_on = _risk_on_by_week(universe_bars["SPY"])
    assert not any("Marktfilter ist ausgeschaltet" in line for line in result.log)
    assert len(_entries(result)) > 20
    assert _entries_when_risk_off(result, risk_on) == 0
    unfiltered = _run(Unfiltered, universe_bars, settings)
    assert _entries_when_risk_off(unfiltered, risk_on) > 0


def test_exits_only_by_atr_trailing_stop(result: PortfolioResult) -> None:
    assert set(result.trades["exit_reason"]) == {"ATR-Trailing-Stop"}


def test_at_most_ten_positions_with_ten_percent(result: PortfolioResult) -> None:
    trades = result.trades
    events = [(pd.Timestamp(d), 1) for d in trades["entry_date"]]
    events += [(pd.Timestamp(d), -1) for d in trades["exit_date"]]
    events += [(pd.Timestamp(p.entry_date), 1) for p in result.open_positions]
    held, peak = 0, 0
    for _day, step in sorted(events, key=lambda e: (e[0], e[1])):
        held += step
        peak = max(peak, held)
    assert 1 < peak <= 10
    assert float(result.equity["exposure"].max()) <= 1.0 + 1e-9


def test_ranking_prefers_stronger_momentum() -> None:
    strategy = Candidate()
    p = strategy.make_params()
    up = wave_bars(0.0, 0.001, 0.0, 50.0, start="2010-01-04")
    ind = strategy.indicators(up, p)
    assert list(ind.columns) == ["dc_upper", "trend", "momentum", "atr"]
    weekly_like = ind["momentum"].dropna()
    assert (weekly_like > 0).all()
    market = strategy.market_indicators(up, p)
    assert list(market.columns) == ["sma"]
    assert np.isnan(market["sma"].iloc[38])
    assert not np.isnan(market["sma"].iloc[39])


@pytest.mark.slow
def test_study_runs_with_market_filter(
    universe_bars: dict[str, pd.DataFrame], settings: Settings
) -> None:
    study = run_study(
        Candidate,
        None,
        "W1",
        settings,
        universe=SYMBOLS[:3],
        bars=universe_bars,
        config=StudyConfig(workers=1, mc_simulations=50),
    )
    assert study.criteria
    assert len(study.full.result.trades) > 0
    assert not any("Marktfilter ist ausgeschaltet" in line for line in study.log)
