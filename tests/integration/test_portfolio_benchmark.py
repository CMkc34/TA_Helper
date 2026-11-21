"""Benchmark: S&P 500 über 25 Jahre in 1W unter 60 Sekunden (B-04)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.portfolio import run_portfolio_backtest
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.data.resample import resample
from trading_helfer.data.universe import all_members
from trading_helfer.strategies.builtin.sma_cross import SmaCross

START, END = "2000-01-01", "2024-12-31"
LIMIT_SECONDS = 60.0


def _synthetic_universe(symbols: list[str]) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("1999-01-04", END)
    rng = np.random.default_rng(7)
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        close = 50.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, len(dates))))
        result[symbol] = pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1000,
            }
        )
    return result


@pytest.mark.slow
def test_sp500_25_years_weekly(tmp_path: Path) -> None:
    symbols = sorted(all_members("SP500", START, END))
    assert len(symbols) > 500
    bars = _synthetic_universe(symbols)
    resample(bars[symbols[0]].iloc[:30], Timeframe.W1)  # Börsenkalender laden (Cache)
    settings = Settings(data_dir=tmp_path)

    started = time.perf_counter()
    result = run_portfolio_backtest(
        SmaCross(),
        None,
        "SP500",
        Timeframe.W1,
        START,
        END,
        settings,
        max_positions=10,
        bars=bars,
    )
    elapsed = time.perf_counter() - started

    assert len(result.trades) > 10
    assert elapsed < LIMIT_SECONDS, f"{elapsed:.1f} s"
