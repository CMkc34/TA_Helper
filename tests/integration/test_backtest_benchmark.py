"""Benchmark: 25 Jahre eines Werts in 1W unter 1 Sekunde, in 1D unter 3 Sekunden."""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.engine import run_backtest
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.data.resample import resample
from trading_helfer.strategies.base import BarContext, Order, Param, Params, Sizing, Strategy

YEARS = 25
LIMITS = {Timeframe.W1: 1.0, Timeframe.D1: 3.0}


class SmaCross(Strategy):
    id = "bench_sma_cross"
    name = "SMA-Kreuzung"
    version = "1.0.0"
    description = "SMA-Kreuzung für den Benchmark."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        "fast": Param(10, 2, 50, 1, "schnell"),
        "slow": Param(30, 5, 200, 1, "langsam"),
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        close = bars["close"]
        return pd.DataFrame(
            {"fast": close.rolling(p.fast).mean(), "slow": close.rolling(p.slow).mean()}
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and ctx.crossed_above("fast", "slow"):
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        if ctx.position is not None and ctx.crossed_below("fast", "slow"):
            return ctx.exit()
        return []


@pytest.mark.slow
@pytest.mark.parametrize("timeframe", [Timeframe.W1, Timeframe.D1])
def test_25_years_single_symbol(tmp_path: Path, timeframe: Timeframe) -> None:
    dates = pd.bdate_range("2000-01-03", "2024-12-31")
    rng = np.random.default_rng(1)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(dates))))
    bars = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000,
        }
    )
    assert len(dates) >= YEARS * 250
    resample(bars.iloc[:30], timeframe)  # Börsenkalender einmalig laden (Cache)
    settings = Settings(data_dir=tmp_path)

    started = time.perf_counter()
    result = run_backtest(
        SmaCross(), None, "BENCH", timeframe, "2000-01-01", "2024-12-31", settings, bars=bars
    )
    elapsed = time.perf_counter() - started

    assert len(result.trades) > 10
    assert elapsed < LIMITS[timeframe], f"{timeframe.value}: {elapsed:.2f} s"
