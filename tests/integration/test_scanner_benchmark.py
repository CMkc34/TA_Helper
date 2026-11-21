"""Benchmark: Scan von 500 Werten auf 1W in unter 20 Sekunden (S-04)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.core.types import Timeframe
from trading_helfer.data.storage import MarketStore
from trading_helfer.scanner.scan import scan
from trading_helfer.strategies.base import BarContext, Order, Param, Params, Sizing, Strategy

SYMBOLS = 500
DAYS = 9_000  # rund 36 Jahre, also Historie ab 1990
LIMIT_SECONDS = 20.0


class SmaCross(Strategy):
    id = "bench_sma_cross"
    name = "SMA-Kreuzung"
    version = "1.0.0"
    description = "Benchmark"
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        "fast": Param(10, 2, 50, 1, "schnell"),
        "slow": Param(30, 10, 200, 1, "langsam"),
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
def test_scan_500_symbols_weekly_under_20_seconds(tmp_path: Path) -> None:
    dates = pd.bdate_range("1990-01-02", periods=DAYS)
    rng = np.random.default_rng(7)
    symbols = [f"S{i:03d}" for i in range(SYMBOLS)]
    with MarketStore(tmp_path / "bench.duckdb") as store:
        for symbol in symbols:
            close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, DAYS)).clip(-90.0, None)
            bars = pd.DataFrame(
                {
                    "date": dates,
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": np.full(DAYS, 1_000, dtype="int64"),
                }
            )
            store.upsert_bars(symbol, bars)

        started = time.perf_counter()
        result = scan(SmaCross, None, symbols, Timeframe.W1, store=store)
        elapsed = time.perf_counter() - started

    print(f"Scan: {elapsed:.2f} s")
    assert not result.attrs["skipped"]
    assert elapsed < LIMIT_SECONDS, f"Scan dauerte {elapsed:.2f} s"
