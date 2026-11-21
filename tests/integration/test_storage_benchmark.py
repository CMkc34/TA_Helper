"""Benchmark: 500 Symbole × 6.500 Tage aus dem Speicher laden."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_helfer.data.storage import MarketStore

SYMBOLS = 500
DAYS = 6_500
LIMIT_SECONDS = 5.0


@pytest.mark.slow
def test_load_500_symbols_x_6500_days_under_5_seconds(tmp_path: Path) -> None:
    dates = pd.bdate_range("1999-01-04", periods=DAYS)
    rng = np.random.default_rng(42)
    with MarketStore(tmp_path / "bench.duckdb") as store:
        for i in range(SYMBOLS):
            close = 50.0 + np.cumsum(rng.normal(0.0, 1.0, DAYS)).clip(-40.0, None)
            bars = pd.DataFrame(
                {
                    "date": dates,
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": np.full(DAYS, 1_000, dtype="int64"),
                    "adj_close": close * 0.9,
                }
            )
            store.upsert_bars(f"S{i:03d}", bars)

    with MarketStore(tmp_path / "bench.duckdb", read_only=True) as store:
        started = time.perf_counter()
        total = sum(len(store.load_bars(f"S{i:03d}")) for i in range(SYMBOLS))
        elapsed = time.perf_counter() - started

    assert total == SYMBOLS * DAYS
    assert elapsed < LIMIT_SECONDS, f"Laden dauerte {elapsed:.2f} s"
