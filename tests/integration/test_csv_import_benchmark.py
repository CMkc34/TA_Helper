"""Benchmark: 3 Mio. Zeilen per CSV-Import in unter 60 Sekunden (D-04)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_helfer.data.providers.csv_import import import_csv
from trading_helfer.data.storage import MarketStore

SYMBOLS = 500
DAYS = 6_000
LIMIT_SECONDS = 60.0


@pytest.mark.slow
def test_import_3_million_rows_under_60_seconds(tmp_path: Path) -> None:
    dates = pd.bdate_range("2000-01-03", periods=DAYS).strftime("%Y-%m-%d")
    rng = np.random.default_rng(7)
    close = (50.0 + rng.normal(0.0, 1.0, (SYMBOLS, DAYS)).cumsum(axis=1)).clip(1.0, None)
    frame = pd.DataFrame(
        {
            "Date": np.tile(dates, SYMBOLS),
            "Symbol": np.repeat([f"S{i:03d}" for i in range(SYMBOLS)], DAYS),
            "Open": close.ravel().round(4),
            "High": (close + 1.0).ravel().round(4),
            "Low": (close - 0.5).clip(0.5, None).ravel().round(4),
            "Close": close.ravel().round(4),
            "Adj Close": (close * 0.9).ravel().round(4),
            "Volume": np.full(SYMBOLS * DAYS, 1_000, dtype="int64"),
        }
    )
    source = tmp_path / "kaggle.csv"
    frame.to_csv(source, index=False)

    with MarketStore(tmp_path / "bench.duckdb") as store:
        started = time.perf_counter()
        report = import_csv(store, source)
        elapsed = time.perf_counter() - started
        count = store.connection.execute("SELECT count(*) FROM bars_daily").fetchall()[0][0]

    assert report.rows_imported == SYMBOLS * DAYS == count
    assert report.errors == []
    assert elapsed < LIMIT_SECONDS, f"Import dauerte {elapsed:.2f} s"
