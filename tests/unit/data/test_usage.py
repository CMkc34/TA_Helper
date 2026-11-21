"""Anfragezähler für Kontingente (Speicher und DuckDB)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from trading_helfer.data.providers.usage import (
    DuckDBRequestCounter,
    MemoryRequestCounter,
    QuotaWindow,
    RequestCounter,
    next_free_slot,
)

MIGRATION = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "trading_helfer"
    / "data"
    / "migrations"
    / "002_provider_requests.sql"
)
NOW = datetime(2024, 1, 15, 12, 0, 0)
HOURLY = QuotaWindow(limit=2, period=timedelta(hours=1), label="pro Stunde")
DAILY = QuotaWindow(limit=3, period=timedelta(days=1), label="pro Tag")


def open_db(path: Path | str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path))
    con.execute(MIGRATION.read_text(encoding="utf-8"))
    return con


@pytest.fixture(params=["memory", "duckdb"])
def counter(request: pytest.FixtureRequest) -> Iterator[RequestCounter]:
    if request.param == "memory":
        yield MemoryRequestCounter()
    else:
        con = open_db()
        yield DuckDBRequestCounter(con)
        con.close()


def test_counters_satisfy_protocol(counter: RequestCounter) -> None:
    assert isinstance(counter, RequestCounter)


def test_record_and_read_sorted_per_provider(counter: RequestCounter) -> None:
    counter.record("tiingo", NOW)
    counter.record("tiingo", NOW - timedelta(minutes=5))
    counter.record("other", NOW)
    assert counter.timestamps_since("tiingo", NOW - timedelta(hours=1)) == [
        NOW - timedelta(minutes=5),
        NOW,
    ]
    assert counter.timestamps_since("tiingo", NOW - timedelta(minutes=1)) == [NOW]
    assert counter.timestamps_since("missing", NOW - timedelta(days=1)) == []


def test_old_entries_are_pruned(counter: RequestCounter) -> None:
    counter.record("tiingo", NOW - timedelta(days=3))
    counter.record("tiingo", NOW)
    assert counter.timestamps_since("tiingo", NOW - timedelta(days=10)) == [NOW]


def test_next_free_slot_allows_below_limits(counter: RequestCounter) -> None:
    counter.record("tiingo", NOW - timedelta(minutes=10))
    assert next_free_slot(counter, "tiingo", (HOURLY, DAILY), NOW) is None
    assert next_free_slot(counter, "tiingo", (), NOW) is None


def test_next_free_slot_hourly(counter: RequestCounter) -> None:
    counter.record("tiingo", NOW - timedelta(minutes=50))
    counter.record("tiingo", NOW - timedelta(minutes=20))
    assert next_free_slot(counter, "tiingo", (HOURLY,), NOW) == (
        NOW + timedelta(minutes=10),
        HOURLY,
    )
    assert next_free_slot(counter, "tiingo", (HOURLY,), NOW + timedelta(minutes=10)) is None


def test_next_free_slot_picks_longest_block(counter: RequestCounter) -> None:
    counter.record("tiingo", NOW - timedelta(hours=5))
    counter.record("tiingo", NOW - timedelta(minutes=30))
    counter.record("tiingo", NOW - timedelta(minutes=10))
    # Stunde frei in 30 min, Tag erst in 19 h → Tag blockiert länger.
    assert next_free_slot(counter, "tiingo", (HOURLY, DAILY), NOW) == (
        NOW - timedelta(hours=5) + timedelta(days=1),
        DAILY,
    )


def test_duckdb_counter_survives_restart(tmp_path: Path) -> None:
    db = tmp_path / "market.duckdb"
    con = open_db(db)
    DuckDBRequestCounter(con).record("tiingo", NOW)
    con.close()

    con = duckdb.connect(str(db))
    try:
        assert DuckDBRequestCounter(con).timestamps_since("tiingo", NOW - timedelta(hours=1)) == [
            NOW
        ]
    finally:
        con.close()


def test_migration_is_idempotent() -> None:
    con = open_db()
    con.execute(MIGRATION.read_text(encoding="utf-8"))
    assert con.execute("SELECT count(*) FROM provider_requests").fetchone() == (0,)
    con.close()
