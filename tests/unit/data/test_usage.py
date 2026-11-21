"""Anfragezähler für Kontingente (Speicher und DuckDB)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from trading_helfer.data.providers.usage import (
    DuckDBRequestCounter,
    MemoryRequestCounter,
    QuotaWindow,
    RequestCounter,
    next_free_slot,
)
from trading_helfer.data.schema import discover_migrations
from trading_helfer.data.storage import MarketStore

NOW = datetime(2024, 1, 15, 12, 0, 0)
HOURLY = QuotaWindow(limit=2, period=timedelta(hours=1), label="pro Stunde")
DAILY = QuotaWindow(limit=3, period=timedelta(days=1), label="pro Tag")


@pytest.fixture(params=["memory", "duckdb"])
def counter(request: pytest.FixtureRequest) -> Iterator[RequestCounter]:
    if request.param == "memory":
        yield MemoryRequestCounter()
    else:
        with MarketStore(":memory:") as store:
            yield DuckDBRequestCounter(store.connection)


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
    with MarketStore(db) as store:
        assert 2 in store.applied_migrations
        DuckDBRequestCounter(store.connection).record("tiingo", NOW)

    with MarketStore(db) as store:
        assert store.applied_migrations == []
        counter = DuckDBRequestCounter(store.connection)
        assert counter.timestamps_since("tiingo", NOW - timedelta(hours=1)) == [NOW]


def test_migration_follows_numbering() -> None:
    names = [(m.version, m.name) for m in discover_migrations()]
    assert (2, "provider_requests") in names
