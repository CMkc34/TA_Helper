"""DuckDB-Speicher (MarketStore)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_helfer.core.bars import BarsValidationError
from trading_helfer.data.schema import discover_migrations
from trading_helfer.data.storage import (
    LOADED_BAR_COLUMNS,
    MarketStore,
    StorageError,
)

EXPECTED_TABLES = {
    "schema_version",
    "instruments",
    "index_membership",
    "bars_daily",
    "corporate_actions",
    "fundamentals",
    "option_snapshots",
    "update_log",
    "backtest_runs",
    "backtest_trades",
    "backtest_equity",
    "signals",
    "provider_requests",
}
ALL_VERSIONS = [m.version for m in discover_migrations()]
LATEST_VERSION = ALL_VERSIONS[-1]


def make_bars(
    n: int = 5,
    *,
    start: str = "2024-01-02",
    base: float = 100.0,
    adj_ratio: float | None = 1.0,
) -> pd.DataFrame:
    """Synthetische Kursreihe mit konstantem Aufwärtstrend."""
    dates = pd.bdate_range(start, periods=n)
    close = base + np.arange(n, dtype="float64")
    data: dict[str, Any] = {
        "date": dates,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.arange(1, n + 1, dtype="int64") * 1000,
    }
    if adj_ratio is not None:
        data["adj_close"] = close * adj_ratio
    return pd.DataFrame(data)


def split_bars() -> pd.DataFrame:
    """Split 4:1 am 2024-01-08: Rohkurs fällt von 400 auf 100, bereinigt bleibt er bei 100."""
    dates = pd.bdate_range("2024-01-02", periods=8)  # 02.–05. vor, 08.–11. nach dem Split
    close = np.array([400.0, 404.0, 408.0, 412.0, 104.0, 105.0, 106.0, 107.0])
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 4.0,
            "low": close - 4.0,
            "close": close,
            "volume": [100, 100, 100, 100, 400, 400, 400, 400],
            "adj_close": np.where(dates < "2024-01-08", close / 4.0, close),
        }
    )


@pytest.fixture
def store() -> Iterator[MarketStore]:
    with MarketStore(":memory:") as market_store:
        yield market_store


def table_names(store: MarketStore) -> set[str]:
    rows = store.connection.execute("SELECT table_name FROM duckdb_tables()").fetchall()
    return {str(row[0]) for row in rows}


# ---------------------------------------------------------------- Migrationen


def test_new_database_has_all_tables(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "market.duckdb"
    with MarketStore(db) as store:
        assert store.applied_migrations == ALL_VERSIONS
        assert store.schema_version() == LATEST_VERSION
        assert table_names(store) == EXPECTED_TABLES
        assert store.path == str(db)
    assert db.exists()


def test_reopening_runs_no_migration_twice(tmp_path: Path) -> None:
    db = tmp_path / "market.duckdb"
    with MarketStore(db) as store:
        store.upsert_bars("AAPL", make_bars())
    with MarketStore(db) as store:
        assert store.applied_migrations == []
        assert store.schema_version() == LATEST_VERSION
        rows = store.connection.execute("SELECT count(*) FROM schema_version").fetchall()
        assert rows[0][0] == len(ALL_VERSIONS)
        assert len(store.load_bars("AAPL")) == 5


def test_read_only_skips_migrations(tmp_path: Path) -> None:
    db = tmp_path / "market.duckdb"
    MarketStore(db).close()
    with MarketStore(db, read_only=True) as store:
        assert store.applied_migrations == []
        assert store.schema_version() == LATEST_VERSION


def test_new_migration_is_applied_on_next_open(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_a.sql").write_text("CREATE TABLE a (x INTEGER);", encoding="utf-8")
    db = tmp_path / "m.duckdb"
    with MarketStore(db, migrations_dir=migrations) as store:
        assert store.applied_migrations == [1]
    (migrations / "002_b.sql").write_text("CREATE TABLE b (x INTEGER);", encoding="utf-8")
    with MarketStore(db, migrations_dir=migrations) as store:
        assert store.applied_migrations == [2]
        assert store.schema_version() == 2


INSERT_SIGNAL = (
    "INSERT INTO signals (scan_id, symbol, date, signal) "
    "VALUES ('s', 'A', DATE '2024-01-02', 'BUY')"
)


def insert_then_fail(store: MarketStore) -> None:
    with store.transaction() as conn:
        conn.execute(INSERT_SIGNAL)
        raise RuntimeError("Abbruch")


def test_transaction_rolls_back_on_error(store: MarketStore) -> None:
    with pytest.raises(RuntimeError, match="Abbruch"):
        insert_then_fail(store)
    assert store.connection.execute("SELECT count(*) FROM signals").fetchall() == [(0,)]

    with store.transaction() as conn:
        conn.execute(INSERT_SIGNAL)
    assert store.connection.execute("SELECT count(*) FROM signals").fetchall() == [(1,)]


def test_update_log_id_is_generated(store: MarketStore) -> None:
    conn = store.connection
    conn.execute("INSERT INTO update_log (started_at, provider) VALUES (now(), 'yfinance')")
    conn.execute("INSERT INTO update_log (started_at, provider) VALUES (now(), 'tiingo')")
    assert conn.execute("SELECT id FROM update_log ORDER BY id").fetchall() == [(1,), (2,)]


# ---------------------------------------------------------------- Kurse


def test_upsert_overwrites_existing_days_and_adds_new(store: MarketStore) -> None:
    store.upsert_bars("AAPL", make_bars(5, base=100.0), source="yfinance")
    # Überschneidung: 2024-01-05 und 2024-01-08 neu bewertet, 2024-01-09/10 neu.
    update = make_bars(4, start="2024-01-05", base=200.0)
    assert store.upsert_bars("AAPL", update, source="tiingo") == 4

    loaded = store.load_bars("AAPL", adjusted=False)
    assert len(loaded) == 7
    assert loaded["date"].tolist() == list(pd.bdate_range("2024-01-02", periods=7))
    assert loaded["close"].tolist() == [100.0, 101.0, 102.0, 200.0, 201.0, 202.0, 203.0]
    sources = store.connection.execute("SELECT source FROM bars_daily ORDER BY date").fetchall()
    assert [s[0] for s in sources] == ["yfinance"] * 3 + ["tiingo"] * 4


def test_split_example_adjusted_vs_raw(store: MarketStore) -> None:
    store.upsert_bars("SPLT", split_bars())

    raw = store.load_bars("SPLT", adjusted=False)
    adjusted = store.load_bars("SPLT")

    assert raw["close"].tolist() == [400.0, 404.0, 408.0, 412.0, 104.0, 105.0, 106.0, 107.0]
    assert adjusted["close"].tolist() == [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    assert adjusted["adj_factor"].tolist() == [0.25] * 4 + [1.0] * 4
    assert adjusted["high"].iloc[0] == pytest.approx(101.0)
    assert adjusted["low"].iloc[0] == pytest.approx(99.0)
    assert adjusted["open"].iloc[0] == pytest.approx(100.0)
    pd.testing.assert_series_equal(adjusted["volume"], raw["volume"])
    pd.testing.assert_series_equal(adjusted["adj_close"], raw["adj_close"])
    for column in ("open", "high", "low", "close"):
        np.testing.assert_allclose(adjusted[column], raw[column] * raw["adj_factor"])


def test_load_bars_schema_and_dtypes(store: MarketStore) -> None:
    store.upsert_bars("AAPL", make_bars())
    loaded = store.load_bars("AAPL")
    assert tuple(loaded.columns) == LOADED_BAR_COLUMNS
    assert str(loaded["date"].dtype) == "datetime64[ns]"
    assert str(loaded["volume"].dtype) == "int64"
    assert isinstance(loaded.index, pd.RangeIndex)


def test_load_bars_unknown_symbol_is_empty_with_columns(store: MarketStore) -> None:
    loaded = store.load_bars("NOPE")
    assert loaded.empty
    assert tuple(loaded.columns) == LOADED_BAR_COLUMNS
    assert str(loaded["date"].dtype) == "datetime64[ns]"
    assert str(loaded["close"].dtype) == "float64"


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2024-01-03", None, ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]),
        (None, dt.date(2024, 1, 3), ["2024-01-02", "2024-01-03"]),
        (pd.Timestamp("2024-01-04"), "2024-01-05", ["2024-01-04", "2024-01-05"]),
        ("2024-02-01", None, []),
    ],
)
def test_load_bars_date_range_is_inclusive(
    store: MarketStore, start: Any, end: Any, expected: list[str]
) -> None:
    store.upsert_bars("AAPL", make_bars())
    loaded = store.load_bars("AAPL", start=start, end=end)
    assert loaded["date"].dt.strftime("%Y-%m-%d").tolist() == expected


def test_missing_adj_close_means_factor_one(store: MarketStore) -> None:
    store.upsert_bars("AAPL", make_bars(adj_ratio=None))
    loaded = store.load_bars("AAPL")
    assert loaded["adj_factor"].tolist() == [1.0] * 5
    assert loaded["adj_close"].tolist() == loaded["close"].tolist()


def test_symbols_are_normalized(store: MarketStore) -> None:
    store.upsert_bars(" aapl ", make_bars())
    assert store.symbols() == ["AAPL"]
    assert len(store.load_bars("Aapl")) == 5
    assert store.last_date("aapl") == dt.date(2024, 1, 8)


def test_source_column_in_frame_wins(store: MarketStore) -> None:
    bars = make_bars(2)
    bars["source"] = ["csv", None]
    store.upsert_bars("AAPL", bars, source="fallback")
    rows = store.connection.execute("SELECT source FROM bars_daily ORDER BY date").fetchall()
    assert [r[0] for r in rows] == ["csv", "fallback"]


def test_last_date_without_bars_is_none(store: MarketStore) -> None:
    assert store.last_date("AAPL") is None


def test_upsert_empty_bars_writes_nothing(store: MarketStore) -> None:
    assert store.upsert_bars("AAPL", make_bars().iloc[0:0]) == 0
    assert store.symbols() == []


@pytest.mark.parametrize("symbol", ["", "   "])
def test_invalid_symbol_is_rejected(store: MarketStore, symbol: str) -> None:
    with pytest.raises(StorageError, match="Symbol"):
        store.upsert_bars(symbol, make_bars())


def test_invalid_bars_are_rejected(store: MarketStore) -> None:
    with pytest.raises(BarsValidationError):
        store.upsert_bars("AAPL", make_bars().drop(columns="close"))


def test_zero_close_is_rejected(store: MarketStore) -> None:
    bars = make_bars()
    bars.loc[2, "close"] = 0.0
    with pytest.raises(StorageError, match="Schlusskurs 0"):
        store.upsert_bars("AAPL", bars)
    assert store.symbols() == []


@pytest.mark.parametrize("value", ["kein Datum", "2024-13-45"])
def test_invalid_start_date_is_rejected(store: MarketStore, value: str) -> None:
    with pytest.raises(StorageError, match="Ungültiges Datum"):
        store.load_bars("AAPL", start=value)


def test_missing_date_is_rejected(store: MarketStore) -> None:
    with pytest.raises(StorageError, match="Ungültiges Datum"):
        store.load_bars("AAPL", end=pd.NaT)  # type: ignore[arg-type]


# ---------------------------------------------------------------- Stammdaten


def test_instruments_upsert_keeps_unmentioned_columns(store: MarketStore) -> None:
    first = pd.DataFrame(
        {
            "symbol": ["aapl", "msft"],
            "name": ["Apple", "Microsoft"],
            "is_active": [True, True],
            "first_date": ["1990-01-02", dt.date(1990, 1, 2)],
            "source": ["yfinance", "yfinance"],
        }
    )
    assert store.upsert_instruments(first) == 2
    store.upsert_instruments(pd.DataFrame({"symbol": ["AAPL"], "sector": ["Technology"]}))

    apple = store.get_instrument("aapl")
    assert apple == {
        "symbol": "AAPL",
        "name": "Apple",
        "exchange": None,
        "sector": "Technology",
        "industry": None,
        "currency": None,
        "is_active": True,
        "first_date": dt.date(1990, 1, 2),
        "last_date": None,
        "source": "yfinance",
    }
    assert store.get_instrument("MSFT") is not None
    assert store.get_instrument("XYZ") is None


def test_instruments_only_symbol_column(store: MarketStore) -> None:
    store.upsert_instruments(pd.DataFrame({"symbol": ["AAPL"]}))
    store.upsert_instruments(pd.DataFrame({"symbol": ["AAPL"]}))
    instrument = store.get_instrument("AAPL")
    assert instrument is not None
    assert instrument["name"] is None


def test_instruments_empty_frame(store: MarketStore) -> None:
    assert store.upsert_instruments(pd.DataFrame({"symbol": []})) == 0


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"name": ["Apple"]}), "symbol fehlt"),
        (pd.DataFrame({"symbol": ["A"], "color": ["rot"]}), "unbekannte Spalte"),
        (pd.DataFrame({"symbol": ["A", "a"]}), "doppelte"),
        (pd.DataFrame({"symbol": ["A"], "first_date": ["kein Datum"]}), "Datumswerte"),
    ],
)
def test_instruments_invalid_input(store: MarketStore, frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(StorageError, match=message):
        store.upsert_instruments(frame)


# ---------------------------------------------------------------- Corporate Actions


def test_actions_upsert_and_load(store: MarketStore) -> None:
    actions = pd.DataFrame(
        {
            "date": ["2024-01-08", "2023-11-10"],
            "type": ["Split", " DIVIDEND "],
            "value": [4.0, 0.24],
        }
    )
    assert store.upsert_actions("aapl", actions) == 2
    store.upsert_actions(
        "AAPL", pd.DataFrame({"date": ["2023-11-10"], "type": ["dividend"], "value": [0.25]})
    )

    loaded = store.load_actions("AAPL")
    assert loaded["date"].dt.strftime("%Y-%m-%d").tolist() == ["2023-11-10", "2024-01-08"]
    assert loaded["type"].tolist() == ["dividend", "split"]
    assert loaded["value"].tolist() == [0.25, 4.0]


def test_actions_empty_frame(store: MarketStore) -> None:
    assert store.upsert_actions("AAPL", pd.DataFrame({"date": [], "type": [], "value": []})) == 0
    assert store.load_actions("AAPL").empty


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"date": ["2024-01-02"], "type": ["split"]}), "Pflichtspalte"),
        (pd.DataFrame({"date": ["2024-01-02"], "type": ["merger"], "value": [1.0]}), "unbekannte"),
        (pd.DataFrame({"date": ["2024-01-02"], "type": ["split"], "value": [None]}), "fehlenden"),
        (
            pd.DataFrame({"date": ["2024-01-02"] * 2, "type": ["split"] * 2, "value": [2.0, 3.0]}),
            "doppelte",
        ),
    ],
)
def test_actions_invalid_input(store: MarketStore, frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(StorageError, match=message):
        store.upsert_actions("AAPL", frame)


# ---------------------------------------------------------------- Fundamentaldaten


def fundamentals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period_end": ["2023-09-30", "2023-12-31", "2023-12-31", "2024-03-31"],
            "period_type": ["Q", "Q", "fy", "Q"],
            "metric": ["eps", "eps", "eps", "eps"],
            "value": [1.0, 2.0, 6.0, 1.5],
            "reported_date": ["2023-11-02", "2024-02-01", "2024-02-01", None],
        }
    )


def test_load_fundamentals_as_of_returns_only_reported_values(store: MarketStore) -> None:
    store.upsert_fundamentals("AAPL", fundamentals_frame(), source="yfinance")

    everything = store.load_fundamentals("AAPL")
    assert len(everything) == 4
    assert list(everything.columns) == [
        "period_end",
        "period_type",
        "metric",
        "value",
        "reported_date",
        "source",
    ]

    before = store.load_fundamentals("AAPL", as_of="2024-01-31")
    assert before["period_end"].dt.strftime("%Y-%m-%d").tolist() == ["2023-09-30"]

    on_day = store.load_fundamentals("AAPL", as_of=dt.date(2024, 2, 1))
    assert on_day["period_type"].tolist() == ["Q", "FY", "Q"]
    assert (on_day["reported_date"] <= pd.Timestamp("2024-02-01")).all()
    # Wert ohne reported_date fehlt bei Stichtag (Veröffentlichung unbekannt).
    later = store.load_fundamentals("AAPL", as_of="2030-01-01")
    assert pd.Timestamp("2024-03-31") not in later["period_end"].tolist()
    assert set(later["source"]) == {"yfinance"}


def test_fundamentals_upsert_overwrites(store: MarketStore) -> None:
    store.upsert_fundamentals("AAPL", fundamentals_frame())
    revised = fundamentals_frame().iloc[[0]].assign(value=1.1, source="tiingo")
    assert store.upsert_fundamentals("AAPL", revised) == 1
    loaded = store.load_fundamentals("AAPL")
    first = loaded.iloc[0]
    assert first["value"] == pytest.approx(1.1)
    assert first["source"] == "tiingo"


def test_fundamentals_empty(store: MarketStore) -> None:
    assert store.upsert_fundamentals("AAPL", fundamentals_frame().iloc[0:0]) == 0
    loaded = store.load_fundamentals("AAPL")
    assert loaded.empty
    assert str(loaded["period_end"].dtype) == "datetime64[ns]"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"drop": "reported_date"}, "Pflichtspalte"),
        ({"period_type": "H1"}, "period_type"),
        ({"metric": None}, "fehlendem Schlüssel"),
        ({"duplicate": True}, "doppelte"),
    ],
)
def test_fundamentals_invalid_input(
    store: MarketStore, change: dict[str, Any], message: str
) -> None:
    frame = fundamentals_frame()
    if "drop" in change:
        frame = frame.drop(columns=change["drop"])
    elif "duplicate" in change:
        frame = pd.concat([frame, frame.iloc[[0]]])
    else:
        frame = frame.assign(**change)
    with pytest.raises(StorageError, match=message):
        store.upsert_fundamentals("AAPL", frame)


def test_timezone_aware_dates_are_accepted(store: MarketStore) -> None:
    actions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-08 00:00"]).tz_localize("America/New_York"),
            "type": ["split"],
            "value": [4.0],
        }
    )
    store.upsert_actions("AAPL", actions)
    assert store.load_actions("AAPL")["date"].tolist() == [pd.Timestamp("2024-01-08")]
