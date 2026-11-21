"""Tests für die Indexmitglieder (trading_helfer.data.universe)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from trading_helfer.data import universe
from trading_helfer.data.universe import (
    MEMBERSHIP_COLUMNS,
    SURVIVORSHIP_WARNING,
    UniverseError,
    all_members,
    load_history,
    members_on,
    membership_intervals,
    reliable_since,
    survivorship_risk,
)

# Kleine Historie: heute A, B, D, E.
#   2020-01-02: +C, -X     2021-06-01: +D, -C     2022-03-01: +E (ohne Entfernung)
CURRENT = "symbol\nA\nB\nD\nE\n"
CHANGES = (
    "date,added,removed,quelle\n2021-06-01,D,C,test\n2020-01-02,C,X,test\n2022-03-01,E,,test\n"
)


def _write(directory: Path, prefix: str, current: str, changes: str) -> Path:
    (directory / f"{prefix}_current.csv").write_text(current, encoding="utf-8")
    (directory / f"{prefix}_changes.csv").write_text(changes, encoding="utf-8")
    return directory


@pytest.fixture
def small(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fixture-Historie als Standardordner (Cache wird geleert)."""
    _write(tmp_path, "sp500", CURRENT, CHANGES)
    _write(tmp_path, "ndx100", "symbol\nA\n", "date,added,removed,quelle\n")
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    universe._load_default.cache_clear()
    yield tmp_path
    universe._load_default.cache_clear()


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        ("2026-01-01", {"A", "B", "D", "E"}),
        ("2022-03-01", {"A", "B", "D", "E"}),  # Aufnahme gilt ab dem Stichtag
        ("2022-02-28", {"A", "B", "D"}),
        ("2021-06-01", {"A", "B", "D"}),
        ("2021-05-31", {"A", "B", "C"}),
        ("2020-01-02", {"A", "B", "C"}),
        ("2020-01-01", {"A", "B", "X"}),
        ("1990-01-01", {"A", "B", "X"}),  # vor der ersten Änderung: früheste Liste
    ],
)
def test_members_on(small: Path, day: str, expected: set[str]) -> None:
    assert members_on("SP500", day) == expected


def test_members_on_accepts_date_types(small: Path) -> None:
    assert members_on("sp500", dt.date(2021, 5, 31)) == {"A", "B", "C"}
    assert members_on("SP500", pd.Timestamp("2021-05-31")) == {"A", "B", "C"}
    assert members_on("SP500", dt.datetime(2021, 5, 31, 12)) == {"A", "B", "C"}


def test_all_members(small: Path) -> None:
    assert all_members("SP500", "2019-01-01", "2026-01-01") == {"A", "B", "C", "D", "E", "X"}
    assert all_members("SP500", "2020-01-02", "2021-06-01") == {"A", "B", "C", "D"}
    assert all_members("SP500", "2023-01-01", "2023-12-31") == {"A", "B", "D", "E"}
    with pytest.raises(UniverseError, match="vor dem Start"):
        all_members("SP500", "2023-01-01", "2022-01-01")


def test_survivorship(small: Path) -> None:
    assert reliable_since("SP500") == dt.date(2020, 1, 2)
    assert survivorship_risk("SP500", "2020-01-01")
    assert not survivorship_risk("SP500", "2020-01-02")
    assert reliable_since("NDX100") == dt.date.max
    assert survivorship_risk("NDX100", "2024-01-01")
    assert "Survivorship-Risiko" in SURVIVORSHIP_WARNING


def test_membership_intervals(small: Path) -> None:
    frame = membership_intervals("SP500")
    assert tuple(frame.columns) == MEMBERSHIP_COLUMNS
    rows = {r.symbol: (r.start_date, r.end_date) for r in frame.itertuples()}
    base = dt.date(2020, 1, 1)
    assert rows == {
        "A": (base, None),
        "B": (base, None),
        "C": (dt.date(2020, 1, 2), dt.date(2021, 6, 1)),
        "D": (dt.date(2021, 6, 1), None),
        "E": (dt.date(2022, 3, 1), None),
        "X": (base, dt.date(2020, 1, 2)),
    }
    assert set(frame["index_code"]) == {"SP500"}
    ndx = membership_intervals("NDX100")
    assert ndx["symbol"].tolist() == ["A"]
    assert ndx["start_date"].isna().all()


def test_load_history_directory_and_cache(small: Path) -> None:
    history = load_history("SP500", small)
    assert history.first_change == dt.date(2020, 1, 2)
    assert len(history.changes) == 3
    assert load_history("SP500") is load_history(" sp500 ")


@pytest.mark.parametrize(
    ("current", "changes", "message"),
    [
        (CURRENT, "date,added,removed,quelle\n2020-13-01,A,,t\n", "ungültiges Datum"),
        (CURRENT, "date,added,removed,quelle\n2020-01-01,,,t\n", "weder added noch removed"),
        (CURRENT, "date,added,quelle\n", "Spalten fehlen"),
        (CURRENT, "date,added,removed,quelle\n2020-01-01,Z,,t\n", "nicht Mitglied"),
        (CURRENT, "date,added,removed,quelle\n2020-01-01,,A,t\n", "weiterhin Mitglied"),
    ],
)
def test_invalid_data(tmp_path: Path, current: str, changes: str, message: str) -> None:
    _write(tmp_path, "sp500", current, changes)
    with pytest.raises(UniverseError, match=message):
        load_history("SP500", tmp_path)


def test_unknown_index_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(UniverseError, match="Unbekannter Index"):
        load_history("DAX")
    with pytest.raises(UniverseError, match="fehlt"):
        load_history("SP500", tmp_path)
    with pytest.raises(UniverseError, match="Ungültiges Datum"):
        load_history("SP500").members_on("kein Datum")


# --- echte CSV-Dateien -----------------------------------------------------------------


def _month_starts(start: dt.date, end: dt.date) -> list[dt.date]:
    months = pd.date_range(start, end, freq="MS")
    return [stamp.date() for stamp in months]


def test_sp500_plausible_member_count() -> None:
    """Für jeden Monatsersten ab 2000: 495–510 Mitglieder (2000–2004: 490–510).

    Die freie Datenbasis enthält für 2000–2004 nur 491–494 Werte (siehe
    ``data/universe/SOURCES.md``); die Grenze dafür ist mit der Person abgestimmt.
    """
    history = load_history("SP500")
    last = history.changes[-1].date
    outside = []
    for day in _month_starts(dt.date(2000, 1, 1), last):
        low = 490 if day.year < 2005 else 495
        count = len(history.members_on(day))
        if not low <= count <= 510:
            outside.append((day, count))
    assert outside == []


def test_ndx100_plausible_member_count() -> None:
    history = load_history("NDX100")
    for day in _month_starts(dt.date(2003, 6, 1), history.changes[-1].date):
        # mehrere Aktiengattungen (GOOG/GOOGL, FOX/FOXA …) erhöhen die Zahl
        assert 98 <= len(history.members_on(day)) <= 110, day


def test_real_data_known_facts() -> None:
    assert "META" in members_on("SP500", "2023-01-03")
    assert "FB" in members_on("SP500", "2015-01-02")
    assert "AAPL" in members_on("NDX100", "2010-01-04")
    assert reliable_since("SP500") == dt.date(2000, 1, 1)
    assert reliable_since("NDX100") == dt.date(2003, 5, 1)
    assert survivorship_risk("SP500", "1995-01-01")
    assert not survivorship_risk("SP500", "2005-01-01")
    current = load_history("SP500").current
    assert members_on("SP500", dt.date.today()) == set(current)
    assert len(all_members("SP500", "2000-01-01", dt.date.today())) > len(current)
