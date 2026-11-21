"""Datenqualitätsprüfung (D-08) mit synthetischen Reihen."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_helfer.app import main
from trading_helfer.core import config
from trading_helfer.data.quality import (
    REPORT_COLUMNS,
    Check,
    Issue,
    Severity,
    check_all,
    check_bars,
    check_membership,
    check_symbol,
    issues_to_frame,
)
from trading_helfer.data.storage import MarketStore

# 2024-01-02 bis 2024-01-31: 21 XNYS-Handelstage (2024-01-15 ist Martin Luther King Day).
START = "2024-01-02"
END = "2024-01-31"


def sessions(start: str = START, end: str = END) -> pd.DatetimeIndex:
    """XNYS-Handelstage ohne Kalenderabhängigkeit im Test: Werktage ohne Feiertage 2024."""
    holidays = pd.to_datetime(["2024-01-15", "2024-02-19", "2024-03-29"])
    days = pd.bdate_range(start, end)
    return days.difference(holidays)


def clean_bars(start: str = START, end: str = END, base: float = 100.0) -> pd.DataFrame:
    """Sägezahn um ``base`` ohne Lücken, Sprünge oder ungültige Kurse."""
    dates = sessions(start, end)
    n = len(dates)
    close = base + (np.arange(n) % 5).astype("float64")
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000, dtype="int64"),
        }
    )


def checks(issues: list[Issue]) -> list[Check]:
    return [i.check for i in issues]


def no_actions() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime([]), "type": [], "value": []})


# ---------------------------------------------------------------------- saubere Daten


def test_clean_series_reports_nothing() -> None:
    assert check_bars("AAA", clean_bars(), no_actions()) == []


def test_empty_series_reports_nothing() -> None:
    assert check_bars("AAA", clean_bars().iloc[0:0]) == []


def test_single_bar_reports_nothing() -> None:
    assert check_bars("AAA", clean_bars().iloc[:1]) == []


# ---------------------------------------------------------------------- Lücken


def test_gap_detected_and_grouped() -> None:
    bars = clean_bars()
    drop = pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-22"])
    bars = bars[~bars["date"].isin(drop)]
    issues = check_bars("AAA", bars)
    assert checks(issues) == [Check.GAP, Check.GAP]
    first, second = issues
    assert first.date == dt.date(2024, 1, 8)
    assert first.severity is Severity.WARNING
    assert "3 Handelstage fehlen" in first.detail
    assert "2024-01-10" in first.detail
    assert second.date == dt.date(2024, 1, 22)
    assert second.detail == "1 Handelstag fehlt (2024-01-22)."


def test_holiday_and_weekend_are_no_gap() -> None:
    # 2024-01-15 (Feiertag) und Wochenenden fehlen in clean_bars bereits.
    assert Check.GAP not in checks(check_bars("AAA", clean_bars()))


# ---------------------------------------------------------------------- High < Low


def test_high_below_low_detected() -> None:
    bars = clean_bars()
    bars.loc[3, ["high", "low"]] = [90.0, 95.0]
    issues = check_bars("AAA", bars)
    assert checks(issues) == [Check.HIGH_LOW]
    assert issues[0].date == bars["date"].iloc[3].date()
    assert issues[0].severity is Severity.ERROR
    assert issues[0].detail == "Hoch 90 liegt unter Tief 95."


def test_high_equal_low_is_ok() -> None:
    bars = clean_bars()
    bars.loc[3, ["open", "high", "low", "close"]] = 101.0
    assert check_bars("AAA", bars) == []


# ---------------------------------------------------------------------- Kurse ≤ 0


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_non_positive_price_detected(column: str) -> None:
    bars = clean_bars()
    bars.loc[5, column] = 0.0 if column != "low" else -1.0
    issues = [i for i in check_bars("AAA", bars) if i.check is Check.NON_POSITIVE]
    assert len(issues) == 1
    assert issues[0].date == bars["date"].iloc[5].date()
    assert issues[0].severity is Severity.ERROR
    assert column in issues[0].detail


def test_small_positive_price_is_ok() -> None:
    bars = clean_bars(base=0.01)
    bars["low"] = bars["close"] * 0.5
    bars["open"] = bars["close"]
    # Sägezahn 0.01 … 4.01 springt stark; nur die Kurs-Prüfung ist hier relevant.
    assert Check.NON_POSITIVE not in checks(check_bars("AAA", bars))


def test_non_positive_close_does_not_trigger_jump() -> None:
    bars = clean_bars()
    bars.loc[5, "close"] = 0.0
    assert Check.JUMP not in checks(check_bars("AAA", bars))


# ---------------------------------------------------------------------- Sprünge


def test_jump_without_action_detected() -> None:
    bars = clean_bars()
    bars.loc[10:, ["open", "high", "low", "close"]] *= 0.25
    issues = check_bars("AAA", bars, no_actions())
    assert checks(issues) == [Check.JUMP]
    assert issues[0].date == bars["date"].iloc[10].date()
    assert issues[0].severity is Severity.WARNING
    assert "-7" in issues[0].detail
    assert "ohne Corporate Action" in issues[0].detail


def test_upward_jump_detected() -> None:
    bars = clean_bars()
    bars.loc[10:, ["open", "high", "low", "close"]] *= 2.0
    assert checks(check_bars("AAA", bars)) == [Check.JUMP]


def test_split_4_to_1_explains_jump() -> None:
    bars = clean_bars()
    bars.loc[10:, ["open", "high", "low", "close"]] *= 0.25
    split_day = bars["date"].iloc[10]
    actions = pd.DataFrame({"date": [split_day], "type": ["SPLIT"], "value": [4.0]})
    assert check_bars("AAA", bars, actions) == []


def test_action_between_bars_explains_jump() -> None:
    # Fehlt der Split-Tag selbst, erklärt eine Action zwischen den beiden Kursen den Sprung.
    bars = clean_bars()
    bars.loc[10:, ["open", "high", "low", "close"]] *= 0.25
    split_day = bars["date"].iloc[10]
    bars = bars.drop(index=10).reset_index(drop=True)
    actions = pd.DataFrame({"date": [split_day], "type": ["SPLIT"], "value": [4.0]})
    assert checks(check_bars("AAA", bars, actions)) == [Check.GAP]


def test_action_elsewhere_does_not_explain_jump() -> None:
    bars = clean_bars()
    bars.loc[10:, ["open", "high", "low", "close"]] *= 0.25
    other_day = bars["date"].iloc[3]
    actions = pd.DataFrame({"date": [other_day], "type": ["SPLIT"], "value": [4.0]})
    assert checks(check_bars("AAA", bars, actions)) == [Check.JUMP]


def test_jump_below_threshold_is_ok() -> None:
    bars = clean_bars()
    bars.loc[10:, ["open", "high", "low", "close"]] *= 1.45
    assert check_bars("AAA", bars) == []
    assert checks(check_bars("AAA", bars, jump_threshold=0.3)) == [Check.JUMP]


def test_unsorted_input_is_sorted() -> None:
    bars = clean_bars().iloc[::-1].reset_index(drop=True)
    assert check_bars("AAA", bars) == []


# ---------------------------------------------------------------------- Indexmitglieder


def membership(start: str, end: str | None, index_code: str = "SP500") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index_code": [index_code],
            "start_date": [pd.Timestamp(start).date()],
            "end_date": [pd.Timestamp(end).date() if end else None],
        }
    )


def test_former_member_without_data_detected() -> None:
    issues = check_membership("OLD", clean_bars().iloc[0:0], membership("2024-01-02", "2024-01-31"))
    assert checks(issues) == [Check.MISSING_DATA]
    assert issues[0].severity is Severity.ERROR
    assert issues[0].date == dt.date(2024, 1, 2)
    assert "SP500" in issues[0].detail
    assert "2024-01-02 bis 2024-01-31" in issues[0].detail


def test_member_with_full_data_is_ok() -> None:
    assert check_membership("AAA", clean_bars(), membership("2024-01-02", "2024-01-31")) == []


def test_member_with_partial_data_warns() -> None:
    bars = clean_bars(start="2024-01-02", end="2024-01-12")
    issues = check_membership("AAA", bars, membership("2024-01-02", "2024-01-31"))
    assert checks(issues) == [Check.MISSING_DATA]
    assert issues[0].severity is Severity.WARNING
    assert "nur teilweise" in issues[0].detail


def test_partial_data_within_tolerance_is_ok() -> None:
    bars = clean_bars(start="2024-01-05", end="2024-01-29")
    assert check_membership("AAA", bars, membership("2024-01-02", "2024-01-31")) == []


def test_membership_before_history_start_is_ignored() -> None:
    issues = check_membership(
        "OLD",
        clean_bars(),
        membership("1980-01-02", "1985-12-31"),
        history_start=dt.date(1990, 1, 1),
    )
    assert issues == []


def test_open_membership_uses_today() -> None:
    bars = clean_bars()
    today = dt.date(2024, 2, 1)
    assert check_membership("AAA", bars, membership("2024-01-02", None), today=today) == []
    later = dt.date(2024, 3, 1)
    issues = check_membership("AAA", bars, membership("2024-01-02", None), today=later)
    assert [i.severity for i in issues] == [Severity.WARNING]


def test_open_membership_without_today_excludes_running_day() -> None:
    today = pd.Timestamp(dt.date.today())
    start = today - pd.Timedelta(days=30)
    dates = pd.bdate_range(start, today - pd.Timedelta(days=1))
    bars = pd.DataFrame({"date": dates})
    # Feiertage im Zeitraum fehlen in bdate_range nicht, also keine Fehlmeldung möglich.
    issues = check_membership("AAA", bars, membership(start.date().isoformat(), None))
    assert issues == []


def test_no_membership_reports_nothing() -> None:
    empty = membership("2024-01-02", None).iloc[0:0]
    assert check_membership("AAA", clean_bars(), empty) == []


# ---------------------------------------------------------------------- Bericht


def test_issues_to_frame_columns_and_order() -> None:
    issues = [
        Issue("BBB", dt.date(2024, 1, 3), Check.JUMP, Severity.WARNING, "b"),
        Issue("AAA", None, Check.MISSING_DATA, Severity.ERROR, "x"),
        Issue("AAA", dt.date(2024, 1, 5), Check.GAP, Severity.WARNING, "a"),
    ]
    frame = issues_to_frame(issues)
    assert tuple(frame.columns) == REPORT_COLUMNS
    assert list(frame["symbol"]) == ["AAA", "AAA", "BBB"]
    assert list(frame["check"]) == ["GAP", "MISSING_DATA", "JUMP"]
    assert list(frame["severity"]) == ["WARNING", "ERROR", "WARNING"]
    assert frame["date"].dtype == "datetime64[ns]"
    assert pd.isna(frame["date"].iloc[1])


def test_issues_to_frame_empty() -> None:
    frame = issues_to_frame([])
    assert frame.empty
    assert tuple(frame.columns) == REPORT_COLUMNS


# ---------------------------------------------------------------------- Speicher


@pytest.fixture
def store() -> Iterator[MarketStore]:
    with MarketStore(":memory:") as opened:
        yield opened


def fill_store(store: MarketStore) -> None:
    """GOOD sauber, BAD mit High < Low, SPLIT mit 4:1-Split, OLD nur Indexmitglied."""
    store.upsert_bars("GOOD", clean_bars())
    bad = clean_bars()
    bad.loc[4, ["high", "low"]] = [90.0, 95.0]
    store.upsert_bars("BAD", bad)
    split = clean_bars(base=400.0)
    split.loc[10:, ["open", "high", "low", "close"]] /= 4.0
    store.upsert_bars("SPLIT", split)
    store.upsert_actions(
        "SPLIT",
        pd.DataFrame({"date": [split["date"].iloc[10]], "type": ["SPLIT"], "value": [4.0]}),
    )
    store.connection.execute(
        "INSERT INTO index_membership VALUES "
        "('SP500', 'OLD', DATE '2024-01-02', DATE '2024-01-31'), "
        "('SP500', 'GOOD', DATE '2024-01-02', DATE '2024-01-31')"
    )


def test_check_symbol_from_store(store: MarketStore) -> None:
    fill_store(store)
    assert check_symbol("good", store) == []
    assert check_symbol("SPLIT", store) == []
    assert checks(check_symbol("BAD", store)) == [Check.HIGH_LOW]
    assert checks(check_symbol("OLD", store)) == [Check.MISSING_DATA]


def test_check_all_from_store(store: MarketStore) -> None:
    fill_store(store)
    report = check_all(store)
    assert list(report["symbol"]) == ["BAD", "OLD"]
    assert list(report["check"]) == ["HIGH_LOW", "MISSING_DATA"]


def test_check_all_selected_symbols(store: MarketStore) -> None:
    fill_store(store)
    report = check_all(store, symbols=[" good ", "split", ""])
    assert report.empty
    assert tuple(report.columns) == REPORT_COLUMNS


# ---------------------------------------------------------------------- Einstellungen und CLI


@pytest.fixture
def settings_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Leitet Einstellungen und Datenbank nach ``tmp_path`` um."""
    for name in list(os.environ):
        if name.upper().startswith("TH_"):
            monkeypatch.delenv(name)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "user_data_dir", lambda app, appauthor: str(data_dir))
    monkeypatch.setattr(config, "user_config_dir", lambda app, appauthor: str(tmp_path / "config"))
    return data_dir


def make_database(path: Path, *, with_error: bool = True) -> None:
    with MarketStore(path) as store:
        if with_error:
            fill_store(store)
        else:
            store.upsert_bars("GOOD", clean_bars())


def test_check_symbol_opens_database_from_settings(settings_dir: Path) -> None:
    make_database(settings_dir / "market.duckdb")
    assert checks(check_symbol("BAD")) == [Check.HIGH_LOW]
    assert list(check_all()["symbol"]) == ["BAD", "OLD"]


def test_cli_quality_reports_errors(
    settings_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_database(settings_dir / "market.duckdb")
    csv_path = tmp_path / "bericht.csv"
    assert main(["quality", "--csv", str(csv_path)]) == 1
    out = capsys.readouterr().out
    assert "HIGH_LOW" in out
    assert "2 Befunde: 2 Fehler, 0 Warnungen." in out
    saved = pd.read_csv(csv_path)
    assert tuple(saved.columns) == REPORT_COLUMNS
    assert len(saved) == 2


def test_cli_quality_clean_symbols(
    settings_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "other.duckdb"
    make_database(db)
    assert main(["quality", "--db", str(db), "GOOD", "SPLIT"]) == 0
    assert "Keine Auffälligkeiten gefunden." in capsys.readouterr().out


def test_cli_quality_warnings_only(
    settings_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "warn.duckdb"
    with MarketStore(db) as store:
        bars = clean_bars()
        store.upsert_bars("GAPPY", bars.drop(index=5))
    assert main(["quality", "--db", str(db)]) == 0
    assert "1 Befunde: 0 Fehler, 1 Warnungen." in capsys.readouterr().out


def test_cli_quality_missing_database(
    settings_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["quality", "--db", str(tmp_path / "fehlt.duckdb")]) == 2
    assert "nicht gefunden" in capsys.readouterr().err
