"""Zeitrahmen 3D, 1W und 1M aus Tageskursen (D-07)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_helfer.core.bars import BarsValidationError
from trading_helfer.core.types import Timeframe
from trading_helfer.data import resample as resample_module
from trading_helfer.data.resample import (
    PARTIAL_COLUMN,
    ResampleCache,
    resample,
    resample_cached,
)

SESSIONS = resample_module._calendar_sessions(1980, 2030)


def sessions_between(start: str, end: str) -> pd.DatetimeIndex:
    """XNYS-Handelstage von ``start`` bis ``end`` (inklusive)."""
    return SESSIONS[(pd.Timestamp(start) <= SESSIONS) & (pd.Timestamp(end) >= SESSIONS)]


def make_daily(dates: Any, *, adj_close: bool = True) -> pd.DataFrame:
    """Synthetische Tageskurse: Close steigt je Tag um 1, Volumen 100, 200, 300 …"""
    idx = pd.DatetimeIndex(dates)
    n = len(idx)
    close = 100.0 + np.arange(n, dtype="float64")
    data: dict[str, Any] = {
        "date": idx,
        "open": close - 0.5,
        "high": close + 1.0 + (np.arange(n) % 3),
        "low": close - 1.0 - (np.arange(n) % 2),
        "close": close,
        "volume": (np.arange(n, dtype="int64") + 1) * 100,
    }
    if adj_close:
        data["adj_close"] = close / 2
    return pd.DataFrame(data)


def dates_of(df: pd.DataFrame) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in df["date"]]


# --- 1W ---------------------------------------------------------------------------


def test_week_with_good_friday_has_four_days_and_thursday_date() -> None:
    daily = make_daily(sessions_between("2024-03-18", "2024-04-05"))
    weekly = resample(daily, Timeframe.W1)

    assert dates_of(weekly) == ["2024-03-22", "2024-03-28", "2024-04-05"]
    easter = weekly.iloc[1]
    days = daily[(daily["date"] >= "2024-03-25") & (daily["date"] <= "2024-03-28")]
    assert len(days) == 4
    assert easter["open"] == days["open"].iloc[0]
    assert easter["high"] == days["high"].max()
    assert easter["low"] == days["low"].min()
    assert easter["close"] == days["close"].iloc[-1]
    assert easter["volume"] == days["volume"].sum()
    assert easter["adj_close"] == days["adj_close"].iloc[-1]
    assert not weekly[PARTIAL_COLUMN].any()


def test_week_across_year_change_is_one_bar() -> None:
    daily = make_daily(sessions_between("2024-12-23", "2025-01-03"))
    weekly = resample(daily, Timeframe.W1)

    # 2024-12-25 und 2025-01-01 sind Feiertage.
    assert dates_of(weekly) == ["2024-12-27", "2025-01-03"]
    assert weekly["volume"].tolist() == [
        100 + 200 + 300 + 400,
        500 + 600 + 700 + 800,
    ]
    assert weekly["open"].iloc[1] == daily["open"].iloc[4]  # 2024-12-30


def test_week_date_is_last_session_even_if_data_missing_that_day() -> None:
    dates = sessions_between("2024-03-04", "2024-03-15").drop(pd.Timestamp("2024-03-08"))
    weekly = resample(make_daily(dates), Timeframe.W1)

    assert dates_of(weekly) == ["2024-03-08", "2024-03-15"]
    assert not weekly[PARTIAL_COLUMN].any()


def test_week_partial_last_bar() -> None:
    weekly = resample(make_daily(sessions_between("2024-03-04", "2024-03-13")), Timeframe.W1)

    assert dates_of(weekly) == ["2024-03-08", "2024-03-13"]
    assert weekly[PARTIAL_COLUMN].tolist() == [False, True]


# --- 1M ---------------------------------------------------------------------------


def test_month_with_holiday_on_last_weekday() -> None:
    daily = make_daily(sessions_between("2024-02-01", "2024-04-30"))
    monthly = resample(daily, Timeframe.MN1)

    # 2024-03-29 (Karfreitag) ist der letzte Werktag im März, aber kein Handelstag.
    assert dates_of(monthly) == ["2024-02-29", "2024-03-28", "2024-04-30"]
    march = daily[(daily["date"] >= "2024-03-01") & (daily["date"] <= "2024-03-31")]
    assert monthly["volume"].iloc[1] == march["volume"].sum()
    assert monthly["close"].iloc[1] == march["close"].iloc[-1]
    assert not monthly[PARTIAL_COLUMN].any()


def test_month_partial_and_year_change() -> None:
    monthly = resample(make_daily(sessions_between("2023-12-01", "2024-01-10")), Timeframe.MN1)

    assert dates_of(monthly) == ["2023-12-29", "2024-01-10"]
    assert monthly[PARTIAL_COLUMN].tolist() == [False, True]


# --- 3D ---------------------------------------------------------------------------


def test_3d_groups_around_anchor() -> None:
    daily = make_daily(sessions_between("1999-12-20", "2000-01-19"))
    bars = resample(daily, Timeframe.D3)

    # Vor dem Anker rückwärts: [12-20..12-22], [12-23, 12-27, 12-28], [12-29..12-31];
    # ab dem Anker: [01-03..01-05], [01-06, 01-07, 01-10], …, [01-14, 01-18, 01-19].
    assert dates_of(bars) == [
        "1999-12-22",
        "1999-12-28",
        "1999-12-31",
        "2000-01-05",
        "2000-01-10",
        "2000-01-13",
        "2000-01-19",
    ]
    assert (bars["volume"] == daily["volume"].to_numpy().reshape(-1, 3).sum(axis=1)).all()
    assert not bars[PARTIAL_COLUMN].any()


def test_3d_boundaries_do_not_depend_on_first_date() -> None:
    all_days = sessions_between("2010-01-01", "2010-12-31")
    full = resample(make_daily(all_days), Timeframe.D3)
    for offset in range(1, 6):
        shifted = resample(make_daily(all_days[offset:]), Timeframe.D3)
        assert set(dates_of(shifted)) <= set(dates_of(full))
        # Gleiche Gruppengrenzen: das letzte Datum stimmt überein, Länge ±2.
        assert dates_of(shifted)[-1] == dates_of(full)[-1]
        assert len(full) - 2 <= len(shifted) <= len(full)


def test_3d_symbol_starting_mid_group_drops_first_group() -> None:
    # 2000-01-04 liegt mitten in der Gruppe [01-03, 01-04, 01-05].
    daily = make_daily(sessions_between("2000-01-04", "2000-01-13"))
    bars = resample(daily, Timeframe.D3)

    assert dates_of(bars) == ["2000-01-10", "2000-01-13"]
    assert bars["open"].iloc[0] == daily["open"].iloc[2]  # 2000-01-06


def test_3d_partial_last_group() -> None:
    bars = resample(make_daily(sessions_between("2000-01-03", "2000-01-11")), Timeframe.D3)

    assert dates_of(bars) == ["2000-01-05", "2000-01-10", "2000-01-11"]
    assert bars[PARTIAL_COLUMN].tolist() == [False, False, True]


def test_3d_non_session_day_counts_to_next_session() -> None:
    # 2000-01-17 (MLK-Tag) ist kein Handelstag und zählt zur Gruppe von 01-18.
    dates = [*sessions_between("2000-01-03", "2000-01-14"), pd.Timestamp("2000-01-17")]
    dates += list(sessions_between("2000-01-18", "2000-01-19"))
    bars = resample(make_daily(dates), Timeframe.D3)

    assert dates_of(bars)[-1] == "2000-01-19"
    assert bars["volume"].iloc[-1] == 1000 + 1100 + 1200 + 1300


# --- allgemein ----------------------------------------------------------------------


def test_daily_timeframe_returns_validated_copy() -> None:
    daily = make_daily(sessions_between("2024-01-02", "2024-01-05"))
    daily["extra"] = 1
    result = resample(daily, Timeframe.D1)

    assert list(result.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
        PARTIAL_COLUMN,
    ]
    assert not result[PARTIAL_COLUMN].any()
    assert "is_partial" not in daily.columns


@pytest.mark.parametrize("timeframe", list(Timeframe))
def test_empty_input(timeframe: Timeframe) -> None:
    result = resample(make_daily([], adj_close=False), timeframe)

    assert result.empty
    assert list(result.columns) == ["date", "open", "high", "low", "close", "volume", "is_partial"]
    assert result[PARTIAL_COLUMN].dtype == bool


@pytest.mark.parametrize("timeframe", [Timeframe.D3, Timeframe.W1, Timeframe.MN1])
def test_output_schema_and_input_unchanged(timeframe: Timeframe) -> None:
    daily = make_daily(sessions_between("2024-01-02", "2024-03-15"), adj_close=False)
    before = daily.copy()
    result = resample(daily, timeframe)

    pd.testing.assert_frame_equal(daily, before)
    assert list(result.columns) == ["date", "open", "high", "low", "close", "volume", "is_partial"]
    assert str(result["date"].dtype) == "datetime64[ns]"
    assert str(result["volume"].dtype) == "int64"
    assert result[PARTIAL_COLUMN].dtype == bool
    assert isinstance(result.index, pd.RangeIndex)


def test_accepts_timeframe_string_and_rejects_unknown() -> None:
    daily = make_daily(sessions_between("2024-01-02", "2024-01-12"))
    pd.testing.assert_frame_equal(
        resample(daily, "1W"),  # type: ignore[arg-type]
        resample(daily, Timeframe.W1),
    )
    with pytest.raises(ValueError, match="2W"):
        resample(daily, "2W")  # type: ignore[arg-type]


def test_invalid_input_raises() -> None:
    with pytest.raises(BarsValidationError):
        resample(pd.DataFrame({"date": []}), Timeframe.W1)


def test_history_before_1990() -> None:
    weekly = resample(make_daily(sessions_between("1985-01-02", "1985-01-11")), Timeframe.W1)
    assert dates_of(weekly) == ["1985-01-04", "1985-01-11"]


# --- Eigenschaften ------------------------------------------------------------------


@st.composite
def daily_frames(draw: st.DrawFn) -> pd.DataFrame:
    """Zufällige Tageskurse auf einer Teilmenge der Handelstage 1995–2005."""
    pool = sessions_between("1995-01-01", "2005-12-31")
    start = draw(st.integers(0, len(pool) - 2))
    length = draw(st.integers(1, 120))
    days = pool[start : start + length]
    keep = draw(st.lists(st.booleans(), min_size=len(days), max_size=len(days)))
    keep[0] = True
    days = days[np.asarray(keep)]
    n = len(days)
    price = st.floats(1.0, 1000.0, allow_nan=False, allow_infinity=False)
    base = np.asarray(draw(st.lists(price, min_size=n, max_size=n)))
    spread = np.asarray(draw(st.lists(st.floats(0.0, 10.0), min_size=n, max_size=n)))
    volume = draw(st.lists(st.integers(0, 10**9), min_size=n, max_size=n))
    return pd.DataFrame(
        {
            "date": days,
            "open": base,
            "high": base + spread,
            "low": base - spread / 2,
            "close": base + spread / 4,
            "volume": np.asarray(volume, dtype="int64"),
        }
    )


def reference_keys(dates: pd.Series[Any], timeframe: Timeframe) -> list[Any]:
    """Unabhängig berechnete Gruppe je Tag (Woche, Monat oder 3D-Gruppe)."""
    if timeframe is Timeframe.W1:
        return list(dates.dt.to_period("W-FRI"))
    if timeframe is Timeframe.MN1:
        return list(dates.dt.to_period("M"))
    anchor = int(SESSIONS.get_loc(pd.Timestamp("2000-01-03")))
    return [(int(SESSIONS.get_loc(d)) - anchor) // 3 for d in dates]


@settings(max_examples=80, deadline=None)
@given(daily=daily_frames(), timeframe=st.sampled_from([Timeframe.D3, Timeframe.W1, Timeframe.MN1]))
def test_property_aggregation(daily: pd.DataFrame, timeframe: Timeframe) -> None:
    bars = resample(daily, timeframe)

    groups = [g for _, g in daily.groupby(reference_keys(daily["date"], timeframe), sort=True)]
    if timeframe is Timeframe.D3:
        first = int(reference_keys(daily["date"].iloc[:1], timeframe)[0])
        group_start = SESSIONS[int(SESSIONS.get_loc(pd.Timestamp("2000-01-03"))) + first * 3]
        if daily["date"].iloc[0] > group_start:
            groups = groups[1:]
    assert len(bars) == len(groups)
    for bar, days in zip(bars.itertuples(index=False), groups, strict=True):
        assert bar.high == days["high"].max()
        assert bar.low == days["low"].min()
        assert bar.open == days["open"].iloc[0]
        assert bar.close == days["close"].iloc[-1]
        assert bar.volume == days["volume"].sum()
        assert bar.date >= days["date"].iloc[-1]
    if timeframe is not Timeframe.D3:
        assert bars["volume"].sum() == daily["volume"].sum()
    assert bars["date"].is_monotonic_increasing
    assert not bars[PARTIAL_COLUMN].iloc[:-1].any()


# --- Zwischenspeicher ---------------------------------------------------------------


def test_cache_hits_by_symbol_timeframe_and_last_date(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Timeframe] = []
    original = resample_module.resample

    def counting(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
        calls.append(timeframe)
        return original(df, timeframe)

    monkeypatch.setattr(resample_module, "resample", counting)
    cache = ResampleCache(maxsize=2)
    daily = make_daily(sessions_between("2024-01-02", "2024-02-29"))

    first = cache.get("AAA", daily, Timeframe.W1)
    first["close"] = 0.0  # Kopie, Speicher bleibt unverändert
    second = cache.get("AAA", daily, Timeframe.W1)
    assert len(calls) == 1
    assert (second["close"] > 0).all()

    cache.get("AAA", daily, Timeframe.MN1)
    cache.get("BBB", daily, Timeframe.W1)
    assert len(calls) == 3
    assert len(cache) == 2  # ältester Eintrag (AAA, 1W) verdrängt
    cache.get("AAA", daily, Timeframe.W1)
    assert len(calls) == 4

    longer = make_daily(sessions_between("2024-01-02", "2024-03-01"))
    cache.get("AAA", longer, Timeframe.W1)
    assert len(calls) == 5

    cache.clear()
    assert len(cache) == 0


def test_cache_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="maxsize"):
        ResampleCache(maxsize=0)


def test_resample_cached_matches_resample() -> None:
    daily = make_daily(sessions_between("2024-01-02", "2024-02-29"))
    pd.testing.assert_frame_equal(
        resample_cached("AAA", daily, Timeframe.D3), resample(daily, Timeframe.D3)
    )
    empty = make_daily([])
    assert resample_cached("EMPTY", empty, Timeframe.W1).empty
