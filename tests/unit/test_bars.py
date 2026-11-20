"""Validierung von Kursreihen."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_helfer.core.bars import (
    BAR_COLUMNS,
    BAR_DTYPES,
    OPTIONAL_BAR_COLUMNS,
    BarsValidationError,
    empty_bars,
    validate_bars,
)


def make_bars(n: int = 5, *, start: str = "2024-01-02", adj_close: bool = True) -> pd.DataFrame:
    """Synthetische, gültige Kursreihe mit konstantem Aufwärtstrend."""
    dates = pd.bdate_range(start, periods=n).as_unit("ns")
    close = 100.0 + np.arange(n, dtype="float64")
    data: dict[str, Any] = {
        "date": dates,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.arange(1, n + 1, dtype="int64") * 1000,
    }
    if adj_close:
        data["adj_close"] = close * 0.5
    return pd.DataFrame(data)


def assert_schema(df: pd.DataFrame) -> None:
    for name in df.columns:
        if name in BAR_DTYPES:
            assert str(df[name].dtype) == BAR_DTYPES[name], name
    assert isinstance(df.index, pd.RangeIndex)
    assert df.index.start == 0
    assert df.index.step == 1
    assert df["date"].is_monotonic_increasing
    assert df["date"].is_unique


# --- korrekter Fall -------------------------------------------------------------------


def test_valid_bars_pass_unchanged() -> None:
    bars = make_bars()
    result = validate_bars(bars)
    assert_schema(result)
    pd.testing.assert_frame_equal(result, bars)


def test_result_is_a_new_copy() -> None:
    bars = make_bars()
    result = validate_bars(bars)
    assert result is not bars
    result.loc[0, "close"] = -1.0
    assert bars.loc[0, "close"] == 100.0


def test_adj_close_is_optional() -> None:
    bars = make_bars(adj_close=False)
    result = validate_bars(bars)
    assert list(result.columns) == list(BAR_COLUMNS)
    assert_schema(result)


def test_column_order_is_canonical_and_extra_columns_are_kept() -> None:
    bars = make_bars()
    bars["source"] = "test"
    shuffled = bars[["source", "volume", "adj_close", "close", "low", "high", "open", "date"]]
    result = validate_bars(shuffled)
    assert list(result.columns) == [*BAR_COLUMNS, *OPTIONAL_BAR_COLUMNS, "source"]
    assert (result["source"] == "test").all()


def test_index_is_reset_to_range_index() -> None:
    bars = make_bars().set_index(pd.Index([10, 20, 30, 40, 50]))
    result = validate_bars(bars)
    assert_schema(result)


def test_empty_frame_with_columns_is_valid() -> None:
    bars = pd.DataFrame({name: [] for name in BAR_COLUMNS})
    result = validate_bars(bars)
    assert result.empty
    assert_schema(result)


# --- fehlende Spalten -----------------------------------------------------------------


@pytest.mark.parametrize("column", BAR_COLUMNS)
def test_missing_column_raises(column: str) -> None:
    bars = make_bars().drop(columns=[column])
    with pytest.raises(BarsValidationError, match=f"fehlen: {column}") as exc:
        validate_bars(bars)
    assert column in exc.value.reason


def test_missing_columns_are_all_listed() -> None:
    with pytest.raises(BarsValidationError, match="date, open, high, low, close, volume"):
        validate_bars(pd.DataFrame())


def test_error_is_a_value_error() -> None:
    assert issubclass(BarsValidationError, ValueError)


def test_duplicate_column_names_raise() -> None:
    bars = make_bars()
    bars = pd.concat([bars, bars[["close"]]], axis=1)
    with pytest.raises(BarsValidationError, match="Doppelte Spaltennamen: close"):
        validate_bars(bars)


def test_non_dataframe_raises() -> None:
    with pytest.raises(BarsValidationError, match="DataFrame"):
        validate_bars([1, 2, 3])  # type: ignore[arg-type]


# --- unsortierte Daten ----------------------------------------------------------------


def test_unsorted_dates_are_sorted() -> None:
    bars = make_bars()
    shuffled = bars.iloc[[3, 0, 4, 2, 1]]
    result = validate_bars(shuffled)
    assert_schema(result)
    pd.testing.assert_frame_equal(result, bars)


def test_input_is_not_modified() -> None:
    bars = make_bars().iloc[[2, 1, 0]]
    before = bars.copy()
    validate_bars(bars)
    pd.testing.assert_frame_equal(bars, before)


# --- doppelte Daten -------------------------------------------------------------------


def test_identical_duplicate_rows_are_removed() -> None:
    bars = make_bars()
    doubled = pd.concat([bars, bars.iloc[[1, 3]]], ignore_index=True)
    result = validate_bars(doubled)
    pd.testing.assert_frame_equal(result, bars)


def test_conflicting_duplicate_dates_raise() -> None:
    bars = make_bars()
    conflict = bars.iloc[[2]].copy()
    conflict["close"] = 999.0
    with pytest.raises(BarsValidationError, match="2024-01-04"):
        validate_bars(pd.concat([bars, conflict], ignore_index=True))


def test_duplicate_dates_differing_in_extra_column_raise() -> None:
    bars = make_bars()
    bars["source"] = "a"
    conflict = bars.iloc[[0]].copy()
    conflict["source"] = "b"
    with pytest.raises(BarsValidationError, match="abweichenden Werten"):
        validate_bars(pd.concat([bars, conflict], ignore_index=True))


def test_dates_equal_after_normalization_raise_if_values_differ() -> None:
    bars = make_bars(2)
    bars.loc[1, "date"] = pd.Timestamp("2024-01-02 16:00")
    with pytest.raises(BarsValidationError, match="2024-01-02"):
        validate_bars(bars)


def test_duplicates_with_unhashable_extra_column_raise() -> None:
    bars = make_bars(2)
    bars["meta"] = [[1], [2]]
    doubled = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)
    with pytest.raises(BarsValidationError, match="abweichenden Werten"):
        validate_bars(doubled)


def test_many_conflicts_are_abbreviated() -> None:
    bars = make_bars(8)
    conflict = bars.copy()
    conflict["close"] += 1.0
    with pytest.raises(BarsValidationError, match=r"\(und 3 weitere\)"):
        validate_bars(pd.concat([bars, conflict], ignore_index=True))


# --- Typumwandlung --------------------------------------------------------------------


def test_types_are_converted() -> None:
    bars = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02"],
            "open": [1, 2],
            "high": ["3.5", "4.5"],
            "low": np.array([0.5, 1.5], dtype="float32"),
            "close": pd.array([2, 3], dtype="Int64"),
            "volume": [100.0, 200.0],
            "adj_close": ["1.0", "1.5"],
        }
    )
    result = validate_bars(bars)
    assert_schema(result)
    assert list(result["date"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert list(result["open"]) == [2.0, 1.0]
    assert list(result["high"]) == [4.5, 3.5]
    assert list(result["close"]) == [3.0, 2.0]
    assert list(result["volume"]) == [200, 100]
    assert list(result["adj_close"]) == [1.5, 1.0]


def test_python_dates_and_times_are_normalized() -> None:
    bars = make_bars(2)
    bars["date"] = [dt.date(2024, 1, 2), dt.datetime(2024, 1, 3, 15, 30)]
    result = validate_bars(bars)
    assert list(result["date"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert str(result["date"].dtype) == "datetime64[ns]"


def test_timezone_is_removed_keeping_local_date() -> None:
    bars = make_bars(2)
    bars["date"] = pd.to_datetime(["2024-01-02", "2024-01-03"]).tz_localize("America/New_York")
    result = validate_bars(bars)
    assert result["date"].dt.tz is None
    assert list(result["date"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_integer_like_string_volume_is_accepted() -> None:
    bars = make_bars(2)
    bars["volume"] = ["10", "20"]
    assert list(validate_bars(bars)["volume"]) == [10, 20]


def test_nullable_integer_volume_is_accepted() -> None:
    bars = make_bars(2)
    bars["volume"] = pd.array([10, 20], dtype="Int64")
    result = validate_bars(bars)
    assert str(result["volume"].dtype) == "int64"


def test_unparseable_date_raises() -> None:
    bars = make_bars(2)
    bars["date"] = ["2024-01-02", "kein Datum"]
    with pytest.raises(BarsValidationError, match="Datumswerte"):
        validate_bars(bars)


def test_missing_date_raises() -> None:
    bars = make_bars(2)
    bars["date"] = [pd.Timestamp("2024-01-02"), pd.NaT]
    with pytest.raises(BarsValidationError, match="'date' enthält fehlende Werte in 1 Zeile"):
        validate_bars(bars)


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "adj_close", "volume"])
def test_non_numeric_value_raises(column: str) -> None:
    bars = make_bars(2).astype({column: object})
    bars.loc[1, column] = "abc"
    with pytest.raises(BarsValidationError, match=f"'{column}' lässt sich nicht in Zahlen"):
        validate_bars(bars)


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "adj_close", "volume"])
def test_nan_raises(column: str) -> None:
    bars = make_bars(3).astype({column: "float64"})
    bars.loc[1, column] = np.nan
    with pytest.raises(
        BarsValidationError, match=f"'{column}' enthält fehlende Werte am 2024-01-03"
    ):
        validate_bars(bars)


def test_missing_nullable_integer_volume_raises() -> None:
    bars = make_bars(2)
    bars["volume"] = pd.array([10, None], dtype="Int64")
    with pytest.raises(BarsValidationError, match="'volume' enthält fehlende Werte"):
        validate_bars(bars)


@pytest.mark.parametrize("column", ["close", "volume"])
def test_infinite_value_raises(column: str) -> None:
    bars = make_bars(2).astype({column: "float64"})
    bars.loc[0, column] = np.inf
    with pytest.raises(BarsValidationError, match=f"'{column}' enthält unendliche Werte"):
        validate_bars(bars)


def test_fractional_volume_raises() -> None:
    bars = make_bars(2)
    bars["volume"] = [10.0, 10.5]
    with pytest.raises(BarsValidationError, match=r"ganzzahlig.*2024-01-03"):
        validate_bars(bars)


# --- empty_bars -----------------------------------------------------------------------


@pytest.mark.parametrize("with_adj_close", [True, False])
def test_empty_bars_schema(with_adj_close: bool) -> None:
    bars = empty_bars(with_adj_close=with_adj_close)
    expected = list(BAR_COLUMNS) + (list(OPTIONAL_BAR_COLUMNS) if with_adj_close else [])
    assert list(bars.columns) == expected
    assert bars.empty
    assert_schema(bars)
    pd.testing.assert_frame_equal(validate_bars(bars), bars)


# --- Eigenschaften (hypothesis) -------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    order=st.permutations(range(6)),
    duplicates=st.lists(st.integers(min_value=0, max_value=5), max_size=4),
)
def test_shuffled_and_duplicated_rows_yield_canonical_frame(
    order: list[int], duplicates: list[int]
) -> None:
    bars = make_bars(6)
    messy = bars.iloc[[*order, *duplicates]]
    result = validate_bars(messy)
    pd.testing.assert_frame_equal(result, bars)
    pd.testing.assert_frame_equal(validate_bars(result), result)
