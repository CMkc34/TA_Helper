"""Normierung, konservatives reported_date und Speichern der Fundamentaldaten."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import pandas as pd
import pytest

from trading_helfer.data.fundamentals import (
    ANNUAL_REPORT_LAG,
    METRICS,
    QUARTER_REPORT_LAG,
    UPDATE_INTERVAL,
    conservative_reported_date,
    empty_fundamentals,
    normalize_fundamentals,
    update_fundamentals,
)
from trading_helfer.data.providers import FUNDAMENTALS_COLUMNS, FundamentalsProvider
from trading_helfer.data.storage import MarketStore


@pytest.fixture
def store() -> Iterator[MarketStore]:
    with MarketStore(":memory:") as market_store:
        yield market_store


def test_metrics_are_the_fixed_names() -> None:
    assert len(METRICS) == 22
    assert len(set(METRICS)) == 22
    assert {"revenue", "eps_diluted", "long_term_debt", "dividends_paid"} <= set(METRICS)


def test_update_interval_is_weekly() -> None:
    assert UPDATE_INTERVAL.days == 7


# --- reported_date -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("period_end", "period_type", "expected"),
    [
        (date(2023, 12, 31), "Q", "2024-02-14"),
        (date(2023, 12, 31), "FY", "2024-03-15"),
        (date(2024, 3, 31), "q", "2024-05-15"),
        (pd.Timestamp("2023-06-30 00:00", tz="America/New_York"), "Q", "2023-08-14"),
        (pd.Timestamp("2023-06-30 15:30"), " FY ", "2023-09-13"),
    ],
)
def test_conservative_reported_date(period_end: object, period_type: str, expected: str) -> None:
    result = conservative_reported_date(period_end, period_type)  # type: ignore[arg-type]
    assert result == pd.Timestamp(expected)
    assert result.tzinfo is None


def test_conservative_reported_date_lags() -> None:
    assert QUARTER_REPORT_LAG.days == 45
    assert ANNUAL_REPORT_LAG.days == 75


def test_conservative_reported_date_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Periodenart"):
        conservative_reported_date(date(2023, 12, 31), "H1")


# --- Normierung ----------------------------------------------------------------------


def frame(**columns: list[object]) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_normalize_fills_missing_reported_date() -> None:
    raw = frame(
        period_end=["2023-12-31", "2023-12-31", "2023-09-30"],
        period_type=["FY", "Q", "q"],
        metric=["revenue", "revenue", " net_income "],
        value=[100.0, 30.0, 5],
        reported_date=[None, "2024-02-01", None],
    )
    result = normalize_fundamentals(raw)
    assert list(result.columns) == list(FUNDAMENTALS_COLUMNS)
    assert result.to_dict("records") == [
        {
            "period_end": pd.Timestamp("2023-09-30"),
            "period_type": "Q",
            "metric": "net_income",
            "value": 5.0,
            "reported_date": pd.Timestamp("2023-11-14"),
        },
        {
            "period_end": pd.Timestamp("2023-12-31"),
            "period_type": "FY",
            "metric": "revenue",
            "value": 100.0,
            "reported_date": pd.Timestamp("2024-03-15"),
        },
        {
            "period_end": pd.Timestamp("2023-12-31"),
            "period_type": "Q",
            "metric": "revenue",
            "value": 30.0,
            # Datum der Quelle bleibt erhalten.
            "reported_date": pd.Timestamp("2024-02-01"),
        },
    ]
    assert str(result["period_end"].dtype) == "datetime64[ns]"
    assert str(result["reported_date"].dtype) == "datetime64[ns]"


def test_normalize_without_reported_date_column() -> None:
    raw = frame(period_end=["2024-03-31"], period_type=["Q"], metric=["cash"], value=[1.0])
    result = normalize_fundamentals(raw)
    assert result.loc[0, "reported_date"] == pd.Timestamp("2024-05-15")


def test_normalize_replaces_reported_date_before_period_end() -> None:
    raw = frame(
        period_end=["2024-03-31"],
        period_type=["Q"],
        metric=["cash"],
        value=[1.0],
        reported_date=["2024-01-01"],
    )
    assert normalize_fundamentals(raw).loc[0, "reported_date"] == pd.Timestamp("2024-05-15")


def test_normalize_drops_missing_values_instead_of_zero() -> None:
    raw = frame(
        period_end=["2023-12-31"] * 4,
        period_type=["FY"] * 4,
        metric=["revenue", "ebitda", "inventory", "capex"],
        value=[100.0, float("nan"), None, 0.0],
    )
    result = normalize_fundamentals(raw)
    # Echte Null bleibt, fehlende Werte fehlen.
    assert result["metric"].tolist() == ["capex", "revenue"]
    assert result["value"].tolist() == [0.0, 100.0]


def test_normalize_handles_timezones() -> None:
    raw = frame(
        period_end=[pd.Timestamp("2023-12-31", tz="UTC")],
        period_type=["FY"],
        metric=["revenue"],
        value=[1.0],
        reported_date=[pd.Timestamp("2024-02-10 16:00", tz="UTC")],
    )
    result = normalize_fundamentals(raw)
    assert result.loc[0, "period_end"] == pd.Timestamp("2023-12-31")
    assert result.loc[0, "reported_date"] == pd.Timestamp("2024-02-10")


def test_normalize_empty() -> None:
    result = normalize_fundamentals(empty_fundamentals())
    assert result.empty
    assert list(result.columns) == list(FUNDAMENTALS_COLUMNS)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"metric": ["umsatz"]}, "Unbekannte Metrik"),
        ({"metric": [None]}, "Unbekannte Metrik"),
        ({"period_type": ["H1"]}, "Unbekannte Periodenart"),
        ({"period_type": [None]}, "Unbekannte Periodenart"),
        ({"period_end": [None]}, "Periodenende"),
    ],
)
def test_normalize_rejects_invalid(change: dict[str, list[object]], message: str) -> None:
    columns: dict[str, list[object]] = {
        "period_end": ["2023-12-31"],
        "period_type": ["FY"],
        "metric": ["revenue"],
        "value": [1.0],
    }
    columns.update(change)
    with pytest.raises(ValueError, match=message):
        normalize_fundamentals(pd.DataFrame(columns))


def test_normalize_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="metric"):
        normalize_fundamentals(frame(period_end=[], period_type=[], value=[]))


def test_normalize_rejects_duplicates() -> None:
    raw = frame(
        period_end=["2023-12-31", "2023-12-31"],
        period_type=["FY", "FY"],
        metric=["revenue", "revenue"],
        value=[1.0, 2.0],
    )
    with pytest.raises(ValueError, match="Doppelter Eintrag"):
        normalize_fundamentals(raw)


# --- Speichern -----------------------------------------------------------------------


class StubProvider:
    name = "stub"

    def __init__(self, answers: dict[str, pd.DataFrame]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    def fetch_fundamentals(self, symbol: str) -> pd.DataFrame:
        self.calls.append(symbol)
        return self.answers.get(symbol, empty_fundamentals())


def test_update_fundamentals_writes_point_in_time(store: MarketStore) -> None:
    provider = StubProvider(
        {
            "AAA": frame(
                period_end=["2023-12-31", "2023-12-31"],
                period_type=["Q", "FY"],
                metric=["revenue", "revenue"],
                value=[30.0, 100.0],
                reported_date=[None, None],
            )
        }
    )
    assert isinstance(provider, FundamentalsProvider)
    assert update_fundamentals(store, provider, ["AAA", "BBB"]) == {"AAA": 2, "BBB": 0}
    assert provider.calls == ["AAA", "BBB"]

    stored = store.load_fundamentals("AAA")
    assert stored["source"].tolist() == ["stub", "stub"]
    # Vor dem konservativen Datum ist der Wert im Backtest noch nicht sichtbar.
    assert store.load_fundamentals("AAA", as_of="2024-02-13").empty
    visible = store.load_fundamentals("AAA", as_of="2024-02-14")
    assert visible["period_type"].tolist() == ["Q"]
    assert len(store.load_fundamentals("AAA", as_of="2024-03-15")) == 2


def test_update_fundamentals_overwrites_restated_values(store: MarketStore) -> None:
    def answer(value: float) -> StubProvider:
        return StubProvider(
            {
                "AAA": frame(
                    period_end=["2023-12-31"], period_type=["FY"], metric=["cash"], value=[value]
                )
            }
        )

    update_fundamentals(store, answer(1.0), ["AAA"])
    update_fundamentals(store, answer(2.0), ["AAA"])
    assert store.load_fundamentals("AAA")["value"].tolist() == [2.0]
