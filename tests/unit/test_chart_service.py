"""ChartService: Laden, Bereinigen, Zeitrahmen und Zwischenspeicher."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading_helfer.core.bars import BarsValidationError, empty_bars
from trading_helfer.core.types import Timeframe
from trading_helfer.services.chart_service import (
    CHART_COLUMNS,
    ChartService,
    adjust_prices,
    empty_loader,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "chart" / "SYNTH_daily.csv"
EXPECTED_COUNTS = {
    Timeframe.D1: 475,
    Timeframe.D3: 159,
    Timeframe.W1: 99,
    Timeframe.MN1: 23,
}


@pytest.fixture
def daily() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, parse_dates=["date"])


@pytest.fixture
def calls() -> list[str]:
    return []


@pytest.fixture
def service(daily: pd.DataFrame, calls: list[str]) -> ChartService:
    def loader(symbol: str) -> pd.DataFrame:
        calls.append(symbol)
        return daily

    return ChartService(loader)


@pytest.mark.parametrize("timeframe", list(EXPECTED_COUNTS))
def test_candle_count_per_timeframe(service: ChartService, timeframe: Timeframe) -> None:
    data = service.load("SYNTH", timeframe)
    assert len(data) == EXPECTED_COUNTS[timeframe]
    assert tuple(data.bars.columns) == CHART_COLUMNS
    assert data.timeframe is timeframe
    assert data.symbol == "SYNTH"
    assert data.adjusted
    assert not data.is_empty
    assert data.bars["is_partial"].sum() == (0 if timeframe is Timeframe.D1 else 1)
    assert bool(data.bars["is_partial"].iloc[-1]) is (timeframe is not Timeframe.D1)


def test_daily_bars_are_loaded_once(service: ChartService, calls: list[str]) -> None:
    for timeframe in EXPECTED_COUNTS:
        service.load("SYNTH", timeframe)
        service.load("SYNTH", timeframe, adjusted=False)
    assert calls == ["SYNTH"]
    service.invalidate("SYNTH")
    service.load("SYNTH", Timeframe.W1)
    assert calls == ["SYNTH", "SYNTH"]
    service.invalidate()
    service.load("SYNTH", Timeframe.W1)
    assert len(calls) == 3


def test_adjusted_and_raw_prices(service: ChartService, daily: pd.DataFrame) -> None:
    adjusted = service.load("SYNTH", Timeframe.D1).bars
    raw = service.load("SYNTH", Timeframe.D1, adjusted=False).bars
    factor = daily["adj_close"] / daily["close"]
    for column in ("open", "high", "low", "close"):
        pd.testing.assert_series_equal(
            adjusted[column], raw[column] * factor, check_names=False, rtol=1e-12
        )
    # Vor dem Split 4:1 sind rohe Kurse etwa viermal so hoch wie bereinigte.
    assert raw["close"].iloc[0] / adjusted["close"].iloc[0] == pytest.approx(4 / 0.99)
    assert raw["close"].iloc[-1] == pytest.approx(adjusted["close"].iloc[-1])
    pd.testing.assert_series_equal(adjusted["volume"], raw["volume"])


def test_adjusted_weekly_uses_daily_factors(service: ChartService) -> None:
    weekly = service.load("SYNTH", Timeframe.W1).bars
    daily = service.load("SYNTH", Timeframe.D1).bars
    first_week = daily[daily["date"] <= weekly["date"].iloc[0]]
    assert weekly["high"].iloc[0] == pytest.approx(first_week["high"].max())
    assert weekly["close"].iloc[0] == pytest.approx(first_week["close"].iloc[-1])


def test_adjust_prices_without_adj_close(daily: pd.DataFrame) -> None:
    raw = daily.drop(columns=["adj_close"])
    pd.testing.assert_frame_equal(adjust_prices(raw), raw)


def test_adjust_prices_missing_values_count_as_one(daily: pd.DataFrame) -> None:
    frame = daily.head(3).copy()
    frame.loc[1, "adj_close"] = float("nan")
    adjusted = adjust_prices(frame)
    assert adjusted.loc[1, "close"] == frame.loc[1, "close"]
    assert adjusted.loc[0, "close"] == pytest.approx(frame.loc[0, "adj_close"])


def test_default_loader_returns_empty_chart() -> None:
    service = ChartService()
    data = service.load("AAPL", Timeframe.W1)
    assert data.is_empty
    assert len(data) == 0
    assert tuple(data.bars.columns) == CHART_COLUMNS
    assert empty_loader("X").equals(empty_bars())


def test_set_loader_replaces_source(service: ChartService, daily: pd.DataFrame) -> None:
    assert len(service.load("SYNTH", Timeframe.D1)) == 475
    service.set_loader(lambda _symbol: daily.head(10))
    assert len(service.load("SYNTH", Timeframe.D1)) == 10


def test_invalid_data_raises() -> None:
    service = ChartService(lambda _s: pd.DataFrame({"date": ["2024-01-02"]}))
    with pytest.raises(BarsValidationError):
        service.load("BAD", Timeframe.D1)


def test_daily_cache_is_bounded(daily: pd.DataFrame) -> None:
    calls: list[str] = []

    def loader(symbol: str) -> pd.DataFrame:
        calls.append(symbol)
        return daily.head(5)

    service = ChartService(loader)
    for i in range(20):
        service.daily_bars(f"S{i}")
    service.daily_bars("S19")
    service.daily_bars("S0")
    assert calls.count("S19") == 1
    assert calls.count("S0") == 2
