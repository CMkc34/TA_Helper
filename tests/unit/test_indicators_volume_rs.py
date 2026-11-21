"""Volumen-Indikatoren und relative Stärke (RS-Linie, RS-Rang)."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_helfer.core.bars import validate_bars
from trading_helfer.data.universe import IndexChange, IndexHistory
from trading_helfer.indicators import registry, volume
from trading_helfer.indicators.registry import IndicatorError
from trading_helfer.indicators.relative_strength import rs_line, rs_rank, rs_returns

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "indicators"
TOLERANCE = 1e-6

# Referenzspalte -> (Indikator, Ausgabe, erwarteter erster gültiger Index)
CASES = {
    "obv_obv": ("obv", "obv", 0),
    "vwap_vwap": ("vwap", "vwap", 19),
    "cmf_cmf": ("cmf", "cmf", 19),
    "volume_sma_volume_sma": ("volume_sma", "volume_sma", 19),
}


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return validate_bars(pd.read_csv(FIXTURES / "bars.csv"))


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "volume_reference.csv")


@pytest.mark.parametrize("column", sorted(CASES))
def test_matches_reference(bars: pd.DataFrame, reference: pd.DataFrame, column: str) -> None:
    name, output, first_valid = CASES[column]
    result = registry.get(name).compute(bars)[output]
    expected = reference[column].to_numpy()

    assert result.index.equals(bars.index)
    assert result.iloc[:first_valid].isna().all()
    assert result.iloc[first_valid:].notna().all()
    np.testing.assert_allclose(
        result.to_numpy()[first_valid:], expected[first_valid:], rtol=1e-12, atol=TOLERANCE
    )


@pytest.mark.parametrize(
    ("name", "params", "pane"),
    [
        ("obv", {}, "separate"),
        ("vwap", {"period": 20}, "price"),
        ("cmf", {"period": 20}, "separate"),
        ("volume_sma", {"period": 20}, "separate"),
    ],
)
def test_defaults(name: str, params: dict[str, object], pane: str) -> None:
    spec = registry.get(name)
    assert dict(spec.params) == params
    assert spec.pane == pane
    assert spec.outputs == (name,)


def test_obv_small_example() -> None:
    df = pd.DataFrame({"close": [10.0, 11.0, 11.0, 9.0, 12.0], "volume": [5, 3, 7, 2, 4]})
    assert volume.obv(df).tolist() == [5.0, 8.0, 8.0, 6.0, 10.0]


def test_vwap_and_cmf_small_example() -> None:
    df = pd.DataFrame(
        {
            "high": [12.0, 12.0, 6.0],
            "low": [9.0, 6.0, 6.0],
            "close": [12.0, 6.0, 6.0],
            "volume": [100, 300, 50],
        }
    )
    vwap = volume.vwap(df, period=2)
    assert math.isnan(vwap.iloc[0])
    assert vwap.iloc[1] == pytest.approx((11.0 * 100 + 8.0 * 300) / 400)
    cmf = volume.chaikin_money_flow(df, period=2)
    # Multiplikatoren: +1, −1, 0 (High = Low)
    assert cmf.iloc[1] == pytest.approx((100 - 300) / 400)
    assert cmf.iloc[2] == pytest.approx(-300 / 350)


def test_zero_volume_gives_nan() -> None:
    df = pd.DataFrame(
        {"high": [2.0] * 3, "low": [1.0] * 3, "close": [1.5] * 3, "volume": [0, 0, 0]}
    )
    assert volume.vwap(df, period=2).isna().all()
    assert volume.chaikin_money_flow(df, period=2).isna().all()
    assert volume.volume_sma(df, period=2).dropna().tolist() == [0.0, 0.0]


def test_cmf_range(bars: pd.DataFrame) -> None:
    assert volume.chaikin_money_flow(bars).dropna().between(-1, 1).all()


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (volume.vwap, {"period": 0}),
        (volume.chaikin_money_flow, {"period": 1.5}),
        (volume.volume_sma, {"period": -1}),
    ],
)
def test_invalid_parameters(func: object, kwargs: dict[str, object], bars: pd.DataFrame) -> None:
    with pytest.raises(IndicatorError):
        func(bars, **kwargs)  # type: ignore[operator]


def test_missing_volume_column() -> None:
    with pytest.raises(IndicatorError, match="volume"):
        volume.obv(pd.DataFrame({"close": [1.0, 2.0]}))


# --- Relative Stärke -------------------------------------------------------------------


def _series_bars(closes: list[float] | np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"date": dates, "close": np.asarray(closes, dtype=float)})


def test_rs_line_normalized_to_100() -> None:
    symbol = _series_bars([10.0, 11.0, 12.0, 15.0])
    benchmark = _series_bars([100.0, 100.0, 120.0, 100.0])
    result = rs_line(symbol, benchmark)
    assert result.name == "rs_line"
    assert result.index.equals(symbol.index)
    np.testing.assert_allclose(result.to_numpy(), [100.0, 110.0, 100.0, 150.0])


def test_rs_line_with_start_and_missing_benchmark_days() -> None:
    symbol = _series_bars([10.0, 11.0, 12.0, 15.0, 16.0])
    benchmark = _series_bars([100.0, 100.0, 100.0, 100.0, 100.0]).drop(index=[2, 4])
    result = rs_line(symbol, benchmark, start="2024-01-03")
    # Start fällt auf einen Tag ohne Benchmark, also gilt der nächste gemeinsame Tag.
    assert result.isna().tolist() == [True, True, True, False, True]
    assert result.iloc[3] == pytest.approx(100.0)


def test_rs_line_keeps_input_order_and_datetime_index() -> None:
    symbol = _series_bars([10.0, 20.0, 40.0])
    benchmark = _series_bars([1.0, 1.0, 1.0]).set_index("date")
    shuffled = symbol.iloc[[2, 0, 1]]
    result = rs_line(shuffled, benchmark)
    assert result.index.tolist() == [2, 0, 1]
    assert result.tolist() == pytest.approx([400.0, 100.0, 200.0])


def test_rs_line_errors() -> None:
    symbol = _series_bars([1.0, 2.0])
    with pytest.raises(IndicatorError, match="gemeinsamen"):
        rs_line(symbol, _series_bars([1.0, 2.0], start="2030-01-01"))
    with pytest.raises(IndicatorError, match="Datumsindex"):
        rs_line(symbol.drop(columns="date"), symbol)
    with pytest.raises(IndicatorError, match="adj_close"):
        rs_line(symbol, symbol, source="adj_close")
    with pytest.raises(IndicatorError, match="doppelte"):
        rs_line(pd.concat([symbol, symbol]), symbol)
    with pytest.raises(IndicatorError, match="Ungültiges Datum"):
        rs_line(symbol, symbol, start="kein Datum")
    with pytest.raises(IndicatorError, match="ungültige Datumswerte"):
        rs_line(symbol.assign(date=["x", "y"]), symbol)


@pytest.fixture
def three_series() -> dict[str, pd.DataFrame]:
    """Drei Reihen mit bekannter Rangfolge über 10 Kerzen: AAA > BBB > CCC."""
    steps = np.arange(30)
    return {
        "aaa": _series_bars(100.0 * 1.02**steps),  # stark steigend
        "BBB": _series_bars(100.0 * 1.005**steps),  # leicht steigend
        "CCC": _series_bars(100.0 * 0.99**steps),  # fallend
    }


def test_rs_rank_known_order(three_series: dict[str, pd.DataFrame]) -> None:
    date = three_series["BBB"]["date"].iloc[-1]
    result = rs_rank(three_series, date, lookback=10)
    assert result.name == "rs_rank"
    assert result.to_dict() == {"AAA": 100.0, "BBB": 50.0, "CCC": 0.0}


def test_rs_rank_uses_only_data_up_to_date(three_series: dict[str, pd.DataFrame]) -> None:
    universe = dict(three_series)
    # CCC steigt erst nach dem Stichtag stark; das darf den Rang nicht beeinflussen.
    closes = universe["CCC"]["close"].to_numpy().copy()
    closes[20:] *= 10.0
    universe["CCC"] = universe["CCC"].assign(close=closes)
    date = universe["CCC"]["date"].iloc[19]
    assert rs_rank(universe, date, lookback=10)["CCC"] == 0.0
    returns = rs_returns(universe, date, lookback=10)
    assert returns["AAA"] == pytest.approx(1.02**10 - 1)


def test_rs_rank_only_members_on_date(three_series: dict[str, pd.DataFrame]) -> None:
    # AAA kommt am 2024-02-01 in den Index, CCC fliegt am selben Tag heraus.
    history = IndexHistory(
        index_code="TEST",
        current=frozenset({"AAA", "BBB"}),
        changes=(
            IndexChange(
                date=dt.date(2024, 2, 1), added=frozenset({"AAA"}), removed=frozenset({"CCC"})
            ),
        ),
    )
    before = rs_rank(three_series, "2024-01-31", lookback=10, index=history)
    after = rs_rank(three_series, "2024-02-01", lookback=10, index=history)
    assert before.to_dict() == {"BBB": 100.0, "CCC": 0.0}
    assert after.to_dict() == {"AAA": 100.0, "BBB": 0.0}


def test_rs_rank_with_index_code_and_symbol_set(three_series: dict[str, pd.DataFrame]) -> None:
    date = "2024-02-09"
    # Keiner der synthetischen Ticker ist im S&P 500 (point in time).
    assert rs_rank(three_series, date, lookback=10, index="SP500").empty
    assert rs_rank(three_series, date, lookback=10, index=["bbb"]).to_dict() == {"BBB": 100.0}
    with pytest.raises(IndicatorError):
        rs_rank(three_series, date, lookback=10, index="DAX")


def test_rs_rank_ties_short_history_and_lookback(three_series: dict[str, pd.DataFrame]) -> None:
    universe = {**three_series, "DDD": three_series["BBB"], "NEW": _series_bars([1.0, 2.0])}
    date = three_series["BBB"]["date"].iloc[-1]
    result = rs_rank(universe, date, lookback=10)
    assert "NEW" not in result.index
    assert result.to_dict() == {"AAA": 100.0, "BBB": 50.0, "DDD": 50.0, "CCC": 0.0}
    assert list(result.index) == ["AAA", "BBB", "DDD", "CCC"]
    with pytest.raises(IndicatorError, match="lookback"):
        rs_rank(universe, date, lookback=0)
    assert rs_rank({"X": _series_bars([0.0, 1.0, 2.0])}, date, lookback=2).empty
