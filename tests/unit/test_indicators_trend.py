"""Trend-Indikatoren gegen Referenzwerte (TA-Lib bzw. unabhängige Implementierung)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_helfer.core.bars import validate_bars
from trading_helfer.indicators import registry, trend
from trading_helfer.indicators.registry import IndicatorError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "indicators"
TOLERANCE = 1e-6

# Referenzspalte -> (Indikator, Ausgabe, erwarteter erster gültiger Index)
CASES = {
    "sma_20": ("sma", "sma", 19),
    "ema_20": ("ema", "ema", 19),
    "wma_20": ("wma", "wma", 19),
    "macd_macd": ("macd", "macd", 25),
    "macd_signal": ("macd", "signal", 33),
    "macd_hist": ("macd", "hist", 33),
    "adx_adx": ("adx", "adx", 27),
    "adx_plus_di": ("adx", "plus_di", 14),
    "adx_minus_di": ("adx", "minus_di", 14),
    "psar_psar": ("psar", "psar", 1),
    "ichimoku_tenkan": ("ichimoku", "tenkan", 8),
    "ichimoku_kijun": ("ichimoku", "kijun", 25),
    "ichimoku_senkou_a": ("ichimoku", "senkou_a", 51),
    "ichimoku_senkou_b": ("ichimoku", "senkou_b", 77),
    "supertrend_supertrend": ("supertrend", "supertrend", 10),
    "supertrend_direction": ("supertrend", "direction", 10),
}


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return validate_bars(pd.read_csv(FIXTURES / "bars.csv"))


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "trend_reference.csv")


@pytest.mark.parametrize("column", sorted(CASES))
def test_matches_reference(bars: pd.DataFrame, reference: pd.DataFrame, column: str) -> None:
    name, output, first_valid = CASES[column]
    result = registry.get(name).compute(bars)[output]
    expected = reference[column].to_numpy()

    assert result.index.equals(bars.index)
    assert result.iloc[:first_valid].isna().all()
    assert result.iloc[first_valid:].notna().all()
    mask = ~np.isnan(expected)
    assert mask[first_valid:].any()
    np.testing.assert_allclose(result.to_numpy()[mask], expected[mask], rtol=0, atol=TOLERANCE)


def test_reference_data_covers_trend_changes(reference: pd.DataFrame) -> None:
    """Die Testdaten enthalten Trendwechsel, damit SAR und Supertrend umschalten."""
    direction = reference["supertrend_direction"].dropna()
    assert (direction.diff().fillna(0) != 0).sum() >= 3


def test_registry_lists_trend_indicators() -> None:
    names = {spec.name for spec in registry.list_all()}
    assert {"sma", "ema", "wma", "macd", "adx", "ichimoku", "psar", "supertrend"} <= names


@pytest.mark.parametrize(
    ("name", "params", "pane", "outputs"),
    [
        ("sma", {"period": 20, "source": "close"}, "price", ("sma",)),
        (
            "macd",
            {"fast": 12, "slow": 26, "signal": 9, "source": "close"},
            "separate",
            ("macd", "signal", "hist"),
        ),
        ("adx", {"period": 14}, "separate", ("adx", "plus_di", "minus_di")),
        (
            "ichimoku",
            {"tenkan": 9, "kijun": 26, "senkou": 52},
            "price",
            ("tenkan", "kijun", "senkou_a", "senkou_b"),
        ),
        ("psar", {"acceleration": 0.02, "maximum": 0.2}, "price", ("psar",)),
        ("supertrend", {"period": 10, "multiplier": 3.0}, "price", ("supertrend", "direction")),
    ],
)
def test_defaults(
    name: str, params: dict[str, object], pane: str, outputs: tuple[str, ...]
) -> None:
    spec = registry.get(name)
    assert dict(spec.params) == params
    assert spec.pane == pane
    assert spec.outputs == outputs


def test_custom_parameters_and_source(bars: pd.DataFrame) -> None:
    result = trend.sma(bars, period=3, source="high")
    expected = bars["high"].rolling(3).mean()
    np.testing.assert_allclose(result.to_numpy()[2:], expected.to_numpy()[2:], atol=1e-12)
    assert result.name == "sma"


def test_ema_matches_recursive_definition() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 10.0]})
    result = trend.ema(df, period=3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)
    assert result.iloc[4] == pytest.approx(6.5)


def test_wma_small_example() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]})
    result = trend.wma(df, period=3)
    assert result.iloc[2] == pytest.approx((1 + 4 + 9) / 6)
    assert result.iloc[3] == pytest.approx((2 + 6 + 12) / 6)


def test_index_is_preserved_for_custom_index(bars: pd.DataFrame) -> None:
    df = bars.set_index("date")
    for spec in registry.list_all():
        assert spec.compute(df).index.equals(df.index), spec.name


@pytest.mark.parametrize("length", [0, 1, 5])
def test_short_input_is_all_nan(bars: pd.DataFrame, length: int) -> None:
    df = bars.iloc[:length]
    for spec in registry.list_all():
        result = spec.compute(df)
        assert len(result) == length
        # OBV ist kumulativ und hat keine Anlaufphase.
        if spec.name == "obv":
            assert result.notna().all().all()
        elif spec.name != "psar" or length < 2:
            assert result.isna().all().all(), spec.name


def test_atr_wilder_first_value(bars: pd.DataFrame) -> None:
    atr = trend.atr_wilder(bars, period=3)
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    assert atr.iloc[:3].isna().all()
    assert atr.iloc[3] == pytest.approx(tr.iloc[1:4].mean())


def test_psar_starts_short_when_first_move_is_down() -> None:
    df = pd.DataFrame({"high": [10.0, 9.5, 9.0, 8.5], "low": [9.0, 8.0, 7.5, 7.0]})
    result = trend.parabolic_sar(df)
    assert result.iloc[1] == pytest.approx(10.0)  # SAR über dem Kurs: Abwärtstrend


def test_flat_prices_give_zero_di() -> None:
    df = pd.DataFrame({"high": [1.0] * 40, "low": [1.0] * 40, "close": [1.0] * 40})
    result = trend.adx(df)
    assert (result["plus_di"].dropna() == 0).all()
    assert (result["adx"].dropna() == 0).all()


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (trend.sma, {"period": 0}),
        (trend.ema, {"period": 2.5}),
        (trend.macd, {"fast": 26, "slow": 12}),
        (trend.macd, {"signal": -1}),
        (trend.ichimoku, {"kijun": 0}),
        (trend.parabolic_sar, {"acceleration": 0}),
        (trend.supertrend, {"multiplier": 0}),
        (trend.sma, {"source": "adj_close"}),
    ],
)
def test_invalid_parameters(bars: pd.DataFrame, func: object, kwargs: dict[str, object]) -> None:
    with pytest.raises(IndicatorError):
        func(bars, **kwargs)  # type: ignore[operator]
