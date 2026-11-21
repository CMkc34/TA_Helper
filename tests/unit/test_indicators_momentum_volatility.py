"""Momentum- und Volatilitäts-Indikatoren gegen Referenzwerte (TA-Lib bzw. unabhängig)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_helfer.core.bars import validate_bars
from trading_helfer.indicators import momentum, registry, volatility
from trading_helfer.indicators.registry import IndicatorError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "indicators"
TOLERANCE = 1e-6

# Referenzspalte -> (Indikator, Ausgabe, erwarteter erster gültiger Index)
CASES = {
    "rsi_rsi": ("rsi", "rsi", 14),
    "stoch_k": ("stoch", "k", 15),
    "stoch_d": ("stoch", "d", 17),
    "roc_roc": ("roc", "roc", 10),
    "willr_willr": ("willr", "willr", 13),
    "cci_cci": ("cci", "cci", 19),
    "atr_atr": ("atr", "atr", 14),
    "bbands_upper": ("bbands", "upper", 19),
    "bbands_middle": ("bbands", "middle", 19),
    "bbands_lower": ("bbands", "lower", 19),
    "keltner_upper": ("keltner", "upper", 19),
    "keltner_middle": ("keltner", "middle", 19),
    "keltner_lower": ("keltner", "lower", 19),
    "donchian_upper": ("donchian", "upper", 19),
    "donchian_middle": ("donchian", "middle", 19),
    "donchian_lower": ("donchian", "lower", 19),
    "hv_hv": ("hv", "hv", 20),
}


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return validate_bars(pd.read_csv(FIXTURES / "bars.csv"))


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "momentum_volatility_reference.csv")


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


def test_registry_lists_momentum_and_volatility() -> None:
    names = {spec.name for spec in registry.list_all()}
    expected = {"rsi", "stoch", "roc", "willr", "cci", "atr", "bbands", "keltner", "donchian"}
    assert expected | {"hv"} <= names


@pytest.mark.parametrize(
    ("name", "params", "pane", "outputs"),
    [
        ("rsi", {"period": 14, "source": "close"}, "separate", ("rsi",)),
        ("stoch", {"k_period": 14, "k_smooth": 3, "d_period": 3}, "separate", ("k", "d")),
        ("roc", {"period": 10, "source": "close"}, "separate", ("roc",)),
        ("willr", {"period": 14}, "separate", ("willr",)),
        ("cci", {"period": 20}, "separate", ("cci",)),
        ("atr", {"period": 14}, "separate", ("atr",)),
        (
            "bbands",
            {"period": 20, "num_std": 2.0, "source": "close"},
            "price",
            ("upper", "middle", "lower"),
        ),
        (
            "keltner",
            {"period": 20, "multiplier": 2.0, "atr_period": 10},
            "price",
            ("upper", "middle", "lower"),
        ),
        ("donchian", {"period": 20}, "price", ("upper", "middle", "lower")),
        (
            "hv",
            {"period": 20, "annualization": 252, "source": "close"},
            "separate",
            ("hv",),
        ),
    ],
)
def test_defaults(
    name: str, params: dict[str, object], pane: str, outputs: tuple[str, ...]
) -> None:
    spec = registry.get(name)
    assert dict(spec.params) == params
    assert spec.pane == pane
    assert spec.outputs == outputs


def test_oscillator_ranges(bars: pd.DataFrame) -> None:
    rsi = momentum.rsi(bars).dropna()
    stoch = momentum.stochastic(bars).dropna()
    willr = momentum.williams_r(bars).dropna()
    assert rsi.between(0, 100).all()
    assert stoch.stack().between(0, 100).all()
    assert willr.between(-100, 0).all()


def test_flat_prices_follow_talib_conventions() -> None:
    df = pd.DataFrame({"high": [5.0] * 40, "low": [5.0] * 40, "close": [5.0] * 40})
    assert (momentum.rsi(df).dropna() == 50).all()
    assert (momentum.stochastic(df).dropna() == 0).all().all()
    assert (momentum.williams_r(df).dropna() == 0).all()
    assert (momentum.cci(df).dropna() == 0).all()
    assert (momentum.roc(df).dropna() == 0).all()
    assert (volatility.historical_volatility(df).dropna() == 0).all()
    bands = volatility.bollinger(df).dropna()
    assert (bands["upper"] == bands["lower"]).all()


def test_rsi_extremes() -> None:
    up = pd.DataFrame({"close": np.arange(1.0, 31.0)})
    assert (momentum.rsi(up).dropna() == 100).all()
    assert (momentum.rsi(up.iloc[::-1].reset_index(drop=True)).dropna() == 0).all()


def test_roc_small_example() -> None:
    df = pd.DataFrame({"open": [1.0, 2.0, 3.0], "close": [100.0, 110.0, 121.0]})
    result = momentum.roc(df, period=1)
    assert math.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(10.0)
    assert result.iloc[2] == pytest.approx(10.0)
    assert momentum.roc(df, period=2, source="open").iloc[2] == pytest.approx(200.0)


def test_historical_volatility_constant_growth_and_weekly() -> None:
    df = pd.DataFrame({"close": 100.0 * 1.01 ** np.arange(30)})
    assert volatility.historical_volatility(df).dropna().abs().max() < 1e-12
    rng = np.random.default_rng(1)
    noisy = pd.DataFrame({"close": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))})
    daily = volatility.historical_volatility(noisy)
    weekly = volatility.historical_volatility(noisy, annualization=52)
    np.testing.assert_allclose(
        (daily / weekly).dropna().to_numpy(), math.sqrt(252 / 52), rtol=1e-12
    )


def test_atr_matches_trend_helper(bars: pd.DataFrame) -> None:
    from trading_helfer.indicators.trend import atr_wilder

    pd.testing.assert_series_equal(volatility.atr(bars, period=7), atr_wilder(bars, period=7))


def test_donchian_includes_current_bar() -> None:
    df = pd.DataFrame({"high": [1.0, 2.0, 5.0], "low": [0.5, 1.5, 0.1]})
    result = volatility.donchian(df, period=2)
    assert result["upper"].iloc[2] == 5.0
    assert result["lower"].iloc[2] == 0.1
    assert result["middle"].iloc[2] == pytest.approx(2.55)


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (momentum.rsi, {"period": 0}),
        (momentum.stochastic, {"k_smooth": 0}),
        (momentum.roc, {"period": 1.5}),
        (momentum.williams_r, {"period": -3}),
        (momentum.cci, {"period": 0}),
        (momentum.rsi, {"source": "adj_close"}),
        (volatility.atr, {"period": 0}),
        (volatility.bollinger, {"num_std": 0}),
        (volatility.keltner, {"atr_period": 0}),
        (volatility.keltner, {"multiplier": -1.0}),
        (volatility.donchian, {"period": 0}),
        (volatility.historical_volatility, {"period": 1}),
        (volatility.historical_volatility, {"annualization": 0}),
    ],
)
def test_invalid_parameters(bars: pd.DataFrame, func: object, kwargs: dict[str, object]) -> None:
    with pytest.raises(IndicatorError):
        func(bars, **kwargs)  # type: ignore[operator]
