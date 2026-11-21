"""Registry: Registrierung, Abfrage, generischer Lookahead-Test."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_helfer.core.bars import validate_bars
from trading_helfer.indicators import registry
from trading_helfer.indicators.registry import IndicatorError, indicator

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "indicators"


@pytest.fixture
def temp_name() -> Iterator[str]:
    name = "_test_indicator"
    yield name
    registry._unregister(name)


def test_register_and_get(temp_name: str) -> None:
    @indicator(temp_name, params={"period": 3}, pane="price")
    def func(df: pd.DataFrame, period: int = 3) -> pd.Series:
        return df["close"].rolling(period).mean()

    spec = registry.get(temp_name)
    assert spec.func is func
    assert spec.outputs == (temp_name,)
    assert spec.pane == "price"
    assert temp_name in [s.name for s in registry.list_all()]
    result = spec.compute(pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0]}), period=2)
    assert list(result.columns) == [temp_name]
    assert result[temp_name].tolist()[1:] == [1.5, 2.5, 3.5]


def test_params_are_read_only(temp_name: str) -> None:
    indicator(temp_name, params={"period": 3})(lambda df, period=3: df["close"])
    with pytest.raises(TypeError):
        registry.get(temp_name).params["period"] = 5  # type: ignore[index]


def test_duplicate_name_rejected() -> None:
    with pytest.raises(IndicatorError, match="bereits registriert"):
        indicator("sma")(lambda df: df["close"])


def test_invalid_pane_and_outputs() -> None:
    with pytest.raises(IndicatorError, match="Anzeigeart"):
        indicator("x", pane="oben")  # type: ignore[arg-type]
    with pytest.raises(IndicatorError, match="Ausgabe"):
        indicator("x", outputs=[])


def test_unknown_indicator_and_parameter() -> None:
    with pytest.raises(IndicatorError, match="Unbekannter Indikator"):
        registry.get("gibt_es_nicht")
    with pytest.raises(IndicatorError, match="Unbekannte"):
        registry.get("sma").compute(pd.DataFrame({"close": [1.0]}), length=3)


def test_list_all_sorted() -> None:
    names = [s.name for s in registry.list_all()]
    assert names == sorted(names)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return validate_bars(pd.read_csv(FIXTURES / "bars.csv"))


def _perturb_after(bars: pd.DataFrame, t: int, seed: int) -> pd.DataFrame:
    """Verändert alle Kerzen nach ``t`` zufällig, aber konsistent (High ≥ Open/Close ≥ Low)."""
    rng = np.random.default_rng(seed)
    changed = bars.copy()
    n = len(bars) - t - 1
    factor = np.exp(np.cumsum(rng.normal(0, 0.05, n)))
    for col in ("open", "high", "low", "close"):
        changed.loc[t + 1 :, col] = bars.loc[t + 1 :, col].to_numpy() * factor
    changed.loc[t + 1 :, "high"] *= 1.03
    changed.loc[t + 1 :, "low"] *= 0.97
    changed.loc[t + 1 :, "volume"] = rng.integers(1, 10_000_000, n)
    return changed


@pytest.mark.parametrize("spec", registry.list_all(), ids=lambda s: s.name)
@pytest.mark.parametrize("t", [0, 30, 60, 120, 250, 398])
def test_no_lookahead(bars: pd.DataFrame, spec: registry.IndicatorSpec, t: int) -> None:
    """Werte bis einschließlich t ändern sich nicht, wenn Daten nach t verändert werden."""
    original = spec.compute(bars)
    for seed in range(3):
        changed = spec.compute(_perturb_after(bars, t, seed))
        pd.testing.assert_frame_equal(original.iloc[: t + 1], changed.iloc[: t + 1])
