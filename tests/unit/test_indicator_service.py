"""IndicatorService: Parameterfelder, Normalisierung, Berechnung, Vorlagen, Marker."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_helfer.core.markers import Marker, MarkerKind
from trading_helfer.indicators import registry
from trading_helfer.indicators.registry import IndicatorError, indicator
from trading_helfer.services.indicator_service import (
    LINE_COLORS,
    SOURCE_CHOICES,
    TEMPLATES_FILE_NAME,
    IndicatorConfig,
    IndicatorService,
    ParamField,
    TemplateError,
    marker_indices,
)


@pytest.fixture
def service() -> IndicatorService:
    return IndicatorService()


@pytest.fixture
def bars() -> pd.DataFrame:
    n = 60
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )


def test_available_matches_registry(service: IndicatorService) -> None:
    assert [s.name for s in service.available()] == [s.name for s in registry.list_all()]
    assert service.spec("sma").pane == "price"


def test_param_fields_from_metadata(service: IndicatorService) -> None:
    fields = service.param_fields("macd")
    assert [f.name for f in fields] == ["fast", "slow", "signal", "source"]
    assert fields[0] == ParamField("fast", 12, "int")
    assert fields[3].kind == "choice"
    assert fields[3].choices == SOURCE_CHOICES
    psar = service.param_fields("psar")
    assert {f.kind for f in psar} == {"float"}


def test_default_config_assigns_colors_per_visible_output(service: IndicatorService) -> None:
    config = service.default_config("bbands", color_offset=1)
    assert config.params == {"period": 20, "num_std": 2.0, "source": "close"}
    assert config.colors == {
        "upper": LINE_COLORS[1],
        "middle": LINE_COLORS[2],
        "lower": LINE_COLORS[3],
    }
    supertrend = service.default_config("supertrend")
    assert list(supertrend.colors) == ["supertrend"]  # direction ist versteckt


def test_normalize_coerces_and_keeps_colors(service: IndicatorService) -> None:
    config = service.normalize(
        IndicatorConfig(name="sma", params={"period": "50"}, colors={"sma": "#123456"})
    )
    assert config.params == {"period": 50, "source": "close"}
    assert config.colors == {"sma": "#123456"}
    floats = service.normalize(IndicatorConfig(name="psar", params={"acceleration": 1}))
    assert floats.params["acceleration"] == 1.0
    assert isinstance(floats.params["acceleration"], float)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"bogus": 1}, "Unbekannte"),
        ({"period": "abc"}, "muss eine Zahl sein"),
        ({"period": 2.5}, "muss eine Zahl sein"),
        ({"source": "volume"}, "nicht erlaubt"),
    ],
)
def test_normalize_rejects_invalid(
    service: IndicatorService, params: dict[str, object], message: str
) -> None:
    with pytest.raises(IndicatorError, match=message):
        service.normalize(IndicatorConfig(name="sma", params=params))


def test_normalize_unknown_indicator(service: IndicatorService) -> None:
    with pytest.raises(IndicatorError, match="Unbekannter Indikator"):
        service.normalize(IndicatorConfig(name="nope"))


def test_label(service: IndicatorService) -> None:
    assert service.label(service.default_config("sma")) == "SMA(20, close)"
    assert service.label(service.default_config("psar")) == "PSAR(0,02, 0,2)"


def test_label_without_params(service: IndicatorService) -> None:
    @indicator("_noparam", pane="separate")
    def func(df: pd.DataFrame) -> pd.Series:
        return df["close"]

    try:
        assert service.label(IndicatorConfig(name="_noparam")) == "_NOPARAM"
        assert service.param_fields("_noparam") == []
    finally:
        registry._unregister("_noparam")


def test_compute_returns_all_outputs(service: IndicatorService, bars: pd.DataFrame) -> None:
    frame = service.compute(service.default_config("sma"), bars)
    assert list(frame.columns) == ["sma"]
    assert frame["sma"].iloc[19] == pytest.approx(bars["close"].iloc[:20].mean())
    st = service.compute(service.default_config("supertrend"), bars)
    assert list(st.columns) == ["supertrend", "direction"]


def test_compute_wraps_errors(service: IndicatorService, bars: pd.DataFrame) -> None:
    @indicator("_broken", params={"x": 1})
    def broken(df: pd.DataFrame, x: int = 1) -> pd.Series:
        raise ValueError("kaputt")

    try:
        with pytest.raises(IndicatorError, match=r"_BROKEN\(1\) konnte nicht berechnet"):
            service.compute(IndicatorConfig(name="_broken"), bars)
    finally:
        registry._unregister("_broken")
    with pytest.raises(IndicatorError):
        service.compute(IndicatorConfig(name="sma", params={"period": 0}), bars)


# -- Vorlagen ------------------------------------------------------------------------


@pytest.fixture
def templates_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / TEMPLATES_FILE_NAME


def test_templates_survive_restart(templates_path: Path) -> None:
    first = IndicatorService(templates_path)
    assert first.templates_path == templates_path
    assert first.template_names() == []
    configs = [first.default_config("sma"), first.default_config("rsi", 1)]
    assert first.save_template("  Trend 1W ", configs) == "Trend 1W"
    assert templates_path.exists()

    second = IndicatorService(templates_path)
    assert second.template_names() == ["Trend 1W"]
    assert second.template("Trend 1W") == configs


def test_template_overwrite_delete_and_errors(templates_path: Path) -> None:
    svc = IndicatorService(templates_path)
    svc.save_template("b", [svc.default_config("sma")])
    svc.save_template("A", [svc.default_config("ema")])
    svc.save_template("b", [svc.default_config("wma")])
    assert svc.template_names() == ["A", "b"]
    assert svc.template("b")[0].name == "wma"
    copy = svc.template("b")
    copy[0].params["period"] = 99
    assert svc.template("b")[0].params["period"] == 20  # Kopie, nicht das Original
    svc.delete_template("b")
    svc.delete_template("gibt es nicht")
    assert IndicatorService(templates_path).template_names() == ["A"]
    with pytest.raises(TemplateError, match="gibt es nicht"):
        svc.template("b")
    with pytest.raises(TemplateError, match="Namen"):
        svc.save_template("  ", [svc.default_config("sma")])
    with pytest.raises(TemplateError, match="höchstens"):
        svc.save_template("x" * 61, [svc.default_config("sma")])
    with pytest.raises(TemplateError, match="mindestens einen"):
        svc.save_template("leer", [])


def test_broken_template_file_is_ignored(templates_path: Path) -> None:
    templates_path.parent.mkdir(parents=True)
    templates_path.write_text("{kein json", encoding="utf-8")
    assert IndicatorService(templates_path).template_names() == []


def test_unwritable_template_file_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "datei"
    blocker.write_text("", encoding="utf-8")
    svc = IndicatorService(blocker / TEMPLATES_FILE_NAME)
    svc.save_template("x", [svc.default_config("sma")])
    assert svc.template_names() == ["x"]
    assert "können nicht gespeichert" in caplog.text


def test_memory_only_templates() -> None:
    svc = IndicatorService()
    svc.save_template("x", [svc.default_config("sma")])
    assert svc.template_names() == ["x"]


# -- Marker --------------------------------------------------------------------------


def _dates(*days: str) -> np.ndarray:
    return np.array(days, dtype="datetime64[D]")


def test_marker_indices_daily() -> None:
    dates = _dates("2024-01-02", "2024-01-03", "2024-01-04")
    markers = [
        Marker(date(2024, 1, 3), MarkerKind.BUY),
        Marker(date(2024, 1, 4), MarkerKind.SELL),
        Marker(date(2024, 1, 5)),  # nach der letzten Kerze
        Marker(date(2023, 12, 1)),  # weit vor der ersten Kerze
    ]
    assert [(i, m.kind) for i, m in marker_indices(markers, dates)] == [
        (1, MarkerKind.BUY),
        (2, MarkerKind.SELL),
    ]


def test_marker_indices_weekly_maps_into_period() -> None:
    # Wochenkerzen tragen das Datum des letzten Handelstags (Freitag).
    dates = _dates("2024-01-05", "2024-01-12", "2024-01-19")
    markers = [
        Marker(date(2024, 1, 2)),  # in der ersten Woche
        Marker(date(2024, 1, 8)),  # Montag der zweiten Woche
        Marker(date(2024, 1, 19)),
    ]
    assert [i for i, _m in marker_indices(markers, dates)] == [0, 1, 2]
    assert marker_indices(markers, _dates()) == []
    assert [i for i, _m in marker_indices(markers[:1], _dates("2024-01-05"))] == []
