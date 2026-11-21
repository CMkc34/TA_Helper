"""ScannerService (G-09): Strategien, Scan mit Verlauf, Laden, Löschen, Export."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.core.config import Settings
from trading_helfer.core.markers import MarkerKind
from trading_helfer.core.types import SignalType, Timeframe
from trading_helfer.data.storage import MarketStore
from trading_helfer.scanner.scan import RESULT_COLUMNS, ScanError, ScanFilters
from trading_helfer.services.scanner_service import (
    METRIC_CHOICES,
    ScannerService,
    ScanRecord,
    ScanRequest,
    empty_result,
    parse_symbols,
    signal_marker,
)
from trading_helfer.strategies.base import BarContext, Order, Param, Strategy
from trading_helfer.strategies.registry import StrategyInfo, StrategyRegistry


class AlwaysShort(Strategy):
    """Gibt auf jeder Kerze Ausstieg (falls Position) und danach eine Short-Order ab."""

    id = "test_always_short"
    name = "Immer short"
    version = "1.0.0"
    description = "Test"
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D3, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {"qty": Param(1, 1, 10, 1, "Stück")}

    def on_bar(self, ctx: BarContext) -> list[Order]:
        return [*ctx.exit(), ctx.sell_short(qty=1, tag="probe")]


class Another(AlwaysShort):
    id = "test_another"
    name = "Andere"


def wave_bars(days: int = 400, phase: float = 0.0) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=days)
    t = np.arange(days, dtype=float)
    close = 100.0 + 0.05 * t + 10.0 * np.sin(2 * np.pi * (t + phase) / 60.0)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.3,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000,
        }
    )


def make_registry() -> StrategyRegistry:
    registry = StrategyRegistry(include_builtin=False)
    registry._strategies = {
        cls.id: StrategyInfo(id=cls.id, cls=cls, source="test") for cls in (AlwaysShort, Another)
    }
    return registry


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    settings = Settings(data_dir=tmp_path / "data")
    with MarketStore(settings.database_path) as store:
        for i, symbol in enumerate(["AAA", "BBB", "CCC"]):
            store.upsert_bars(symbol, wave_bars(phase=7.0 * i))
        store.upsert_instruments(
            pd.DataFrame(
                {"symbol": ["AAA", "BBB", "CCC"], "sector": ["Technology", "Energy", None]}
            )
        )
    return settings


class Clock:
    def __init__(self) -> None:
        self.now = dt.datetime(2024, 5, 1, 9, 30, 15, 123)

    def __call__(self) -> dt.datetime:
        self.now += dt.timedelta(minutes=1)
        return self.now


@pytest.fixture
def service(settings: Settings) -> ScannerService:
    return ScannerService(settings, make_registry(), clock=Clock())


def test_registry_helper_registers_strategies() -> None:
    assert make_registry().ids() == ["test_always_short", "test_another"]


def test_strategies_sorted_with_timeframes_and_defaults(service: ScannerService) -> None:
    choices = service.strategies()
    assert [c.id for c in choices] == ["test_another", "test_always_short"]
    assert choices[1].timeframes == (Timeframe.D3, Timeframe.W1)
    assert choices[1].defaults == {"qty": 1}


def test_default_registry_loads_builtin_strategies(tmp_path: Path) -> None:
    service = ScannerService(Settings(data_dir=tmp_path))
    ids = [c.id for c in service.strategies()]
    assert ids  # eingebaute Strategien
    assert service.registry is service.registry


def test_sectors(service: ScannerService) -> None:
    assert service.sectors() == ["Energy", "Technology"]


def test_without_database_everything_is_empty(tmp_path: Path) -> None:
    service = ScannerService(Settings(data_dir=tmp_path / "leer"), make_registry())
    assert service.sectors() == []
    assert service.history() == []
    loaded = service.load("x")
    assert loaded.empty
    assert loaded.attrs["scan_id"] == "x"
    service.delete("x")
    assert not (tmp_path / "leer").exists()


def test_run_saves_history_and_signals(service: ScannerService) -> None:
    progress: list[tuple[int, str]] = []
    request = ScanRequest(
        "test_always_short",
        ("AAA", "BBB", "CCC"),
        Timeframe.W1,
        params={"qty": 2},
        filters=ScanFilters(sectors=frozenset({"Technology", "Energy"})),
    )
    result = service.run(request, progress=lambda p, m: progress.append((p, m)))
    assert list(result["symbol"]) == ["AAA", "BBB"]
    assert set(result["signal"]) == {SignalType.SELL.value}
    assert progress[-1] == (100, "2/2")

    history = service.history()
    assert len(history) == 1
    record = history[0]
    assert record.scan_id == result.attrs["scan_id"]
    assert record.strategy_id == "test_always_short"
    assert record.universe == "AAA,BBB,CCC"
    assert record.universe_label == "AAA, BBB, CCC"
    assert record.timeframe is Timeframe.W1
    assert record.as_of is None
    assert record.signal_count == 2
    assert record.params == {"qty": 2}
    assert record.created_at == dt.datetime(2024, 5, 1, 9, 31, 15)

    with MarketStore(service.settings.database_path) as store:
        filters = store.connection.execute("SELECT filters_json FROM scans").fetchone()[0]
    assert json.loads(filters)["sectors"] == ["Energy", "Technology"]


def test_history_newest_first_and_load(service: ScannerService) -> None:
    first = service.run(ScanRequest("test_always_short", ("AAA",), Timeframe.W1))
    second = service.run(
        ScanRequest("test_another", ("BBB", "CCC"), Timeframe.D3, as_of=dt.date(2021, 3, 31))
    )
    history = service.history()
    assert [r.scan_id for r in history] == [second.attrs["scan_id"], first.attrs["scan_id"]]
    assert history[0].as_of == dt.date(2021, 3, 31)
    assert history[0].timeframe is Timeframe.D3

    loaded = service.load(second.attrs["scan_id"])
    assert list(loaded.columns) == list(RESULT_COLUMNS)
    assert list(loaded["symbol"]) == ["BBB", "CCC"]
    assert list(loaded["sector"].fillna("")) == ["Energy", ""]
    assert list(loaded["date"]) == list(second["date"])
    assert loaded.attrs["scan_id"] == second.attrs["scan_id"]


def test_empty_scan_appears_in_history(service: ScannerService) -> None:
    request = ScanRequest(
        "test_always_short",
        ("AAA",),
        Timeframe.W1,
        filters=ScanFilters(signals=frozenset({SignalType.BUY})),
    )
    result = service.run(request)
    assert result.empty
    assert service.history()[0].signal_count == 0
    assert service.load(result.attrs["scan_id"]).empty


def test_delete_removes_scan_and_signals(service: ScannerService) -> None:
    result = service.run(ScanRequest("test_always_short", ("AAA", "BBB"), Timeframe.W1))
    service.delete(result.attrs["scan_id"])
    assert service.history() == []
    assert service.load(result.attrs["scan_id"]).empty


def test_index_universe_is_stored_as_code(
    service: ScannerService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_helfer.scanner import scan as scan_module

    monkeypatch.setattr(scan_module, "members_on", lambda _code, _day: {"AAA"})
    service.run(ScanRequest("test_always_short", "SP500", Timeframe.W1))
    record = service.history()[0]
    assert record.universe == "SP500"
    assert record.universe_label == "S&P 500"


def test_invalid_requests(service: ScannerService) -> None:
    with pytest.raises(ScanError, match="mindestens ein Symbol"):
        service.run(ScanRequest("test_always_short", (), Timeframe.W1))
    with pytest.raises(ScanError, match="Zeitrahmen"):
        service.run(ScanRequest("test_always_short", ("AAA",), Timeframe.D1))
    with pytest.raises(KeyError):
        service.run(ScanRequest("gibt_es_nicht", ("AAA",), Timeframe.W1))
    assert service.history() == []


def test_export_csv(service: ScannerService, tmp_path: Path) -> None:
    result = service.run(ScanRequest("test_always_short", ("AAA",), Timeframe.W1))
    path = service.export_csv(result, tmp_path / "out" / "scan.csv")
    frame = pd.read_csv(path)
    assert list(frame["symbol"]) == ["AAA"]
    assert list(frame.columns[: len(RESULT_COLUMNS)]) == list(RESULT_COLUMNS)


def test_parse_symbols() -> None:
    assert parse_symbols(" aapl, msft;nvda  AAPL ") == ("AAPL", "MSFT", "NVDA")
    assert parse_symbols("") == ()


def test_signal_marker() -> None:
    marker = signal_marker(
        {"date": pd.Timestamp("2024-03-08"), "signal": "BUY", "note": "BUY MARKET"}
    )
    assert marker.date == dt.date(2024, 3, 8)
    assert marker.kind is MarkerKind.BUY
    assert marker.text == "BUY: BUY MARKET"
    assert signal_marker({"date": "2024-03-08", "signal": "EXIT"}).kind is MarkerKind.EXIT
    assert signal_marker({"date": "2024-03-08", "signal": "SELL", "note": None}).text == "SELL"
    assert signal_marker({"date": "2024-03-08", "signal": "?"}).kind is MarkerKind.SIGNAL


def test_helpers() -> None:
    assert list(empty_result().columns) == list(RESULT_COLUMNS)
    assert "market_cap" not in METRIC_CHOICES
    assert "pe_ratio" in METRIC_CHOICES
    record = ScanRecord("id", dt.datetime(2024, 1, 1), "s", "NDX100", Timeframe.W1, None, 0)
    assert record.universe_label == "Nasdaq-100"
    assert ScanRequest("s", "SP500", Timeframe.W1).universe_text == "SP500"
