"""Tests für den DataService (Issue G-02) mit Fake-Kursquelle und temporärer Datenbank."""

from __future__ import annotations

import datetime as dt
import shutil
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from trading_helfer.core.config import ConfigError, Settings, update_config_file
from trading_helfer.data.providers.base import PriceProvider, empty_actions
from trading_helfer.data.quality import REPORT_COLUMNS, Severity
from trading_helfer.data.storage import MarketStore
from trading_helfer.data.update import BENCHMARKS, UpdateError
from trading_helfer.services.data_service import (
    AVAILABLE_PROVIDERS,
    DataService,
    empty_report,
    filter_report,
    mask_secret,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "csv"
START = dt.date(2024, 1, 1)
TODAY = dt.date(2024, 2, 29)


class FakeProvider:
    """Synthetischer Trend an Werktagen; ``JMP`` springt am 2024-02-01 um 80 %."""

    name = "fake"

    def __init__(self) -> None:
        self.symbols: list[str] = []

    def fetch_daily(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        self.symbols.append(symbol)
        days = pd.bdate_range(start, end)
        close = pd.Series(range(len(days)), dtype="float64") + 100.0
        if symbol == "JMP":
            close[days >= pd.Timestamp("2024-02-01")] *= 1.8
        return pd.DataFrame(
            {
                "date": days,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000,
                "adj_close": close,
            }
        )

    def fetch_actions(self, symbol: str) -> pd.DataFrame:
        return empty_actions()


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", start_date=START)


@pytest.fixture
def service(settings: Settings, provider: FakeProvider) -> DataService:
    def factory(_settings: Settings, _store: MarketStore) -> list[PriceProvider]:
        return [provider]

    return DataService(settings, provider_factory=factory)


# ---------------------------------------------------------------------- Hilfsfunktionen


def test_mask_secret_shows_only_last_four_characters() -> None:
    assert mask_secret(None) == ""
    assert mask_secret("") == ""
    assert mask_secret("abc") == "••••••••"
    assert mask_secret("12345678") == "••••••••"
    masked = mask_secret("geheim-key-1234")
    assert masked == "••••••••1234"
    assert "geheim" not in masked


def test_filter_report_by_severity() -> None:
    report = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "date": pd.to_datetime(["2024-01-02", None, "2024-01-03"]),
            "check": ["GAP", "MISSING_DATA", "JUMP"],
            "severity": ["ERROR", "ERROR", "WARNING"],
            "detail": ["x", "y", "z"],
        }
    )
    assert len(filter_report(report, None)) == 3
    assert filter_report(report, Severity.ERROR)["symbol"].tolist() == ["A", "B"]
    assert filter_report(report, Severity.WARNING)["symbol"].tolist() == ["C"]
    assert list(empty_report().columns) == list(REPORT_COLUMNS)
    assert filter_report(empty_report(), Severity.ERROR).empty


# ---------------------------------------------------------------------- Update


def test_without_database_there_is_no_update_and_no_report(service: DataService) -> None:
    assert not service.database_path.exists()
    assert service.last_update() is None
    report = service.quality_report()
    assert report.empty
    assert list(report.columns) == list(REPORT_COLUMNS)


def test_run_update_reports_progress_and_writes_log(
    service: DataService, provider: FakeProvider
) -> None:
    progress: list[tuple[int, str]] = []
    result = service.run_update(
        [], symbols=["AAA"], progress=lambda p, m: progress.append((p, m)), today=TODAY
    )
    assert result.ok == [*BENCHMARKS, "AAA"]
    assert not result.cancelled
    assert provider.symbols == [*BENCHMARKS, "AAA"]
    assert progress == [(33, "SPY (1/3)"), (67, "QQQ (2/3)"), (100, "AAA (3/3)")]

    last = service.last_update()
    assert last is not None
    assert last.finished_at is not None
    assert last.provider == "fake"
    assert last.symbols_ok == 3
    assert last.symbols_failed == 0
    assert last.message == result.summary()


def test_run_update_stops_after_current_symbol_when_cancelled(
    service: DataService, provider: FakeProvider
) -> None:
    done: list[str] = []

    def progress(_percent: int, message: str) -> None:
        done.append(message)

    result = service.run_update(
        [],
        symbols=["AAA", "BBB"],
        progress=progress,
        should_cancel=lambda: len(done) >= 2,
        today=TODAY,
    )
    assert result.cancelled
    assert provider.symbols == list(BENCHMARKS)
    assert result.done == 2
    last = service.last_update()
    assert last is not None
    assert last.message.startswith("Abgebrochen")


def test_run_update_with_unknown_index_fails(service: DataService) -> None:
    with pytest.raises(UpdateError):
        service.run_update(["DAX"], today=TODAY)


def test_quality_report_finds_jump(service: DataService) -> None:
    service.run_update([], symbols=["JMP"], today=TODAY)
    report = service.quality_report()
    jumps = report[report["check"] == "JUMP"]
    assert jumps["symbol"].tolist() == ["JMP"]
    only = service.quality_report(["SPY"])
    assert set(only["symbol"]) <= {"SPY"}


# ---------------------------------------------------------------------- CSV-Import


def test_import_csv_writes_bars(service: DataService, tmp_path: Path) -> None:
    source = tmp_path / "SYNB.csv"
    shutil.copy(FIXTURES / "per_symbol" / "SYNB.csv", source)
    report = service.import_csv(source)
    assert report.symbols == {"SYNB"}
    assert report.rows_imported > 0
    assert not (tmp_path / "SYNB.csv.fehler.csv").exists()
    with MarketStore(service.database_path, read_only=True) as store:
        assert store.symbols() == ["SYNB"]


def test_import_csv_writes_error_file(service: DataService, tmp_path: Path) -> None:
    source = tmp_path / "bad_rows.csv"
    shutil.copy(FIXTURES / "bad_rows.csv", source)
    report = service.import_csv(source)
    assert report.rows_rejected > 0
    assert (tmp_path / "bad_rows.csv.fehler.csv").exists()


# ---------------------------------------------------------------------- Einstellungen


def test_provider_settings_mask_key(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, tiingo_api_key="abcdefgh12345678")
    current = DataService(settings).provider_settings()
    assert current.order == ("yfinance", "tiingo")
    assert current.available == AVAILABLE_PROVIDERS
    assert current.has_tiingo_key
    assert current.tiingo_key_masked.endswith("5678")
    assert "abcd" not in current.tiingo_key_masked


def test_save_provider_settings_in_memory(tmp_path: Path) -> None:
    service = DataService(Settings(data_dir=tmp_path))
    current = service.save_provider_settings(["tiingo", "yfinance"], "  neuer-key-9999 ")
    assert current.order == ("tiingo", "yfinance")
    assert current.tiingo_key_masked.endswith("9999")
    key = service.settings.tiingo_api_key
    assert key is not None
    assert key.get_secret_value() == "neuer-key-9999"

    kept = service.save_provider_settings(["yfinance"], "")
    assert kept.order == ("yfinance",)
    assert kept.tiingo_key_masked.endswith("9999")

    cleared = service.save_provider_settings(["yfinance"], clear_key=True)
    assert not cleared.has_tiingo_key

    with pytest.raises(ConfigError):
        service.save_provider_settings([])


def test_save_provider_settings_writes_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TH_TIINGO_API_KEY", raising=False)
    monkeypatch.delenv("TH_PROVIDERS", raising=False)
    config = tmp_path / "config.toml"
    config.write_text(f'data_dir = "{tmp_path.as_posix()}"\ndefault_capital = 5000.0\n')
    service = DataService(Settings(data_dir=tmp_path), config)

    service.save_provider_settings(["tiingo", "yfinance"], "datei-key-4321")
    data = tomllib.loads(config.read_text(encoding="utf-8"))
    assert data["providers"] == ["tiingo", "yfinance"]
    assert data["tiingo_api_key"] == "datei-key-4321"
    assert data["default_capital"] == 5000.0
    assert service.settings.default_capital == 5000.0

    service.save_provider_settings(["yfinance"], clear_key=True)
    data = tomllib.loads(config.read_text(encoding="utf-8"))
    assert "tiingo_api_key" not in data
    assert data["providers"] == ["yfinance"]


# ---------------------------------------------------------------------- update_config_file


def test_update_config_file_creates_missing_file(tmp_path: Path) -> None:
    config = tmp_path / "neu" / "config.toml"
    settings = update_config_file(config, {"providers": ["tiingo"]})
    assert config.exists()
    assert settings.providers == ["tiingo"]


def test_update_config_file_rejects_invalid_values(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('providers = ["yfinance"]\n')
    with pytest.raises(ConfigError, match="providers"):
        update_config_file(config, {"providers": ["yfinance", "yfinance"]})
    assert config.read_text() == 'providers = ["yfinance"]\n'


def test_update_config_file_rejects_broken_toml(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("providers = [\n")
    with pytest.raises(ConfigError, match="kein gültiges TOML"):
        update_config_file(config, {"providers": ["tiingo"]})


def test_update_config_file_reports_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"

    def fail(*_args: object, **_kwargs: object) -> int:
        raise OSError("schreibgeschützt")

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(ConfigError, match="nicht geschrieben"):
        update_config_file(config, {"providers": ["tiingo"]})
