"""Tests für den CSV-Import (D-04)."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from trading_helfer.app import main
from trading_helfer.data.providers.base import PriceProvider, ProviderError
from trading_helfer.data.providers.csv_import import (
    CsvImportProvider,
    CsvImportReport,
    CsvMappingError,
    RowError,
    import_csv,
    load_mapping,
    read_csv_bars,
    resolve_mapping,
)
from trading_helfer.data.storage import MarketStore

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "csv"


@pytest.fixture
def store() -> MarketStore:
    return MarketStore(":memory:")


def _sources(store: MarketStore, symbol: str) -> set[str]:
    rows = store.connection.execute(
        "SELECT DISTINCT source FROM bars_daily WHERE symbol = ?", [symbol]
    ).fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------- Dateiformen


def test_one_file_per_symbol_directory(store: MarketStore) -> None:
    report = import_csv(store, FIXTURES / "per_symbol")

    assert report.files == ["SYNA.csv", "SYNB.csv"]
    assert report.rows_read == 5
    assert report.rows_imported == 5
    assert report.errors == []
    assert report.symbols == {"SYNA", "SYNB"}
    raw = store.load_bars("SYNA", adjusted=False)
    assert raw["close"].tolist() == [10.5, 11.5, 12.0]
    assert raw["adj_close"].tolist() == [5.25, 5.75, 6.0]
    assert store.load_bars("SYNA")["close"].tolist() == [5.25, 5.75, 6.0]
    # ohne Adj Close gilt adj_close = close
    assert store.load_bars("SYNB")["adj_factor"].tolist() == [1.0, 1.0]
    assert _sources(store, "SYNA") == {"csv:SYNA.csv"}
    assert _sources(store, "SYNB") == {"csv:SYNB.csv"}


def test_single_file_per_symbol(store: MarketStore) -> None:
    report = import_csv(store, FIXTURES / "per_symbol" / "SYNB.csv")
    assert report.rows_imported == 2
    assert store.symbols() == ["SYNB"]


def test_combined_file_with_symbol_column(store: MarketStore) -> None:
    report = import_csv(store, FIXTURES / "combined.csv")

    assert report.rows_imported == 4
    assert report.symbols == {"SYNA", "SYNB"}
    assert store.symbols() == ["SYNA", "SYNB"]
    assert store.load_bars("SYNB")["volume"].tolist() == [300, 400]
    assert _sources(store, "SYNA") == {"csv:combined.csv"}


def test_custom_mapping_from_toml(store: MarketStore) -> None:
    mapping = load_mapping(FIXTURES / "custom_mapping.toml")
    report = import_csv(store, FIXTURES / "custom.csv", mapping, date_format="%d.%m.%Y")

    assert report.rows_imported == 2
    bars = store.load_bars("SYNC")
    assert bars["date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert bars["volume"].tolist() == [100, 200]


def test_reimport_overwrites_existing_days(store: MarketStore, tmp_path: Path) -> None:
    import_csv(store, FIXTURES / "per_symbol" / "SYNB.csv")
    file = tmp_path / "SYNB.csv"
    file.write_text("Date,Open,High,Low,Close,Volume\n2024-01-03,1,2,0.5,1.5,7\n", "utf-8")
    import_csv(store, file)
    bars = store.load_bars("SYNB")
    assert bars["close"].tolist() == [20.5, 1.5]


def test_timezone_offsets_keep_local_day(tmp_path: Path) -> None:
    file = tmp_path / "TZ.csv"
    file.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02 00:00:00-05:00,1,2,0.5,1.5,7\n"
        "2024-07-02 00:00:00-04:00,1,2,0.5,1.5,7\n",
        "utf-8",
    )
    bars, report = read_csv_bars(file)
    assert report.errors == []
    assert bars["date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-07-02")]


def test_single_timezone_offset(tmp_path: Path) -> None:
    file = tmp_path / "TZ.csv"
    file.write_text("Date,Open,High,Low,Close,Volume\n2024-01-02T00:00:00Z,1,2,0.5,1.5,7\n")
    bars, _ = read_csv_bars(file)
    assert bars["date"].tolist() == [pd.Timestamp("2024-01-02")]
    assert str(bars["date"].dtype) == "datetime64[ns]"


# ---------------------------------------------------------------- Mapping


def test_resolve_mapping_detects_common_names() -> None:
    columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert resolve_mapping(columns) == {
        "Date": "date",
        "Ticker": "symbol",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }


def test_resolve_mapping_prefers_symbol_over_name() -> None:
    columns = ["date", "Name", "Symbol", "open", "high", "low", "close", "adj_close", "volume"]
    resolved = resolve_mapping(columns)
    assert resolved["Symbol"] == "symbol"
    assert "Name" not in resolved
    assert resolved["adj_close"] == "adj_close"


def test_resolve_mapping_explicit_wins_over_detection() -> None:
    columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Kurs"]
    resolved = resolve_mapping(columns, {"Kurs": "close"})
    assert resolved["Kurs"] == "close"
    assert "Close" not in resolved


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"Datum": "date"}, "fehlt in der Datei"),
        ({"Date": "datum"}, "Unbekannte Zielspalte"),
        ({"Open": "close", "Close": "close"}, "mehrfach"),
    ],
)
def test_wrong_mapping_is_rejected(mapping: dict[str, str], message: str) -> None:
    columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    with pytest.raises(CsvMappingError, match=message):
        resolve_mapping(columns, mapping)


def test_missing_required_column_without_mapping(store: MarketStore) -> None:
    with pytest.raises(CsvMappingError, match="Pflichtspalte"):
        import_csv(store, FIXTURES / "custom.csv")
    assert store.symbols() == []


def test_wrong_mapping_file_aborts_import(store: MarketStore) -> None:
    with pytest.raises(CsvMappingError, match="fehlt in der Datei"):
        import_csv(store, FIXTURES / "combined.csv", {"Stueck": "volume"})
    assert store.symbols() == []


def test_load_mapping_flat_table(tmp_path: Path) -> None:
    file = tmp_path / "m.toml"
    file.write_text('Datum = "date"\n', "utf-8")
    assert load_mapping(file) == {"Datum": "date"}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("columns = [1, 2]\n", "erwartet wird"),
        ("[columns]\nDatum = 1\n", "erwartet wird"),
        ("[columns\n", "kein gültiges TOML"),
    ],
)
def test_load_mapping_invalid(tmp_path: Path, content: str, message: str) -> None:
    file = tmp_path / "m.toml"
    file.write_text(content, "utf-8")
    with pytest.raises(CsvMappingError, match=message):
        load_mapping(file)


def test_load_mapping_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CsvMappingError, match="nicht lesbar"):
        load_mapping(tmp_path / "fehlt.toml")


# ---------------------------------------------------------------- Fehlerhafte Zeilen


def test_bad_rows_go_to_error_report(store: MarketStore) -> None:
    report = import_csv(store, FIXTURES / "bad_rows.csv")

    assert report.rows_read == 12
    assert report.rows_imported == 2
    assert report.rows_rejected == 9
    reasons = {e.line: e.reason for e in report.errors}
    assert reasons == {
        3: "Datum fehlt oder ist nicht lesbar",
        4: "open fehlt oder ist keine Zahl",
        5: "Symbol fehlt",
        6: "close ist nicht positiv",
        7: "volume ist keine nichtnegative ganze Zahl",
        8: "volume fehlt oder ist keine Zahl",
        11: "Datum doppelt mit abweichenden Werten",
        12: "Datum doppelt mit abweichenden Werten",
        13: "volume ist keine nichtnegative ganze Zahl",
    }
    assert all(e.file == "bad_rows.csv" for e in report.errors)
    assert next(e for e in report.errors if e.line == 5).symbol is None
    assert next(e for e in report.errors if e.line == 3).symbol == "SYNA"
    assert store.load_bars("SYNA")["date"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-10"),
    ]


def test_write_errors_csv(tmp_path: Path) -> None:
    report = CsvImportReport(errors=[RowError("a.csv", 3, None, "Symbol fehlt")])
    out = report.write_errors(tmp_path / "sub" / "fehler.csv")
    assert out.read_text("utf-8").splitlines() == [
        "file,line,symbol,reason",
        "a.csv,3,,Symbol fehlt",
    ]


def test_summary_text() -> None:
    report = CsvImportReport(files=["a.csv"], rows_read=3, rows_imported=2, symbols={"A"})
    report.errors.append(RowError("a.csv", 4, "A", "x"))
    assert report.summary() == (
        "1 Datei(en), 3 Zeilen gelesen, 2 importiert (1 Symbole), 1 abgelehnt."
    )


def test_missing_path_and_empty_directory(store: MarketStore, tmp_path: Path) -> None:
    with pytest.raises(CsvMappingError, match="nicht gefunden"):
        import_csv(store, tmp_path / "fehlt.csv")
    with pytest.raises(CsvMappingError, match="Keine CSV-Dateien"):
        import_csv(store, tmp_path)


def test_empty_and_header_only_files(store: MarketStore, tmp_path: Path) -> None:
    empty = tmp_path / "EMPTY.csv"
    empty.write_text("", "utf-8")
    with pytest.raises(CsvMappingError, match="leer"):
        import_csv(store, empty)
    header = tmp_path / "HEAD.csv"
    header.write_text("Date,Open,High,Low,Close,Volume\n", "utf-8")
    report = import_csv(store, header)
    assert report.rows_read == 0
    assert report.rows_imported == 0


def test_unreadable_file(store: MarketStore, tmp_path: Path) -> None:
    file = tmp_path / "BIN.csv"
    file.write_bytes(b"Date,Open,High,Low,Close,Volume\n\xff\xfe,\xff,1,1,1,1\n")
    with pytest.raises(CsvMappingError, match="nicht lesbar"):
        import_csv(store, file)


# ---------------------------------------------------------------- Provider


def test_provider_fetch_daily_and_actions() -> None:
    provider = CsvImportProvider(FIXTURES / "combined.csv")
    assert isinstance(provider, PriceProvider)
    assert provider.name == "csv"

    bars = provider.fetch_daily("syna", date(2024, 1, 3), date(2024, 12, 31))
    assert list(bars.columns) == ["date", "open", "high", "low", "close", "volume", "adj_close"]
    assert bars["date"].tolist() == [pd.Timestamp("2024-01-03")]
    assert bars["adj_close"].tolist() == [11.5]
    assert provider.fetch_daily("NONE", date(2024, 1, 1), date(2024, 12, 31)).empty
    assert provider.fetch_actions("SYNA").empty
    assert provider.symbols() == ["SYNA", "SYNB"]
    assert provider.report.files == ["combined.csv"]


def test_provider_with_adj_close_and_later_file_wins(tmp_path: Path) -> None:
    shutil.copy(FIXTURES / "per_symbol" / "SYNA.csv", tmp_path / "A1.csv")
    (tmp_path / "A2.csv").write_text(
        "Date,Symbol,Open,High,Low,Close,Adj Close,Volume\n2024-01-04,A1,1,2,0.5,1.5,1.0,9\n",
        "utf-8",
    )
    provider = CsvImportProvider(tmp_path)
    bars = provider.fetch_daily("A1", date(2024, 1, 1), date(2024, 1, 31))
    assert bars["close"].tolist() == [10.5, 11.5, 1.5]
    assert bars["adj_close"].tolist() == [5.25, 5.75, 1.0]


def test_provider_empty_and_errors(tmp_path: Path) -> None:
    header = tmp_path / "HEAD.csv"
    header.write_text("Date,Open,High,Low,Close,Volume\n", "utf-8")
    provider = CsvImportProvider(header)
    assert provider.fetch_daily("HEAD", date(2024, 1, 1), date(2024, 1, 2)).empty
    assert provider.symbols() == []

    broken = CsvImportProvider(tmp_path / "fehlt.csv", {"a": "date"})
    with pytest.raises(ProviderError, match="nicht gefunden"):
        broken.fetch_daily("X", date(2024, 1, 1), date(2024, 1, 2))


# ---------------------------------------------------------------- CLI


def test_cli_import_csv_with_error_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "bad_rows.csv"
    shutil.copy(FIXTURES / "bad_rows.csv", source)
    db = tmp_path / "db.duckdb"

    assert main(["import-csv", str(source), "--db", str(db)]) == 0

    out = capsys.readouterr().out
    assert "2 importiert" in out
    assert "9 abgelehnt" in out
    error_file = tmp_path / "bad_rows.csv.fehler.csv"
    assert error_file.exists()
    assert len(error_file.read_text("utf-8").splitlines()) == 10
    with MarketStore(db, read_only=True) as store:
        assert store.symbols() == ["SYNA"]


def test_cli_import_csv_with_mapping(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "db.duckdb"
    code = main(
        [
            "import-csv",
            str(FIXTURES / "custom.csv"),
            "--mapping",
            str(FIXTURES / "custom_mapping.toml"),
            "--date-format",
            "%d.%m.%Y",
            "--db",
            str(db),
        ]
    )
    assert code == 0
    assert "Fehlerbericht" not in capsys.readouterr().out
    with MarketStore(db, read_only=True) as store:
        assert len(store.load_bars("SYNC")) == 2


def test_cli_import_csv_wrong_mapping(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["import-csv", str(FIXTURES / "custom.csv"), "--db", str(tmp_path / "db.duckdb")])
    assert code == 2
    assert "Pflichtspalte" in capsys.readouterr().err


def test_cli_uses_database_from_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "settings.duckdb"

    class _Settings:
        database_path = db

    monkeypatch.setattr("trading_helfer.core.config.load_settings", lambda: _Settings())
    assert main(["import-csv", str(FIXTURES / "combined.csv")]) == 0
    with MarketStore(db, read_only=True) as store:
        assert store.symbols() == ["SYNA", "SYNB"]
