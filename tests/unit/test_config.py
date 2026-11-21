"""Einstellungen: Standardwerte, TOML-Datei, Umgebungsvariablen und Validierung."""

from __future__ import annotations

import json
import os
import tomllib
from datetime import date
from pathlib import Path

import pytest

from trading_helfer.core import config
from trading_helfer.core.config import (
    ConfigError,
    Settings,
    default_config_path,
    default_data_dir,
    load_settings,
    write_default_config,
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Entfernt ``TH_``-Variablen und leitet die Benutzerordner nach ``tmp_path`` um."""
    for name in list(os.environ):
        if name.upper().startswith("TH_"):
            monkeypatch.delenv(name)
    user_dirs = tmp_path / "user"
    monkeypatch.setattr(
        config, "user_data_dir", lambda app, appauthor: str(user_dirs / "data" / app)
    )
    monkeypatch.setattr(
        config, "user_config_dir", lambda app, appauthor: str(user_dirs / "config" / app)
    )
    return user_dirs


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    return tmp_path / "cfg" / "config.toml"


def toml_str(path: Path) -> str:
    """Pfad als TOML-String (JSON-Escaping ist mit TOML-Basisstrings verträglich)."""
    return json.dumps(str(path))


def write_toml(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- Standardwerte -------------------------------------------------------------------


def test_defaults(isolated_env: Path) -> None:
    s = Settings()
    data_dir = isolated_env / "data" / "trading-helfer"
    assert s.data_dir == data_dir
    assert s.db_path == data_dir / "market.duckdb"
    assert s.database_path == data_dir / "market.duckdb"
    assert s.log_dir == data_dir / "logs"
    assert s.log_path == data_dir / "logs"
    assert s.log_level == "INFO"
    assert s.start_date == date(1990, 1, 1)
    assert s.default_capital == 1000.0
    assert s.allow_fractional is True
    assert s.fee_per_order == 1.0
    assert s.slippage_pct == 0.05
    assert s.tiingo_api_key is None
    assert s.providers == ["yfinance", "tiingo"]
    assert s.secret_values() == []


def test_default_paths_use_platformdirs(isolated_env: Path) -> None:
    assert default_data_dir() == isolated_env / "data" / "trading-helfer"
    assert default_config_path() == isolated_env / "config" / "trading-helfer" / "config.toml"


def test_real_platformdirs_paths_end_with_app_name() -> None:
    # Ohne Umleitung: nur Pfade berechnen, nichts schreiben.
    from platformdirs import user_config_dir, user_data_dir

    assert Path(user_data_dir("trading-helfer", appauthor=False)).name == "trading-helfer"
    assert Path(user_config_dir("trading-helfer", appauthor=False)).name == "trading-helfer"


def test_settings_are_immutable() -> None:
    s = Settings()
    with pytest.raises(ValueError, match="frozen"):
        s.default_capital = 5.0  # type: ignore[misc]


# --- Datei anlegen und lesen ---------------------------------------------------------


def test_missing_config_is_created_with_defaults(config_file: Path, isolated_env: Path) -> None:
    assert not config_file.exists()
    s = load_settings(config_file)
    assert config_file.exists()
    data = tomllib.loads(config_file.read_text(encoding="utf-8"))
    assert data == {
        "data_dir": str(isolated_env / "data" / "trading-helfer"),
        "log_level": "INFO",
        "start_date": date(1990, 1, 1),
        "default_capital": 1000.0,
        "allow_fractional": True,
        "fee_per_order": 1.0,
        "slippage_pct": 0.05,
        "risk_free_rate_pct": 0.0,
        "providers": ["yfinance", "tiingo"],
    }
    assert s == Settings()


def test_load_settings_uses_default_config_path(isolated_env: Path) -> None:
    load_settings()
    assert (isolated_env / "config" / "trading-helfer" / "config.toml").exists()


def test_load_without_create_does_not_write(config_file: Path) -> None:
    s = load_settings(config_file, create=False)
    assert not config_file.exists()
    assert s.default_capital == 1000.0


def test_existing_config_is_read_and_not_overwritten(config_file: Path, tmp_path: Path) -> None:
    text = (
        f"data_dir = {toml_str(tmp_path / 'daten')}\n"
        "default_capital = 2500\n"
        "log_level = 'debug'\n"
        "start_date = 2000-01-03\n"
        "providers = ['Tiingo', 'yfinance', 'csv']\n"
        "tiingo_api_key = 'abc-secret-123'\n"
    )
    write_toml(config_file, text)
    s = load_settings(config_file)
    assert config_file.read_text(encoding="utf-8") == text
    assert s.data_dir == tmp_path / "daten"
    assert s.db_path == tmp_path / "daten" / "market.duckdb"
    assert s.default_capital == 2500.0
    assert s.log_level == "DEBUG"
    assert s.start_date == date(2000, 1, 3)
    assert s.providers == ["tiingo", "yfinance", "csv"]
    assert s.tiingo_api_key is not None
    assert s.tiingo_api_key.get_secret_value() == "abc-secret-123"
    assert "abc-secret-123" not in repr(s)
    assert s.secret_values() == ["abc-secret-123"]


def test_explicit_db_path_and_log_dir(config_file: Path, tmp_path: Path) -> None:
    write_toml(
        config_file,
        f"db_path = {toml_str(tmp_path / 'x.duckdb')}\nlog_dir = {toml_str(tmp_path / 'l')}\n",
    )
    s = load_settings(config_file)
    assert s.db_path == tmp_path / "x.duckdb"
    assert s.log_dir == tmp_path / "l"


def test_tilde_is_expanded() -> None:
    s = Settings(data_dir=Path("~") / "th")
    assert "~" not in str(s.data_dir)
    assert s.data_dir == Path.home() / "th"


def test_write_default_config_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "config.toml"
    assert write_default_config(target) == target
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# Einstellungen für Trading Helfer.")
    assert "tiingo_api_key" in text  # nur als Kommentar
    assert "tiingo_api_key" not in tomllib.loads(text)


# --- Umgebungsvariablen --------------------------------------------------------------


def test_env_overrides_file(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_toml(config_file, "default_capital = 2500\nfee_per_order = 2.5\n")
    monkeypatch.setenv("TH_DEFAULT_CAPITAL", "5000")
    monkeypatch.setenv("TH_ALLOW_FRACTIONAL", "false")
    monkeypatch.setenv("TH_TIINGO_API_KEY", "env-key-9876")
    s = load_settings(config_file)
    assert s.default_capital == 5000.0
    assert s.fee_per_order == 2.5  # aus der Datei
    assert s.allow_fractional is False
    assert s.tiingo_api_key is not None
    assert s.tiingo_api_key.get_secret_value() == "env-key-9876"


def test_env_overrides_defaults_of_new_file(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TH_SLIPPAGE_PCT", "0.2")
    s = load_settings(config_file)
    assert s.slippage_pct == 0.2
    # Die neue Datei enthält die Standardwerte, nicht den Wert aus der Umgebung.
    assert tomllib.loads(config_file.read_text(encoding="utf-8"))["slippage_pct"] == 0.05


def test_env_data_dir_moves_derived_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TH_DATA_DIR", str(tmp_path / "env-data"))
    s = Settings()
    assert s.db_path == tmp_path / "env-data" / "market.duckdb"
    assert s.log_dir == tmp_path / "env-data" / "logs"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("tiingo,yfinance", ["tiingo", "yfinance"]),
        (" csv , YFinance ", ["csv", "yfinance"]),
        ('["tiingo", "csv"]', ["tiingo", "csv"]),
    ],
)
def test_env_providers(raw: str, expected: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TH_PROVIDERS", raw)
    assert Settings().providers == expected


def test_empty_api_key_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TH_TIINGO_API_KEY", "  ")
    assert Settings().tiingo_api_key is None


def test_init_values_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TH_FEE_PER_ORDER", "3")
    assert Settings(fee_per_order=0.0).fee_per_order == 0.0


# --- Validierung ---------------------------------------------------------------------


def test_negative_capital_names_field(config_file: Path) -> None:
    write_toml(config_file, "default_capital = -100\n")
    with pytest.raises(ConfigError) as exc:
        load_settings(config_file)
    message = str(exc.value)
    assert "default_capital" in message
    assert "muss größer als 0 sein" in message
    assert "-100" in message
    assert str(config_file) in message
    assert exc.value.fields == ("default_capital",)


def test_negative_capital_from_env(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TH_DEFAULT_CAPITAL", "-1")
    with pytest.raises(ConfigError, match="default_capital"):
        load_settings(config_file)


@pytest.mark.parametrize(
    ("toml", "field", "fragment"),
    [
        ("fee_per_order = -1\n", "fee_per_order", "muss mindestens 0 sein"),
        ("slippage_pct = 150\n", "slippage_pct", "muss kleiner als 100 sein"),
        ("log_level = 'LAUT'\n", "log_level", "muss einer der Werte"),
        ("allow_fractional = 'vielleicht'\n", "allow_fractional", "true oder false"),
        ("start_date = 'gestern'\n", "start_date", "JJJJ-MM-TT"),
        ("default_capital = 'viel'\n", "default_capital", "muss eine Zahl sein"),
        ("providers = []\n", "providers", "mindestens eine Kursquelle"),
        ("providers = ['tiingo', 'Tiingo']\n", "providers", "mehrfach angegeben: tiingo"),
        ("startkapital = 5\n", "startkapital", "keine bekannte Einstellung"),
    ],
)
def test_invalid_values(config_file: Path, toml: str, field: str, fragment: str) -> None:
    write_toml(config_file, toml)
    with pytest.raises(ConfigError) as exc:
        load_settings(config_file)
    assert field in exc.value.fields
    assert f"- {field}: " in str(exc.value)
    assert fragment in str(exc.value)


def test_single_provider_string_is_list(config_file: Path) -> None:
    write_toml(config_file, "providers = 'tiingo'\n")
    assert load_settings(config_file).providers == ["tiingo"]


def test_invalid_providers_json_from_env(
    monkeypatch: pytest.MonkeyPatch, config_file: Path
) -> None:
    monkeypatch.setenv("TH_PROVIDERS", "[tiingo")
    with pytest.raises(ConfigError, match="providers: keine gültige Liste"):
        load_settings(config_file)


def test_invalid_api_key_type_does_not_echo_value(config_file: Path) -> None:
    write_toml(config_file, "tiingo_api_key = 123456789\n")
    with pytest.raises(ConfigError) as exc:
        load_settings(config_file)
    assert "tiingo_api_key" in str(exc.value)
    assert "123456789" not in str(exc.value)


def test_several_errors_are_listed(config_file: Path) -> None:
    write_toml(config_file, "default_capital = 0\nfee_per_order = -2\n")
    with pytest.raises(ConfigError) as exc:
        load_settings(config_file)
    assert exc.value.fields == ("default_capital", "fee_per_order")
    assert str(exc.value).startswith("Ungültige Einstellungen")


def test_invalid_toml(config_file: Path) -> None:
    write_toml(config_file, "default_capital = = 3\n")
    with pytest.raises(ConfigError, match="kein gültiges TOML"):
        load_settings(config_file)


def test_unwritable_config_location(tmp_path: Path) -> None:
    blocker = tmp_path / "datei"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="kann nicht angelegt werden"):
        load_settings(blocker / "config.toml")


def test_config_error_is_value_error() -> None:
    err = ConfigError("Fehler")
    assert isinstance(err, ValueError)
    assert err.fields == ()
