"""Zentrale, geprüfte Einstellungen der App.

Die Einstellungen stammen (absteigende Priorität) aus

1. direkt übergebenen Werten (``Settings(...)``),
2. Umgebungsvariablen mit Präfix ``TH_`` (z. B. ``TH_DEFAULT_CAPITAL=5000``),
3. der Datei ``config.toml`` im Benutzer-Konfigordner,
4. den Standardwerten dieser Klasse.

Einstiegspunkt ist :func:`load_settings`; sie legt eine fehlende ``config.toml``
mit Standardwerten an und übersetzt Validierungsfehler in :class:`ConfigError`.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Final, Literal

import tomli_w
from platformdirs import user_config_dir, user_data_dir
from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_core import ErrorDetails
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

__all__ = [
    "APP_NAME",
    "CONFIG_FILE_NAME",
    "DB_FILE_NAME",
    "DEFAULT_PROVIDERS",
    "ENV_PREFIX",
    "ConfigError",
    "LogLevel",
    "Settings",
    "default_config_path",
    "default_data_dir",
    "load_settings",
    "update_config_file",
    "write_default_config",
]

APP_NAME: Final = "trading-helfer"
"""Name des App-Ordners unter den Benutzerverzeichnissen."""

ENV_PREFIX: Final = "TH_"
"""Präfix der Umgebungsvariablen, die Einstellungen überschreiben."""

CONFIG_FILE_NAME: Final = "config.toml"
DB_FILE_NAME: Final = "market.duckdb"
LOG_DIR_NAME: Final = "logs"
PLUGINS_DIR_NAME: Final = "plugins"
DEFAULT_PROVIDERS: Final = ("yfinance", "tiingo")

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_SECRET_FIELDS: Final = frozenset({"tiingo_api_key"})

# Pfad der TOML-Datei, die beim nächsten Erzeugen von ``Settings`` gelesen wird.
_active_config_file: ContextVar[Path | None] = ContextVar("_active_config_file", default=None)


class ConfigError(ValueError):
    """Ungültige Konfiguration; die Meldung ist deutsch und nennt die betroffenen Felder."""

    def __init__(self, message: str, fields: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.fields = fields
        """Namen der fehlerhaften Felder (leer, wenn kein Feld zuordenbar ist)."""


def default_data_dir() -> Path:
    """Gibt den Standard-Datenordner zurück (Benutzer-Datenordner laut ``platformdirs``)."""
    return Path(user_data_dir(APP_NAME, appauthor=False))


def default_config_path() -> Path:
    """Gibt den Standardpfad der ``config.toml`` im Benutzer-Konfigordner zurück."""
    return Path(user_config_dir(APP_NAME, appauthor=False)) / CONFIG_FILE_NAME


class Settings(BaseSettings):
    """Geprüfte Einstellungen der App.

    ``db_path``, ``log_dir`` und ``plugins_dir`` werden aus ``data_dir`` abgeleitet, wenn sie nicht
    ausdrücklich gesetzt sind.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        extra="forbid",
        validate_default=True,
        frozen=True,
    )

    data_dir: Path = Field(
        default_factory=default_data_dir, description="Ordner für Datenbank und Logs"
    )
    db_path: Path | None = Field(
        default=None, description="DuckDB-Datei; Standard: data_dir/market.duckdb"
    )
    log_dir: Path | None = Field(default=None, description="Log-Ordner; Standard: data_dir/logs")
    plugins_dir: Path | None = Field(
        default=None, description="Ordner für Strategie-Plugins; Standard: data_dir/plugins"
    )
    log_level: LogLevel = Field(default="INFO", description="Mindest-Level für Logausgaben")
    start_date: date = Field(default=date(1990, 1, 1), description="Frühestes Kursdatum")
    default_capital: float = Field(default=1000.0, gt=0, description="Startkapital in USD")
    allow_fractional: bool = Field(default=True, description="Bruchstücke von Aktien erlauben")
    fee_per_order: float = Field(default=1.0, ge=0, description="Gebühr je Order in USD")
    slippage_pct: float = Field(default=0.05, ge=0, lt=100, description="Slippage in Prozent")
    risk_free_rate_pct: float = Field(
        default=0.0, gt=-100, lt=100, description="Risikofreier Zins p. a. in Prozent"
    )
    tiingo_api_key: SecretStr | None = Field(default=None, description="API-Key für Tiingo")
    providers: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_PROVIDERS),
        description="Reihenfolge der Kursquellen",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Legt die Quellen fest: direkte Werte, dann ``TH_``-Variablen, dann TOML-Datei."""
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        config_file = _active_config_file.get()
        if config_file is not None:
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=config_file))
        return tuple(sources)

    @field_validator("data_dir", "db_path", "log_dir", "plugins_dir", mode="after")
    @classmethod
    def _expand_user(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("tiingo_api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("providers", mode="before")
    @classmethod
    def _parse_providers(cls, value: Any) -> Any:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError("keine gültige Liste") from exc
            return [part for part in text.split(",") if part.strip()]
        return value

    @field_validator("providers", mode="after")
    @classmethod
    def _check_providers(cls, value: list[str]) -> list[str]:
        names = [name.strip().lower() for name in value]
        if not names or any(not name for name in names):
            raise ValueError("mindestens eine Kursquelle angeben")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Kursquelle mehrfach angegeben: {', '.join(duplicates)}")
        return names

    @model_validator(mode="after")
    def _derive_paths(self) -> Settings:
        # Das Modell ist eingefroren; abgeleitete Pfade werden einmalig hier gesetzt.
        if self.db_path is None:
            object.__setattr__(self, "db_path", self.data_dir / DB_FILE_NAME)
        if self.log_dir is None:
            object.__setattr__(self, "log_dir", self.data_dir / LOG_DIR_NAME)
        if self.plugins_dir is None:
            object.__setattr__(self, "plugins_dir", self.data_dir / PLUGINS_DIR_NAME)
        return self

    @property
    def database_path(self) -> Path:
        """Pfad der DuckDB-Datei (immer gesetzt)."""
        assert self.db_path is not None
        return self.db_path

    @property
    def log_path(self) -> Path:
        """Pfad des Log-Ordners (immer gesetzt)."""
        assert self.log_dir is not None
        return self.log_dir

    @property
    def plugins_path(self) -> Path:
        """Pfad des Plugin-Ordners für Strategien (immer gesetzt)."""
        assert self.plugins_dir is not None
        return self.plugins_dir

    def secret_values(self) -> list[str]:
        """Gibt alle gesetzten Geheimnisse im Klartext zurück (für die Log-Maskierung)."""
        secrets: list[str] = []
        if self.tiingo_api_key is not None:
            secrets.append(self.tiingo_api_key.get_secret_value())
        return secrets


def _default_file_values() -> dict[str, Any]:
    """Standardwerte, wie sie in eine neue ``config.toml`` geschrieben werden."""
    return {
        "data_dir": str(default_data_dir()),
        "log_level": "INFO",
        "start_date": date(1990, 1, 1),
        "default_capital": 1000.0,
        "allow_fractional": True,
        "fee_per_order": 1.0,
        "slippage_pct": 0.05,
        "risk_free_rate_pct": 0.0,
        "providers": list(DEFAULT_PROVIDERS),
    }


_HEADER: Final = """\
# Einstellungen für Trading Helfer.
# Umgebungsvariablen mit Präfix TH_ (z. B. TH_DEFAULT_CAPITAL=5000) haben Vorrang.
#
# Optionale Einträge:
# db_path = "..."          # Standard: <data_dir>/market.duckdb
# log_dir = "..."          # Standard: <data_dir>/logs
# tiingo_api_key = "..."   # API-Key für Tiingo (besser per TH_TIINGO_API_KEY setzen)

"""


def write_default_config(path: Path) -> Path:
    """Schreibt eine ``config.toml`` mit Standardwerten nach ``path`` und gibt ``path`` zurück.

    Fehlende Elternordner werden angelegt; eine vorhandene Datei wird überschrieben.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + tomli_w.dumps(_default_file_values()), encoding="utf-8")
    return path


def update_config_file(path: Path, values: dict[str, Any]) -> Settings:
    """Ändert einzelne Einträge der ``config.toml`` und lädt die Einstellungen neu.

    Die übrigen Einträge der Datei bleiben erhalten; Kommentare werden durch den
    Standardkopf ersetzt. Ein Wert ``None`` entfernt den Eintrag (es gilt dann der
    Standardwert). Die Werte werden vor dem Schreiben geprüft, eine ungültige Änderung
    lässt die Datei unverändert.

    Args:
        path: Pfad der TOML-Datei; fehlt sie, wird sie mit Standardwerten angelegt.
        values: Zu setzende Einträge, z. B. ``{"providers": ["tiingo", "yfinance"]}``.

    Returns:
        Die neu geladenen Einstellungen.

    Raises:
        ConfigError: wenn die Datei kein gültiges TOML ist, nicht geschrieben werden kann
            oder die neuen Werte ungültig sind.
    """
    if path.exists():
        try:
            data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Konfigurationsdatei {path} ist kein gültiges TOML: {exc}") from exc
    else:
        data = _default_file_values()
    for key, value in values.items():
        if value is None:
            data.pop(key, None)
        elif isinstance(value, SecretStr):
            data[key] = value.get_secret_value()
        else:
            data[key] = value
    try:
        Settings.model_validate(data)
    except ValidationError as exc:
        raise _to_config_error(exc, path) from exc
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_HEADER + tomli_w.dumps(data), encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Konfigurationsdatei {path} kann nicht geschrieben werden: {exc}"
        ) from exc
    return load_settings(path, create=False)


@contextmanager
def _reading_config_file(path: Path) -> Iterator[None]:
    token = _active_config_file.set(path)
    try:
        yield
    finally:
        _active_config_file.reset(token)


_GERMAN_MESSAGES: Final[dict[str, str]] = {
    "greater_than": "muss größer als {gt} sein",
    "greater_than_equal": "muss mindestens {ge} sein",
    "less_than": "muss kleiner als {lt} sein",
    "less_than_equal": "darf höchstens {le} sein",
    "float_parsing": "muss eine Zahl sein",
    "float_type": "muss eine Zahl sein",
    "bool_parsing": "muss true oder false sein",
    "bool_type": "muss true oder false sein",
    "date_parsing": "muss ein Datum im Format JJJJ-MM-TT sein",
    "date_from_datetime_parsing": "muss ein Datum im Format JJJJ-MM-TT sein",
    "date_type": "muss ein Datum im Format JJJJ-MM-TT sein",
    "literal_error": "muss einer der Werte {expected} sein",
    "list_type": "muss eine Liste sein",
    "string_type": "muss ein Text sein",
    "path_type": "muss ein Pfad sein",
    "extra_forbidden": "ist keine bekannte Einstellung",
}


def _format_value(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _describe_error(error: ErrorDetails) -> tuple[str, str]:
    field = ".".join(str(part) for part in error["loc"]) or "<unbekannt>"
    template = _GERMAN_MESSAGES.get(error["type"])
    if template is not None:
        ctx = {key: _format_value(val) for key, val in (error.get("ctx") or {}).items()}
        text = template.format(**ctx)
    else:
        text = error["msg"].removeprefix("Value error, ")
    root = str(error["loc"][0]) if error["loc"] else ""
    if root not in _SECRET_FIELDS and error["type"] != "extra_forbidden":
        text += f" (Wert: {error['input']!r})"
    return field, text


def _to_config_error(exc: ValidationError, config_path: Path | None) -> ConfigError:
    details = [_describe_error(error) for error in exc.errors()]
    fields = tuple(dict.fromkeys(field for field, _ in details))
    where = f" (Datei: {config_path}, Umgebungsvariablen {ENV_PREFIX}*)" if config_path else ""
    lines = [f"Ungültige Einstellungen{where}:"]
    lines += [f"- {field}: {text}" for field, text in details]
    return ConfigError("\n".join(lines), fields)


def load_settings(config_path: Path | None = None, *, create: bool = True) -> Settings:
    """Lädt die Einstellungen aus ``config_path`` und den ``TH_``-Umgebungsvariablen.

    Args:
        config_path: Pfad der TOML-Datei; Standard ist :func:`default_config_path`.
        create: Legt die Datei mit Standardwerten an, wenn sie fehlt.

    Raises:
        ConfigError: wenn die Datei kein gültiges TOML ist oder Werte ungültig sind.
    """
    path = config_path if config_path is not None else default_config_path()
    if not path.exists() and create:
        try:
            write_default_config(path)
        except OSError as exc:
            raise ConfigError(
                f"Konfigurationsdatei {path} kann nicht angelegt werden: {exc}"
            ) from exc
    try:
        with _reading_config_file(path):
            return Settings()
    except ValidationError as exc:
        raise _to_config_error(exc, path) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Konfigurationsdatei {path} ist kein gültiges TOML: {exc}") from exc
