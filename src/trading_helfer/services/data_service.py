"""Fassade für den Daten-Tab: Datenupdate, Qualitätsbericht, CSV-Import und Quellen.

Das Modul importiert kein Qt. Es verbindet :func:`~trading_helfer.data.update.update_universe`,
:func:`~trading_helfer.data.quality.check_all` und
:func:`~trading_helfer.data.providers.csv_import.import_csv` mit der Oberfläche und
speichert Provider-Reihenfolge und Tiingo-Key in der ``config.toml``.

Jede Methode öffnet die Datenbank selbst und schließt sie wieder, damit sie in einem
Hintergrund-Thread laufen kann.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
from pydantic import SecretStr, ValidationError

from trading_helfer.core.config import (
    DEFAULT_PROVIDERS,
    ConfigError,
    Settings,
    update_config_file,
)
from trading_helfer.data.providers.base import PriceProvider
from trading_helfer.data.providers.csv_import import CsvImportReport, import_csv
from trading_helfer.data.quality import REPORT_COLUMNS, Severity, check_all
from trading_helfer.data.storage import MarketStore
from trading_helfer.data.update import UpdateResult, build_price_providers, update_universe

__all__ = [
    "AVAILABLE_PROVIDERS",
    "INDEX_LABELS",
    "DataService",
    "LastUpdate",
    "ProviderFactory",
    "ProviderSettings",
    "SeverityFilter",
    "empty_report",
    "filter_report",
    "mask_secret",
]

logger = logging.getLogger(__name__)

INDEX_LABELS: Final[dict[str, str]] = {"SP500": "S&P 500", "NDX100": "Nasdaq-100"}
"""Indizes, die der Daten-Tab zur Wahl stellt, mit Anzeigenamen."""

AVAILABLE_PROVIDERS: Final[tuple[str, ...]] = DEFAULT_PROVIDERS
"""Alle bekannten Kursquellen."""

SeverityFilter = Severity | None
"""Filter für den Qualitätsbericht: eine Schwere oder ``None`` für alle."""

ProviderFactory = Callable[[Settings, MarketStore], list[PriceProvider]]
"""Legt die Kursquellen für ein Update an (Standard: :func:`build_price_providers`)."""

ProgressFn = Callable[[int, str], None]
"""Fortschritt in Prozent (0–100) und kurze Meldung."""

_MASK_CHAR: Final = "•"
_VISIBLE_KEY_CHARS: Final = 4


def mask_secret(value: str | None) -> str:
    """Maskiert ein Geheimnis für die Anzeige; nur die letzten vier Zeichen bleiben sichtbar.

    Kurze Werte (bis acht Zeichen) werden ganz maskiert. ``None`` oder leer ergibt ``""``.
    """
    if not value:
        return ""
    if len(value) <= 2 * _VISIBLE_KEY_CHARS:
        return _MASK_CHAR * 8
    return _MASK_CHAR * 8 + value[-_VISIBLE_KEY_CHARS:]


def empty_report() -> pd.DataFrame:
    """Leerer Qualitätsbericht mit allen Spalten aus :data:`REPORT_COLUMNS`."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in REPORT_COLUMNS})


def filter_report(report: pd.DataFrame, severity: SeverityFilter) -> pd.DataFrame:
    """Zeilen des Berichts mit der Schwere ``severity``; ``None`` liefert alle Zeilen."""
    if severity is None:
        return report
    return report[report["severity"] == severity.value].reset_index(drop=True)


@dataclass(frozen=True)
class LastUpdate:
    """Letzter Eintrag aus ``update_log``.

    Attributes:
        started_at: Beginn des Laufs.
        finished_at: Ende des Laufs; ``None``, wenn der Lauf nicht sauber endete.
        provider: Kursquellen des Laufs, durch Komma getrennt.
        symbols_ok: Anzahl erfolgreich aktualisierter Symbole.
        symbols_failed: Anzahl fehlgeschlagener Symbole.
        message: Zusammenfassung des Laufs.
    """

    started_at: dt.datetime
    finished_at: dt.datetime | None
    provider: str
    symbols_ok: int
    symbols_failed: int
    message: str


@dataclass(frozen=True)
class ProviderSettings:
    """Einstellungen der Kursquellen für die Anzeige.

    Attributes:
        order: Aktive Quellen in Reihenfolge.
        available: Alle bekannten Quellen.
        tiingo_key_masked: Maskierter Tiingo-Key; ``""``, wenn keiner gesetzt ist.
    """

    order: tuple[str, ...]
    available: tuple[str, ...]
    tiingo_key_masked: str

    @property
    def has_tiingo_key(self) -> bool:
        """``True``, wenn ein Tiingo-Key gesetzt ist."""
        return bool(self.tiingo_key_masked)


class DataService:
    """Datenupdate, Qualitätsbericht, CSV-Import und Quellen-Einstellungen für den Daten-Tab.

    Args:
        settings: Aktuelle Einstellungen (Datenbankpfad, Quellen, Tiingo-Key).
        config_path: ``config.toml``, in die Änderungen der Quellen geschrieben werden;
            ``None`` ändert die Einstellungen nur im Speicher.
        provider_factory: Legt die Kursquellen für ein Update an; für Tests austauschbar.
    """

    def __init__(
        self,
        settings: Settings,
        config_path: Path | None = None,
        *,
        provider_factory: ProviderFactory = build_price_providers,
    ) -> None:
        self._settings = settings
        self._config_path = config_path
        self._provider_factory = provider_factory

    @property
    def settings(self) -> Settings:
        """Die aktuell gültigen Einstellungen."""
        return self._settings

    @property
    def database_path(self) -> Path:
        """Pfad der DuckDB-Datei."""
        return self._settings.database_path

    # -- Datenupdate ------------------------------------------------------------------

    def run_update(
        self,
        index_codes: Sequence[str],
        *,
        symbols: Sequence[str] = (),
        progress: ProgressFn | None = None,
        should_cancel: Callable[[], bool] | None = None,
        today: dt.date | None = None,
    ) -> UpdateResult:
        """Aktualisiert die Kurse der gewählten Indizes (plus Benchmarks und ``symbols``).

        Args:
            index_codes: Indizes, z. B. ``["SP500"]``; siehe :data:`INDEX_LABELS`.
            symbols: Zusätzliche Symbole.
            progress: Erhält nach jedem Symbol Prozent und ``"SYMBOL (erledigt/gesamt)"``.
            should_cancel: Wird vor jedem Symbol gefragt; ``True`` beendet den Lauf nach
                dem aktuellen Symbol.
            today: Letzter zu ladender Tag; Standard ist heute.

        Returns:
            Ergebnis des Laufs (auch bei Abbruch, dann mit ``cancelled = True``).

        Raises:
            UpdateError: keine oder unbekannte Kursquelle, unbekannter Index.
        """

        def report(done: int, total: int, symbol: str) -> None:
            if progress is not None:
                percent = round(100 * done / total) if total else 100
                progress(percent, f"{symbol} ({done}/{total})")

        path = self.database_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with MarketStore(path) as store:
            return update_universe(
                list(index_codes),
                report,
                store=store,
                providers=self._provider_factory(self._settings, store),
                symbols=symbols,
                start=self._settings.start_date,
                today=today,
                should_cancel=should_cancel,
            )

    def last_update(self) -> LastUpdate | None:
        """Jüngster Eintrag aus ``update_log``; ``None`` ohne Datenbank oder ohne Eintrag."""
        if not self.database_path.exists():
            return None
        with MarketStore(self.database_path, read_only=True) as store:
            row = store.connection.execute(
                "SELECT started_at, finished_at, provider, symbols_ok, symbols_failed, message "
                "FROM update_log ORDER BY started_at DESC, id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        started, finished, provider, ok, failed, message = row
        return LastUpdate(
            started_at=started,
            finished_at=finished,
            provider=provider or "",
            symbols_ok=int(ok or 0),
            symbols_failed=int(failed or 0),
            message=message or "",
        )

    # -- Qualität ---------------------------------------------------------------------

    def quality_report(self, symbols: Sequence[str] | None = None) -> pd.DataFrame:
        """Qualitätsbericht aller (oder der angegebenen) Symbole.

        Returns:
            Bericht mit den Spalten :data:`REPORT_COLUMNS`; leer ohne Datenbank.
        """
        if not self.database_path.exists():
            return empty_report()
        with MarketStore(self.database_path, read_only=True) as store:
            return check_all(
                store,
                symbols=list(symbols) if symbols is not None else None,
                history_start=self._settings.start_date,
            )

    # -- CSV-Import -------------------------------------------------------------------

    def import_csv(self, path: Path, *, date_format: str | None = None) -> CsvImportReport:
        """Importiert Tageskurse aus einer CSV-Datei oder einem Ordner.

        Abgelehnte Zeilen werden neben die Datei als ``<name>.fehler.csv`` geschrieben.

        Raises:
            CsvMappingError: Die Datei fehlt, ist unlesbar oder die Spalten passen nicht.
        """
        db = self.database_path
        db.parent.mkdir(parents=True, exist_ok=True)
        with MarketStore(db) as store:
            report = import_csv(store, path, date_format=date_format)
        if report.errors:
            report.write_errors(path.with_name(path.name + ".fehler.csv"))
        return report

    # -- Einstellungen ----------------------------------------------------------------

    def provider_settings(self) -> ProviderSettings:
        """Aktuelle Reihenfolge der Kursquellen und maskierter Tiingo-Key."""
        key = self._settings.tiingo_api_key
        return ProviderSettings(
            order=tuple(self._settings.providers),
            available=AVAILABLE_PROVIDERS,
            tiingo_key_masked=mask_secret(key.get_secret_value() if key is not None else None),
        )

    def save_provider_settings(
        self, order: Sequence[str], tiingo_key: str | None = None, *, clear_key: bool = False
    ) -> ProviderSettings:
        """Speichert Reihenfolge der Kursquellen und optional einen neuen Tiingo-Key.

        Args:
            order: Aktive Quellen in Reihenfolge (mindestens eine).
            tiingo_key: Neuer Key; ``None`` oder leer lässt den bisherigen Key stehen.
            clear_key: Entfernt den gespeicherten Key.

        Returns:
            Die neuen Einstellungen für die Anzeige.

        Raises:
            ConfigError: ungültige Werte oder die Datei kann nicht geschrieben werden.
        """
        values: dict[str, object] = {"providers": list(order)}
        key = (tiingo_key or "").strip()
        if clear_key:
            values["tiingo_api_key"] = None
        elif key:
            values["tiingo_api_key"] = SecretStr(key)
        if self._config_path is not None:
            self._settings = update_config_file(self._config_path, dict(values))
        else:
            data = self._settings.model_dump()
            data.update(values)
            try:
                self._settings = Settings.model_validate(data)
            except ValidationError as exc:
                raise ConfigError(f"Ungültige Einstellungen: {exc}") from exc
        logger.info("Kursquellen gespeichert: %s", ", ".join(self._settings.providers))
        return self.provider_settings()
