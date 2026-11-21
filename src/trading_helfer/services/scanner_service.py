"""Fassade für den Scanner-Tab: Strategien, Scan, Verlauf früherer Scans und CSV-Export.

Das Modul importiert kein Qt. Es verbindet :func:`~trading_helfer.scanner.scan.scan` mit
der Oberfläche und speichert zu jedem Scan seine Kopfdaten (Strategie, Universum,
Zeitrahmen, Stichtag, Filter) in der Tabelle ``scans``, damit der Tab den Verlauf
anzeigen kann. Die Signale selbst speichert der Scanner in ``signals``.

Jede Methode öffnet die Datenbank selbst und schließt sie wieder, damit sie in einem
Hintergrund-Thread laufen kann.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pandas as pd

from trading_helfer.core.config import Settings
from trading_helfer.core.markers import Marker, MarkerKind
from trading_helfer.core.types import SignalType, Timeframe
from trading_helfer.data.storage import MarketStore
from trading_helfer.fundamentals.metrics import METRIC_LABELS
from trading_helfer.scanner.scan import RESULT_COLUMNS, ScanError, ScanFilters, export_csv, scan
from trading_helfer.strategies.registry import StrategyRegistry

__all__ = [
    "CUSTOM_UNIVERSE",
    "INDEX_LABELS",
    "METRIC_CHOICES",
    "ScanRecord",
    "ScanRequest",
    "ScannerService",
    "StrategyChoice",
    "empty_result",
    "parse_symbols",
    "signal_marker",
]

logger = logging.getLogger(__name__)

INDEX_LABELS: Final[dict[str, str]] = {"SP500": "S&P 500", "NDX100": "Nasdaq-100"}
"""Indizes, die der Scanner-Tab zur Wahl stellt, mit Anzeigenamen."""

CUSTOM_UNIVERSE: Final = "CUSTOM"
"""Schlüssel für eine eigene Symbolliste statt eines Index."""

METRIC_CHOICES: Final[dict[str, str]] = {
    name: label for name, label in METRIC_LABELS.items() if name != "market_cap"
}
"""Kennzahlen für den Kennzahlfilter (Marktkapitalisierung hat eigene Felder)."""

ProgressFn = Callable[[int, str], None]
"""Fortschritt in Prozent (0–100) und kurze Meldung."""

_MARKER_KINDS: Final[dict[str, MarkerKind]] = {
    SignalType.BUY.value: MarkerKind.BUY,
    SignalType.SELL.value: MarkerKind.SELL,
    SignalType.EXIT.value: MarkerKind.EXIT,
}


@dataclass(frozen=True)
class StrategyChoice:
    """Eine Strategie zur Auswahl im Scanner-Tab.

    Attributes:
        id: ID der Strategie.
        name: Anzeigename.
        timeframes: Unterstützte Zeitrahmen.
        defaults: Standardwerte der Parameter.
    """

    id: str
    name: str
    timeframes: tuple[Timeframe, ...]
    defaults: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScanRequest:
    """Eingaben eines Scans aus dem Scanner-Tab.

    Attributes:
        strategy_id: ID der Strategie.
        universe: Index-Code (``"SP500"``, ``"NDX100"``) oder Symbolliste.
        timeframe: Zeitrahmen.
        as_of: Stichtag; ``None`` = alle gespeicherten Kurse.
        params: Parameter der Strategie; fehlende = Standard.
        filters: Filter nach Sektor, Marktkapitalisierung, Kennzahlen und Signalart.
    """

    strategy_id: str
    universe: str | tuple[str, ...]
    timeframe: Timeframe
    as_of: dt.date | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    filters: ScanFilters = field(default_factory=ScanFilters)

    @property
    def universe_text(self) -> str:
        """Universum als Text für den Verlauf (Index-Code oder Symbole mit Komma)."""
        if isinstance(self.universe, str):
            return self.universe
        return ",".join(self.universe)


@dataclass(frozen=True)
class ScanRecord:
    """Kopfdaten eines gespeicherten Scans.

    Attributes:
        scan_id: ID des Scans (wie in der Tabelle ``signals``).
        created_at: Zeitpunkt des Scans (lokale Zeit).
        strategy_id: ID der Strategie.
        universe: Index-Code oder Symbole mit Komma.
        timeframe: Zeitrahmen.
        as_of: Stichtag; ``None`` = alle Kurse zum Zeitpunkt des Scans.
        signal_count: Anzahl Signale nach den Filtern.
        params: Parameter, mit denen der Scan lief.
    """

    scan_id: str
    created_at: dt.datetime
    strategy_id: str
    universe: str
    timeframe: Timeframe
    as_of: dt.date | None
    signal_count: int
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def universe_label(self) -> str:
        """Anzeigename des Universums (Indexname oder Symbolliste)."""
        return INDEX_LABELS.get(self.universe, self.universe.replace(",", ", "))


def parse_symbols(text: str) -> tuple[str, ...]:
    """Zerlegt eine Symbolliste (Komma, Semikolon oder Leerzeichen) in Großbuchstaben.

    Doppelte Symbole fallen weg; die Reihenfolge bleibt erhalten.
    """
    parts = text.replace(",", " ").replace(";", " ").split()
    return tuple(dict.fromkeys(p.strip().upper() for p in parts if p.strip()))


def empty_result() -> pd.DataFrame:
    """Leeres Scan-Ergebnis mit den festen Spalten."""
    return pd.DataFrame(columns=list(RESULT_COLUMNS))


def signal_marker(row: Mapping[str, Any]) -> Marker:
    """Chart-Marker für eine Zeile des Scan-Ergebnisses (Signalkerze, Art, Hinweis)."""
    day = pd.Timestamp(row["date"]).date()
    signal = str(row.get("signal", ""))
    note = row.get("note")
    text = signal if not isinstance(note, str) or not note else f"{signal}: {note}"
    return Marker(date=day, kind=_MARKER_KINDS.get(signal, MarkerKind.SIGNAL), text=text)


class ScannerService:
    """Logik hinter dem Scanner-Tab.

    Args:
        settings: Einstellungen (Datenbank, Plugin-Ordner, Broker und Kosten).
        registry: Strategie-Registry; Standard ist eine Registry mit den eingebauten
            Strategien und den Plugins aus ``settings.plugins_path`` (beim ersten
            Zugriff geladen).
        clock: Liefert den Zeitpunkt eines Scans (für Tests austauschbar).
    """

    def __init__(
        self,
        settings: Settings,
        registry: StrategyRegistry | None = None,
        *,
        clock: Callable[[], dt.datetime] | None = None,
    ) -> None:
        self.settings = settings
        self._registry = registry
        self._registry_loaded = registry is not None
        self._clock = clock if clock is not None else _now
        self._lock = threading.Lock()

    # -- Auswahl ---------------------------------------------------------------------

    @property
    def registry(self) -> StrategyRegistry:
        """Die Strategie-Registry (wird beim ersten Zugriff geladen)."""
        with self._lock:
            if self._registry is None:
                self._registry = StrategyRegistry(self.settings.plugins_path)
            if not self._registry_loaded:
                report = self._registry.load()
                for error in report.errors:
                    logger.warning("Strategie nicht geladen: %s", error)
                self._registry_loaded = True
            return self._registry

    def strategies(self) -> list[StrategyChoice]:
        """Alle geladenen Strategien, sortiert nach Anzeigename."""
        choices = [
            StrategyChoice(
                info.id,
                info.name,
                tuple(info.cls.timeframes),
                {name: param.default for name, param in info.cls.params.items()},
            )
            for info in self.registry.list_all()
        ]
        return sorted(choices, key=lambda c: (c.name.casefold(), c.id))

    def sectors(self) -> list[str]:
        """Alle Sektoren der gespeicherten Werte, alphabetisch (leer ohne Datenbank)."""
        if not self.settings.database_path.exists():
            return []
        with MarketStore(self.settings.database_path) as store:
            rows = store.connection.execute(
                "SELECT DISTINCT sector FROM instruments WHERE sector IS NOT NULL "
                "AND trim(sector) <> '' ORDER BY sector"
            ).fetchall()
        return [str(row[0]) for row in rows]

    # -- Scan ------------------------------------------------------------------------

    def run(self, request: ScanRequest, progress: ProgressFn | None = None) -> pd.DataFrame:
        """Führt einen Scan aus und speichert Signale und Kopfdaten.

        Args:
            request: Eingaben des Scans.
            progress: Erhält Prozent und Meldung (z. B. „12/500“) nach jedem Wert.

        Returns:
            Scan-Ergebnis wie :func:`~trading_helfer.scanner.scan.scan`;
            ``attrs["scan_id"]`` ist die ID im Verlauf.

        Raises:
            ScanError: Ungültige Eingabe (z. B. leere Symbolliste, falscher Zeitrahmen).
            StrategyNotFoundError: Unbekannte Strategie.
        """
        if not isinstance(request.universe, str) and not request.universe:
            raise ScanError("Bitte mindestens ein Symbol eingeben.")
        strategy = self.registry.create(request.strategy_id)

        def report(done: int, total: int) -> None:
            if progress is not None:
                percent = round(100 * done / total) if total else 100
                progress(percent, f"{done}/{total}")

        with MarketStore(self.settings.database_path) as store:
            result = scan(
                strategy,
                dict(request.params),
                request.universe,
                request.timeframe,
                request.as_of,
                request.filters,
                store=store,
                settings=self.settings,
                save=True,
                progress=report,
            )
            self._save_record(store, request, result)
        return result

    def _save_record(self, store: MarketStore, request: ScanRequest, result: pd.DataFrame) -> None:
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO scans (scan_id, created_at, strategy_id, params_json, universe, "
                "timeframe, as_of, filters_json, signal_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    str(result.attrs["scan_id"]),
                    self._clock().replace(microsecond=0),
                    request.strategy_id,
                    json.dumps(dict(request.params), sort_keys=True, default=str),
                    request.universe_text,
                    request.timeframe.value,
                    request.as_of,
                    json.dumps(_filters_dict(request.filters), sort_keys=True),
                    len(result),
                ],
            )

    # -- Verlauf ---------------------------------------------------------------------

    def history(self, limit: int = 100) -> list[ScanRecord]:
        """Frühere Scans, neueste zuerst (leer ohne Datenbank)."""
        if not self.settings.database_path.exists():
            return []
        with MarketStore(self.settings.database_path) as store:
            rows = store.connection.execute(
                "SELECT scan_id, created_at, strategy_id, universe, timeframe, as_of, "
                "signal_count, params_json FROM scans ORDER BY created_at DESC, scan_id LIMIT ?",
                [limit],
            ).fetchall()
        return [
            ScanRecord(
                scan_id=str(scan_id),
                created_at=created,
                strategy_id=str(strategy_id),
                universe=str(universe),
                timeframe=Timeframe(timeframe),
                as_of=as_of,
                signal_count=int(count),
                params=json.loads(params) if params else {},
            )
            for scan_id, created, strategy_id, universe, timeframe, as_of, count, params in rows
        ]

    def load(self, scan_id: str) -> pd.DataFrame:
        """Lädt die Signale eines früheren Scans.

        Returns:
            DataFrame mit :data:`~trading_helfer.scanner.scan.RESULT_COLUMNS`; Kurs,
            Sektor und Marktkapitalisierung stehen nicht in der Datenbank. Sektor kommt
            aus den Stammdaten, Kurs und Marktkapitalisierung bleiben leer.
            ``attrs["scan_id"]`` ist die ID.
        """
        if not self.settings.database_path.exists():
            result = empty_result()
            result.attrs["scan_id"] = scan_id
            return result
        with MarketStore(self.settings.database_path) as store:
            frame: pd.DataFrame = store.connection.execute(
                "SELECT s.symbol, CAST(s.date AS TIMESTAMP_NS) AS date, s.signal, s.strength, "
                "CAST(NULL AS DOUBLE) AS close, i.sector, CAST(NULL AS DOUBLE) AS market_cap, "
                "s.note FROM signals s LEFT JOIN instruments i ON i.symbol = s.symbol "
                "WHERE s.scan_id = ? ORDER BY s.symbol",
                [scan_id],
            ).df()
        order = {SignalType.BUY.value: 0, SignalType.SELL.value: 1, SignalType.EXIT.value: 2}
        frame = frame.sort_values(
            ["signal", "symbol"], key=lambda s: s.map(order) if s.name == "signal" else s
        ).reset_index(drop=True)
        frame = frame[list(RESULT_COLUMNS)]
        frame.attrs["scan_id"] = scan_id
        return frame

    def delete(self, scan_id: str) -> None:
        """Löscht einen früheren Scan samt Signalen."""
        if not self.settings.database_path.exists():
            return
        with MarketStore(self.settings.database_path) as store, store.transaction() as conn:
            conn.execute("DELETE FROM signals WHERE scan_id = ?", [scan_id])
            conn.execute("DELETE FROM scans WHERE scan_id = ?", [scan_id])

    # -- Export ----------------------------------------------------------------------

    def export_csv(self, result: pd.DataFrame, path: str | Path) -> Path:
        """Speichert ein Scan-Ergebnis als CSV und gibt den Pfad zurück."""
        return export_csv(result, path)


def _now() -> dt.datetime:
    return dt.datetime.now()


def _filters_dict(filters: ScanFilters) -> dict[str, Any]:
    return {
        "sectors": sorted(filters.sectors) if filters.sectors is not None else None,
        "min_market_cap": filters.min_market_cap,
        "max_market_cap": filters.max_market_cap,
        "metrics": {name: list(bounds) for name, bounds in sorted(filters.metrics.items())},
        "signals": sorted(s.value for s in filters.signals) if filters.signals else None,
    }
