"""Zähler für Anfragen an Datenquellen mit festen Kontingenten.

Manche Quellen begrenzen die Zahl der Anfragen je Stunde und je Tag (Tiingo Starter:
50 pro Stunde, 1.000 pro Tag). Damit diese Grenzen auch über Neustarts der App hinweg
eingehalten werden, führt :class:`DuckDBRequestCounter` die Anfragen in der Tabelle
``provider_requests`` (Migration ``002_provider_requests.sql``). Für Tests und
kurzlebige Läufe gibt es :class:`MemoryRequestCounter`.

Gezählt wird über gleitende Fenster: „pro Stunde“ heißt „in den letzten 60 Minuten“.
Das ist strenger als ein Zurücksetzen zur vollen Stunde und damit auf der sicheren Seite.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

import duckdb

__all__ = [
    "DuckDBRequestCounter",
    "MemoryRequestCounter",
    "QuotaWindow",
    "RequestCounter",
    "next_free_slot",
]

_RETENTION = timedelta(days=2)
"""Ältere Einträge werden beim Schreiben gelöscht; kein Kontingent reicht weiter zurück."""


@dataclass(frozen=True, slots=True)
class QuotaWindow:
    """Höchstens ``limit`` Anfragen innerhalb von ``period``."""

    limit: int
    period: timedelta
    label: str
    """Deutsche Bezeichnung für Meldungen, z. B. ``"pro Stunde"``."""


@runtime_checkable
class RequestCounter(Protocol):
    """Speichert Zeitpunkte von Anfragen je Provider."""

    def record(self, provider: str, at: datetime) -> None:
        """Vermerkt eine Anfrage von ``provider`` zum Zeitpunkt ``at`` (UTC, ohne Zone)."""
        ...

    def timestamps_since(self, provider: str, since: datetime) -> list[datetime]:
        """Alle Anfragezeitpunkte von ``provider`` ab ``since``, aufsteigend sortiert."""
        ...


class MemoryRequestCounter:
    """Zähler im Arbeitsspeicher (geht beim Beenden verloren)."""

    def __init__(self) -> None:
        """Legt einen leeren Zähler an."""
        self._entries: defaultdict[str, list[datetime]] = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, provider: str, at: datetime) -> None:
        """Vermerkt eine Anfrage von ``provider`` zum Zeitpunkt ``at``."""
        with self._lock:
            entries = self._entries[provider]
            entries.append(at)
            entries.sort()
            cutoff = at - _RETENTION
            self._entries[provider] = [t for t in entries if t >= cutoff]

    def timestamps_since(self, provider: str, since: datetime) -> list[datetime]:
        """Alle Anfragezeitpunkte von ``provider`` ab ``since``, aufsteigend sortiert."""
        with self._lock:
            return [t for t in self._entries.get(provider, []) if t >= since]


class DuckDBRequestCounter:
    """Zähler in der DuckDB-Tabelle ``provider_requests``.

    Die Tabelle legt die Migration ``002_provider_requests.sql`` an; dieser Zähler
    verändert das Schema nicht.
    """

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        """Nutzt eine offene DuckDB-Verbindung, in der App ``MarketStore.connection``.

        Args:
            connection: Verbindung zu einer Datenbank mit Tabelle ``provider_requests``.
        """
        self._con = connection
        self._lock = threading.Lock()

    def record(self, provider: str, at: datetime) -> None:
        """Vermerkt eine Anfrage und löscht Einträge, die älter als zwei Tage sind."""
        with self._lock:
            self._con.execute(
                "INSERT INTO provider_requests (provider, requested_at) VALUES (?, ?)",
                [provider, at],
            )
            self._con.execute(
                "DELETE FROM provider_requests WHERE provider = ? AND requested_at < ?",
                [provider, at - _RETENTION],
            )

    def timestamps_since(self, provider: str, since: datetime) -> list[datetime]:
        """Alle Anfragezeitpunkte von ``provider`` ab ``since``, aufsteigend sortiert."""
        with self._lock:
            rows = self._con.execute(
                "SELECT requested_at FROM provider_requests "
                "WHERE provider = ? AND requested_at >= ? ORDER BY requested_at",
                [provider, since],
            ).fetchall()
        return [row[0] for row in rows]


def next_free_slot(
    counter: RequestCounter,
    provider: str,
    windows: tuple[QuotaWindow, ...],
    now: datetime,
) -> tuple[datetime, QuotaWindow] | None:
    """Prüft, ob jetzt eine weitere Anfrage erlaubt ist.

    Args:
        counter: Zähler mit den bisherigen Anfragen.
        provider: Name des Providers.
        windows: Kontingente, die alle eingehalten werden müssen.
        now: Aktueller Zeitpunkt (UTC, ohne Zone).

    Returns:
        ``None``, wenn die Anfrage erlaubt ist; sonst den frühesten Zeitpunkt, ab dem
        sie wieder erlaubt ist, und das Kontingent, das am längsten blockiert.
    """
    if not windows:
        return None
    longest = max(w.period for w in windows)
    stamps = counter.timestamps_since(provider, now - longest)
    blocked: tuple[datetime, QuotaWindow] | None = None
    for window in windows:
        inside = [t for t in stamps if t > now - window.period]
        if len(inside) < window.limit:
            continue
        # Frei wird ein Platz, wenn die älteste zählende Anfrage aus dem Fenster fällt.
        resume_at = inside[len(inside) - window.limit] + window.period
        if blocked is None or resume_at > blocked[0]:
            blocked = (resume_at, window)
    return blocked
