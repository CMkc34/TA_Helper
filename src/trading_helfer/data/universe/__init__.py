"""Indexmitglieder mit Historie (point in time) für S&P 500 und Nasdaq-100.

Die Daten liegen als CSV neben diesem Modul und werden vom Skript
``scripts/build_universe.py`` gepflegt; die App liest nur die CSV.

Je Index gibt es zwei Dateien:

- ``<code>_current.csv`` mit der Spalte ``symbol``: die aktuelle Mitgliederliste.
- ``<code>_changes.csv`` mit den Spalten ``date, added, removed, quelle``: eine Zeile je
  Aufnahme und/oder Entfernung. Eine Änderung gilt ab ihrem Datum (am Stichtag ist der
  aufgenommene Wert schon Mitglied, der entfernte nicht mehr).

Die Mitglieder an einem Stichtag werden rückwärts rekonstruiert: Ausgehend von der
aktuellen Liste werden alle Änderungen nach dem Stichtag zurückgenommen. Vor dem ersten
bekannten Änderungsdatum gilt die früheste rekonstruierte Liste. Backtests, die vor
:func:`reliable_since` beginnen, tragen das Survivorship-Risiko
(:func:`survivorship_risk`, :data:`SURVIVORSHIP_WARNING`).

Symbole stehen in App-Schreibweise (``BRK.B``) und so, wie sie am jeweiligen Tag
gehandelt wurden (z. B. ``FB`` vor 2022, ``META`` danach).
"""

from __future__ import annotations

import csv
import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

import pandas as pd

__all__ = [
    "CHANGE_COLUMNS",
    "INDEX_CODES",
    "MEMBERSHIP_COLUMNS",
    "SURVIVORSHIP_PREFIX",
    "SURVIVORSHIP_WARNING",
    "UNIVERSE_DIR",
    "IndexChange",
    "IndexHistory",
    "UniverseError",
    "all_members",
    "load_history",
    "members_on",
    "membership_intervals",
    "reliable_since",
    "survivorship_risk",
]

UNIVERSE_DIR: Final = Path(__file__).resolve().parent
"""Ordner mit den Universum-CSV-Dateien."""

INDEX_CODES: Final[dict[str, str]] = {"SP500": "sp500", "NDX100": "ndx100"}
"""Bekannte Index-Codes und der Präfix ihrer CSV-Dateien."""

CHANGE_COLUMNS: Final[tuple[str, ...]] = ("date", "added", "removed", "quelle")
"""Spalten der Änderungsdateien ``<code>_changes.csv``."""

MEMBERSHIP_COLUMNS: Final[tuple[str, ...]] = ("index_code", "symbol", "start_date", "end_date")
"""Spalten von :func:`membership_intervals` (wie Tabelle ``index_membership``)."""

RELIABLE_FROM: Final = dt.date(2000, 1, 1)
"""Vor diesem Datum gelten Mitgliederlisten laut Spezifikation grundsätzlich als unsicher."""

SURVIVORSHIP_PREFIX: Final = "Survivorship-Risiko"
"""Anfang aller Hinweise auf ein Survivorship-Risiko in Protokollen und Berichten."""

SURVIVORSHIP_WARNING: Final = (
    f"{SURVIVORSHIP_PREFIX}: Der Backtest beginnt vor dem Zeitraum, für den die "
    "Indexmitglieder verlässlich rekonstruiert sind. Frühere Mitglieder können fehlen, "
    "die Ergebnisse wirken dadurch eventuell zu gut."
)
"""Hinweistext für Berichte von Backtests, die vor :func:`reliable_since` beginnen."""

DateLike = dt.date | str | pd.Timestamp


class UniverseError(ValueError):
    """Fehler beim Lesen oder Auswerten der Universum-Daten."""


@dataclass(frozen=True, slots=True)
class IndexChange:
    """Alle Aufnahmen und Entfernungen eines Index an einem Tag."""

    date: dt.date
    added: frozenset[str]
    removed: frozenset[str]


@dataclass(frozen=True, slots=True)
class IndexHistory:
    """Aktuelle Mitglieder plus chronologisch sortierte Änderungen eines Index."""

    index_code: str
    current: frozenset[str]
    changes: tuple[IndexChange, ...]

    def members_on(self, date: DateLike) -> set[str]:
        """Mitglieder am Stichtag ``date`` (Änderungen gelten ab ihrem Datum)."""
        day = _to_date(date)
        members = set(self.current)
        for change in reversed(self.changes):
            if change.date <= day:
                break
            members -= change.added
            members |= change.removed
        return members

    def all_members(self, start: DateLike, end: DateLike) -> set[str]:
        """Alle Werte, die an mindestens einem Tag in ``[start, end]`` Mitglied waren."""
        first, last = _to_date(start), _to_date(end)
        if last < first:
            raise UniverseError(f"Ende {last} liegt vor dem Start {first}.")
        members = self.members_on(first)
        for change in self.changes:
            if first < change.date <= last:
                members |= change.added
        return members

    @property
    def first_change(self) -> dt.date | None:
        """Datum der frühesten bekannten Änderung (``None`` ohne Änderungen)."""
        return self.changes[0].date if self.changes else None


def load_history(index_code: str, directory: Path | None = None) -> IndexHistory:
    """Liest aktuelle Liste und Änderungen eines Index aus den CSV-Dateien.

    Args:
        index_code: ``"SP500"`` oder ``"NDX100"`` (Groß-/Kleinschreibung egal).
        directory: Ordner mit den CSV-Dateien; Standard ist :data:`UNIVERSE_DIR`.

    Returns:
        Die geprüfte Historie. Ergebnisse für den Standardordner werden zwischengespeichert.

    Raises:
        UniverseError: Unbekannter Index, fehlende Datei oder widersprüchliche Daten.
    """
    code = _normalize_code(index_code)
    if directory is None:
        return _load_default(code)
    return _read_history(code, Path(directory))


def members_on(index_code: str, date: DateLike) -> set[str]:
    """Mitglieder des Index am Stichtag ``date``.

    Vor der ersten bekannten Änderung gilt die früheste rekonstruierte Liste.
    """
    return load_history(index_code).members_on(date)


def all_members(index_code: str, start: DateLike, end: DateLike) -> set[str]:
    """Alle Werte, die zwischen ``start`` und ``end`` (inklusive) Mitglied des Index waren."""
    return load_history(index_code).all_members(start, end)


def reliable_since(index_code: str) -> dt.date:
    """Erster Tag, ab dem die Mitgliederliste als verlässlich gilt.

    Das ist das spätere Datum aus erster bekannter Änderung und dem 01.01.2000.
    """
    first = load_history(index_code).first_change
    return max(first, RELIABLE_FROM) if first is not None else dt.date.max


def survivorship_risk(index_code: str, start: DateLike) -> bool:
    """``True``, wenn ein Backtest ab ``start`` den Hinweis „Survivorship-Risiko“ braucht."""
    return _to_date(start) < reliable_since(index_code)


def membership_intervals(index_code: str) -> pd.DataFrame:
    """Mitgliedschaften als Intervalle im Format der Tabelle ``index_membership``.

    Returns:
        DataFrame mit :data:`MEMBERSHIP_COLUMNS`; ``end_date`` ist der erste Tag ohne
        Mitgliedschaft oder ``None`` für aktuelle Mitglieder. Mitglieder der frühesten
        rekonstruierten Liste beginnen am Tag vor der ersten Änderung.
    """
    history = load_history(index_code)
    code = history.index_code
    first = history.first_change
    if first is None:
        rows = [(code, symbol, None, None) for symbol in sorted(history.current)]
        return pd.DataFrame(rows, columns=list(MEMBERSHIP_COLUMNS))
    base = first - dt.timedelta(days=1)
    open_since = dict.fromkeys(history.members_on(base), base)
    records: list[tuple[str, str, dt.date | None, dt.date | None]] = []
    for change in history.changes:
        for symbol in sorted(change.removed):
            records.append((code, symbol, open_since.pop(symbol), change.date))
        for symbol in sorted(change.added):
            open_since[symbol] = change.date
    records.extend((code, symbol, start, None) for symbol, start in open_since.items())
    frame = pd.DataFrame(records, columns=list(MEMBERSHIP_COLUMNS))
    return frame.sort_values(["symbol", "start_date"], ignore_index=True)


@cache
def _load_default(code: str) -> IndexHistory:
    return _read_history(code, UNIVERSE_DIR)


def _normalize_code(index_code: str) -> str:
    code = index_code.strip().upper()
    if code not in INDEX_CODES:
        known = ", ".join(INDEX_CODES)
        raise UniverseError(f"Unbekannter Index {index_code!r}; bekannt sind: {known}.")
    return code


def _read_history(code: str, directory: Path) -> IndexHistory:
    prefix = INDEX_CODES[code]
    current = frozenset(_read_current(directory / f"{prefix}_current.csv"))
    changes = _read_changes(directory / f"{prefix}_changes.csv")
    history = IndexHistory(code, current, changes)
    _check_consistency(history)
    return history


def _read_current(path: Path) -> set[str]:
    rows = _read_csv(path, ("symbol",))
    return {_normalize_symbol(row["symbol"]) for row in rows if row["symbol"].strip()}


def _read_changes(path: Path) -> tuple[IndexChange, ...]:
    by_date: dict[dt.date, tuple[set[str], set[str]]] = {}
    for line, row in enumerate(_read_csv(path, CHANGE_COLUMNS), start=2):
        try:
            day = dt.date.fromisoformat(row["date"].strip())
        except ValueError as exc:
            raise UniverseError(f"{path.name}, Zeile {line}: ungültiges Datum.") from exc
        added, removed = row["added"].strip(), row["removed"].strip()
        if not added and not removed:
            raise UniverseError(f"{path.name}, Zeile {line}: weder added noch removed.")
        entry = by_date.setdefault(day, (set(), set()))
        if added:
            entry[0].add(_normalize_symbol(added))
        if removed:
            entry[1].add(_normalize_symbol(removed))
    return tuple(
        IndexChange(day, frozenset(added), frozenset(removed))
        for day, (added, removed) in sorted(by_date.items())
    )


def _read_csv(path: Path, columns: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise UniverseError(f"Universum-Datei fehlt: {path}.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            raise UniverseError(f"{path.name}: Spalten fehlen: {', '.join(sorted(missing))}.")
        return [{key: value or "" for key, value in row.items()} for row in reader]


def _check_consistency(history: IndexHistory) -> None:
    """Prüft, dass sich die Änderungen widerspruchsfrei rückwärts anwenden lassen."""
    members = set(history.current)
    for change in reversed(history.changes):
        missing = change.added - members
        if missing:
            raise UniverseError(
                f"{history.index_code}: {', '.join(sorted(missing))} wurde am {change.date} "
                "aufgenommen, ist danach aber nicht Mitglied."
            )
        members -= change.added
        present = change.removed & members
        if present:
            raise UniverseError(
                f"{history.index_code}: {', '.join(sorted(present))} wurde am {change.date} "
                "entfernt, ist danach aber weiterhin Mitglied."
            )
        members |= change.removed


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _to_date(value: DateLike) -> dt.date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise UniverseError(f"Ungültiges Datum: {value!r}.") from exc
