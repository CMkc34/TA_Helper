"""Datenqualitätsprüfung der gespeicherten Tageskurse (Issue D-08).

Geprüft wird je Symbol (``docs/SPEC.md``, Abschnitt „Daten-Provider“):

- **Lücken** gegen den Börsenkalender (XNYS): fehlende Handelstage zwischen dem ersten
  und dem letzten gespeicherten Tag.
- **High < Low** an einem Tag.
- **Kurse ≤ 0** in ``open``, ``high``, ``low`` oder ``close``.
- **Sprünge > 50 %** im Schlusskurs von einem Tag zum nächsten ohne Corporate Action
  (Split oder Dividende) im Zeitraum dazwischen. Geprüft werden die Rohkurse, weil ein
  Split dort einen echten Sprung erzeugt, der durch die Corporate Action erklärt ist.
- **Fehlende Daten für Indexmitglieder** (auch ehemalige): keine oder nur teilweise
  Kurse während der Mitgliedschaft (``docs/SPEC.md``, Abschnitt „Datenquellen“).
- **Fremdkurse**: Kurse, die wahrscheinlich zu einem anderen Wert unter demselben
  (später neu vergebenen) Ticker gehören (:func:`check_foreign_prices`). Solche Werte
  schließt :func:`refresh_exclusions` aus Backtests aus (Tabelle ``symbol_exclusions``).

Die Prüfungen sind reine Funktionen auf DataFrames (:func:`check_bars`,
:func:`check_membership`); :func:`check_symbol` und :func:`check_all` lesen die Daten
aus dem :class:`~trading_helfer.data.storage.MarketStore`. Das Ergebnis lässt sich mit
:func:`issues_to_frame` als Bericht-DataFrame ausgeben.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
import pandas as pd

from trading_helfer.core.bars import PRICE_COLUMNS
from trading_helfer.data.resample import _calendar_sessions
from trading_helfer.data.storage import (
    EXCLUSION_ALLOWED,
    EXCLUSION_AUTO,
    EXCLUSION_MANUAL,
    MarketStore,
)

__all__ = [
    "DEFAULT_FOREIGN_JUMP_THRESHOLD",
    "DEFAULT_FOREIGN_MIN_JUMPS",
    "DEFAULT_JUMP_THRESHOLD",
    "DEFAULT_MEMBERSHIP_TOLERANCE",
    "REPORT_COLUMNS",
    "Check",
    "Issue",
    "Severity",
    "check_all",
    "check_bars",
    "check_foreign_prices",
    "check_membership",
    "check_symbol",
    "issues_to_frame",
    "refresh_exclusions",
]

DEFAULT_JUMP_THRESHOLD: Final = 0.5
"""Relative Kursänderung von Tag zu Tag, ab der ein Sprung gemeldet wird (50 %)."""

DEFAULT_FOREIGN_JUMP_THRESHOLD: Final = 0.8
"""Tagessprung, der für die Prüfung auf Fremdkurse zählt (80 %)."""

DEFAULT_FOREIGN_MIN_JUMPS: Final = 3
"""So viele Sprünge über :data:`DEFAULT_FOREIGN_JUMP_THRESHOLD` ohne Corporate Action
machen eine Kursreihe zu Fremdkursen. Echte Werte im Index haben das praktisch nie."""

DEFAULT_MEMBERSHIP_TOLERANCE: Final = 5
"""Handelstage, um die Kurse am Rand einer Indexmitgliedschaft fehlen dürfen."""

REPORT_COLUMNS: Final[tuple[str, ...]] = ("symbol", "date", "check", "severity", "detail")
"""Spalten des Qualitätsberichts (``severity`` = Schwere)."""


class Check(StrEnum):
    """Art einer Prüfung (Wert in der Spalte ``check`` des Berichts)."""

    GAP = "GAP"
    """Fehlende Handelstage laut Börsenkalender."""
    HIGH_LOW = "HIGH_LOW"
    """Tageshoch liegt unter dem Tagestief."""
    NON_POSITIVE = "NON_POSITIVE"
    """Kurs kleiner oder gleich 0."""
    JUMP = "JUMP"
    """Kurssprung über der Schwelle ohne Corporate Action."""
    MISSING_DATA = "MISSING_DATA"
    """Keine oder unvollständige Kurse während einer Indexmitgliedschaft."""
    FOREIGN_PRICES = "FOREIGN_PRICES"
    """Kurse gehören wahrscheinlich zu einem anderen Wert unter demselben Ticker."""


class Severity(StrEnum):
    """Schwere eines Befunds."""

    ERROR = "ERROR"
    """Daten sind sicher falsch oder fehlen; Ergebnisse damit sind unzuverlässig."""
    WARNING = "WARNING"
    """Daten sind verdächtig und sollten geprüft werden."""


@dataclass(frozen=True, slots=True)
class Issue:
    """Ein Befund der Qualitätsprüfung.

    Attributes:
        symbol: Tickersymbol.
        date: Betroffener Tag (bei Lücken der erste fehlende Tag); ``None``, wenn der
            Befund keinen einzelnen Tag betrifft.
        check: Art der Prüfung.
        severity: Schwere des Befunds.
        detail: Beschreibung auf Deutsch.
    """

    symbol: str
    date: dt.date | None
    check: Check
    severity: Severity
    detail: str


# ---------------------------------------------------------------------- Prüfungen


def check_bars(
    symbol: str,
    bars: pd.DataFrame,
    actions: pd.DataFrame | None = None,
    *,
    jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
) -> list[Issue]:
    """Prüft eine Tageskursreihe auf Lücken, High < Low, Kurse ≤ 0 und Sprünge.

    Args:
        symbol: Tickersymbol für die Befunde.
        bars: Rohkurse (nicht bereinigt) mit ``date, open, high, low, close``.
        actions: Corporate Actions mit ``date`` (und ``type, value``); ``None`` = keine.
        jump_threshold: Relative Änderung des Schlusskurses, ab der ein Sprung gilt.

    Returns:
        Befunde, sortiert nach Datum und Prüfung; leer für saubere Daten.
    """
    if bars.empty:
        return []
    frame = bars.loc[:, ["date", *PRICE_COLUMNS]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    issues = [
        *_check_gaps(symbol, frame),
        *_check_high_low(symbol, frame),
        *_check_non_positive(symbol, frame),
        *_check_jumps(symbol, frame, actions, jump_threshold),
    ]
    return _sorted(issues)


def check_membership(
    symbol: str,
    bars: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    history_start: dt.date | None = None,
    today: dt.date | None = None,
    tolerance: int = DEFAULT_MEMBERSHIP_TOLERANCE,
) -> list[Issue]:
    """Prüft, ob für jede Indexmitgliedschaft des Symbols Kurse vorhanden sind.

    Gerade für ehemalige Mitglieder liefern Provider oft keine Daten mehr; das wird hier
    sichtbar gemacht statt stillschweigend ignoriert.

    Args:
        symbol: Tickersymbol.
        bars: Kurse des Symbols mit Spalte ``date`` (Rohkurse oder bereinigt).
        memberships: Mitgliedschaften mit ``index_code, start_date, end_date``
            (``end_date`` leer = heute noch Mitglied).
        history_start: Frühestes erwartetes Kursdatum; Mitgliedschaft davor wird
            nicht geprüft. ``None`` = keine Untergrenze.
        today: Bezugstag für laufende Mitgliedschaften; Standard ist der letzte
            abgeschlossene Handelstag vor heute.
        tolerance: Handelstage, die am Anfang oder Ende der Mitgliedschaft fehlen dürfen.

    Returns:
        ``MISSING_DATA``-Befunde: ``ERROR`` ohne jeden Kurs in der Mitgliedschaft,
        ``WARNING`` bei Kursen, die die Mitgliedschaft nur teilweise abdecken.
    """
    if memberships.empty:
        return []
    dates = pd.DatetimeIndex(pd.to_datetime(bars["date"])).normalize().sort_values()
    reference = pd.Timestamp(today if today is not None else dt.date.today())
    lower = pd.Timestamp(history_start) if history_start is not None else None
    issues: list[Issue] = []
    for index_code, start_value, end_value in memberships.loc[
        :, ["index_code", "start_date", "end_date"]
    ].itertuples(index=False, name=None):
        start = _timestamp(start_value)
        end = reference if pd.isna(end_value) else min(_timestamp(end_value), reference)
        if lower is not None:
            start = max(start, lower)
        sessions = _sessions_between(start, end)
        if today is None and len(sessions) > 0 and sessions[-1] >= reference:
            # Der laufende Handelstag ist noch nicht abgeschlossen.
            sessions = sessions[sessions < reference]
        if len(sessions) == 0:
            continue
        first, last = sessions[0], sessions[-1]
        inside = dates[(dates >= first) & (dates <= last)]
        span = f"{first.date().isoformat()} bis {last.date().isoformat()}"
        if len(inside) == 0:
            issues.append(
                Issue(
                    symbol,
                    first.date(),
                    Check.MISSING_DATA,
                    Severity.ERROR,
                    f"Keine Kurse während der Mitgliedschaft in {index_code} ({span}).",
                )
            )
            continue
        missing_head = int(sessions.searchsorted(inside[0]))
        missing_tail = len(sessions) - 1 - int(sessions.searchsorted(inside[-1]))
        if missing_head > tolerance or missing_tail > tolerance:
            issues.append(
                Issue(
                    symbol,
                    first.date(),
                    Check.MISSING_DATA,
                    Severity.WARNING,
                    f"Kurse decken die Mitgliedschaft in {index_code} ({span}) nur "
                    f"teilweise ab: {inside[0].date().isoformat()} bis "
                    f"{inside[-1].date().isoformat()}.",
                )
            )
    return _sorted(issues)


def check_foreign_prices(
    symbol: str,
    bars: pd.DataFrame,
    actions: pd.DataFrame | None = None,
    memberships: pd.DataFrame | None = None,
    *,
    history_start: dt.date | None = None,
    jump_threshold: float = DEFAULT_FOREIGN_JUMP_THRESHOLD,
    min_jumps: int = DEFAULT_FOREIGN_MIN_JUMPS,
) -> list[Issue]:
    """Erkennt Kurse, die wahrscheinlich zu einem anderen Wert gehören (Fremdkurse).

    Ticker werden nach einem Delisting oft neu vergeben; Provider liefern dann unter dem
    alten Symbol die Kurse der neuen Firma. Zwei Muster gelten als Fremdkurse:

    - mindestens ``min_jumps`` Tagessprünge über ``jump_threshold`` ohne Corporate
      Action (Kursreihen zweier Firmen aneinandergehängt oder vermischt),
    - die Kurse beginnen erst nach dem Ende der letzten Indexmitgliedschaft.

    Args:
        symbol: Tickersymbol.
        bars: Rohkurse mit ``date`` und ``close`` (für die Sprünge auch ``open, high,
            low``).
        actions: Corporate Actions mit ``date``; ``None`` = keine.
        memberships: Mitgliedschaften mit ``start_date, end_date``; ``None`` = keine.
        history_start: Mitgliedschaften, die davor enden, zählen nicht.
        jump_threshold: Relative Änderung, ab der ein Sprung zählt.
        min_jumps: Mindestzahl Sprünge für den Befund.

    Returns:
        Höchstens ein ``FOREIGN_PRICES``-Befund (``ERROR``); leer für unauffällige Kurse.
    """
    if bars.empty:
        return []
    frame = bars.loc[:, ["date", *PRICE_COLUMNS]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    jumps = _check_jumps(symbol, frame, actions, jump_threshold)
    if len(jumps) >= max(1, min_jumps):
        first = jumps[0].date
        assert first is not None
        return [
            Issue(
                symbol,
                first,
                Check.FOREIGN_PRICES,
                Severity.ERROR,
                f"Wahrscheinlich Kurse eines anderen Werts unter demselben Ticker: "
                f"{len(jumps)} Tagessprünge über {jump_threshold:.0%} ohne Corporate Action "
                f"(erster am {first.isoformat()}).",
            )
        ]
    if memberships is None or memberships.empty or memberships["end_date"].isna().any():
        return []
    ends = pd.to_datetime(memberships["end_date"])
    if history_start is not None:
        ends = ends[ends >= pd.Timestamp(history_start)]
    if ends.empty:
        return []
    last_end = ends.max()
    first_bar = frame["date"].iloc[0]
    if first_bar < last_end:
        return []
    return [
        Issue(
            symbol,
            first_bar.date(),
            Check.FOREIGN_PRICES,
            Severity.ERROR,
            f"Kurse beginnen erst am {first_bar.date().isoformat()}, nach dem Ende der "
            f"letzten Indexmitgliedschaft ({last_end.date().isoformat()}); der Ticker "
            "wurde wahrscheinlich neu vergeben.",
        )
    ]


def refresh_exclusions(
    store: MarketStore,
    symbols: Iterable[str] | None = None,
    *,
    history_start: dt.date | None = None,
) -> dict[str, str]:
    """Bewertet die automatischen Ausschlüsse wegen Fremdkursen neu.

    Für jedes Symbol wird :func:`check_foreign_prices` ausgeführt. Ein Befund legt einen
    Ausschluss mit Quelle ``auto`` an, ein unauffälliges Symbol verliert seinen
    ``auto``-Ausschluss (z. B. nach Neuladen aus einer besseren Quelle). Einträge mit
    Quelle ``manuell`` oder ``freigegeben`` bleiben unverändert.

    Args:
        store: Beschreibbar geöffneter Speicher.
        symbols: Nur diese Symbole bewerten; ``None`` = alle mit Kursen oder Mitgliedschaft.
        history_start: Siehe :func:`check_foreign_prices`.

    Returns:
        Die bewerteten Symbole, die jetzt automatisch ausgeschlossen sind, mit Grund.
    """
    selected = (
        _all_symbols(store)
        if symbols is None
        else sorted({s.strip().upper() for s in symbols if s.strip()})
    )
    existing = store.load_exclusions()
    sources = dict(zip(existing["symbol"], existing["source"], strict=True))
    flagged: dict[str, str] = {}
    for sym in selected:
        if sources.get(sym) in (EXCLUSION_MANUAL, EXCLUSION_ALLOWED):
            continue
        issues = check_foreign_prices(
            sym,
            store.load_bars(sym, adjusted=False),
            store.load_actions(sym),
            _memberships(store, sym),
            history_start=history_start,
        )
        if issues:
            flagged[sym] = issues[0].detail
            store.set_exclusion(sym, issues[0].detail, EXCLUSION_AUTO)
        elif sources.get(sym) == EXCLUSION_AUTO:
            store.remove_exclusion(sym)
    return flagged


def check_symbol(
    symbol: str,
    store: MarketStore | None = None,
    *,
    jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
    history_start: dt.date | None = None,
    today: dt.date | None = None,
) -> list[Issue]:
    """Prüft alle gespeicherten Daten eines Symbols.

    Args:
        symbol: Tickersymbol.
        store: Geöffneter Speicher; ``None`` öffnet die Datenbank aus den Einstellungen
            nur lesend.
        jump_threshold: Schwelle für Kurssprünge, siehe :func:`check_bars`.
        history_start: Frühestes erwartetes Kursdatum, siehe :func:`check_membership`;
            ``None`` = ``start_date`` aus den Einstellungen, wenn ``store`` fehlt,
            sonst keine Untergrenze.
        today: Bezugstag für laufende Indexmitgliedschaften.

    Returns:
        Befunde, sortiert nach Datum und Prüfung; leer für saubere Daten.
    """
    with _opened(store, history_start) as (opened, lower):
        return _check_one(
            opened, symbol, jump_threshold=jump_threshold, history_start=lower, today=today
        )


def check_all(
    store: MarketStore | None = None,
    *,
    symbols: Iterable[str] | None = None,
    jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
    history_start: dt.date | None = None,
    today: dt.date | None = None,
) -> pd.DataFrame:
    """Prüft alle Symbole mit Kursen oder Indexmitgliedschaft.

    Args:
        store: Geöffneter Speicher; ``None`` öffnet die Datenbank aus den Einstellungen
            nur lesend.
        symbols: Nur diese Symbole prüfen; ``None`` = alle.
        jump_threshold: Schwelle für Kurssprünge, siehe :func:`check_bars`.
        history_start: Frühestes erwartetes Kursdatum, siehe :func:`check_symbol`.
        today: Bezugstag für laufende Indexmitgliedschaften.

    Returns:
        Bericht mit den Spalten :data:`REPORT_COLUMNS`, sortiert nach Symbol, Datum und
        Prüfung; leer (mit allen Spalten) für saubere Daten.
    """
    with _opened(store, history_start) as (opened, lower):
        if symbols is None:
            selected = _all_symbols(opened)
        else:
            selected = sorted({s.strip().upper() for s in symbols if s.strip()})
        issues: list[Issue] = []
        for symbol in selected:
            issues.extend(
                _check_one(
                    opened,
                    symbol,
                    jump_threshold=jump_threshold,
                    history_start=lower,
                    today=today,
                )
            )
    return issues_to_frame(issues)


def issues_to_frame(issues: Sequence[Issue]) -> pd.DataFrame:
    """Wandelt Befunde in den Bericht-DataFrame mit den Spalten :data:`REPORT_COLUMNS`.

    ``date`` ist ``datetime64[ns]`` (``NaT`` für Befunde ohne Tag), ``check`` und
    ``severity`` sind die Zeichenkettenwerte von :class:`Check` und :class:`Severity`.
    """
    ordered = sorted(issues, key=lambda i: (i.symbol, *_sort_key(i)))
    frame = pd.DataFrame(
        {
            "symbol": pd.Series([i.symbol for i in ordered], dtype="object"),
            "date": pd.to_datetime(pd.Series([i.date for i in ordered], dtype="object")).astype(
                "datetime64[ns]"
            ),
            "check": pd.Series([i.check.value for i in ordered], dtype="object"),
            "severity": pd.Series([i.severity.value for i in ordered], dtype="object"),
            "detail": pd.Series([i.detail for i in ordered], dtype="object"),
        }
    )
    return frame.loc[:, list(REPORT_COLUMNS)]


# ---------------------------------------------------------------------- Einzelprüfungen


def _check_gaps(symbol: str, frame: pd.DataFrame) -> list[Issue]:
    first, last = frame["date"].iloc[0], frame["date"].iloc[-1]
    sessions = _sessions_between(first, last)
    missing = sessions.difference(pd.DatetimeIndex(frame["date"]))
    if len(missing) == 0:
        return []
    positions = sessions.get_indexer(missing)
    # Aufeinanderfolgende fehlende Handelstage zu einer Lücke zusammenfassen.
    breaks = np.flatnonzero(np.diff(positions) != 1) + 1
    issues = []
    for block in np.split(np.arange(len(missing)), breaks):
        start, end = missing[block[0]], missing[block[-1]]
        count = len(block)
        if count == 1:
            detail = f"1 Handelstag fehlt ({start.date().isoformat()})."
        else:
            detail = (
                f"{count} Handelstage fehlen ({start.date().isoformat()} bis "
                f"{end.date().isoformat()})."
            )
        issues.append(Issue(symbol, start.date(), Check.GAP, Severity.WARNING, detail))
    return issues


def _check_high_low(symbol: str, frame: pd.DataFrame) -> list[Issue]:
    bad = frame[frame["high"] < frame["low"]]
    return [
        Issue(
            symbol,
            _timestamp(day).date(),
            Check.HIGH_LOW,
            Severity.ERROR,
            f"Hoch {high:g} liegt unter Tief {low:g}.",
        )
        for day, high, low in bad.loc[:, ["date", "high", "low"]].itertuples(index=False, name=None)
    ]


def _check_non_positive(symbol: str, frame: pd.DataFrame) -> list[Issue]:
    prices = frame.loc[:, list(PRICE_COLUMNS)]
    bad_rows = (prices <= 0).any(axis=1)
    issues = []
    for idx in np.flatnonzero(bad_rows.to_numpy()):
        row = frame.iloc[idx]
        parts = [f"{col} = {row[col]:g}" for col in PRICE_COLUMNS if row[col] <= 0]
        issues.append(
            Issue(
                symbol,
                _timestamp(row["date"]).date(),
                Check.NON_POSITIVE,
                Severity.ERROR,
                f"Kurs kleiner oder gleich 0: {', '.join(parts)}.",
            )
        )
    return issues


def _check_jumps(
    symbol: str, frame: pd.DataFrame, actions: pd.DataFrame | None, threshold: float
) -> list[Issue]:
    close = frame["close"].to_numpy(dtype="float64")
    if len(close) < 2:
        return []
    prev = close[:-1]
    curr = close[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        change = curr / prev - 1.0
    # Kurse ≤ 0 meldet eine eigene Prüfung; dort ist die Änderung nicht sinnvoll.
    valid = (prev > 0) & (curr > 0)
    candidates = np.flatnonzero(valid & (np.abs(change) > threshold)) + 1
    if len(candidates) == 0:
        return []
    action_dates = (
        pd.DatetimeIndex([])
        if actions is None or actions.empty
        else pd.DatetimeIndex(pd.to_datetime(actions["date"])).normalize()
    )
    dates = pd.DatetimeIndex(frame["date"])
    issues = []
    for idx in candidates.tolist():
        prev_day, day = dates[idx - 1], dates[idx]
        if ((action_dates > prev_day) & (action_dates <= day)).any():
            continue
        issues.append(
            Issue(
                symbol,
                day.date(),
                Check.JUMP,
                Severity.WARNING,
                f"Schlusskurs ändert sich um {change[idx - 1]:+.1%} "
                f"({close[idx - 1]:g} → {close[idx]:g}) ohne Corporate Action.",
            )
        )
    return issues


# ---------------------------------------------------------------------- Hilfen


def _check_one(
    store: MarketStore,
    symbol: str,
    *,
    jump_threshold: float,
    history_start: dt.date | None,
    today: dt.date | None,
) -> list[Issue]:
    sym = symbol.strip().upper()
    bars = store.load_bars(sym, adjusted=False)
    actions = store.load_actions(sym)
    memberships = _memberships(store, sym)
    issues = check_bars(sym, bars, actions, jump_threshold=jump_threshold)
    issues += check_membership(sym, bars, memberships, history_start=history_start, today=today)
    issues += check_foreign_prices(sym, bars, actions, memberships, history_start=history_start)
    return _sorted(issues)


def _memberships(store: MarketStore, symbol: str) -> pd.DataFrame:
    result: pd.DataFrame = store.connection.execute(
        "SELECT index_code, start_date, end_date FROM index_membership "
        "WHERE symbol = ? ORDER BY start_date, index_code",
        [symbol],
    ).df()
    return result


def _all_symbols(store: MarketStore) -> list[str]:
    rows = store.connection.execute(
        "SELECT symbol FROM bars_daily UNION SELECT symbol FROM index_membership ORDER BY 1"
    ).fetchall()
    return [str(row[0]) for row in rows]


@contextmanager
def _opened(
    store: MarketStore | None, history_start: dt.date | None
) -> Iterator[tuple[MarketStore, dt.date | None]]:
    """Liefert ``store`` oder öffnet die Datenbank aus den Einstellungen nur lesend."""
    if store is not None:
        yield store, history_start
        return
    from trading_helfer.core.config import load_settings

    settings = load_settings()
    lower = history_start if history_start is not None else settings.start_date
    with MarketStore(settings.database_path, read_only=True) as opened:
        yield opened, lower


def _sessions_between(first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    """XNYS-Handelstage von ``first`` bis ``last`` (beide einschließlich)."""
    if last < first:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    sessions = _calendar_sessions(first.year, last.year)
    return sessions[(sessions >= first) & (sessions <= last)]


def _timestamp(value: Any) -> pd.Timestamp:
    """Wandelt einen Datumswert aus einem DataFrame in einen ``Timestamp``."""
    return pd.Timestamp(value)


def _sort_key(issue: Issue) -> tuple[Any, ...]:
    return (issue.date is None, issue.date or dt.date.min, list(Check).index(issue.check))


def _sorted(issues: list[Issue]) -> list[Issue]:
    return sorted(issues, key=_sort_key)
