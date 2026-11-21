"""Tägliche Schnappschüsse der Optionsketten (Issue O-01).

Kostenlose historische Optionsdaten gibt es nicht (siehe ``docs/SPEC.md``, Abschnitt
„Datenquellen“). Die App baut sich deshalb ab Start eine eigene Historie auf: Einmal
am Tag werden die aktuellen Optionsketten der Watchlist abgerufen und in der Tabelle
``option_snapshots`` gespeichert.

Regeln:

- Ein Schnappschuss je Symbol und Tag. Ein zweiter Lauf am selben Tag ersetzt ihn
  vollständig (ein Symbol = eine Transaktion), es entstehen keine doppelten Zeilen.
- Am Wochenende gilt der vorherige Freitag als Stichtag (Yahoo liefert dann den
  Schlussstand vom Freitag).
- Der Fehler eines Symbols blockiert die anderen nicht. Ein Symbol ohne Optionen ist
  kein Fehler.
- Jeder Lauf steht mit Zusammenfassung in ``update_log`` (``provider`` =
  ``<quelle>:optionen``).
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from trading_helfer.data.providers.base import (
    OPTION_CHAIN_COLUMNS,
    OPTION_TYPES,
    OptionChainProvider,
    ProviderError,
)
from trading_helfer.data.storage import MarketStore

__all__ = [
    "SNAPSHOT_COLUMNS",
    "OptionSnapshotError",
    "SnapshotProgress",
    "SnapshotResult",
    "build_option_provider",
    "load_option_snapshot",
    "save_option_snapshot",
    "snapshot_date_for",
    "snapshot_dates",
    "snapshot_options",
    "validate_option_chain",
]

SNAPSHOT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "snapshot_date",
    *OPTION_CHAIN_COLUMNS,
    "source",
)
"""Spalten der Tabelle ``option_snapshots`` in dieser Reihenfolge."""

_QUOTED_COLUMNS: Final = ", ".join(f'"{c}"' for c in SNAPSHOT_COLUMNS)

LOG_SUFFIX: Final = ":optionen"
"""Anhang an den Providernamen in ``update_log.provider``."""

_KEY: Final = ("expiry", "strike", "type")
_PRICE_COLUMNS: Final = ("bid", "ask", "last", "iv", "underlying_price")
_COUNT_COLUMNS: Final = ("volume", "open_interest")

SnapshotProgress = Callable[[int, int, str], None]
"""Fortschritt als ``(erledigt, gesamt, symbol)``; wird nach jedem Symbol aufgerufen."""

_log = logging.getLogger(__name__)


class OptionSnapshotError(ValueError):
    """Ungültige Optionskette oder ungültige Eingabe; die Meldung ist deutsch."""


@dataclass
class SnapshotResult:
    """Ergebnis eines Laufs von :func:`snapshot_options`."""

    log_id: int
    """``id`` des Eintrags in ``update_log``."""
    snapshot_date: dt.date
    """Stichtag der Schnappschüsse."""
    total: int
    """Anzahl der Symbole dieses Laufs."""
    ok: list[str] = field(default_factory=list)
    """Erfolgreich gespeicherte Symbole (auch solche ohne Optionen)."""
    failed: dict[str, str] = field(default_factory=dict)
    """Fehlgeschlagene Symbole mit Grund."""
    no_options: list[str] = field(default_factory=list)
    """Symbole ohne Optionen (Teilmenge von :attr:`ok`)."""
    rows_written: int = 0
    """Anzahl gespeicherter Kontrakte."""
    cancelled: bool = False
    """``True``, wenn der Lauf abgebrochen wurde."""

    @property
    def done(self) -> int:
        """Anzahl bearbeiteter Symbole (erfolgreich oder fehlgeschlagen)."""
        return len(self.ok) + len(self.failed)

    def summary(self) -> str:
        """Kurze deutsche Zusammenfassung, wie sie in ``update_log.message`` steht."""
        parts = [
            f"Optionen {self.snapshot_date.isoformat()}: {len(self.ok)} von {self.total} "
            f"Symbolen gespeichert, {self.rows_written} Kontrakte"
        ]
        if self.failed:
            parts.append(f"{len(self.failed)} fehlgeschlagen ({', '.join(sorted(self.failed))})")
        if self.no_options:
            parts.append(f"{len(self.no_options)} ohne Optionen")
        text = ", ".join(parts) + "."
        if self.cancelled:
            text = "Abgebrochen: " + text
        return text


def build_option_provider() -> OptionChainProvider:
    """Quelle der Optionsketten; derzeit liefert nur yfinance Optionen."""
    from trading_helfer.data.providers.yfinance_provider import YFinanceProvider

    return YFinanceProvider()


def snapshot_date_for(day: dt.date) -> dt.date:
    """Stichtag eines Schnappschusses: ``day`` oder am Wochenende der Freitag davor.

    Args:
        day: Kalendertag des Abrufs.

    Returns:
        Stichtag (Montag bis Freitag).
    """
    weekday = day.weekday()
    if weekday >= 5:  # Samstag = 5, Sonntag = 6
        return day - dt.timedelta(days=weekday - 4)
    return day


def validate_option_chain(chain: pd.DataFrame) -> pd.DataFrame:
    """Prüft eine Optionskette und bringt sie auf feste Datentypen.

    Regeln: alle Spalten aus
    :data:`~trading_helfer.data.providers.base.OPTION_CHAIN_COLUMNS` vorhanden,
    ``type`` ist ``CALL`` oder ``PUT`` (Groß-/Kleinschreibung egal), ``expiry`` ist ein
    Datum, ``strike`` > 0, keine negativen Preise, Umsätze oder Volatilitäten und kein
    Kontrakt (Verfall, Basispreis, Art) doppelt.

    Args:
        chain: Optionskette, z. B. von
            :meth:`~trading_helfer.data.providers.base.OptionChainProvider.fetch_option_chain`.

    Returns:
        Neue Tabelle mit den Spalten in fester Reihenfolge, sortiert nach Verfall, Art
        und Basispreis.

    Raises:
        OptionSnapshotError: Eine Regel ist verletzt.
    """
    missing = [c for c in OPTION_CHAIN_COLUMNS if c not in chain.columns]
    if missing:
        raise OptionSnapshotError(f"Optionskette: Spalte(n) fehlen: {', '.join(missing)}.")
    frame = chain.loc[:, list(OPTION_CHAIN_COLUMNS)].copy()
    try:
        expiry = pd.to_datetime(frame["expiry"])
    except (TypeError, ValueError) as exc:
        raise OptionSnapshotError("Optionskette: ungültiger Verfallstag.") from exc
    if getattr(expiry.dt, "tz", None) is not None:
        expiry = expiry.dt.tz_localize(None)
    if expiry.isna().any():
        raise OptionSnapshotError("Optionskette: Verfallstag fehlt.")
    frame["expiry"] = expiry.dt.normalize().astype("datetime64[ns]")
    frame["type"] = frame["type"].astype("string").str.strip().str.upper().astype("object")
    bad_types = sorted({str(t) for t in frame["type"] if t not in OPTION_TYPES})
    if bad_types:
        raise OptionSnapshotError(
            f"Optionskette: unbekannte Art {', '.join(bad_types)} (erlaubt: CALL, PUT)."
        )
    frame["strike"] = _numeric(frame["strike"], "strike")
    if frame["strike"].isna().any() or (frame["strike"] <= 0).any():
        raise OptionSnapshotError("Optionskette: Basispreis fehlt oder ist nicht positiv.")
    for column in _PRICE_COLUMNS:
        frame[column] = _numeric(frame[column], column)
    for column in _COUNT_COLUMNS:
        values = _numeric(frame[column], column)
        frame[column] = values.round().astype("Int64")
    for column in (*_PRICE_COLUMNS, *_COUNT_COLUMNS):
        if (frame[column].dropna() < 0).any():
            raise OptionSnapshotError(f"Optionskette: negative Werte in Spalte {column}.")
    if frame.duplicated(subset=list(_KEY)).any():
        raise OptionSnapshotError(
            "Optionskette: Kontrakt doppelt (gleicher Verfall, Basispreis und Art)."
        )
    frame = frame.sort_values(["expiry", "type", "strike"], kind="mergesort")
    return frame.reset_index(drop=True)


def save_option_snapshot(
    store: MarketStore,
    symbol: str,
    snapshot_date: dt.date,
    chain: pd.DataFrame,
    *,
    source: str | None = None,
) -> int:
    """Speichert die Optionskette eines Symbols als Schnappschuss eines Tags.

    Ein vorhandener Schnappschuss desselben Symbols und Tags wird in derselben
    Transaktion ersetzt.

    Args:
        store: Offener, beschreibbarer Speicher.
        symbol: Symbol des Basiswerts.
        snapshot_date: Stichtag.
        chain: Optionskette (siehe :func:`validate_option_chain`); leer löscht nur.
        source: Name der Quelle.

    Returns:
        Anzahl gespeicherter Kontrakte.

    Raises:
        OptionSnapshotError: Symbol leer oder Kette ungültig.
    """
    sym = _normalize_symbol(symbol)
    frame = validate_option_chain(chain)
    frame.insert(0, "snapshot_date", snapshot_date)
    frame.insert(0, "symbol", sym)
    frame["source"] = source
    frame["expiry"] = [ts.date() for ts in frame["expiry"]]
    for column in _COUNT_COLUMNS:  # DuckDB liest pandas-Int64 mit NA korrekt als NULL
        frame[column] = frame[column].astype("Int64")
    with store.transaction() as conn:
        conn.execute(
            "DELETE FROM option_snapshots WHERE symbol = ? AND snapshot_date = ?",
            [sym, snapshot_date],
        )
        if frame.empty:
            return 0
        conn.register("_option_source", frame)
        try:
            conn.execute(
                f"INSERT INTO option_snapshots ({_QUOTED_COLUMNS}) "
                f"SELECT {_QUOTED_COLUMNS} FROM _option_source"
            )
        finally:
            conn.unregister("_option_source")
    return len(frame)


def snapshot_dates(store: MarketStore, symbol: str | None = None) -> list[dt.date]:
    """Alle Stichtage mit gespeicherten Schnappschüssen, aufsteigend.

    Args:
        store: Offener Speicher.
        symbol: Nur Stichtage dieses Symbols; ``None`` = aller Symbole.
    """
    if symbol is None:
        rows = store.connection.execute(
            "SELECT DISTINCT snapshot_date FROM option_snapshots ORDER BY 1"
        ).fetchall()
    else:
        rows = store.connection.execute(
            "SELECT DISTINCT snapshot_date FROM option_snapshots WHERE symbol = ? ORDER BY 1",
            [_normalize_symbol(symbol)],
        ).fetchall()
    return [row[0] for row in rows]


def load_option_snapshot(
    store: MarketStore,
    symbol: str,
    snapshot_date: dt.date | None = None,
    *,
    expiry: dt.date | None = None,
) -> pd.DataFrame:
    """Lädt den Schnappschuss eines Symbols.

    Args:
        store: Offener Speicher.
        symbol: Symbol des Basiswerts.
        snapshot_date: Stichtag; ``None`` = jüngster gespeicherter Stichtag.
        expiry: Nur diesen Verfallstag.

    Returns:
        Spalten :data:`SNAPSHOT_COLUMNS`, sortiert nach Verfall, Art und Basispreis;
        ``snapshot_date`` und ``expiry`` als ``datetime64[ns]``. Leer, wenn nichts
        gespeichert ist.
    """
    sym = _normalize_symbol(symbol)
    if snapshot_date is None:
        dates = snapshot_dates(store, sym)
        if not dates:
            return _empty_snapshot()
        snapshot_date = dates[-1]
    sql = f"SELECT {_QUOTED_COLUMNS} FROM option_snapshots WHERE symbol = ? AND snapshot_date = ?"
    params: list[Any] = [sym, snapshot_date]
    if expiry is not None:
        sql += " AND expiry = ?"
        params.append(expiry)
    sql += " ORDER BY expiry, type, strike"
    result: pd.DataFrame = store.connection.execute(sql, params).df()
    if result.empty:
        return _empty_snapshot()
    return _with_snapshot_dtypes(result)


def snapshot_options(
    symbols: Sequence[str],
    progress_cb: SnapshotProgress | None = None,
    *,
    store: MarketStore,
    provider: OptionChainProvider,
    today: dt.date | None = None,
    max_expiries: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> SnapshotResult:
    """Ruft die Optionsketten aller ``symbols`` ab und speichert sie als Schnappschuss.

    Args:
        symbols: Basiswerte, z. B. die Symbole der Watchlist; Duplikate werden entfernt.
        progress_cb: Erhält nach jedem Symbol ``(erledigt, gesamt, symbol)``.
        store: Offener, beschreibbarer Speicher.
        provider: Quelle der Optionsketten.
        today: Kalendertag des Abrufs; Standard ist heute. Stichtag siehe
            :func:`snapshot_date_for`.
        max_expiries: Nur die nächsten ``n`` Verfallstage je Symbol; ``None`` = alle.
        should_cancel: Wird vor jedem Symbol gefragt; ``True`` beendet den Lauf sauber.

    Returns:
        Ergebnis mit Erfolgen, Fehlern und der ``id`` in ``update_log``.

    Raises:
        OptionSnapshotError: Keine Symbole angegeben.
        KeyboardInterrupt: Bei Strg+C, nachdem der Lauf protokolliert wurde.
    """
    unique = list(dict.fromkeys(_normalize_symbol(s) for s in symbols if s.strip()))
    if not unique:
        raise OptionSnapshotError("Keine Symbole für den Options-Schnappschuss angegeben.")
    day = snapshot_date_for(today if today is not None else dt.date.today())
    log_id = _start_log(store, provider.name + LOG_SUFFIX)
    result = SnapshotResult(log_id=log_id, snapshot_date=day, total=len(unique))
    try:
        for symbol in unique:
            if should_cancel is not None and should_cancel():
                result.cancelled = True
                break
            _snapshot_symbol(store, provider, symbol, day, max_expiries, result)
            if progress_cb is not None:
                progress_cb(result.done, result.total, symbol)
    except KeyboardInterrupt:
        result.cancelled = True
        raise
    finally:
        _finish_log(store, result)
        _log.info("Options-Schnappschuss %d: %s", log_id, result.summary())
    return result


# ---------------------------------------------------------------------- intern


def _snapshot_symbol(
    store: MarketStore,
    provider: OptionChainProvider,
    symbol: str,
    day: dt.date,
    max_expiries: int | None,
    result: SnapshotResult,
) -> None:
    """Speichert ein Symbol; Fehler landen in ``result.failed``."""
    try:
        chain = provider.fetch_option_chain(symbol, max_expiries=max_expiries)
        if chain.empty:
            result.ok.append(symbol)
            result.no_options.append(symbol)
            return
        written = save_option_snapshot(store, symbol, day, chain, source=provider.name)
    except (ProviderError, OptionSnapshotError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, ProviderError) else str(exc)
        result.failed[symbol] = reason
        _log.warning("Options-Schnappschuss: %s fehlgeschlagen: %s", symbol, reason)
        return
    result.ok.append(symbol)
    result.rows_written += written


def _numeric(values: pd.Series[Any], column: str) -> pd.Series[Any]:
    converted = pd.to_numeric(values, errors="coerce").astype("float64")
    if np.isinf(converted.to_numpy()).any():
        raise OptionSnapshotError(f"Optionskette: unendliche Werte in Spalte {column}.")
    return converted


def _empty_snapshot() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype="object") for c in SNAPSHOT_COLUMNS})
    return _with_snapshot_dtypes(frame)


def _with_snapshot_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, list(SNAPSHOT_COLUMNS)].copy()
    for column in ("snapshot_date", "expiry"):
        out[column] = pd.to_datetime(out[column]).astype("datetime64[ns]")
    for column in ("strike", *_PRICE_COLUMNS):
        out[column] = out[column].astype("float64")
    for column in _COUNT_COLUMNS:
        out[column] = out[column].astype("Int64")
    for column in ("symbol", "type", "source"):
        out[column] = out[column].astype("object")
    return out.reset_index(drop=True)


def _start_log(store: MarketStore, provider_name: str) -> int:
    rows = store.connection.execute(
        "INSERT INTO update_log (started_at, provider, symbols_ok, symbols_failed, message) "
        "VALUES (?, ?, 0, 0, ?) RETURNING id",
        [_now(), provider_name, "läuft"],
    ).fetchall()
    return int(rows[0][0])


def _finish_log(store: MarketStore, result: SnapshotResult) -> None:
    store.connection.execute(
        "UPDATE update_log SET finished_at = ?, symbols_ok = ?, symbols_failed = ?, message = ? "
        "WHERE id = ?",
        [_now(), len(result.ok), len(result.failed), result.summary(), result.log_id],
    )


def _now() -> dt.datetime:
    return dt.datetime.now().replace(microsecond=0)


def _normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol.strip():
        raise OptionSnapshotError(f"Ungültiges Symbol: {symbol!r}.")
    return symbol.strip().upper()
