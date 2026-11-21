"""Inkrementelles Datenupdate für Indexmitglieder, eigene Symbole und Benchmarks.

:func:`update_universe` bringt die Tageskurse aller Symbole auf den neuesten Stand
(siehe ``docs/SPEC.md``, Abschnitt „Module im Detail › Daten-Provider“):

- Ohne gespeicherte Kurse wird die ganze Historie ab ``start`` geladen, dazu Splits und
  Dividenden und (falls die Quelle sie liefert) die Stammdaten.
- Mit gespeicherten Kursen wird ab ``last_date − 5 Tage`` geladen; überlappende Tage
  werden überschrieben. Das ist genau eine Anfrage je Symbol.
- Weicht ``adj_factor`` an einem überlappenden Tag vom gespeicherten Wert ab (neuer
  Split oder neue Dividende), wird die ganze Historie des Symbols neu geladen.
- Die Kursquellen werden in der Reihenfolge aus den Einstellungen versucht; schlägt
  eine fehl, kommt die nächste dran. Ein leeres Ergebnis ist kein Fehler.
- Der Fehler eines Symbols blockiert die anderen nicht.
- Jedes Symbol wird in einer eigenen Transaktion geschrieben. Ein Abbruch (Strg+C)
  hinterlässt deshalb eine konsistente Datenbank; der Lauf steht mit Meldung
  „abgebrochen“ in ``update_log``.

Die Benchmarks :data:`BENCHMARKS` (SPY, QQQ) werden immer mitgeladen.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import pandas as pd

from trading_helfer.core.bars import ADJ_CLOSE_COLUMN, BarsValidationError, validate_bars
from trading_helfer.data.providers.base import InstrumentProvider, PriceProvider, ProviderError
from trading_helfer.data.storage import MarketStore, StorageError
from trading_helfer.data.universe import (
    INDEX_CODES,
    MEMBERSHIP_COLUMNS,
    UniverseError,
    all_members,
    membership_intervals,
)

if TYPE_CHECKING:
    from trading_helfer.core.config import Settings

__all__ = [
    "BENCHMARKS",
    "INCREMENTAL_OVERLAP",
    "ProgressCallback",
    "UpdateError",
    "UpdateResult",
    "build_price_providers",
    "collect_symbols",
    "update_universe",
]

BENCHMARKS: Final[tuple[str, ...]] = ("SPY", "QQQ")
"""Benchmarks, die bei jedem Update mitgeladen werden."""

INCREMENTAL_OVERLAP: Final = dt.timedelta(days=5)
"""Überlappung beim inkrementellen Laden: ab ``last_date − 5 Tage``."""

ADJ_FACTOR_TOLERANCE: Final = 1e-5
"""Relative Abweichung von ``adj_factor``, ab der die Historie neu geladen wird."""

DEFAULT_START: Final = dt.date(1990, 1, 1)
"""Standard-Startdatum der Historie (wie ``Settings.start_date``)."""

MAX_LISTED_FAILURES: Final = 20
"""Höchstzahl fehlgeschlagener Symbole, die die Meldung in ``update_log`` aufzählt."""

ProgressCallback = Callable[[int, int, str], None]
"""Fortschritt als ``(erledigt, gesamt, symbol)``; wird nach jedem Symbol aufgerufen."""

_log = logging.getLogger(__name__)


class UpdateError(ValueError):
    """Das Update kann nicht starten (z. B. unbekannte Kursquelle oder unbekannter Index)."""


@dataclass
class UpdateResult:
    """Ergebnis eines Laufs von :func:`update_universe`."""

    log_id: int
    """``id`` des Eintrags in ``update_log``."""
    total: int
    """Anzahl der Symbole dieses Laufs."""
    ok: list[str] = field(default_factory=list)
    """Erfolgreich aktualisierte Symbole (auch solche ohne neue Kurse)."""
    failed: dict[str, str] = field(default_factory=dict)
    """Fehlgeschlagene Symbole mit Grund."""
    no_data: list[str] = field(default_factory=list)
    """Symbole, für die keine Quelle Kurse hatte (Teilmenge von :attr:`ok`)."""
    reloaded: list[str] = field(default_factory=list)
    """Symbole, deren Historie wegen geändertem ``adj_factor`` neu geladen wurde."""
    rows_written: int = 0
    """Anzahl geschriebener Kurszeilen."""
    cancelled: bool = False
    """``True``, wenn der Lauf abgebrochen wurde."""

    @property
    def done(self) -> int:
        """Anzahl bearbeiteter Symbole (erfolgreich oder fehlgeschlagen)."""
        return len(self.ok) + len(self.failed)

    def summary(self) -> str:
        """Kurze deutsche Zusammenfassung, wie sie in ``update_log.message`` steht."""
        parts = [f"{len(self.ok)} von {self.total} Symbolen aktualisiert"]
        if self.failed:
            names = sorted(self.failed)
            listed = ", ".join(names[:MAX_LISTED_FAILURES])
            if len(names) > MAX_LISTED_FAILURES:
                listed += f" und {len(names) - MAX_LISTED_FAILURES} weitere"
            parts.append(f"{len(self.failed)} fehlgeschlagen ({listed})")
        if self.no_data:
            parts.append(f"{len(self.no_data)} ohne Kurse")
        if self.reloaded:
            parts.append(f"{len(self.reloaded)} neu geladen (Split/Dividende)")
        text = ", ".join(parts) + "."
        if self.cancelled:
            text = "Abgebrochen: " + text
        return text


@dataclass(frozen=True)
class _Fetched:
    """Kurse eines Symbols von einer Quelle."""

    provider: PriceProvider
    bars: pd.DataFrame


def build_price_providers(settings: Settings, store: MarketStore) -> list[PriceProvider]:
    """Legt die Kursquellen in der Reihenfolge von ``settings.providers`` an.

    Args:
        settings: Einstellungen (Reihenfolge der Quellen, Tiingo-API-Key).
        store: Offener Speicher; Tiingo zählt seine Anfragen in dieser Datenbank.

    Returns:
        Liste der Provider.

    Raises:
        UpdateError: Eine Quelle ist unbekannt.
    """
    from trading_helfer.data.providers.tiingo_provider import TiingoProvider
    from trading_helfer.data.providers.usage import DuckDBRequestCounter
    from trading_helfer.data.providers.yfinance_provider import YFinanceProvider

    providers: list[PriceProvider] = []
    for name in settings.providers:
        if name == YFinanceProvider.name:
            providers.append(YFinanceProvider())
        elif name == TiingoProvider.name:
            counter = DuckDBRequestCounter(store.connection)
            providers.append(TiingoProvider.from_settings(settings, counter=counter))
        else:
            raise UpdateError(
                f"Unbekannte Kursquelle {name!r} in den Einstellungen "
                f"(möglich: {YFinanceProvider.name}, {TiingoProvider.name})."
            )
    return providers


def collect_symbols(
    index_codes: Iterable[str],
    symbols: Iterable[str] = (),
    *,
    start: dt.date = DEFAULT_START,
    end: dt.date | None = None,
) -> list[str]:
    """Alle Symbole eines Updates: Indexmitglieder, eigene Symbole und Benchmarks.

    Indexmitglieder sind alle Werte, die zwischen ``start`` und ``end`` Mitglied waren,
    also auch frühere Mitglieder.

    Args:
        index_codes: Indizes, z. B. ``["SP500", "NDX100"]``.
        symbols: Zusätzliche Symbole.
        start: Beginn des Zeitraums für frühere Mitglieder.
        end: Ende des Zeitraums; Standard ist heute.

    Returns:
        Symbole ohne Duplikate: Benchmarks zuerst, danach alphabetisch.

    Raises:
        UpdateError: Ein Index ist unbekannt.
    """
    end = end if end is not None else dt.date.today()
    found: set[str] = set()
    for code in index_codes:
        try:
            found |= all_members(code, start, end)
        except UniverseError as exc:
            raise UpdateError(str(exc)) from exc
    found |= {_normalize(s) for s in symbols if s.strip()}
    rest = sorted(found - set(BENCHMARKS))
    return [*BENCHMARKS, *rest]


def update_universe(
    index_codes: Sequence[str],
    progress_cb: ProgressCallback | None = None,
    *,
    store: MarketStore,
    providers: Sequence[PriceProvider],
    symbols: Sequence[str] = (),
    start: dt.date = DEFAULT_START,
    today: dt.date | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> UpdateResult:
    """Bringt die Kurse aller Mitglieder von ``index_codes`` und von ``symbols`` auf Stand.

    Die Benchmarks :data:`BENCHMARKS` werden immer mitgeladen. Für jeden Index werden
    außerdem die Mitgliedschaften in ``index_membership`` geschrieben.

    Args:
        index_codes: Indizes, z. B. ``["SP500", "NDX100"]``; leer = nur ``symbols``.
        progress_cb: Erhält nach jedem Symbol ``(erledigt, gesamt, symbol)``.
        store: Offener, beschreibbarer Speicher.
        providers: Kursquellen in Reihenfolge; bei einem Fehler kommt die nächste dran.
        symbols: Zusätzliche Symbole.
        start: Erster Tag der Historie beim ersten Laden.
        today: Letzter zu ladender Tag; Standard ist heute.
        should_cancel: Wird vor jedem Symbol gefragt; ``True`` beendet den Lauf sauber.

    Returns:
        Das Ergebnis mit Erfolgen, Fehlern und der ``id`` in ``update_log``.

    Raises:
        UpdateError: Keine Kursquelle oder ein unbekannter Index.
        KeyboardInterrupt: Bei Strg+C, nachdem der Lauf protokolliert wurde.
    """
    if not providers:
        raise UpdateError("Keine Kursquelle angegeben.")
    end = today if today is not None else dt.date.today()
    all_symbols = collect_symbols(index_codes, symbols, start=start, end=end)
    provider_names = ",".join(p.name for p in providers)
    log_id = _start_log(store, provider_names)
    result = UpdateResult(log_id=log_id, total=len(all_symbols))
    try:
        for code in index_codes:
            _write_membership(store, code, start)
        for symbol in all_symbols:
            if should_cancel is not None and should_cancel():
                result.cancelled = True
                break
            _update_symbol(store, providers, symbol, start, end, result)
            if progress_cb is not None:
                progress_cb(result.done, result.total, symbol)
    except KeyboardInterrupt:
        result.cancelled = True
        raise
    finally:
        _finish_log(store, result)
        _log.info("Datenupdate %d: %s", log_id, result.summary())
    return result


# ---------------------------------------------------------------------- je Symbol


def _update_symbol(
    store: MarketStore,
    providers: Sequence[PriceProvider],
    symbol: str,
    start: dt.date,
    end: dt.date,
    result: UpdateResult,
) -> None:
    """Aktualisiert ein Symbol; Fehler landen in ``result.failed``."""
    try:
        last = store.last_date(symbol)
        if last is None:
            written = _full_load(store, providers, symbol, start, end, first_load=True)
        else:
            written = _incremental(store, providers, symbol, start, end, last, result)
    except (ProviderError, BarsValidationError, StorageError, ValueError) as exc:
        result.failed[symbol] = str(exc)
        _log.warning("Datenupdate: %s fehlgeschlagen: %s", symbol, exc)
        return
    result.ok.append(symbol)
    if written is None:
        result.no_data.append(symbol)
    else:
        result.rows_written += written


def _incremental(
    store: MarketStore,
    providers: Sequence[PriceProvider],
    symbol: str,
    start: dt.date,
    end: dt.date,
    last: dt.date,
    result: UpdateResult,
) -> int | None:
    """Lädt ab ``last − 5 Tage``; bei geändertem ``adj_factor`` die ganze Historie."""
    since = min(last - INCREMENTAL_OVERLAP, end)
    fetched = _fetch_daily(providers, symbol, since, end)
    if fetched is None or fetched.bars.empty:
        return 0
    if _adj_factor_changed(store, symbol, fetched.bars):
        _log.info("Datenupdate: %s – adj_factor geändert, lade ganze Historie neu.", symbol)
        written = _full_load(store, providers, symbol, start, end, first_load=False)
        result.reloaded.append(symbol)
        return written if written is not None else 0
    with store.transaction():
        count = store.upsert_bars(symbol, fetched.bars, source=fetched.provider.name)
        _write_instrument_dates(store, symbol, fetched.provider.name)
    return count


def _full_load(
    store: MarketStore,
    providers: Sequence[PriceProvider],
    symbol: str,
    start: dt.date,
    end: dt.date,
    *,
    first_load: bool,
) -> int | None:
    """Lädt die ganze Historie und ersetzt die gespeicherten Kurse des Symbols.

    Returns:
        Anzahl geschriebener Zeilen oder ``None``, wenn keine Quelle Kurse hatte.
    """
    fetched = _fetch_daily(providers, symbol, start, end)
    if fetched is None:
        return None
    provider = fetched.provider
    actions = provider.fetch_actions(symbol)
    info = _fetch_instrument(provider, symbol) if first_load else None
    with store.transaction() as conn:
        conn.execute("DELETE FROM bars_daily WHERE symbol = ?", [_normalize(symbol)])
        conn.execute("DELETE FROM corporate_actions WHERE symbol = ?", [_normalize(symbol)])
        count = store.upsert_bars(symbol, fetched.bars, source=provider.name)
        store.upsert_actions(symbol, actions)
        if info is not None:
            store.upsert_instruments(info)
        _write_instrument_dates(store, symbol, provider.name)
    return count


def _fetch_daily(
    providers: Sequence[PriceProvider], symbol: str, start: dt.date, end: dt.date
) -> _Fetched | None:
    """Fragt die Quellen der Reihe nach, bis eine ohne Fehler antwortet.

    Returns:
        Kurse der ersten fehlerfreien Quelle oder ``None``, wenn diese keine Kurse hat.

    Raises:
        ProviderError: Alle Quellen sind fehlgeschlagen (Gründe zusammengefasst).
    """
    reasons: list[str] = []
    for provider in providers:
        try:
            bars = validate_bars(provider.fetch_daily(symbol, start, end))
        except (ProviderError, BarsValidationError) as exc:
            _log.info("Datenupdate: %s bei %s fehlgeschlagen: %s", symbol, provider.name, exc)
            reasons.append(f"{provider.name}: {_reason(exc)}")
            continue
        if bars.empty:
            return None
        return _Fetched(provider, bars)
    raise ProviderError(",".join(p.name for p in providers), symbol, "; ".join(reasons))


def _fetch_instrument(provider: PriceProvider, symbol: str) -> pd.DataFrame | None:
    """Stammdaten von der Quelle, falls sie welche liefert; Fehler werden nur protokolliert."""
    if not isinstance(provider, InstrumentProvider):
        return None
    try:
        info = provider.fetch_instrument(symbol)
    except ProviderError as exc:
        _log.info("Datenupdate: Stammdaten für %s fehlen: %s", symbol, exc)
        return None
    if info is None:
        return None
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "name": info.name,
                "exchange": info.exchange,
                "sector": info.sector,
                "industry": info.industry,
                "currency": info.currency,
            }
        ]
    )


def _adj_factor_changed(store: MarketStore, symbol: str, bars: pd.DataFrame) -> bool:
    """``True``, wenn ``adj_factor`` an einem überlappenden Tag abweicht."""
    if ADJ_CLOSE_COLUMN not in bars.columns:
        return False
    first = bars["date"].min()
    stored = store.load_bars(symbol, start=first.date(), adjusted=False)
    if stored.empty:
        return False
    new = pd.DataFrame({"date": bars["date"], "new_factor": bars[ADJ_CLOSE_COLUMN] / bars["close"]})
    merged = stored[["date", "adj_factor"]].merge(new, on="date", how="inner")
    return any(
        not math.isclose(old, fresh, rel_tol=ADJ_FACTOR_TOLERANCE)
        for old, fresh in zip(merged["adj_factor"], merged["new_factor"], strict=True)
    )


def _write_instrument_dates(store: MarketStore, symbol: str, source: str) -> None:
    """Setzt ``first_date``, ``last_date`` und ``source`` aus den gespeicherten Kursen."""
    sym = _normalize(symbol)
    rows = store.connection.execute(
        "SELECT min(date), max(date) FROM bars_daily WHERE symbol = ?", [sym]
    ).fetchall()
    first, last = rows[0]
    store.upsert_instruments(
        pd.DataFrame([{"symbol": sym, "first_date": first, "last_date": last, "source": source}])
    )


# ---------------------------------------------------------------------- Index und Protokoll


def _write_membership(store: MarketStore, index_code: str, start: dt.date) -> None:
    """Ersetzt die Mitgliedschaften eines Index in ``index_membership``."""
    code = index_code.strip().upper()
    if code not in INDEX_CODES:
        raise UpdateError(f"Unbekannter Index {index_code!r} (möglich: {', '.join(INDEX_CODES)}).")
    frame = membership_intervals(code).loc[:, list(MEMBERSHIP_COLUMNS)].copy()
    frame["start_date"] = frame["start_date"].where(frame["start_date"].notna(), start)
    conn = store.connection
    with store.transaction():
        conn.execute("DELETE FROM index_membership WHERE index_code = ?", [code])
        conn.register("_membership_source", frame)
        try:
            conn.execute(
                "INSERT INTO index_membership (index_code, symbol, start_date, end_date) "
                "SELECT index_code, symbol, CAST(start_date AS DATE), CAST(end_date AS DATE) "
                "FROM _membership_source"
            )
        finally:
            conn.unregister("_membership_source")


def _start_log(store: MarketStore, provider_names: str) -> int:
    rows = store.connection.execute(
        "INSERT INTO update_log (started_at, provider, symbols_ok, symbols_failed, message) "
        "VALUES (?, ?, 0, 0, ?) RETURNING id",
        [_now(), provider_names, "läuft"],
    ).fetchall()
    return int(rows[0][0])


def _finish_log(store: MarketStore, result: UpdateResult) -> None:
    store.connection.execute(
        "UPDATE update_log SET finished_at = ?, symbols_ok = ?, symbols_failed = ?, message = ? "
        "WHERE id = ?",
        [_now(), len(result.ok), len(result.failed), result.summary(), result.log_id],
    )


def _now() -> dt.datetime:
    return dt.datetime.now().replace(microsecond=0)


def _normalize(symbol: str) -> str:
    return symbol.strip().upper()


def _reason(exc: Exception) -> str:
    return exc.reason if isinstance(exc, ProviderError) else str(exc)
