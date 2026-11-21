"""Scanner: wendet eine Strategie auf ein Universum an (docs/SPEC.md, „Scanner“).

Je Wert wird die Strategie wie in einem Einzelwert-Backtest Kerze für Kerze über die
abgeschlossenen Kerzen bis ``as_of`` geführt; Orders laufen dabei über den simulierten
Broker (:class:`~trading_helfer.backtest.broker_sim.SimBroker`), damit Position, Cash und
Equity im Kontext genau wie im Backtest aussehen. Ein Signal ist jede Order, die die
Strategie auf der **letzten abgeschlossenen Kerze** abgibt:

- ``BUY``: Order eröffnet oder vergrößert eine Long-Position.
- ``SELL``: Order eröffnet oder vergrößert eine Short-Position.
- ``EXIT``: Order schließt eine offene Position (ganz oder teilweise).

Gibt eine Strategie auf derselben Kerze mehrere Orders ab, zählt die erste eröffnende
Order, sonst die erste schließende. Ein Signal auf Kerze ``t`` entspricht damit genau
einem Einstieg des Backtests zum Open von ``t+1``.

Regeln:

- Simulationsfenster: Standard sind die letzten :data:`DEFAULT_HISTORY_YEARS` Jahre vor
  dem Stichtag (Einschwingphase für Indikatoren und Positionen); ``start`` oder
  ``history_years=None`` ändern das. Ein Scan stimmt mit einem Backtest überein, der am
  selben Tag beginnt.
- Nur abgeschlossene Kerzen: Die letzte, noch unvollständige Kerze (``is_partial``)
  wird verworfen. Mit ``as_of`` werden nur Tageskurse bis einschließlich ``as_of``
  geladen; die Kerze, in die ``as_of`` fällt, zählt also nur, wenn sie an diesem Tag
  abgeschlossen ist.
- Universum point in time: Bei einem Index-Code dürfen Einstiege nur an Tagen entstehen,
  an denen der Wert Indexmitglied war. Gescannt werden die Mitglieder am Stichtag.
- ``strength``: Liefert :meth:`Strategy.indicators` eine Spalte ``strength``, ist das
  ihr Wert auf der Signalkerze, sonst leer.
- Ergebnis-Spalten: :data:`RESULT_COLUMNS` und danach alle Indikatoren der Strategie auf
  der Signalkerze.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from trading_helfer.backtest.broker_sim import Bar, SimBroker
from trading_helfer.backtest.costs import CostModel
from trading_helfer.core.config import Settings
from trading_helfer.core.types import SignalType, Timeframe
from trading_helfer.data.resample import PARTIAL_COLUMN, resample
from trading_helfer.data.storage import MarketStore
from trading_helfer.data.universe import INDEX_CODES, members_on, membership_intervals
from trading_helfer.fundamentals.metrics import METRIC_NAMES, compute_metrics
from trading_helfer.strategies.base import BarContext, Order, OrderSide, Params, Strategy

__all__ = [
    "DEFAULT_HISTORY_YEARS",
    "RESULT_COLUMNS",
    "STRENGTH_COLUMN",
    "ScanError",
    "ScanFilters",
    "export_csv",
    "load_signals",
    "scan",
    "signal_of",
]

logger = logging.getLogger(__name__)

RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "date",
    "signal",
    "strength",
    "close",
    "sector",
    "market_cap",
    "note",
)
"""Feste Spalten des Scan-Ergebnisses; danach folgen die Indikatoren der Strategie."""

DEFAULT_HISTORY_YEARS: Final = 10
"""Standard-Länge der Simulation vor dem Stichtag in Jahren (Einschwingphase)."""

STRENGTH_COLUMN: Final = "strength"
"""Indikatorspalte, aus der die Signalstärke gelesen wird."""

_ATR_COLUMN: Final = "atr"
_BAR_FIELDS: Final = ("open", "high", "low", "close")

ProgressCallback = Callable[[int, int], None]
"""Fortschritt: ``(erledigt, gesamt)``."""


class ScanError(ValueError):
    """Ungültige Scan-Eingabe; Text auf Deutsch."""


@dataclass(frozen=True)
class ScanFilters:
    """Filter für das Scan-Ergebnis.

    Attributes:
        sectors: Nur Werte dieser Sektoren (Groß-/Kleinschreibung egal); ``None`` = alle.
        min_market_cap: Mindest-Marktkapitalisierung in USD.
        max_market_cap: Höchste Marktkapitalisierung in USD.
        metrics: Kennzahl → ``(min, max)``; ``None`` = keine Grenze. Namen wie in
            :data:`~trading_helfer.fundamentals.metrics.METRIC_NAMES`.
        signals: Nur diese Signalarten; ``None`` = alle.

    Ein Wert ohne Angabe (fehlender Sektor, nicht berechenbare Kennzahl) fällt heraus,
    sobald für dieses Merkmal ein Filter gesetzt ist.
    """

    sectors: frozenset[str] | None = None
    min_market_cap: float | None = None
    max_market_cap: float | None = None
    metrics: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    signals: frozenset[SignalType] | None = None

    def __post_init__(self) -> None:
        unknown = sorted(set(self.metrics) - set(METRIC_NAMES))
        if unknown:
            raise ScanError(
                f"Unbekannte Kennzahl(en) im Filter: {', '.join(unknown)}. "
                f"Bekannt: {', '.join(METRIC_NAMES)}."
            )
        bounds = [(self.min_market_cap, self.max_market_cap), *self.metrics.values()]
        for low, high in bounds:
            if low is not None and high is not None and low > high:
                raise ScanError(f"Filter: Untergrenze {low} liegt über Obergrenze {high}.")

    @property
    def needs_metrics(self) -> bool:
        """``True``, wenn Kennzahlen (inkl. Marktkapitalisierung) gefiltert werden."""
        return (
            self.min_market_cap is not None
            or self.max_market_cap is not None
            or any(lo is not None or hi is not None for lo, hi in self.metrics.values())
        )

    def match_sector(self, sector: str | None) -> bool:
        """Prüft den Sektorfilter."""
        if self.sectors is None:
            return True
        if not sector:
            return False
        return sector.casefold() in {s.casefold() for s in self.sectors}

    def match_metrics(self, values: Mapping[str, float | None]) -> bool:
        """Prüft Marktkapitalisierung und Kennzahlen."""
        bounds = dict(self.metrics)
        if self.min_market_cap is not None or self.max_market_cap is not None:
            bounds["market_cap"] = (self.min_market_cap, self.max_market_cap)
        for name, (low, high) in bounds.items():
            if low is None and high is None:
                continue
            value = values.get(name)
            if value is None or not math.isfinite(value):
                return False
            if (low is not None and value < low) or (high is not None and value > high):
                return False
        return True


def signal_of(order: Order) -> SignalType:
    """Signalart einer Order: BUY (Long-Einstieg), SELL (Short-Einstieg) oder EXIT."""
    if not order.side.is_entry:
        return SignalType.EXIT
    return SignalType.BUY if order.side is OrderSide.BUY else SignalType.SELL


def scan(
    strategy: Strategy | type[Strategy],
    params: Params | Mapping[str, Any] | None,
    universe: str | Iterable[str],
    timeframe: Timeframe | str,
    as_of: dt.date | str | None = None,
    filters: ScanFilters | None = None,
    *,
    store: MarketStore,
    settings: Settings | None = None,
    costs: CostModel | None = None,
    start: dt.date | str | None = None,
    history_years: int | None = DEFAULT_HISTORY_YEARS,
    save: bool = True,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Listet alle Werte eines Universums mit Signal auf der letzten abgeschlossenen Kerze.

    Args:
        strategy: Strategie (Instanz oder Klasse).
        params: Parameterwerte; fehlende Werte = Standard, ``None`` = alle Standard.
        universe: Index-Code (``"SP500"``, ``"NDX100"``) oder Liste von Symbolen.
        timeframe: Zeitrahmen (``Timeframe`` oder Wert wie ``"1W"``).
        as_of: Stichtag; ``None`` = alle gespeicherten Kurse.
        filters: Filter nach Sektor, Marktkapitalisierung, Kennzahlen und Signalart.
        store: Offener Kursspeicher.
        settings: Startkapital, Bruchstücke und Kosten des simulierten Brokers;
            ``None`` = Standardeinstellungen.
        costs: Kostenmodell; ``None`` = aus ``settings``.
        start: Erster Tageskurs der Simulation (wie Beginn eines Backtests); hat
            Vorrang vor ``history_years``.
        history_years: Ohne ``start`` beginnt die Simulation so viele Jahre vor dem
            Stichtag (``as_of`` bzw. letzter gespeicherter Tag); ``None`` = ab Beginn der
            Daten. Ein Backtest mit demselben Beginn liefert dieselben Einstiege.
        save: Ergebnis in Tabelle ``signals`` speichern.
        progress: Wird nach jedem Wert mit ``(erledigt, gesamt)`` aufgerufen.

    Returns:
        DataFrame mit :data:`RESULT_COLUMNS` plus Indikatorspalten, sortiert nach Signal
        und Symbol. ``attrs["scan_id"]`` ist die ID des Scans, ``attrs["skipped"]``
        nennt je übersprungenem Wert den Grund (keine Kurse, Fehler der Strategie).

    Raises:
        ScanError: Zeitrahmen passt nicht zur Strategie, unbekannter Index oder
            ungültige Parameter.
    """
    instance = strategy() if isinstance(strategy, type) else strategy
    tf = _timeframe(timeframe)
    if tf not in instance.timeframes:
        supported = ", ".join(t.value for t in instance.timeframes)
        raise ScanError(
            f"Strategie {instance.id!r} unterstützt den Zeitrahmen {tf.value} nicht "
            f"(nur {supported})."
        )
    try:
        p = params if isinstance(params, Params) else instance.make_params(params)
    except ValueError as exc:
        raise ScanError(str(exc)) from exc
    filters = filters if filters is not None else ScanFilters()
    settings = settings if settings is not None else Settings()
    costs = costs if costs is not None else CostModel.from_settings(settings)
    end = _to_date(as_of) if as_of is not None else None
    index_code, symbols = _resolve_universe(universe, end, store)
    if start is not None:
        first: dt.date | None = _to_date(start)
    elif history_years is not None:
        reference = end if end is not None else _last_stored_day(store)
        first = _years_before(reference, history_years)
    else:
        first = None

    sectors = _sectors(store, symbols)
    symbols = [s for s in symbols if filters.match_sector(sectors.get(s))]
    membership = _membership(index_code) if index_code is not None else None

    rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    total = len(symbols)
    for done, symbol in enumerate(symbols, start=1):
        try:
            row = _scan_symbol(
                instance,
                p,
                symbol,
                tf,
                store=store,
                settings=settings,
                costs=costs,
                start=first,
                end=end,
                member_intervals=None if membership is None else membership.get(symbol, []),
            )
        except _SkipError as exc:
            skipped[symbol] = str(exc)
            row = None
        except Exception as exc:  # Fehler einer Strategie darf den Scan nicht abbrechen
            logger.warning("Scan %s: Fehler der Strategie: %s", symbol, exc)
            skipped[symbol] = f"Fehler der Strategie: {type(exc).__name__}: {exc}"
            row = None
        if row is not None and (filters.signals is None or row["signal"] in filters.signals):
            rows.append(row)
        if progress is not None:
            progress(done, total)

    result = _finish(rows, sectors, filters, store)
    scan_id = uuid.uuid4().hex
    result.attrs["scan_id"] = scan_id
    result.attrs["skipped"] = skipped
    if save:
        _save(store, scan_id, instance.id, result)
    return result


def export_csv(result: pd.DataFrame, path: str | Path) -> Path:
    """Speichert ein Scan-Ergebnis als CSV (UTF-8, Komma, Datum ``YYYY-MM-DD``).

    Returns:
        Pfad der geschriebenen Datei.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False, date_format="%Y-%m-%d")
    return target


def load_signals(store: MarketStore, scan_id: str | None = None) -> pd.DataFrame:
    """Lädt gespeicherte Signale eines Scans (``None`` = alle Scans).

    Returns:
        DataFrame mit den Spalten der Tabelle ``signals``, sortiert nach Scan, Symbol, Datum.
    """
    sql = (
        "SELECT scan_id, symbol, CAST(date AS TIMESTAMP_NS) AS date, strategy_id, signal, "
        "strength, note FROM signals"
    )
    args: list[Any] = []
    if scan_id is not None:
        sql += " WHERE scan_id = ?"
        args.append(scan_id)
    sql += " ORDER BY scan_id, symbol, date"
    frame: pd.DataFrame = store.connection.execute(sql, args).df()
    return frame


# ---------------------------------------------------------------------- Interna


class _SkipError(Exception):
    """Wert wird übersprungen (Grund im Text)."""


def _timeframe(value: Timeframe | str) -> Timeframe:
    if isinstance(value, Timeframe):
        return value
    text = str(value).strip()
    for tf in Timeframe:
        if text.upper() in (tf.name, tf.value.upper()):
            return tf
    known = ", ".join(f"{tf.name}/{tf.value}" for tf in Timeframe)
    raise ScanError(f"Unbekannter Zeitrahmen {value!r}. Bekannt: {known}.")


def _to_date(value: dt.date | str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ScanError(f"Ungültiges Datum {value!r} (erwartet JJJJ-MM-TT).") from exc


def _resolve_universe(
    universe: str | Iterable[str], end: dt.date | None, store: MarketStore
) -> tuple[str | None, list[str]]:
    if isinstance(universe, str):
        code = universe.strip().upper()
        if code not in INDEX_CODES:
            raise ScanError(
                f"Unbekannter Index {universe!r}. Bekannt: {', '.join(INDEX_CODES)}; "
                "alternativ eine Liste von Symbolen."
            )
        day = end if end is not None else _last_stored_day(store)
        return code, sorted(members_on(code, day))
    symbols = sorted({s.strip().upper() for s in universe if s and s.strip()})
    return None, symbols


def _last_stored_day(store: MarketStore) -> dt.date:
    row = store.connection.execute("SELECT max(date) FROM bars_daily").fetchone()
    value = row[0] if row else None
    return value if isinstance(value, dt.date) else dt.date.today()


def _years_before(day: dt.date, years: int) -> dt.date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:  # 29. Februar
        return day.replace(year=day.year - years, day=28)


def _membership(code: str) -> dict[str, list[tuple[dt.date | None, dt.date | None]]]:
    intervals = membership_intervals(code)
    result: dict[str, list[tuple[dt.date | None, dt.date | None]]] = {}
    for row in intervals.itertuples(index=False):
        start = row.start_date if isinstance(row.start_date, dt.date) else None
        stop = row.end_date if isinstance(row.end_date, dt.date) else None
        result.setdefault(str(row.symbol), []).append((start, stop))
    return result


def _member_mask(
    dates: pd.Series[Any], intervals: list[tuple[dt.date | None, dt.date | None]] | None
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    if intervals is None:
        return np.ones(len(dates), dtype=bool)
    values = dates.to_numpy(dtype="datetime64[ns]")
    mask = np.zeros(len(values), dtype=bool)
    for start, stop in intervals:
        inside = np.ones(len(values), dtype=bool)
        if start is not None:
            inside &= values >= np.datetime64(start, "ns")
        if stop is not None:
            inside &= values < np.datetime64(stop, "ns")
        mask |= inside
    return mask


def _sectors(store: MarketStore, symbols: list[str]) -> dict[str, str | None]:
    if not symbols:
        return {}
    rows = store.connection.execute(
        "SELECT symbol, sector FROM instruments WHERE list_contains(?, symbol)", [symbols]
    ).fetchall()
    return {str(sym): (str(sec) if sec is not None else None) for sym, sec in rows}


def _closed_bars(
    store: MarketStore,
    symbol: str,
    timeframe: Timeframe,
    start: dt.date | None,
    end: dt.date | None,
) -> pd.DataFrame:
    daily = store.load_bars(symbol, start=start, end=end)
    if daily.empty:
        raise _SkipError("keine Kurse")
    daily = daily.drop(columns=["adj_factor"], errors="ignore")
    bars = resample(daily, timeframe)
    closed = bars.loc[~bars[PARTIAL_COLUMN].to_numpy(dtype=bool)]
    if closed.empty:
        raise _SkipError("keine abgeschlossene Kerze")
    return closed.drop(columns=[PARTIAL_COLUMN]).reset_index(drop=True)


def _scan_symbol(
    strategy: Strategy,
    p: Params,
    symbol: str,
    timeframe: Timeframe,
    *,
    store: MarketStore,
    settings: Settings,
    costs: CostModel,
    start: dt.date | None,
    end: dt.date | None,
    member_intervals: list[tuple[dt.date | None, dt.date | None]] | None,
) -> dict[str, Any] | None:
    bars = _closed_bars(store, symbol, timeframe, start, end)
    series = strategy.prepare(bars, p)
    member: list[bool] = _member_mask(bars["date"], member_intervals).tolist()
    dates: list[dt.date] = [ts.date() for ts in pd.DatetimeIndex(bars["date"])]
    o, h, low, c = (bars[f].to_numpy(dtype=float).tolist() for f in _BAR_FIELDS)
    atr = series.ind_at(len(bars) - 1)
    atr_values = atr.column(_ATR_COLUMN) if _ATR_COLUMN in atr else None

    broker = SimBroker(settings.default_capital, costs, allow_fractional=settings.allow_fractional)
    last = len(bars) - 1
    orders: list[Order] = []
    for t in range(len(bars)):
        bar_atr = _finite(atr_values[t]) if atr_values is not None else None
        position = broker.position(symbol)
        # Ohne Position und offene Orders hat die Kerze für den Broker keine Wirkung.
        if t > 0 and (position is not None or broker.open_orders(symbol)):
            broker.process_bar(symbol, Bar(dates[t], o[t], h[t], low[t], c[t]), atr=bar_atr)
            position = broker.position(symbol)
        cash = broker.cash
        ctx = BarContext(
            symbol=symbol,
            date=dates[t],
            timeframe=timeframe,
            bars=series.bars_at(t),
            ind=series.ind_at(t),
            position=position,
            cash=cash,
            equity=cash if position is None else broker.equity(),
            is_member=member[t],
        )
        orders = [order for order in strategy.on_bar(ctx) if member[t] or not order.side.is_entry]
        if t == last:
            break
        for order in orders:
            broker.submit(order, date=dates[t], close=c[t], atr=bar_atr)

    chosen = _pick(orders)
    if chosen is None:
        return None
    indicators = series.ind_at(last).row()
    strength = indicators.get(STRENGTH_COLUMN)
    row: dict[str, Any] = {
        "symbol": symbol,
        "date": pd.Timestamp(dates[last]),
        "signal": signal_of(chosen),
        "strength": _finite(strength),
        "close": float(c[last]),
        "note": _note(chosen),
    }
    row.update({k: v for k, v in indicators.items() if k != STRENGTH_COLUMN})
    return row


def _pick(orders: list[Order]) -> Order | None:
    for order in orders:
        if order.side.is_entry:
            return order
    return orders[0] if orders else None


def _note(order: Order) -> str:
    parts = [order.side.value, order.type.value]
    if order.stop_price is not None:
        parts.append(f"Stop {order.stop_price:g}")
    if order.limit_price is not None:
        parts.append(f"Limit {order.limit_price:g}")
    if order.tag:
        parts.append(order.tag)
    return " ".join(parts)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finish(
    rows: list[dict[str, Any]],
    sectors: Mapping[str, str | None],
    filters: ScanFilters,
    store: MarketStore,
) -> pd.DataFrame:
    kept: list[dict[str, Any]] = []
    for row in rows:
        values = compute_metrics(row["symbol"], row["date"].date(), store=store)
        if not filters.match_metrics(values):
            continue
        row["sector"] = sectors.get(row["symbol"])
        row["market_cap"] = values.get("market_cap")
        kept.append(row)
    extra: list[str] = []
    for row in kept:
        extra.extend(k for k in row if k not in RESULT_COLUMNS and k not in extra)
    frame = pd.DataFrame(kept, columns=[*RESULT_COLUMNS, *extra])
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    for column in ("strength", "close", "market_cap"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["signal"] = frame["signal"].map(lambda s: SignalType(s).value).astype("object")
    order = {SignalType.BUY.value: 0, SignalType.SELL.value: 1, SignalType.EXIT.value: 2}
    frame = frame.sort_values(
        ["signal", "symbol"], key=lambda s: s.map(order) if s.name == "signal" else s
    )
    return frame.reset_index(drop=True)


def _save(store: MarketStore, scan_id: str, strategy_id: str, result: pd.DataFrame) -> None:
    if result.empty:
        return
    records = [
        (
            scan_id,
            str(symbol),
            pd.Timestamp(day).date(),
            strategy_id,
            str(signal),
            _finite(strength),
            str(note),
        )
        for symbol, day, signal, strength, note in zip(
            result["symbol"],
            result["date"],
            result["signal"],
            result["strength"],
            result["note"],
            strict=True,
        )
    ]
    with store.transaction() as conn:
        conn.executemany(
            "INSERT INTO signals (scan_id, symbol, date, strategy_id, signal, strength, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            records,
        )
