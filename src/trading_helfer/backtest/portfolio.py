"""Portfolio-Backtest: eine Strategie über alle Werte eines Universums mit gemeinsamem Kapital.

Ablauf (siehe Spezifikation, Backtest-Engine):

1. Kurse aller Werte laden; Zeitrahmen umrechnen und Indikatoren je Wert vorab berechnen,
   parallel mit :mod:`concurrent.futures` (je Wert eine eigene Kopie der Strategie, damit
   kein Zustand geteilt wird).
2. Gemeinsame Zeitachse = alle Kerzendaten der abgeschlossenen Kerzen aller Werte. Die
   unvollständige letzte Kerze eines Werts wird auf den nächsten Termin der Zeitachse
   gelegt (Delisting mitten in der Woche) oder verworfen, wenn es keinen gibt.
3. Je Termin ``t`` der Zeitachse:

   a. Der Broker führt die auf ``t-1`` abgegebenen Orders aus. Zuerst Werte mit offener
      Position (alphabetisch), damit Verkäufe Kapital freigeben, danach die übrigen in
      der Rangfolge ihrer Einstiegssignale.
   b. Endet die Kurshistorie eines Werts vor dem letzten Termin (Delisting), wird eine
      offene Position zum letzten Close geschlossen (ohne Slippage, mit Gebühr) und alle
      Orders des Werts werden gestrichen.
   c. Die Strategie wird je Wert mit Kerze ``t`` aufgerufen (alphabetisch);
      ``ctx.is_member`` zeigt die Indexmitgliedschaft am Tag ``t``. Werte, die am Tag
      ``t`` weder Mitglied sind noch eine Position oder offene Order haben, werden
      übersprungen, weil sie nichts handeln dürfen.
   d. Ausstiegs-Orders werden immer angenommen; eine aus dem Index gefallene Position
      darf regulär geschlossen werden. Eröffnende Orders für Werte ohne Position gelten
      nur für Mitglieder am Tag ``t``. Gibt es mehr Signale als freie Plätze
      (``max_positions``), entscheidet die Signalstärke (``Order.strength``, sonst die
      Indikatorspalte ``strength``, sonst 0; bei Gleichstand das Symbol alphabetisch).
      Alle Orders eines nicht berücksichtigten Werts aus dieser Kerze werden verworfen.

Freie Plätze = ``max_positions`` minus Werte mit Position oder offener Einstiegs-Order,
plus Werte, deren ganze Position auf ``t`` per Market-Order geschlossen wird. Zukäufe zu
einer bestehenden Position belegen keinen weiteren Platz.

Bilanz: Summe der Trade-PnL (nach allen Kosten) plus nicht realisierter Gewinn offener
Positionen ergibt Endkapital minus Startkapital.

Beispiel::

    result = run_portfolio_backtest(SmaCross, None, "SP500", Timeframe.W1,
                                    "2000-01-01", "2024-12-31", settings, max_positions=10)
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import logging
import math
import os
import pickle
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import date as Date  # noqa: N812
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from trading_helfer.backtest.broker_sim import Bar, SimBroker
from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import (
    BARS_PER_YEAR,
    EQUITY_COLUMNS,
    ORDER_COLUMNS,
    TRADE_COLUMNS,
    BacktestError,
    BacktestResult,
    DateInput,
    LookaheadError,
    StrategyError,
)
from trading_helfer.core.bars import DATE_COLUMN
from trading_helfer.core.types import OrderType, Timeframe
from trading_helfer.data.resample import PARTIAL_COLUMN, resample
from trading_helfer.data.universe import (
    SURVIVORSHIP_WARNING,
    IndexHistory,
    load_history,
    survivorship_risk,
)
from trading_helfer.indicators.volatility import atr as atr_indicator
from trading_helfer.indicators.volatility import historical_volatility
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Params,
    PreparedSeries,
    Strategy,
)

if TYPE_CHECKING:
    from trading_helfer.core.config import Settings

__all__ = [
    "DEFAULT_MAX_POSITIONS",
    "DELISTING_REASON",
    "STRENGTH_COLUMN",
    "PortfolioResult",
    "run_portfolio_backtest",
]

DEFAULT_MAX_POSITIONS: Final = 10
"""Standard für die maximale Anzahl gleichzeitiger Positionen."""

STRENGTH_COLUMN: Final = "strength"
"""Indikatorspalte, die als Signalstärke dient, wenn die Order keine ``strength`` hat."""

DELISTING_REASON: Final = "Delisting"
"""``exit_reason`` von Positionen, die wegen Ende der Kurshistorie geschlossen wurden."""

logger = logging.getLogger(__name__)

_PROCESS_MIN_SYMBOLS: Final = 64
"""Ab dieser Anzahl Werte lohnt sich die Vorberechnung in Prozessen."""

_ATR_PERIOD: Final = 14
_VOL_PERIOD: Final = 20

Universe = str | IndexHistory | Iterable[str]
"""Index-Code (``"SP500"``, ``"NDX100"``), geladene Index-Historie oder feste Symbolliste."""


@dataclass(frozen=True)
class PortfolioResult(BacktestResult):
    """Ergebnis eines Portfolio-Backtests.

    Wie :class:`~trading_helfer.backtest.engine.BacktestResult`; ``symbol`` enthält den
    Namen des Universums. Zusätzlich:

    Attributes:
        universe: Name des Universums (Index-Code oder ``"Liste"``).
        symbols: Alle Werte mit Kursen im Lauf, alphabetisch.
        max_positions: Maximale Anzahl gleichzeitiger Positionen.
        delisted: Werte, deren Kurshistorie vor dem Ende endete, mit Datum der letzten Kerze.
    """

    universe: str = ""
    symbols: tuple[str, ...] = ()
    max_positions: int = DEFAULT_MAX_POSITIONS
    delisted: dict[str, Date] = field(default_factory=dict)

    @property
    def open_pnl(self) -> float:
        """Nicht realisierter Gewinn: Endkapital − Startkapital − Summe der Trade-PnL."""
        realized = float(self.trades["pnl"].sum()) if len(self.trades) else 0.0
        return self.final_equity - self.initial_capital - realized

    def fingerprint(self) -> str:
        """SHA-256 über alle Ergebnisdaten; gleiche Eingaben ergeben denselben Wert."""
        h = hashlib.sha256(super().fingerprint().encode())
        extra = (
            self.universe,
            self.symbols,
            self.max_positions,
            sorted((k, v.isoformat()) for k, v in self.delisted.items()),
        )
        h.update(repr(extra).encode())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Öffentliche Funktion
# ---------------------------------------------------------------------------


def run_portfolio_backtest(
    strategy: Strategy | type[Strategy],
    params: Mapping[str, Any] | None,
    universe: Universe,
    timeframe: Timeframe,
    start: DateInput,
    end: DateInput,
    settings: Settings,
    *,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    bars: Mapping[str, pd.DataFrame] | None = None,
    costs: CostModel | None = None,
    workers: int | None = None,
) -> PortfolioResult:
    """Führt einen Portfolio-Backtest über alle Werte eines Universums aus.

    Args:
        strategy: Strategie (Instanz oder Klasse). Je Wert wird eine eigene Kopie verwendet.
        params: Parameterwerte; fehlende erhalten den Standardwert, ``None`` = alle Standard.
        universe: Index-Code (``"SP500"``, ``"NDX100"``), :class:`IndexHistory` oder eine
            feste Liste von Symbolen (dann ist jeder Wert immer Mitglied).
        timeframe: Zeitrahmen; muss in ``strategy.timeframes`` stehen.
        start: Erster Termin, an dem gehandelt werden darf (einschließlich).
        end: Letzter Tag der Daten (einschließlich).
        settings: Einstellungen (Startkapital, Bruchstücke, Kosten, Datenbank).
        max_positions: Maximale Anzahl gleichzeitig offener Positionen (≥ 1).
        bars: Bereinigte Tageskurse je Symbol statt Laden aus der Datenbank. Symbole des
            Universums ohne Eintrag werden übersprungen.
        costs: Kostenmodell statt der Kosten aus ``settings``.
        workers: Anzahl Prozesse bzw. Threads für die Vorberechnung; ``None`` = Anzahl
            CPU-Kerne, ``1`` = ohne Parallelisierung.

    Returns:
        Ergebnis mit Trades aller Werte, gemeinsamer Equity-Kurve, Orders und Protokoll.

    Raises:
        BacktestError: Ungültige Eingaben oder keine Kurse im Zeitraum.
        StrategyError: Die Strategie löst einen Fehler aus oder liefert ungültige Orders.
        LookaheadError: Die Strategie greift auf Daten nach ``t`` zu.
        trading_helfer.strategies.base.ParamError: Ungültige Parameter.
    """
    template = strategy() if isinstance(strategy, type) else strategy
    timeframe = Timeframe(timeframe)
    start_day, end_day = _to_day(start, "start"), _to_day(end, "end")
    if start_day > end_day:
        raise BacktestError(f"Beginn {start_day} liegt nach dem Ende {end_day}.")
    if max_positions < 1:
        raise BacktestError(f"max_positions muss mindestens 1 sein, erhalten: {max_positions}.")
    if timeframe not in template.timeframes:
        supported = ", ".join(t.value for t in template.timeframes) or "keine"
        raise BacktestError(
            f"Strategie {template.id!r} unterstützt den Zeitrahmen {timeframe.value} nicht "
            f"(unterstützt: {supported})."
        )
    p = template.make_params(params)
    log: list[str] = []
    name, candidates, is_member = _resolve_universe(universe, start_day, end_day, log)
    daily = _collect_bars(candidates, bars, settings, end_day, log)
    if not daily:
        raise BacktestError(f"Keine Kurse für das Universum {name} bis {end_day}.")

    prepared = _prepare_all(template, p, daily, timeframe, start_day, workers)
    data = [d for d in prepared if d is not None]
    skipped = sorted(set(daily) - {d.symbol for d in data})
    if skipped:
        log.append(
            f"{len(skipped)} Wert(e) ohne Kerzen im Zeitraum übersprungen: "
            + ", ".join(skipped[:20])
            + (" …" if len(skipped) > 20 else "")
        )
    if not data:
        raise BacktestError(
            f"Keine abgeschlossenen {timeframe.value}-Kerzen im Zeitraum {start_day} bis "
            f"{end_day} für das Universum {name}."
        )
    broker = SimBroker(
        settings.default_capital,
        costs if costs is not None else CostModel.from_settings(settings),
        allow_fractional=settings.allow_fractional,
    )
    run = _PortfolioRun(
        template, p, name, timeframe, start_day, end_day, broker, data, is_member, max_positions
    )
    run.log.extend(log)
    run.loop()
    return run.result()


# ---------------------------------------------------------------------------
# Vorbereitung
# ---------------------------------------------------------------------------


def _to_day(value: DateInput, name: str) -> Date:
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise BacktestError(f"{name}: kein gültiges Datum: {value!r}.") from exc
    if pd.isna(ts):
        raise BacktestError(f"{name}: kein gültiges Datum: {value!r}.")
    return ts.date()


def _resolve_universe(
    universe: Universe, start: Date, end: Date, log: list[str]
) -> tuple[str, list[str], Callable[[Date], frozenset[str] | None]]:
    """Name, Kandidaten und Mitgliedschaftsfunktion (``None`` = alle sind Mitglied)."""
    if isinstance(universe, str):
        history = load_history(universe)
    elif isinstance(universe, IndexHistory):
        history = universe
    else:
        symbols = sorted({str(s) for s in universe})
        if not symbols:
            raise BacktestError("Das Universum enthält keine Symbole.")
        return "Liste", symbols, lambda _day: None

    if isinstance(universe, str) and survivorship_risk(history.index_code, start):
        log.append(SURVIVORSHIP_WARNING)
    cache: dict[Date, frozenset[str]] = {}

    def members(day: Date) -> frozenset[str]:
        found = cache.get(day)
        if found is None:
            found = frozenset(history.members_on(day))
            cache[day] = found
        return found

    return history.index_code, sorted(history.all_members(start, end)), members


def _collect_bars(
    symbols: Sequence[str],
    bars: Mapping[str, pd.DataFrame] | None,
    settings: Settings,
    end: Date,
    log: list[str],
) -> dict[str, pd.DataFrame]:
    if bars is not None:
        found = {s: _until(bars[s], end) for s in symbols if s in bars}
    else:
        found = _load_daily(settings, symbols, end)
    missing = [s for s in symbols if s not in found]
    if missing:
        log.append(
            f"{len(missing)} Wert(e) des Universums ohne Kurse: "
            + ", ".join(missing[:20])
            + (" …" if len(missing) > 20 else "")
        )
    return found


def _until(daily: pd.DataFrame, end: Date) -> pd.DataFrame:
    """Tageskurse bis einschließlich ``end`` (wie beim Laden aus der Datenbank)."""
    if DATE_COLUMN not in daily.columns or daily.empty:
        return daily
    dates = pd.to_datetime(daily[DATE_COLUMN])
    return daily.loc[(dates <= pd.Timestamp(end)).to_numpy()]


def _load_daily(settings: Settings, symbols: Sequence[str], end: Date) -> dict[str, pd.DataFrame]:
    from trading_helfer.data.storage import MarketStore

    path = Path(settings.database_path)
    if not path.exists():
        raise BacktestError(f"Datenbank {path} existiert nicht; bitte zuerst Daten laden.")
    result: dict[str, pd.DataFrame] = {}
    with MarketStore(path, read_only=True) as store:
        for symbol in symbols:
            frame = store.load_bars(symbol, end=end, adjusted=True)
            if len(frame):
                result[symbol] = frame
    return result


def _prepare_all(
    template: Strategy,
    params: Params,
    daily: Mapping[str, pd.DataFrame],
    timeframe: Timeframe,
    start: Date,
    workers: int | None,
) -> list[_SymbolData | None]:
    """Bereitet alle Werte vor: in Prozessen, wenn möglich, sonst in Threads.

    Prozesse umgehen die GIL; sie setzen voraus, dass sich die Strategie mit
    :mod:`pickle` übertragen lässt und ihr Modul in einem neuen Prozess importierbar ist
    (Windows startet Prozesse neu). Sonst (z. B. dynamisch geladene Plugins) und bei
    wenigen Werten laufen Threads. Der Weg ändert das Ergebnis nicht.
    """
    symbols = sorted(daily)
    values = params.to_dict()  # Params selbst lässt sich nicht pickeln
    jobs = [(template, values, s, daily[s], timeframe, start) for s in symbols]
    count = workers if workers is not None else (os.cpu_count() or 1)
    if count <= 1:
        return [_prepare_job(job) for job in jobs]
    if len(jobs) >= _PROCESS_MIN_SYMBOLS and _process_safe(template):
        chunk = max(1, len(jobs) // (count * 4))
        try:
            with ProcessPoolExecutor(max_workers=count) as pool:
                return list(pool.map(_prepare_job, jobs, chunksize=chunk))
        except (BrokenProcessPool, ImportError, AttributeError, pickle.PicklingError):
            logger.warning("Prozess-Pool nicht verfügbar; Vorberechnung in Threads.")
    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(_prepare_job, jobs))


def _process_safe(template: Strategy) -> bool:
    """``True``, wenn die Strategie in einem frisch gestarteten Prozess ladbar ist."""
    module = type(template).__module__
    if module == "__main__":
        return False
    try:
        if importlib.util.find_spec(module) is None:
            return False
        pickle.dumps(template)
    except Exception:
        return False
    return True


def _prepare_job(
    job: tuple[Strategy, dict[str, Any], str, pd.DataFrame, Timeframe, Date],
) -> _SymbolData | None:
    template, values, symbol, daily, timeframe, start = job
    params = template.make_params(values)
    return _prepare_symbol(template, params, symbol, daily, timeframe, start)


@dataclass
class _SymbolData:
    """Vorberechnete Kerzen, Indikatoren und Hilfsreihen eines Werts."""

    symbol: str
    strategy: Strategy
    dates: list[Date]
    day_values: np.ndarray[Any, Any]
    partial_last: bool
    open: np.ndarray[Any, Any]
    high: np.ndarray[Any, Any]
    low: np.ndarray[Any, Any]
    close: np.ndarray[Any, Any]
    atr: np.ndarray[Any, Any]
    vol_pct: np.ndarray[Any, Any]
    series: PreparedSeries
    strength: np.ndarray[Any, Any] | None
    signal_start: int | None
    grid_pos: np.ndarray[Any, Any] = field(default_factory=lambda: np.empty(0, dtype=np.int64))


def _prepare_symbol(
    template: Strategy,
    params: Params,
    symbol: str,
    daily: pd.DataFrame,
    timeframe: Timeframe,
    start: Date,
) -> _SymbolData | None:
    """Zeitrahmen umrechnen und Indikatoren berechnen (läuft in einem Thread)."""
    if daily.empty:
        return None
    frame = resample(daily, timeframe)
    if frame.empty:
        return None
    partial_last = bool(frame[PARTIAL_COLUMN].iloc[-1])
    frame = frame.drop(columns=[PARTIAL_COLUMN]).reset_index(drop=True)
    day_index = pd.DatetimeIndex(frame[DATE_COLUMN])
    if day_index[-1] < pd.Timestamp(start):
        return None
    strategy = copy.deepcopy(template)
    try:
        ind = strategy.indicators(frame.copy(), params)
    except Exception as exc:
        raise StrategyError(
            f"Strategie {strategy.id!r}: Indikatoren für {symbol} konnten nicht berechnet "
            f"werden: {exc}"
        ) from exc
    if not isinstance(ind, pd.DataFrame):
        raise StrategyError(
            f"Strategie {strategy.id!r}: indicators() muss ein DataFrame liefern, "
            f"erhalten: {type(ind).__name__}."
        )
    ind = ind.reset_index(drop=True)
    try:
        series = PreparedSeries(frame, ind)
    except ValueError as exc:
        raise StrategyError(f"Strategie {strategy.id!r} ({symbol}): {exc}") from exc
    first = int(np.searchsorted(day_index.values, np.datetime64(pd.Timestamp(start))))
    complete = ~ind.isna().any(axis=1).to_numpy()
    complete[:first] = False
    hits = np.flatnonzero(complete)
    strength = (
        pd.to_numeric(ind[STRENGTH_COLUMN], errors="coerce").to_numpy(dtype=float)
        if STRENGTH_COLUMN in ind.columns
        else None
    )
    hv = historical_volatility(frame, _VOL_PERIOD, annualization=BARS_PER_YEAR[timeframe])
    return _SymbolData(
        symbol=symbol,
        strategy=strategy,
        dates=[d.date() for d in day_index],
        day_values=day_index.values,
        partial_last=partial_last,
        open=frame["open"].to_numpy(dtype=float),
        high=frame["high"].to_numpy(dtype=float),
        low=frame["low"].to_numpy(dtype=float),
        close=frame["close"].to_numpy(dtype=float),
        atr=atr_indicator(frame, _ATR_PERIOD).to_numpy(dtype=float),
        vol_pct=hv.to_numpy(dtype=float) * 100.0,
        series=series,
        strength=strength,
        signal_start=int(hits[0]) if hits.size else None,
    )


# ---------------------------------------------------------------------------
# Lauf
# ---------------------------------------------------------------------------


@dataclass
class _Submitted:
    pending_id: int
    submitted: Date
    order: Order


@dataclass
class _Engine:
    """Von der Engine (vor dem Broker) verworfene Order."""

    submitted: Date
    order: Order
    reason: str


class _PortfolioRun:
    """Zustand eines Laufs (eine Instanz je Aufruf von :func:`run_portfolio_backtest`)."""

    def __init__(
        self,
        strategy: Strategy,
        params: Params,
        universe: str,
        timeframe: Timeframe,
        start: Date,
        end: Date,
        broker: SimBroker,
        data: list[_SymbolData],
        is_member: Callable[[Date], frozenset[str] | None],
        max_positions: int,
    ) -> None:
        self.strategy = strategy
        self.params = params
        self.universe = universe
        self.timeframe = timeframe
        self.start = start
        self.end = end
        self.broker = broker
        self.data = data
        self.by_symbol = {d.symbol: d for d in data}
        self.is_member = is_member
        self.max_positions = max_positions
        self.log: list[str] = []
        self.submitted: list[_Submitted] = []
        self.engine_rejections: list[_Engine] = []
        self.equity_rows: list[tuple[Any, ...]] = []
        self.delisted: dict[str, Date] = {}
        self.entry_rank: dict[str, int] = {}
        self._rank_counter = 0
        self.warmup_end: Date | None = None
        self._build_grid()

    # -- Zeitachse ------------------------------------------------------------

    def _build_grid(self) -> None:
        complete: list[np.ndarray[Any, Any]] = []
        for d in self.data:
            values = d.day_values[:-1] if d.partial_last else d.day_values
            complete.append(values)
        grid = np.unique(np.concatenate(complete)) if complete else np.empty(0)
        grid = grid[grid >= np.datetime64(pd.Timestamp(self.start))]
        if grid.size == 0:
            raise BacktestError(
                f"Keine abgeschlossenen {self.timeframe.value}-Kerzen im Zeitraum "
                f"{self.start} bis {self.end}."
            )
        self.grid = grid
        self.grid_dates = [pd.Timestamp(g).date() for g in grid]
        events: list[list[tuple[_SymbolData, int]]] = [[] for _ in range(len(grid))]
        last_index = len(grid) - 1
        for d in self.data:  # alphabetisch
            pos = np.searchsorted(grid, d.day_values, side="left")
            d.grid_pos = pos
            first = int(np.searchsorted(d.day_values, grid[0], side="left"))
            for i in range(first, len(pos)):
                g = int(pos[i])
                if g > last_index:
                    break  # unvollständige Kerze nach dem letzten Termin
                events[g].append((d, i))
        self.events = events
        if self.data:
            starts = [d.dates[d.signal_start] for d in self.data if d.signal_start is not None]
            self.warmup_end = min(starts) if starts else None
            if self.warmup_end is None:
                self.log.append("Indikatoren werden im Zeitraum nie vollständig; keine Signale.")

    # -- Schleife -------------------------------------------------------------

    def loop(self) -> None:
        last = len(self.grid) - 1
        for g, entries in enumerate(self.events):
            day = self.grid_dates[g]
            self._execute(entries)
            active: list[tuple[_SymbolData, int]] = []
            for d, i in entries:
                is_last_bar = i == len(d.dates) - 1 or int(d.grid_pos[i + 1]) > last
                if is_last_bar and g < last:
                    self._delist(d, i)
                else:
                    active.append((d, i))
            self._signals(active, day)
            self._record_equity(day)

    def _execute(self, entries: list[tuple[_SymbolData, int]]) -> None:
        broker = self.broker
        held = set(broker.positions)
        busy = held | {p.symbol for p in broker.open_orders()}
        entries = [e for e in entries if e[0].symbol in busy]

        def order_key(item: tuple[_SymbolData, int]) -> tuple[int, int, str]:
            sym = item[0].symbol
            if sym in held:
                return (0, 0, sym)
            return (1, self.entry_rank.get(sym, 1 << 62), sym)

        for d, i in sorted(entries, key=order_key):
            bar = Bar(d.dates[i], d.open[i], d.high[i], d.low[i], d.close[i])
            broker.process_bar(d.symbol, bar, atr=_finite(d.atr[i]))
            if not broker.open_orders(d.symbol):
                self.entry_rank.pop(d.symbol, None)

    def _delist(self, d: _SymbolData, i: int) -> None:
        day = d.dates[i]
        self.delisted[d.symbol] = day
        fill = self.broker.force_close(d.symbol, day, float(d.close[i]), DELISTING_REASON)
        self.entry_rank.pop(d.symbol, None)
        if fill is not None:
            self.submitted.append(
                _Submitted(fill.id, day, Order(d.symbol, fill.side, tag=DELISTING_REASON))
            )
            self.log.append(
                f"{day}: {d.symbol} hat keine weiteren Kurse (Delisting); Position zum "
                f"letzten Close {d.close[i]:.4f} geschlossen."
            )

    def _signals(self, active: list[tuple[_SymbolData, int]], day: Date) -> None:
        broker = self.broker
        members = self.is_member(day)
        held = set(broker.positions)
        exits: list[tuple[_SymbolData, int, Order]] = []
        candidates: dict[str, tuple[float, _SymbolData, int, list[Order]]] = {}
        freed = 0
        waiting = {p.symbol for p in broker.open_orders()}
        cash, equity = broker.cash, broker.equity()  # Orders ändern beides erst bei Ausführung
        for d, i in active:
            if d.signal_start is None or i < d.signal_start:
                continue
            member = members is None or d.symbol in members
            if not member and d.symbol not in held and d.symbol not in waiting:
                continue  # kann weder eröffnen noch schließen
            orders = self._call_strategy(d, i, member, cash, equity)
            if not orders:
                continue
            if d.symbol in held:
                for order in orders:
                    exits.append((d, i, order))
                    if (
                        not order.side.is_entry
                        and order.type is OrderType.MARKET
                        and order.qty is None
                    ):
                        freed += 1
                continue
            if not any(o.side.is_entry for o in orders):
                for order in orders:
                    exits.append((d, i, order))
                continue
            if not member:
                for order in orders:
                    self.engine_rejections.append(
                        _Engine(day, order, f"{d.symbol} ist am {day} kein Indexmitglied.")
                    )
                continue
            candidates[d.symbol] = (self._strength(d, i, orders), d, i, orders)

        for d, i, order in exits:
            self._submit(d, i, order)

        pending_entries = {p.symbol for p in broker.open_orders() if p.order.side.is_entry} - held
        occupied = len(held | pending_entries)
        free = max(0, self.max_positions - occupied + min(freed, len(held)))
        ranked = sorted(candidates.values(), key=lambda c: (-c[0], c[1].symbol))
        taken = 0
        for strength, d, i, orders in ranked:
            if d.symbol not in pending_entries:
                if taken >= free:
                    reason = (
                        f"Kein freier Platz (max. {self.max_positions} Positionen); "
                        f"Signalstärke {strength:g}."
                    )
                    for order in orders:
                        self.engine_rejections.append(_Engine(day, order, reason))
                    continue
                taken += 1
            self._rank_counter += 1
            self.entry_rank.setdefault(d.symbol, self._rank_counter)
            for order in orders:
                self._submit(d, i, order)

    def _strength(self, d: _SymbolData, i: int, orders: list[Order]) -> float:
        values = [o.strength for o in orders if o.side.is_entry and o.strength is not None]
        if values:
            return float(max(values))
        if d.strength is not None and math.isfinite(d.strength[i]):
            return float(d.strength[i])
        return 0.0

    def _call_strategy(
        self, d: _SymbolData, i: int, member: bool, cash: float, equity: float
    ) -> list[Order]:
        ctx = BarContext.at(
            d.series,
            i,
            symbol=d.symbol,
            timeframe=self.timeframe,
            position=self.broker.position(d.symbol),
            cash=cash,
            equity=equity,
            is_member=member,
        )
        name = d.strategy.id
        day = d.dates[i]
        try:
            orders = d.strategy.on_bar(ctx)
        except IndexError as exc:
            raise LookaheadError(
                f"Strategie {name!r} greift am {day} ({d.symbol}) auf Daten nach der aktuellen "
                f"Kerze zu: {exc}"
            ) from exc
        except Exception as exc:
            raise StrategyError(
                f"Strategie {name!r} ist am {day} ({d.symbol}) fehlgeschlagen: {exc}"
            ) from exc
        if orders is None:
            return []
        if not isinstance(orders, list | tuple):
            raise StrategyError(
                f"Strategie {name!r}: on_bar() muss eine Liste von Orders liefern, "
                f"erhalten: {type(orders).__name__}."
            )
        for order in orders:
            if not isinstance(order, Order):
                raise StrategyError(
                    f"Strategie {name!r}: on_bar() lieferte {type(order).__name__} statt Order."
                )
            if order.symbol != d.symbol:
                raise StrategyError(
                    f"Strategie {name!r}: Order für {order.symbol!r} auf der Kerze von "
                    f"{d.symbol!r}; je Kerze sind nur Orders für den eigenen Wert erlaubt."
                )
        return list(orders)

    def _submit(self, d: _SymbolData, i: int, order: Order) -> None:
        broker = self.broker
        day = d.dates[i]
        pending = broker.submit(
            order,
            date=day,
            close=float(d.close[i]),
            atr=_finite(d.atr[i]),
            volatility=_finite(d.vol_pct[i]),
        )
        pending_id = pending.id if pending is not None else broker.rejections[-1].id
        self.submitted.append(_Submitted(pending_id, day, order))

    def _record_equity(self, day: Date) -> None:
        broker = self.broker
        equity = broker.equity()
        exposure_value = 0.0
        for sym, pos in broker.positions.items():
            price = broker.last_price(sym)
            exposure_value += pos.qty * (price if price is not None else pos.entry_price)
        exposure = exposure_value / equity if equity > 0 else 0.0
        self.equity_rows.append((day, equity, broker.cash, exposure))

    # -- Ergebnis -------------------------------------------------------------

    def result(self) -> PortfolioResult:
        broker = self.broker
        equity = pd.DataFrame(self.equity_rows, columns=list(EQUITY_COLUMNS[:-1]))
        equity[DATE_COLUMN] = pd.to_datetime(equity[DATE_COLUMN])
        peak = equity["equity"].cummax()
        equity["drawdown"] = (equity["equity"] / peak - 1.0).where(peak > 0, 0.0)
        trades = pd.DataFrame(
            [
                (
                    tr.symbol,
                    tr.side.value,
                    pd.Timestamp(tr.entry_date),
                    tr.entry_price,
                    pd.Timestamp(tr.exit_date),
                    tr.exit_price,
                    tr.qty,
                    tr.pnl,
                    tr.pnl_pct,
                    tr.fees,
                    tr.exit_reason,
                )
                for tr in broker.trades
            ],
            columns=list(TRADE_COLUMNS),
        )
        for rejection in broker.rejections:
            self.log.append(
                f"{rejection.date}: Order {rejection.id} {rejection.order.symbol} "
                f"({rejection.order.side.value} {rejection.order.type.value}) verworfen: "
                f"{rejection.reason}"
            )
        skipped = len(self.engine_rejections)
        if skipped:
            self.log.append(
                f"{skipped} Order(s) vor dem Broker verworfen (kein Mitglied oder kein Platz)."
            )
        open_orders = broker.open_orders()
        if open_orders:
            self.log.append(f"{len(open_orders)} Order(s) am Ende noch offen.")
        open_positions = list(broker.positions.values())
        if open_positions:
            self.log.append(
                f"{len(open_positions)} Position(en) am Ende offen, bewertet zum letzten Close."
            )
        final_equity = float(equity["equity"].iloc[-1]) if len(equity) else broker.equity()
        return PortfolioResult(
            strategy_id=self.strategy.id,
            strategy_version=self.strategy.version,
            params=self.params.to_dict(),
            symbol=self.universe,
            timeframe=self.timeframe,
            start=self.start,
            end=self.end,
            settings={
                "capital": broker.initial_capital,
                "allow_fractional": broker.allow_fractional,
                "fee_per_order": broker.costs.fee_per_order,
                "fee_pct": broker.costs.fee_pct,
                "slippage_pct": broker.costs.slippage_pct,
                "borrow_fee_pct": broker.costs.borrow_fee_pct,
                "max_positions": self.max_positions,
            },
            initial_capital=broker.initial_capital,
            final_equity=final_equity,
            trades=trades,
            equity=equity.loc[:, list(EQUITY_COLUMNS)],
            orders=self._orders_frame(),
            log=self.log,
            open_positions=open_positions,
            warmup_end=self.warmup_end,
            universe=self.universe,
            symbols=tuple(d.symbol for d in self.data),
            max_positions=self.max_positions,
            delisted=dict(self.delisted),
        )

    def _orders_frame(self) -> pd.DataFrame:
        broker = self.broker
        fills = {f.id: f for f in broker.fills}
        rejections = {r.id: r for r in broker.rejections}
        open_ids = {p.id for p in broker.open_orders()}
        rows: list[tuple[Any, ...]] = []
        for s in self.submitted:
            fill = fills.get(s.pending_id)
            rejection = rejections.get(s.pending_id)
            if fill is not None:
                status = "FILLED"
            elif rejection is not None:
                status = "REJECTED"
            elif s.pending_id in open_ids:
                status = "OPEN"
            else:
                status = "CANCELLED"
            rows.append(
                _order_row(
                    s.pending_id,
                    s.submitted,
                    s.order,
                    status,
                    fill,
                    rejection.reason if rejection else "",
                )
            )
        for e in self.engine_rejections:
            rows.append(_order_row(0, e.submitted, e.order, "REJECTED", None, e.reason))
        frame = pd.DataFrame(rows, columns=list(ORDER_COLUMNS))
        return frame.sort_values(["submitted", "symbol", "order_id"], kind="stable").reset_index(
            drop=True
        )


def _order_row(
    order_id: int, submitted: Date, o: Order, status: str, fill: Any, reason: str
) -> tuple[Any, ...]:
    sizing = f"{o.sizing.method.value}:{o.sizing.value!r}" if o.sizing else ""
    return (
        order_id,
        pd.Timestamp(submitted),
        o.symbol,
        o.side.value,
        o.type.value,
        o.qty if o.qty is not None else math.nan,
        sizing,
        o.stop_price if o.stop_price is not None else math.nan,
        o.limit_price if o.limit_price is not None else math.nan,
        o.tag,
        status,
        pd.Timestamp(fill.date) if fill else pd.NaT,
        fill.price if fill else math.nan,
        fill.qty if fill else math.nan,
        fill.fee if fill else math.nan,
        reason,
    )


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None
