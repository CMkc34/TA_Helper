"""Ereignisgesteuerte Backtest-Engine für einen Wert.

Ablauf (siehe Spezifikation, Backtest-Engine):

1. Tageskurse bis ``end`` laden (bereinigt) und in den Zeitrahmen umrechnen. Die
   unvollständige letzte Kerze (``is_partial``) wird nicht verwendet.
2. Indikatoren einmal vektorisiert über die ganze Reihe berechnen
   (:meth:`Strategy.indicators`). Kerzen vor ``start`` dienen nur dem Anlauf der
   Indikatoren; gehandelt wird ab der ersten Kerze mit Datum ``>= start``.
3. Je abgeschlossener Kerze ``t``: zuerst führt der Broker die auf ``t-1`` abgegebenen
   Orders auf Kerze ``t`` aus (Market zum Open, Stop/Limit aus High/Low), danach sieht
   die Strategie alle Daten bis einschließlich Close von ``t`` und gibt neue Orders ab.
4. Warm-up: :meth:`Strategy.on_bar` wird erst ab der ersten Kerze aufgerufen, auf der
   kein Indikator NaN ist.

Offene Positionen am Ende bleiben offen und werden zum letzten Close bewertet; sie
stehen in :attr:`BacktestResult.open_positions`.

Beispiel::

    result = run_backtest(SmaCross(), {"fast": 10}, "AAPL", Timeframe.W1,
                          "2000-01-01", "2024-12-31", settings)
    print(result.final_equity, len(result.trades))
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date as Date  # noqa: N812
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from trading_helfer.backtest.broker_sim import Bar, PendingOrder, SimBroker
from trading_helfer.backtest.costs import CostModel
from trading_helfer.core.bars import DATE_COLUMN
from trading_helfer.core.types import Timeframe
from trading_helfer.data.resample import PARTIAL_COLUMN, resample
from trading_helfer.indicators.volatility import atr as atr_indicator
from trading_helfer.indicators.volatility import historical_volatility
from trading_helfer.strategies.base import (
    POSITION_SUFFIX,
    RANK_SUFFIX,
    BarContext,
    Order,
    Params,
    Position,
    PreparedSeries,
    Strategy,
)

if TYPE_CHECKING:
    from trading_helfer.core.config import Settings

__all__ = [
    "BARS_PER_YEAR",
    "EQUITY_COLUMNS",
    "ORDER_COLUMNS",
    "TRADE_COLUMNS",
    "BacktestError",
    "BacktestResult",
    "LookaheadError",
    "StrategyError",
    "run_backtest",
]

DateInput = Date | str | pd.Timestamp
"""Erlaubte Datumsangaben für ``start`` und ``end``."""

BARS_PER_YEAR: Final[Mapping[Timeframe, int]] = {
    Timeframe.D1: 252,
    Timeframe.D3: 84,
    Timeframe.W1: 52,
    Timeframe.MN1: 12,
}
"""Kerzen pro Jahr je Zeitrahmen (Annualisierung der Volatilität)."""

TRADE_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "side",
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "qty",
    "pnl",
    "pnl_pct",
    "fees",
    "exit_reason",
)
"""Spalten der Trade-Liste (wie Tabelle ``backtest_trades`` ohne Schlüssel)."""

EQUITY_COLUMNS: Final[tuple[str, ...]] = ("date", "equity", "cash", "exposure", "drawdown")
"""Spalten der Equity-Kurve (wie Tabelle ``backtest_equity`` ohne ``run_id``)."""

ORDER_COLUMNS: Final[tuple[str, ...]] = (
    "order_id",
    "submitted",
    "symbol",
    "side",
    "type",
    "qty",
    "sizing",
    "stop_price",
    "limit_price",
    "tag",
    "status",
    "fill_date",
    "fill_price",
    "fill_qty",
    "fee",
    "reason",
)
"""Spalten der Order-Liste. ``status``: FILLED, REJECTED, CANCELLED oder OPEN."""

_ATR_PERIOD: Final = 14
_VOL_PERIOD: Final = 20


class BacktestError(ValueError):
    """Backtest konnte nicht ausgeführt werden; Text auf Deutsch."""


class StrategyError(BacktestError):
    """Die Strategie hat auf einer Kerze einen Fehler ausgelöst oder Ungültiges geliefert."""


class LookaheadError(StrategyError):
    """Die Strategie hat auf Daten nach der aktuellen Kerze ``t`` zugegriffen."""


@dataclass(frozen=True)
class BacktestResult:
    """Ergebnis eines Backtests für einen Wert.

    Attributes:
        strategy_id: ID der Strategie.
        strategy_version: Version der Strategie.
        params: Verwendete (geprüfte) Parameter.
        symbol: Wertpapier.
        timeframe: Zeitrahmen.
        start: Gewünschter Beginn.
        end: Gewünschtes Ende.
        settings: Kapital- und Kosteneinstellungen des Laufs (für ``settings_json``).
        initial_capital: Startkapital.
        final_equity: Kapital zum letzten Close (offene Positionen bewertet).
        trades: Abgeschlossene Trades, Spalten :data:`TRADE_COLUMNS`.
        equity: Equity-Kurve je gehandelter Kerze, Spalten :data:`EQUITY_COLUMNS`;
            ``exposure`` = Positionswert / Kapital, ``drawdown`` = Kapital / Höchststand − 1
            (0 oder negativ).
        orders: Alle abgegebenen Orders mit Ergebnis, Spalten :data:`ORDER_COLUMNS`.
        log: Protokoll des Laufs (Warm-up, verworfene Orders …), auf Deutsch.
        open_positions: Am Ende noch offene Positionen.
        warmup_end: Datum der ersten Kerze, auf der die Strategie aufgerufen wurde,
            oder ``None``, wenn die Indikatoren nie vollständig waren.
        last_close: Close der letzten abgeschlossenen Kerze.
        last_indicators: Indikatorwerte der Strategie auf der letzten Kerze (für den
            Scanner; nicht Teil von :meth:`fingerprint`).
    """

    strategy_id: str
    strategy_version: str
    params: dict[str, Any]
    symbol: str
    timeframe: Timeframe
    start: Date
    end: Date
    settings: dict[str, Any]
    initial_capital: float
    final_equity: float
    trades: pd.DataFrame
    equity: pd.DataFrame
    orders: pd.DataFrame
    log: list[str] = field(default_factory=list)
    open_positions: list[Position] = field(default_factory=list)
    warmup_end: Date | None = None
    last_close: float | None = None
    last_indicators: dict[str, Any] = field(default_factory=dict)

    @property
    def total_return_pct(self) -> float:
        """Gesamtrendite in Prozent bezogen auf das Startkapital."""
        return (self.final_equity / self.initial_capital - 1.0) * 100.0

    def fingerprint(self) -> str:
        """SHA-256 über alle Ergebnisdaten; gleiche Eingaben ergeben denselben Wert."""
        h = hashlib.sha256()
        header = (
            self.strategy_id,
            self.strategy_version,
            sorted((k, repr(v)) for k, v in self.params.items()),
            self.symbol,
            self.timeframe.value,
            self.start.isoformat(),
            self.end.isoformat(),
            sorted((k, repr(v)) for k, v in self.settings.items()),
            repr(self.initial_capital),
            repr(self.final_equity),
            [repr(p) for p in self.open_positions],
            repr(self.warmup_end),
        )
        h.update(repr(header).encode())
        for frame in (self.trades, self.equity, self.orders):
            h.update(_frame_bytes(frame))
        h.update("\n".join(self.log).encode())
        return h.hexdigest()


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    rows = [repr(tuple(frame.columns))]
    rows.extend(repr(tuple(row)) for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows).encode()


# ---------------------------------------------------------------------------
# Öffentliche Funktion
# ---------------------------------------------------------------------------


def run_backtest(
    strategy: Strategy | type[Strategy],
    params: Mapping[str, Any] | None,
    symbol: str,
    timeframe: Timeframe,
    start: DateInput,
    end: DateInput,
    settings: Settings,
    *,
    bars: pd.DataFrame | None = None,
    costs: CostModel | None = None,
) -> BacktestResult:
    """Führt einen ereignisgesteuerten Backtest für einen Wert aus.

    Args:
        strategy: Strategie (Instanz oder Klasse, die ohne Argumente erzeugt wird).
        params: Parameterwerte; fehlende erhalten den Standardwert, ``None`` = alle Standard.
        symbol: Wertpapier.
        timeframe: Zeitrahmen; muss in ``strategy.timeframes`` stehen.
        start: Erste Kerze, auf der gehandelt werden darf (einschließlich).
        end: Letzter Tag der Daten (einschließlich).
        settings: Einstellungen (Startkapital, Bruchstücke, Gebühr, Slippage, Datenbank).
        bars: Bereinigte Tageskurse statt Laden aus der Datenbank (z. B. für Tests).
        costs: Kostenmodell statt der Kosten aus ``settings``.

    Returns:
        Ergebnis mit Trades, Equity-Kurve, Orders und Protokoll.

    Raises:
        BacktestError: Ungültige Eingaben oder keine Kurse im Zeitraum.
        StrategyError: Die Strategie löst einen Fehler aus oder liefert ungültige Orders.
        LookaheadError: Die Strategie greift auf Daten nach ``t`` zu.
        trading_helfer.strategies.base.ParamError: Ungültige Parameter.
    """
    instance = strategy() if isinstance(strategy, type) else strategy
    timeframe = Timeframe(timeframe)
    start_day, end_day = _to_day(start, "start"), _to_day(end, "end")
    if start_day > end_day:
        raise BacktestError(f"Beginn {start_day} liegt nach dem Ende {end_day}.")
    if timeframe not in instance.timeframes:
        supported = ", ".join(t.value for t in instance.timeframes) or "keine"
        raise BacktestError(
            f"Strategie {instance.id!r} unterstützt den Zeitrahmen {timeframe.value} nicht "
            f"(unterstützt: {supported})."
        )
    p = instance.make_params(params)
    broker = SimBroker(
        settings.default_capital,
        costs if costs is not None else CostModel.from_settings(settings),
        allow_fractional=settings.allow_fractional,
    )
    daily = bars.copy() if bars is not None else _load_daily(settings, symbol, end_day)
    run = _Run(instance, p, symbol, timeframe, start_day, end_day, broker)
    run.prepare(daily)
    run.loop()
    return run.result()


# ---------------------------------------------------------------------------
# Interna
# ---------------------------------------------------------------------------


def _to_day(value: DateInput, name: str) -> Date:
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise BacktestError(f"{name}: kein gültiges Datum: {value!r}.") from exc
    if pd.isna(ts):
        raise BacktestError(f"{name}: kein gültiges Datum: {value!r}.")
    return ts.date()


def _load_daily(settings: Settings, symbol: str, end: Date) -> pd.DataFrame:
    from trading_helfer.data.storage import MarketStore

    path = Path(settings.database_path)
    if not path.exists():
        raise BacktestError(f"Datenbank {path} existiert nicht; bitte zuerst Daten laden.")
    with MarketStore(path, read_only=True) as store:
        return store.load_bars(symbol, end=end, adjusted=True)


@dataclass
class _Submitted:
    pending_id: int
    submitted: Date
    order: Order


class _Run:
    """Zustand eines Laufs (eine Instanz je Aufruf von :func:`run_backtest`)."""

    def __init__(
        self,
        strategy: Strategy,
        params: Params,
        symbol: str,
        timeframe: Timeframe,
        start: Date,
        end: Date,
        broker: SimBroker,
    ) -> None:
        self.strategy = strategy
        self.params = params
        self.symbol = symbol
        self.timeframe = timeframe
        self.start = start
        self.end = end
        self.broker = broker
        self.log: list[str] = []
        self.submitted: list[_Submitted] = []
        self.equity_rows: list[tuple[Any, ...]] = []
        self.warmup_end: Date | None = None

    # -- Vorbereitung ---------------------------------------------------------

    def prepare(self, daily: pd.DataFrame) -> None:
        if DATE_COLUMN in daily.columns:
            dates = pd.to_datetime(daily[DATE_COLUMN])
            daily = daily.loc[(dates <= pd.Timestamp(self.end)).to_numpy()]
        frame = resample(daily, self.timeframe)
        if len(frame) and bool(frame[PARTIAL_COLUMN].iloc[-1]):
            last = pd.Timestamp(frame[DATE_COLUMN].iloc[-1]).date()
            self.log.append(f"Unvollständige letzte Kerze vom {last} wird nicht verwendet.")
            frame = frame.iloc[:-1]
        frame = frame.drop(columns=[PARTIAL_COLUMN]).reset_index(drop=True)
        if frame.empty:
            raise BacktestError(
                f"Keine abgeschlossenen {self.timeframe.value}-Kerzen für {self.symbol} "
                f"bis {self.end}."
            )
        day_index = pd.DatetimeIndex(frame[DATE_COLUMN])
        first = int(np.searchsorted(day_index.values, np.datetime64(pd.Timestamp(self.start))))
        if first >= len(frame):
            raise BacktestError(
                f"Keine Kerzen für {self.symbol} im Zeitraum {self.start} bis {self.end}."
            )
        self.frame = frame
        self.first = first
        self.dates = [d.date() for d in day_index]
        self.open = frame["open"].to_numpy(dtype=float)
        self.high = frame["high"].to_numpy(dtype=float)
        self.low = frame["low"].to_numpy(dtype=float)
        self.close = frame["close"].to_numpy(dtype=float)
        self.atr = atr_indicator(frame, _ATR_PERIOD).to_numpy(dtype=float)
        hv = historical_volatility(
            frame, _VOL_PERIOD, annualization=BARS_PER_YEAR[self.timeframe]
        ).to_numpy(dtype=float)
        self.vol_pct = hv * 100.0

        try:
            ind = self.strategy.indicators(frame.copy(), self.params)
        except Exception as exc:
            raise StrategyError(
                f"Strategie {self.strategy.id!r}: Indikatoren konnten nicht berechnet werden: {exc}"
            ) from exc
        if not isinstance(ind, pd.DataFrame):
            raise StrategyError(
                f"Strategie {self.strategy.id!r}: indicators() muss ein DataFrame liefern, "
                f"erhalten: {type(ind).__name__}."
            )
        try:
            self.series = PreparedSeries(frame, _single_ranks(self.strategy, ind))
        except ValueError as exc:
            raise StrategyError(f"Strategie {self.strategy.id!r}: {exc}") from exc
        self.signal_start = self._warmup_index(ind)
        if self.signal_start is None:
            self.log.append("Indikatoren werden im Zeitraum nie vollständig; keine Signale.")
        else:
            self.warmup_end = self.dates[self.signal_start]
            if self.signal_start > self.first:
                self.log.append(f"Warm-up der Indikatoren bis {self.warmup_end}.")

    def _warmup_index(self, ind: pd.DataFrame) -> int | None:
        complete = ~ind.isna().any(axis=1).to_numpy()
        complete[: self.first] = False
        hits = np.flatnonzero(complete)
        return int(hits[0]) if hits.size else None

    # -- Schleife -------------------------------------------------------------

    def loop(self) -> None:
        broker = self.broker
        for t in range(self.first, len(self.frame)):
            day = self.dates[t]
            bar = Bar(day, self.open[t], self.high[t], self.low[t], self.close[t])
            broker.process_bar(self.symbol, bar, atr=_finite(self.atr[t]))
            if self.signal_start is not None and t >= self.signal_start:
                self._call_strategy(t, day)
            self._record_equity(day)

    def _call_strategy(self, t: int, day: Date) -> None:
        broker = self.broker
        ctx = BarContext.at(
            self.series,
            t,
            symbol=self.symbol,
            timeframe=self.timeframe,
            position=broker.position(self.symbol),
            cash=broker.cash,
            equity=broker.equity(),
        )
        name = self.strategy.id
        try:
            orders = self.strategy.on_bar(ctx)
        except IndexError as exc:
            raise LookaheadError(
                f"Strategie {name!r} greift am {day} auf Daten nach der aktuellen Kerze zu: {exc}"
            ) from exc
        except Exception as exc:
            raise StrategyError(f"Strategie {name!r} ist am {day} fehlgeschlagen: {exc}") from exc
        if orders is None:
            return
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
            if order.symbol != self.symbol:
                raise StrategyError(
                    f"Strategie {name!r}: Order für {order.symbol!r}, gehandelt wird nur "
                    f"{self.symbol!r}."
                )
            self._submit(order, t, day)

    def _submit(self, order: Order, t: int, day: Date) -> None:
        broker = self.broker
        pending: PendingOrder | None = broker.submit(
            order,
            date=day,
            close=self.close[t],
            atr=_finite(self.atr[t]),
            volatility=_finite(self.vol_pct[t]),
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

    def result(self) -> BacktestResult:
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
                f"{rejection.date}: Order {rejection.id} ({rejection.order.side.value} "
                f"{rejection.order.type.value}) verworfen: {rejection.reason}"
            )
        open_orders = broker.open_orders(self.symbol)
        if open_orders:
            self.log.append(f"{len(open_orders)} Order(s) am Ende noch offen.")
        open_positions = list(broker.positions.values())
        if open_positions:
            self.log.append(
                f"{len(open_positions)} Position(en) am Ende offen, bewertet zum letzten Close."
            )

        final_equity = float(equity["equity"].iloc[-1]) if len(equity) else broker.equity()
        return BacktestResult(
            strategy_id=self.strategy.id,
            strategy_version=self.strategy.version,
            params=self.params.to_dict(),
            symbol=self.symbol,
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
            },
            initial_capital=broker.initial_capital,
            final_equity=final_equity,
            trades=trades,
            equity=equity.loc[:, list(EQUITY_COLUMNS)],
            orders=self._orders_frame(open_orders),
            log=self.log,
            open_positions=open_positions,
            warmup_end=self.warmup_end,
            last_close=float(self.close[-1]),
            last_indicators=self.series.ind_at(len(self.frame) - 1).row(),
        )

    def _orders_frame(self, open_orders: list[PendingOrder]) -> pd.DataFrame:
        broker = self.broker
        fills = {f.id: f for f in broker.fills}
        rejections = {r.id: r for r in broker.rejections}
        open_ids = {p.id for p in open_orders}
        rows: list[tuple[Any, ...]] = []
        for s in self.submitted:
            o = s.order
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
            sizing = f"{o.sizing.method.value}:{o.sizing.value!r}" if o.sizing else ""
            rows.append(
                (
                    s.pending_id,
                    pd.Timestamp(s.submitted),
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
                    rejection.reason if rejection else "",
                )
            )
        return pd.DataFrame(rows, columns=list(ORDER_COLUMNS))


def _single_ranks(strategy: Strategy, ind: pd.DataFrame) -> pd.DataFrame:
    """Rang-Spalten im Einzelwert-Backtest: der Wert ist der einzige (Rang 100, Platz 1)."""
    ind = ind.reset_index(drop=True)
    columns = tuple(strategy.rank_columns)
    if not columns:
        return ind
    missing = [c for c in columns if c not in ind.columns]
    if missing:
        raise StrategyError(
            f"Strategie {strategy.id!r}: rank_columns {missing} fehlen in den Indikatoren."
        )
    extra = {}
    for c in columns:
        valid = np.isfinite(pd.to_numeric(ind[c], errors="coerce").to_numpy(dtype=float))
        extra[c + RANK_SUFFIX] = np.where(valid, 100.0, np.nan)
        extra[c + POSITION_SUFFIX] = np.where(valid, 1.0, np.nan)
    return ind.assign(**extra)


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None
