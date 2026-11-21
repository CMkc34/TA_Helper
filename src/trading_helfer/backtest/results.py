"""Backtest-Ergebnisse in der Datenbank speichern, auflisten, laden und löschen.

Ein Lauf wird auf drei Tabellen verteilt (siehe Spezifikation, Datenmodell):

* ``backtest_runs``: Kopfdaten (Strategie, Parameter, Universum, Zeitraum, Einstellungen,
  Kennzahlen) sowie Herkunft (Hash des Strategie-Quellcodes, Datenstand) und alle
  übrigen Ergebnisteile in ``details_json`` (Orders, Protokoll, offene Positionen).
* ``backtest_trades``: Trade-Liste.
* ``backtest_equity``: Equity-Kurve.

Speichern und Laden ist eine exakte Rundreise: :meth:`BacktestResult.fingerprint` des
geladenen Ergebnisses ist gleich dem des gespeicherten.

Beispiel::

    with MarketStore(settings.database_path) as market:
        runs = BacktestRunStore(market)
        run_id = runs.save_run(result, source_hash=strategy_source_hash(strategy))
        for summary in runs.list_runs(RunFilter(strategy_id="trend_sma_cross")):
            print(summary.run_id, summary.total_return_pct)
        loaded = runs.load_run(run_id)
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import inspect
import json
import math
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

from trading_helfer.backtest.engine import EQUITY_COLUMNS, TRADE_COLUMNS, BacktestResult
from trading_helfer.backtest.metrics import (
    BenchmarkComparison,
    MetricsError,
    PerformanceMetrics,
    metrics_for_result,
)
from trading_helfer.core.types import Side, Timeframe
from trading_helfer.strategies.base import Position, Strategy

if TYPE_CHECKING:
    from trading_helfer.data.storage import MarketStore

__all__ = [
    "BacktestRunStore",
    "DataSnapshot",
    "RunFilter",
    "RunNotFoundError",
    "RunSummary",
    "StoredRun",
    "data_snapshot",
    "strategy_source_hash",
]

_DETAILS_VERSION: Final = 1
_DATE_DTYPE: Final = "datetime64[s]"


class RunNotFoundError(KeyError):
    """Zur angegebenen ``run_id`` gibt es keinen gespeicherten Lauf."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "Lauf nicht gefunden."


@dataclass(frozen=True)
class DataSnapshot:
    """Datenstand der Kursdatenbank zum Zeitpunkt eines Laufs.

    Attributes:
        last_date: Letztes Datum in ``bars_daily`` oder ``None``, wenn leer.
        symbol_count: Anzahl verschiedener Symbole in ``bars_daily``.
    """

    last_date: dt.date | None
    symbol_count: int


@dataclass(frozen=True)
class RunFilter:
    """Filter für :meth:`BacktestRunStore.list_runs`; ``None`` heißt „egal“.

    Attributes:
        strategy_id: Nur Läufe dieser Strategie.
        strategy_version: Nur Läufe dieser Strategie-Version.
        universe: Nur Läufe dieses Universums (bei Einzelwerten das Symbol).
        timeframe: Nur Läufe in diesem Zeitrahmen.
        created_from: Nur Läufe, die zu diesem Zeitpunkt oder später gespeichert wurden.
        created_to: Nur Läufe, die zu diesem Zeitpunkt oder früher gespeichert wurden.
        limit: Höchstens so viele Läufe (neueste zuerst).
    """

    strategy_id: str | None = None
    strategy_version: str | None = None
    universe: str | None = None
    timeframe: Timeframe | str | None = None
    created_from: dt.datetime | None = None
    created_to: dt.datetime | None = None
    limit: int | None = None


@dataclass(frozen=True)
class RunSummary:
    """Kopfdaten eines gespeicherten Laufs für Listen (ohne Trades und Equity-Kurve).

    Attributes:
        run_id: ID des Laufs (UUID als Text).
        strategy_id: ID der Strategie.
        strategy_version: Version der Strategie.
        params: Parameterwerte.
        universe: Universum bzw. Symbol.
        timeframe: Zeitrahmen.
        start: Gewünschter Beginn.
        end: Gewünschtes Ende.
        settings: Kapital- und Kosteneinstellungen.
        metrics: Gespeicherte Kennzahlen als Wörterbuch (leer, wenn keine).
        created_at: Zeitpunkt des Speicherns.
        strategy_source_hash: SHA-256 des Strategie-Quellcodes oder ``None``.
        data: Datenstand der Kursdatenbank.
    """

    run_id: str
    strategy_id: str
    strategy_version: str
    params: dict[str, Any]
    universe: str
    timeframe: Timeframe
    start: dt.date
    end: dt.date
    settings: dict[str, Any]
    metrics: dict[str, Any]
    created_at: dt.datetime
    strategy_source_hash: str | None
    data: DataSnapshot

    @property
    def total_return_pct(self) -> float | None:
        """Gesamtrendite aus den Kennzahlen oder ``None``."""
        value = self.metrics.get("total_return_pct")
        return None if value is None else float(value)


@dataclass(frozen=True)
class StoredRun:
    """Vollständig geladener Lauf.

    Attributes:
        summary: Kopfdaten wie in :meth:`BacktestRunStore.list_runs`.
        result: Wiederhergestelltes Backtest-Ergebnis.
        metrics: Wiederhergestellte Kennzahlen oder ``None``, wenn keine gespeichert sind.
    """

    summary: RunSummary
    result: BacktestResult
    metrics: PerformanceMetrics | None = field(default=None)

    @property
    def run_id(self) -> str:
        """ID des Laufs."""
        return self.summary.run_id


# ---------------------------------------------------------------------------
# Herkunft
# ---------------------------------------------------------------------------


def strategy_source_hash(strategy: Strategy | type[Strategy]) -> str | None:
    """SHA-256 über den Quellcode der Datei, in der die Strategie definiert ist.

    Damit lässt sich später erkennen, ob ein gespeicherter Lauf mit dem heutigen Code
    der Strategie entstanden ist. Ist der Quellcode nicht verfügbar (z. B. interaktiv
    definiert), wird der Quellcode der Klasse verwendet; gibt es auch den nicht, ``None``.

    Args:
        strategy: Strategie-Instanz oder -Klasse.

    Returns:
        Hexadezimaler SHA-256-Wert oder ``None``.
    """
    cls = strategy if isinstance(strategy, type) else type(strategy)
    module = sys.modules.get(cls.__module__)
    for obj in (module, cls):
        if obj is None:
            continue
        try:
            source = inspect.getsource(obj)
        except (OSError, TypeError):
            continue
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
    return None


def data_snapshot(store: MarketStore) -> DataSnapshot:
    """Liest den aktuellen Datenstand (letztes Datum, Anzahl Symbole) aus ``bars_daily``.

    Args:
        store: Geöffnete Datenbank.

    Returns:
        Datenstand; bei leerer Tabelle ``last_date=None`` und ``symbol_count=0``.
    """
    rows = store.connection.execute(
        "SELECT max(date), count(DISTINCT symbol) FROM bars_daily"
    ).fetchall()
    last, count = rows[0] if rows else (None, 0)
    return DataSnapshot(last_date=last, symbol_count=int(count or 0))


# ---------------------------------------------------------------------------
# Speicher
# ---------------------------------------------------------------------------


class BacktestRunStore:
    """Speichert Backtest-Läufe in den Tabellen ``backtest_*`` einer :class:`MarketStore`."""

    def __init__(self, store: MarketStore) -> None:
        """Verwendet die Verbindung von ``store`` (Migrationen müssen angewendet sein).

        Args:
            store: Geöffnete, beschreibbare Datenbank.
        """
        self._store = store

    def save_run(
        self,
        result: BacktestResult,
        *,
        metrics: PerformanceMetrics | None = None,
        source_hash: str | None = None,
        snapshot: DataSnapshot | None = None,
        universe: str | None = None,
        run_id: str | uuid.UUID | None = None,
        created_at: dt.datetime | None = None,
    ) -> str:
        """Speichert einen Lauf mit Trades, Equity-Kurve und Kennzahlen.

        Args:
            result: Ergebnis von :func:`~trading_helfer.backtest.engine.run_backtest`.
            metrics: Kennzahlen; ohne Angabe werden sie mit
                :func:`~trading_helfer.backtest.metrics.metrics_for_result` berechnet
                (leer, wenn das nicht möglich ist, z. B. ohne Equity-Kurve).
            source_hash: Hash des Strategie-Quellcodes, z. B. von
                :func:`strategy_source_hash`.
            snapshot: Datenstand; ohne Angabe wird er mit :func:`data_snapshot` gelesen.
            universe: Universum; ohne Angabe das Symbol des Ergebnisses.
            run_id: Eigene ID (UUID); ohne Angabe wird eine neue erzeugt.
            created_at: Zeitpunkt des Speicherns; ohne Angabe jetzt (UTC, ohne Zeitzone).

        Returns:
            Die ``run_id`` als Text.

        Raises:
            ValueError: ``run_id`` ist keine gültige UUID oder bereits vergeben.
        """
        rid = _parse_run_id(run_id) if run_id is not None else str(uuid.uuid4())
        if self._exists(rid):
            raise ValueError(f"Lauf {rid} ist bereits gespeichert.")
        if metrics is None:
            metrics = _try_metrics(result)
        if snapshot is None:
            snapshot = data_snapshot(self._store)
        if created_at is None:
            created_at = dt.datetime.now(dt.UTC).replace(tzinfo=None)

        details = {
            "version": _DETAILS_VERSION,
            "symbol": result.symbol,
            "initial_capital": result.initial_capital,
            "final_equity": result.final_equity,
            "warmup_end": _iso(result.warmup_end),
            "log": list(result.log),
            "open_positions": [_position_to_json(p) for p in result.open_positions],
            "orders": _frame_to_json(result.orders),
            "trade_dtypes": _dtypes(result.trades),
            "equity_dtypes": _dtypes(result.equity),
        }
        trades = _with_run_id(result.trades, TRADE_COLUMNS, rid)
        trades.insert(1, "trade_id", range(1, len(trades) + 1))
        equity = _with_run_id(result.equity, EQUITY_COLUMNS, rid)

        with self._store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO backtest_runs (
                    run_id, strategy_id, strategy_version, params_json, universe, timeframe,
                    start, "end", settings_json, metrics_json, created_at,
                    strategy_source_hash, data_last_date, data_symbol_count, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    rid,
                    result.strategy_id,
                    result.strategy_version,
                    _dumps(result.params),
                    universe if universe is not None else result.symbol,
                    result.timeframe.value,
                    result.start,
                    result.end,
                    _dumps(result.settings),
                    None if metrics is None else _dumps(_metrics_to_json(metrics)),
                    created_at,
                    source_hash,
                    snapshot.last_date,
                    snapshot.symbol_count,
                    _dumps(details),
                ],
            )
            if not trades.empty:
                conn.register("_bt_trades", trades)
                conn.execute(
                    "INSERT INTO backtest_trades (run_id, trade_id, "
                    + ", ".join(TRADE_COLUMNS)
                    + ") SELECT * FROM _bt_trades"
                )
                conn.unregister("_bt_trades")
            if not equity.empty:
                conn.register("_bt_equity", equity)
                conn.execute(
                    "INSERT INTO backtest_equity (run_id, "
                    + ", ".join(EQUITY_COLUMNS)
                    + ") SELECT * FROM _bt_equity"
                )
                conn.unregister("_bt_equity")
        return rid

    def list_runs(self, filter: RunFilter | None = None) -> list[RunSummary]:
        """Listet gespeicherte Läufe, neueste zuerst.

        Args:
            filter: Optionaler Filter (Strategie, Version, Universum, Zeitrahmen, Zeitraum
                des Speicherns, Anzahl).

        Returns:
            Kopfdaten der passenden Läufe.

        Raises:
            ValueError: ``limit`` ist kleiner als 1.
        """
        f = filter if filter is not None else RunFilter()
        where: list[str] = []
        args: list[Any] = []
        for column, value in (
            ("strategy_id", f.strategy_id),
            ("strategy_version", f.strategy_version),
            ("universe", f.universe),
            ("timeframe", None if f.timeframe is None else Timeframe(f.timeframe).value),
        ):
            if value is not None:
                where.append(f"{column} = ?")
                args.append(value)
        if f.created_from is not None:
            where.append("created_at >= ?")
            args.append(f.created_from)
        if f.created_to is not None:
            where.append("created_at <= ?")
            args.append(f.created_to)
        sql = f"SELECT {_SUMMARY_COLUMNS} FROM backtest_runs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, run_id"
        if f.limit is not None:
            if f.limit < 1:
                raise ValueError(f"limit muss mindestens 1 sein, nicht {f.limit}.")
            sql += f" LIMIT {int(f.limit)}"
        rows = self._store.connection.execute(sql, args).fetchall()
        return [_summary_from_row(row) for row in rows]

    def load_run(self, run_id: str | uuid.UUID) -> StoredRun:
        """Lädt einen Lauf vollständig mit Trades, Equity-Kurve, Orders und Kennzahlen.

        Args:
            run_id: ID des Laufs.

        Returns:
            Geladener Lauf; ``result`` entspricht exakt dem gespeicherten Ergebnis.

        Raises:
            RunNotFoundError: Es gibt keinen Lauf mit dieser ID.
            ValueError: ``run_id`` ist keine gültige UUID.
        """
        rid = _parse_run_id(run_id)
        conn = self._store.connection
        rows = conn.execute(
            f"SELECT {_SUMMARY_COLUMNS}, details_json FROM backtest_runs WHERE run_id = ?",
            [rid],
        ).fetchall()
        if not rows:
            raise RunNotFoundError(f"Kein gespeicherter Lauf mit der ID {rid}.")
        row = rows[0]
        summary = _summary_from_row(row[:-1])
        details = json.loads(row[-1]) if row[-1] else {}

        trades = conn.execute(
            "SELECT "
            + ", ".join(TRADE_COLUMNS)
            + " FROM backtest_trades WHERE run_id = ? ORDER BY trade_id",
            [rid],
        ).df()
        equity = conn.execute(
            "SELECT "
            + ", ".join(EQUITY_COLUMNS)
            + " FROM backtest_equity WHERE run_id = ? ORDER BY date",
            [rid],
        ).df()
        trades = _restore_dtypes(trades, details.get("trade_dtypes"))
        equity = _restore_dtypes(equity, details.get("equity_dtypes"))
        orders = _frame_from_json(details["orders"]) if "orders" in details else pd.DataFrame()
        initial = float(details.get("initial_capital", summary.settings.get("capital", 0.0)))
        final = details.get("final_equity")
        if final is None:
            final = float(equity["equity"].iloc[-1]) if not equity.empty else initial

        result = BacktestResult(
            strategy_id=summary.strategy_id,
            strategy_version=summary.strategy_version,
            params=summary.params,
            symbol=str(details.get("symbol", summary.universe)),
            timeframe=summary.timeframe,
            start=summary.start,
            end=summary.end,
            settings=summary.settings,
            initial_capital=initial,
            final_equity=float(final),
            trades=trades,
            equity=equity,
            orders=orders,
            log=list(details.get("log", [])),
            open_positions=[_position_from_json(p) for p in details.get("open_positions", [])],
            warmup_end=_date_or_none(details.get("warmup_end")),
        )
        metrics = _metrics_from_json(summary.metrics) if summary.metrics else None
        return StoredRun(summary=summary, result=result, metrics=metrics)

    def delete_run(self, run_id: str | uuid.UUID) -> bool:
        """Löscht einen Lauf mit seinen Trades und seiner Equity-Kurve.

        Args:
            run_id: ID des Laufs.

        Returns:
            ``True``, wenn der Lauf existierte und gelöscht wurde, sonst ``False``.

        Raises:
            ValueError: ``run_id`` ist keine gültige UUID.
        """
        rid = _parse_run_id(run_id)
        if not self._exists(rid):
            return False
        with self._store.transaction() as conn:
            conn.execute("DELETE FROM backtest_trades WHERE run_id = ?", [rid])
            conn.execute("DELETE FROM backtest_equity WHERE run_id = ?", [rid])
            conn.execute("DELETE FROM backtest_runs WHERE run_id = ?", [rid])
        return True

    def _exists(self, rid: str) -> bool:
        rows = self._store.connection.execute(
            "SELECT count(*) FROM backtest_runs WHERE run_id = ?", [rid]
        ).fetchall()
        return bool(rows[0][0])


# ---------------------------------------------------------------------------
# Interna
# ---------------------------------------------------------------------------

_SUMMARY_COLUMNS: Final = (
    "run_id, strategy_id, strategy_version, params_json, universe, timeframe, start, "
    '"end", settings_json, metrics_json, created_at, strategy_source_hash, '
    "data_last_date, data_symbol_count"
)


def _summary_from_row(row: tuple[Any, ...]) -> RunSummary:
    (
        rid,
        strategy_id,
        version,
        params_json,
        universe,
        timeframe,
        start,
        end,
        settings_json,
        metrics_json,
        created_at,
        source_hash,
        last_date,
        symbol_count,
    ) = row
    return RunSummary(
        run_id=str(rid),
        strategy_id=strategy_id,
        strategy_version=version or "",
        params=json.loads(params_json) if params_json else {},
        universe=universe or "",
        timeframe=Timeframe(timeframe),
        start=start,
        end=end,
        settings=json.loads(settings_json) if settings_json else {},
        metrics=json.loads(metrics_json) if metrics_json else {},
        created_at=created_at,
        strategy_source_hash=source_hash,
        data=DataSnapshot(last_date=last_date, symbol_count=int(symbol_count or 0)),
    )


def _parse_run_id(run_id: str | uuid.UUID) -> str:
    try:
        return str(run_id if isinstance(run_id, uuid.UUID) else uuid.UUID(str(run_id)))
    except ValueError as exc:
        raise ValueError(f"Ungültige Lauf-ID: {run_id!r}.") from exc


def _try_metrics(result: BacktestResult) -> PerformanceMetrics | None:
    try:
        return metrics_for_result(result)
    except MetricsError:
        return None


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, dt.date | pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy-Skalare
        return value.item()
    raise TypeError(f"Nicht als JSON speicherbar: {type(value).__name__}")


def _iso(value: dt.date | None) -> str | None:
    return None if value is None else value.isoformat()


def _date_or_none(value: str | None) -> dt.date | None:
    return None if value is None else dt.date.fromisoformat(value)


def _position_to_json(position: Position) -> dict[str, Any]:
    return {
        "symbol": position.symbol,
        "side": position.side.value,
        "qty": float(position.qty),
        "entry_price": float(position.entry_price),
        "entry_date": position.entry_date.isoformat(),
    }


def _position_from_json(data: Mapping[str, Any]) -> Position:
    return Position(
        symbol=data["symbol"],
        side=Side(data["side"]),
        qty=float(data["qty"]),
        entry_price=float(data["entry_price"]),
        entry_date=dt.date.fromisoformat(data["entry_date"]),
    )


def _dtypes(frame: pd.DataFrame) -> list[list[str]]:
    """Spalten und Typen in Reihenfolge (als Liste, weil JSON-Schlüssel sortiert werden)."""
    return [[str(c), str(t)] for c, t in frame.dtypes.items()]


def _with_run_id(frame: pd.DataFrame, columns: tuple[str, ...], rid: str) -> pd.DataFrame:
    out = pd.DataFrame({c: frame[c] if c in frame.columns else None for c in columns})
    out = out.reset_index(drop=True)
    out.insert(0, "run_id", rid)
    return out


def _restore_dtypes(frame: pd.DataFrame, dtypes: list[list[str]] | None) -> pd.DataFrame:
    if not dtypes:
        return frame
    out = frame.loc[:, [c for c, _ in dtypes if c in frame.columns]].copy()
    for column, dtype in dtypes:
        if column in out.columns:
            out[column] = _cast(out[column], dtype)
    return out.reset_index(drop=True)


def _cast(series: pd.Series, dtype: str) -> pd.Series:
    target = pd.api.types.pandas_dtype(dtype)
    if dtype.startswith("datetime64"):
        return pd.to_datetime(series).astype(target)
    return series.astype(target)


def _frame_to_json(frame: pd.DataFrame) -> dict[str, Any]:
    data = []
    for row in frame.itertuples(index=False, name=None):
        data.append([_cell(v) for v in row])
    return {"columns": [str(c) for c in frame.columns], "dtypes": _dtypes(frame), "data": data}


def _cell(value: Any) -> Any:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp | dt.date):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _frame_from_json(data: Mapping[str, Any]) -> pd.DataFrame:
    columns = list(data["columns"])
    frame = pd.DataFrame(data["data"], columns=columns)
    if frame.empty:
        frame = pd.DataFrame({c: pd.Series(dtype=object) for c in columns})
    return _restore_dtypes(frame, data["dtypes"])


def _metrics_to_json(metrics: PerformanceMetrics) -> dict[str, Any]:
    out = dataclasses.asdict(metrics)
    out["timeframe"] = metrics.timeframe.value
    return out


def _metrics_from_json(data: Mapping[str, Any]) -> PerformanceMetrics | None:
    names = {f.name for f in dataclasses.fields(PerformanceMetrics)}
    if not names.issubset({*data, "benchmark"}):
        return None
    values = {k: v for k, v in data.items() if k in names}
    values["timeframe"] = Timeframe(values["timeframe"])
    bench = values.get("benchmark")
    values["benchmark"] = None if bench is None else BenchmarkComparison(**bench)
    return PerformanceMetrics(**values)
