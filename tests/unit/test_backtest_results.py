"""Tests für das Speichern und Laden von Backtest-Ergebnissen (B-06)."""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.engine import BacktestResult, run_backtest
from trading_helfer.backtest.metrics import compute_metrics, metrics_for_result
from trading_helfer.backtest.results import (
    BacktestRunStore,
    DataSnapshot,
    RunFilter,
    RunNotFoundError,
    data_snapshot,
    strategy_source_hash,
)
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.data.storage import MarketStore
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    Params,
    Sizing,
    Strategy,
)

TIMEFRAMES = [Timeframe.D1, Timeframe.D3, Timeframe.W1, Timeframe.MN1]


def _bars(close: np.ndarray, start: str = "2020-01-02") -> pd.DataFrame:
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(close)),
            "open": opens,
            "high": np.maximum(opens, close) * 1.01,
            "low": np.minimum(opens, close) * 0.99,
            "close": close,
            "volume": 1000,
        }
    )


def _sawtooth(cycles: int = 6, half: int = 7) -> np.ndarray:
    up = np.linspace(100.0, 120.0, half, endpoint=False)
    down = np.linspace(120.0, 100.0, half, endpoint=False)
    return np.tile(np.concatenate([up, down]), cycles)


class SmaCross(Strategy):
    id = "test_sma_cross"
    name = "SMA-Kreuzung"
    version = "1.0.0"
    description = "SMA-Kreuzung für Tests."
    timeframes: ClassVar[list[Timeframe]] = TIMEFRAMES
    params: ClassVar[dict[str, Param]] = {
        "fast": Param(3, 1, 50, 1, "schnell"),
        "slow": Param(8, 2, 200, 1, "langsam"),
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        close = bars["close"]
        return pd.DataFrame(
            {"fast": close.rolling(p.fast).mean(), "slow": close.rolling(p.slow).mean()}
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and ctx.crossed_above("fast", "slow"):
            return [ctx.buy(sizing=Sizing.percent_equity(100), tag="Kreuzung oben")]
        if ctx.position is not None and ctx.crossed_below("fast", "slow"):
            return ctx.exit(tag="Kreuzung unten")
        return []


class Idle(Strategy):
    id = "test_idle"
    name = "Untätig"
    version = "0.1.0"
    description = "Handelt nie."
    timeframes: ClassVar[list[Timeframe]] = TIMEFRAMES

    def on_bar(self, ctx: BarContext) -> list[Order]:
        return []


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def market() -> MarketStore:
    store = MarketStore(":memory:")
    yield store  # type: ignore[misc]
    store.close()


@pytest.fixture
def runs(market: MarketStore) -> BacktestRunStore:
    return BacktestRunStore(market)


def _run(
    settings: Settings,
    strategy: Strategy | None = None,
    params: dict[str, int] | None = None,
    timeframe: Timeframe = Timeframe.W1,
    symbol: str = "SAW",
) -> BacktestResult:
    return run_backtest(
        strategy or SmaCross(),
        params if params is not None else {"fast": 2, "slow": 5},
        symbol,
        timeframe,
        "2020-01-01",
        "2030-01-01",
        settings,
        bars=_bars(_sawtooth()),
    )


# ---------------------------------------------------------------------------
# Akzeptanzkriterien
# ---------------------------------------------------------------------------


def test_round_trip_gives_same_result(settings: Settings, runs: BacktestRunStore) -> None:
    result = _run(settings)
    assert len(result.trades) > 0  # aussagekräftiger Lauf
    assert result.open_positions
    metrics = metrics_for_result(result)
    run_id = runs.save_run(result, source_hash="abc")

    loaded = runs.load_run(run_id)

    assert loaded.run_id == run_id
    assert loaded.result.fingerprint() == result.fingerprint()
    pd.testing.assert_frame_equal(loaded.result.trades, result.trades)
    pd.testing.assert_frame_equal(loaded.result.equity, result.equity)
    pd.testing.assert_frame_equal(loaded.result.orders, result.orders)
    assert loaded.result.open_positions == result.open_positions
    assert loaded.result.log == result.log
    assert loaded.result.warmup_end == result.warmup_end
    assert loaded.result.params == result.params
    assert loaded.result.settings == result.settings
    assert loaded.metrics == metrics
    assert loaded.summary.strategy_source_hash == "abc"
    assert loaded.summary.total_return_pct == pytest.approx(result.total_return_pct)


def test_filter_by_strategy(settings: Settings, runs: BacktestRunStore) -> None:
    sma_a = runs.save_run(_run(settings))
    sma_b = runs.save_run(_run(settings, params={"fast": 2, "slow": 6}))
    idle = runs.save_run(_run(settings, strategy=Idle(), params={}))

    sma = runs.list_runs(RunFilter(strategy_id="test_sma_cross"))
    assert {s.run_id for s in sma} == {sma_a, sma_b}
    assert all(s.strategy_id == "test_sma_cross" for s in sma)
    assert [s.run_id for s in runs.list_runs(RunFilter(strategy_id="test_idle"))] == [idle]
    assert runs.list_runs(RunFilter(strategy_id="gibt_es_nicht")) == []
    assert len(runs.list_runs()) == 3


# ---------------------------------------------------------------------------
# Weitere Eigenschaften
# ---------------------------------------------------------------------------


def test_list_order_and_other_filters(settings: Settings, runs: BacktestRunStore) -> None:
    t0 = dt.datetime(2026, 1, 1, 12, 0)
    first = runs.save_run(_run(settings), created_at=t0)
    second = runs.save_run(
        _run(settings, timeframe=Timeframe.D3, symbol="ABC"),
        created_at=t0 + dt.timedelta(hours=1),
    )
    third = runs.save_run(_run(settings), created_at=t0 + dt.timedelta(hours=2), universe="SP500")

    assert [s.run_id for s in runs.list_runs()] == [third, second, first]
    assert [s.run_id for s in runs.list_runs(RunFilter(limit=1))] == [third]
    assert [s.run_id for s in runs.list_runs(RunFilter(timeframe="3D"))] == [second]
    assert [s.run_id for s in runs.list_runs(RunFilter(universe="ABC"))] == [second]
    assert [s.run_id for s in runs.list_runs(RunFilter(universe="SP500"))] == [third]
    assert [s.run_id for s in runs.list_runs(RunFilter(strategy_version="1.0.0"))] == [
        third,
        second,
        first,
    ]
    window = RunFilter(
        created_from=t0 + dt.timedelta(minutes=30), created_to=t0 + dt.timedelta(hours=1)
    )
    assert [s.run_id for s in runs.list_runs(window)] == [second]
    with pytest.raises(ValueError, match="limit"):
        runs.list_runs(RunFilter(limit=0))


def test_summary_contents(settings: Settings, runs: BacktestRunStore) -> None:
    result = _run(settings)
    created = dt.datetime(2026, 2, 3, 4, 5, 6)
    run_id = runs.save_run(
        result,
        snapshot=DataSnapshot(dt.date(2024, 12, 31), 503),
        created_at=created,
        source_hash=strategy_source_hash(SmaCross),
    )
    (summary,) = runs.list_runs()
    assert summary.run_id == run_id
    assert summary.timeframe is Timeframe.W1
    assert summary.universe == "SAW"
    assert summary.start == dt.date(2020, 1, 1)
    assert summary.end == dt.date(2030, 1, 1)
    assert summary.params == {"fast": 2, "slow": 5}
    assert summary.settings["capital"] == 1000.0
    assert summary.created_at == created
    assert summary.data == DataSnapshot(dt.date(2024, 12, 31), 503)
    assert summary.strategy_source_hash == strategy_source_hash(SmaCross())
    assert summary.metrics["num_trades"] == len(result.trades)


def test_delete_run(settings: Settings, runs: BacktestRunStore, market: MarketStore) -> None:
    keep = runs.save_run(_run(settings))
    gone = runs.save_run(_run(settings))

    assert runs.delete_run(gone) is True
    assert runs.delete_run(gone) is False
    assert [s.run_id for s in runs.list_runs()] == [keep]
    with pytest.raises(RunNotFoundError, match="Kein gespeicherter Lauf"):
        runs.load_run(gone)
    for table in ("backtest_trades", "backtest_equity"):
        rows = market.connection.execute(
            f"SELECT count(*) FROM {table} WHERE run_id = ?", [gone]
        ).fetchall()
        assert rows == [(0,)]
    assert runs.load_run(keep).result.trades.shape[0] > 0


def test_run_without_trades(settings: Settings, runs: BacktestRunStore) -> None:
    result = _run(settings, strategy=Idle(), params={})
    assert result.trades.empty
    assert result.orders.empty
    loaded = runs.load_run(runs.save_run(result))
    assert loaded.result.fingerprint() == result.fingerprint()
    pd.testing.assert_frame_equal(loaded.result.trades, result.trades)
    pd.testing.assert_frame_equal(loaded.result.orders, result.orders)


def test_metrics_with_benchmark_round_trip(settings: Settings, runs: BacktestRunStore) -> None:
    result = _run(settings)
    bench = pd.Series(
        np.linspace(100.0, 130.0, len(result.equity)),
        index=pd.DatetimeIndex(result.equity["date"]),
    )
    metrics = compute_metrics(
        result.equity, result.trades, result.timeframe, initial_capital=1000.0, benchmark=bench
    )
    assert metrics.benchmark is not None
    loaded = runs.load_run(runs.save_run(result, metrics=metrics))
    assert loaded.metrics == metrics


def test_metrics_missing_when_not_computable(settings: Settings, runs: BacktestRunStore) -> None:
    result = _run(settings)
    empty = replace(result, equity=result.equity.iloc[0:0])
    loaded = runs.load_run(runs.save_run(empty))
    assert loaded.metrics is None
    assert loaded.summary.metrics == {}
    assert loaded.summary.total_return_pct is None
    assert loaded.result.equity.empty


def test_run_id_handling(settings: Settings, runs: BacktestRunStore) -> None:
    rid = uuid.UUID("12345678-1234-5678-1234-567812345678")
    assert runs.save_run(_run(settings), run_id=rid) == str(rid)
    assert runs.load_run(rid).run_id == str(rid)
    assert runs.load_run(str(rid).upper()).run_id == str(rid)
    with pytest.raises(ValueError, match="bereits gespeichert"):
        runs.save_run(_run(settings), run_id=rid)
    with pytest.raises(ValueError, match="Ungültige Lauf-ID"):
        runs.load_run("kein-uuid")
    with pytest.raises(ValueError, match="Ungültige Lauf-ID"):
        runs.delete_run("kein-uuid")
    with pytest.raises(RunNotFoundError) as info:
        runs.load_run(uuid.UUID(int=1))
    assert "Kein gespeicherter Lauf" in str(info.value)
    assert str(RunNotFoundError()) == "Lauf nicht gefunden."


def test_persists_across_reopen(settings: Settings, tmp_path: Path) -> None:
    result = _run(settings)
    path = tmp_path / "runs.duckdb"
    with MarketStore(path) as market:
        run_id = BacktestRunStore(market).save_run(result)
    with MarketStore(path, read_only=True) as market:
        loaded = BacktestRunStore(market).load_run(run_id)
    assert loaded.result.fingerprint() == result.fingerprint()


def test_data_snapshot_reads_bars(market: MarketStore, settings: Settings) -> None:
    assert data_snapshot(market) == DataSnapshot(None, 0)
    market.upsert_bars("AAA", _bars(np.linspace(10.0, 20.0, 5)))
    market.upsert_bars("BBB", _bars(np.linspace(10.0, 20.0, 7)))
    snap = data_snapshot(market)
    assert snap == DataSnapshot(dt.date(2020, 1, 10), 2)

    runs = BacktestRunStore(market)
    run_id = runs.save_run(_run(settings))
    assert runs.load_run(run_id).summary.data == snap


def test_strategy_source_hash() -> None:
    by_class = strategy_source_hash(SmaCross)
    assert by_class is not None
    assert len(by_class) == 64
    assert strategy_source_hash(SmaCross()) == by_class
    # Klassen derselben Datei teilen den Hash der Datei.
    assert strategy_source_hash(Idle) == by_class

    namespace: dict[str, object] = {}
    exec(  # Klasse ohne Quelldatei
        "from trading_helfer.strategies.base import Strategy\n"
        "class Dyn(Strategy):\n"
        "    id = 'dyn'\n"
        "    def on_bar(self, ctx):\n"
        "        return []\n",
        namespace,
    )
    dyn = namespace["Dyn"]
    assert isinstance(dyn, type)
    dyn.__module__ = "nicht_vorhanden"
    assert strategy_source_hash(dyn) is None


def test_legacy_row_without_details(runs: BacktestRunStore, market: MarketStore) -> None:
    rid = str(uuid.UUID(int=7))
    market.connection.execute(
        """
        INSERT INTO backtest_runs (run_id, strategy_id, strategy_version, params_json, universe,
            timeframe, start, "end", settings_json, metrics_json)
        VALUES (?, 'x', NULL, NULL, NULL, '1W', '2020-01-01', '2020-12-31', '{"capital": 500.0}',
            '{"total_return_pct": 1.0}')
        """,
        [rid],
    )
    market.connection.execute(
        "INSERT INTO backtest_equity VALUES (?, '2020-06-30', 505.0, 505.0, 0.0, 0.0)", [rid]
    )
    loaded = runs.load_run(rid)
    assert loaded.result.initial_capital == 500.0
    assert loaded.result.final_equity == 505.0
    assert loaded.result.orders.empty
    assert loaded.metrics is None  # unvollständige Kennzahlen
    assert loaded.summary.total_return_pct == 1.0
    assert loaded.summary.params == {}
    assert loaded.summary.data == DataSnapshot(None, 0)


def test_params_with_numpy_and_dates(settings: Settings, runs: BacktestRunStore) -> None:
    result = replace(
        _run(settings), params={"n": np.int64(3), "x": np.float64(0.5), "d": dt.date(2020, 1, 2)}
    )
    loaded = runs.load_run(runs.save_run(result))
    assert loaded.summary.params == {"n": 3, "x": 0.5, "d": "2020-01-02"}
    with pytest.raises(TypeError, match="Nicht als JSON speicherbar"):
        runs.save_run(replace(result, params={"o": object()}))
    assert len(runs.list_runs()) == 1  # fehlgeschlagener Lauf hinterlässt nichts
