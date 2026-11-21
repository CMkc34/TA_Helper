"""Tests für den BacktestService (G-06): Lauf, Universum, Speichern, Vergleich, Export."""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.engine import (
    EQUITY_COLUMNS,
    ORDER_COLUMNS,
    TRADE_COLUMNS,
    BacktestError,
    BacktestResult,
)
from trading_helfer.backtest.metrics import compute_metrics
from trading_helfer.backtest.portfolio import PortfolioResult
from trading_helfer.backtest.results import RunNotFoundError
from trading_helfer.core.config import Settings
from trading_helfer.core.markers import MarkerKind
from trading_helfer.core.types import Timeframe
from trading_helfer.services.backtest_service import (
    COMPARE_METRICS,
    BacktestOutcome,
    BacktestRequest,
    BacktestService,
    UniverseKind,
    compare_runs,
    format_metric_value,
    metric_value,
    trade_markers,
    trades_to_csv,
)
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    ParamError,
    Params,
    Sizing,
    Strategy,
)
from trading_helfer.strategies.registry import (
    StrategyInfo,
    StrategyNotFoundError,
    StrategyRegistry,
)


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


def _sawtooth(cycles: int = 8, half: int = 7) -> np.ndarray:
    up = np.linspace(100.0, 120.0, half, endpoint=False)
    down = np.linspace(120.0, 100.0, half, endpoint=False)
    return np.tile(np.concatenate([up, down]), cycles)


class Cross(Strategy):
    id = "test_cross"
    name = "Test-Kreuzung"
    version = "1.0.0"
    description = "SMA-Kreuzung für Tests."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        "fast": Param(2, 1, 20, 1, "schnell"),
        "slow": Param(5, 2, 50, 1, "langsam"),
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        close = bars["close"]
        return pd.DataFrame(
            {"f": close.rolling(p.fast).mean(), "s": close.rolling(p.slow).mean()},
            index=bars.index,
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and ctx.crossed_above("f", "s"):
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        if ctx.position is not None and ctx.crossed_below("f", "s"):
            return ctx.exit()
        return []


BARS = {
    "AAA": _bars(_sawtooth()),
    "MSFT": _bars(_sawtooth() * 2),
    "SPY": _bars(np.linspace(100.0, 130.0, len(_sawtooth()))),
}


def _registry() -> StrategyRegistry:
    registry = StrategyRegistry(None, include_builtin=False)
    registry._strategies["test_cross"] = StrategyInfo("test_cross", Cross, "test")
    return registry


def _loader(symbol: str) -> pd.DataFrame:
    return BARS.get(symbol, pd.DataFrame(columns=["date", "open", "high", "low", "close"]))


@pytest.fixture
def service(tmp_path: Path) -> BacktestService:
    settings = Settings(data_dir=tmp_path / "data")
    return BacktestService(
        settings, _registry(), bars_loader=_loader, symbols_loader=lambda: sorted(BARS)
    )


def _request(**kw: object) -> BacktestRequest:
    base: dict[str, object] = {
        "strategy_id": "test_cross",
        "params": {"fast": 2, "slow": 5},
        "universe": "AAA",
        "timeframe": Timeframe.D1,
        "start": dt.date(2020, 1, 1),
        "end": dt.date(2021, 1, 1),
    }
    base.update(kw)
    return BacktestRequest(**base)  # type: ignore[arg-type]


def test_run_single_symbol_saves_and_reloads(service: BacktestService) -> None:
    progress: list[int] = []
    outcome = service.run(_request(), progress=lambda p, _m: progress.append(p))
    assert progress[-1] == 100
    assert outcome.run_id is not None
    assert outcome.metrics is not None
    assert outcome.metrics.num_trades == len(outcome.result.trades) > 0
    assert outcome.metrics.benchmark is not None
    assert outcome.monthly is not None
    assert outcome.symbols == ["AAA"]
    bench = outcome.benchmark_equity()
    assert bench is not None
    assert len(bench) == len(outcome.result.equity)

    runs = service.list_runs()
    assert [r.run_id for r in runs] == [outcome.run_id]
    loaded = service.load_run(outcome.run_id)
    assert loaded.result.fingerprint() == outcome.result.fingerprint()
    assert loaded.benchmark_symbol == "SPY"
    assert loaded.benchmark is not None
    assert service.delete_run(outcome.run_id)
    assert service.list_runs() == []
    with pytest.raises(RunNotFoundError):
        service.load_run(outcome.run_id)


def test_costs_and_capital_are_used(service: BacktestService) -> None:
    cheap = service.run(_request(save=False, fee_per_order=0.0, slippage_pct=0.0))
    dear = service.run(_request(save=False, fee_per_order=5.0, capital=2000.0))
    assert cheap.run_id is None
    assert dear.result.initial_capital == 2000.0
    assert dear.result.trades["fees"].sum() > cheap.result.trades["fees"].sum()


def test_run_without_benchmark(service: BacktestService) -> None:
    outcome = service.run(_request(save=False, benchmark=""))
    assert outcome.benchmark is None
    assert outcome.benchmark_equity() is None
    assert outcome.metrics is not None
    assert outcome.metrics.benchmark is None
    missing = service.run(_request(save=False, benchmark="NOPE"))
    assert missing.benchmark is None


def test_run_errors(service: BacktestService) -> None:
    with pytest.raises(BacktestError, match="Symbol"):
        service.run(_request(universe=" "))
    with pytest.raises(BacktestError, match="keine Kurse"):
        service.run(_request(universe="ZZZ"))
    with pytest.raises(BacktestError, match="Startkapital"):
        service.run(_request(capital=0.0))
    with pytest.raises(StrategyNotFoundError):
        service.run(_request(strategy_id="gibt_es_nicht"))
    with pytest.raises(ParamError):
        service.run(_request(params={"fast": 999}))


def test_run_index_universe(service: BacktestService) -> None:
    # Vom Nasdaq-100 hat der Fake-Speicher nur MSFT; AAA ist kein Mitglied.
    progress: list[str] = []
    outcome = service.run(
        _request(
            universe="ndx100",
            universe_kind=UniverseKind.INDEX,
            capital=1000.0,
            max_positions=3,
        ),
        progress=lambda _p, m: progress.append(m),
    )
    result = outcome.result
    assert isinstance(result, PortfolioResult)
    assert outcome.universe == "NDX100"
    assert result.symbols == ("MSFT",)
    assert result.max_positions == 3
    assert result.initial_capital == 1000.0
    assert set(result.trades["symbol"]) == {"MSFT"}
    assert any("Portfolio-Backtest" in m for m in progress)
    assert service.list_runs()[0].universe == "NDX100"
    loaded = service.load_run(outcome.run_id or "").result
    assert loaded.final_equity == pytest.approx(result.final_equity)
    assert len(loaded.trades) == len(result.trades)


def test_run_index_errors(service: BacktestService, tmp_path: Path) -> None:
    with pytest.raises(BacktestError, match="Unbekannter Index"):
        service.run(_request(universe="DAX", universe_kind=UniverseKind.INDEX))
    empty = BacktestService(
        service.settings, _registry(), bars_loader=_loader, symbols_loader=lambda: []
    )
    with pytest.raises(BacktestError, match="keine Kurse"):
        empty.run(_request(universe="SP500", universe_kind=UniverseKind.INDEX))
    broken = BacktestService(
        service.settings,
        _registry(),
        bars_loader=lambda s: _loader("ZZZ"),
        symbols_loader=lambda: ["MSFT"],
    )
    with pytest.raises(BacktestError, match="Keine Kurse"):
        broken.run(_request(universe="NDX100", universe_kind=UniverseKind.INDEX))


def test_reload_strategies(tmp_path: Path) -> None:
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    svc = BacktestService(Settings(data_dir=tmp_path / "d", plugins_dir=plugins))
    assert "plug_cross" not in [s.id for s in svc.strategies()]
    (plugins / "bad.py").write_text("this is not python", encoding="utf-8")
    (plugins / "plug.py").write_text(
        (Path(__file__).parents[1] / "fixtures/good_strategies/sma_cross.py").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    svc.reload_strategies()
    assert len(svc.strategies()) > 3


def test_strategy_queries_and_defaults(service: BacktestService) -> None:
    assert [s.id for s in service.strategies()] == ["test_cross"]
    assert set(service.strategy_params("test_cross")) == {"fast", "slow"}
    assert service.strategy_timeframes("test_cross") == [Timeframe.D1, Timeframe.W1]
    req = service.default_request("test_cross", "AAA")
    assert req.params == {"fast": 2, "slow": 5}
    assert req.timeframe is Timeframe.D1
    assert req.capital == service.settings.default_capital
    assert req.fee_per_order == service.settings.fee_per_order


def test_default_service_without_database(tmp_path: Path) -> None:
    svc = BacktestService(Settings(data_dir=tmp_path / "leer"))
    assert "trend_sma_cross" in [s.id for s in svc.strategies()]
    assert svc.list_runs() == []
    with pytest.raises(BacktestError, match="keine Kurse"):
        svc.run(
            BacktestRequest(
                "trend_sma_cross",
                {},
                "AAPL",
                Timeframe.W1,
                dt.date(2020, 1, 1),
                dt.date(2021, 1, 1),
            )
        )
    assert svc._symbols_from_store() == []


def test_markers_and_csv(service: BacktestService, tmp_path: Path) -> None:
    outcome = service.run(_request(save=False))
    trades = outcome.result.trades
    markers = trade_markers(trades)
    closed = int(trades["exit_date"].notna().sum())
    assert len(markers) == len(trades) + closed
    assert markers[0].kind is MarkerKind.BUY
    assert markers[0].price == pytest.approx(float(trades["entry_price"].iloc[0]))
    assert any(m.kind is MarkerKind.EXIT for m in markers)
    assert trade_markers(trades, "OTHER") == []
    short = trades.copy()
    short["side"] = "SHORT"
    assert trade_markers(short)[0].kind is MarkerKind.SELL

    path = tmp_path / "trades.csv"
    assert service.export_trades(outcome, path) == len(trades)
    text = path.read_text(encoding="utf-8-sig")
    header = text.splitlines()[0]
    assert header.startswith("symbol;side;entry_date")
    assert "," in text.splitlines()[1]  # Dezimalkomma
    back = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    assert len(back) == len(trades)
    assert trades_to_csv(trades.iloc[:0], tmp_path / "leer.csv") == 0


def test_compare_and_format(service: BacktestService) -> None:
    a = service.run(_request(save=False)).metrics
    b = service.run(_request(save=False, universe="MSFT")).metrics
    rows = compare_runs(a, b)
    assert len(rows) == len(COMPARE_METRICS)
    assert rows[0][0] == "Gesamtrendite"
    none_rows = compare_runs(a, None)
    assert all(r[2] == "–" and r[3] == "" for r in none_rows)
    assert metric_value(None, "sharpe") is None
    assert format_metric_value(1234.5, "money") == "1.234,50 $"
    assert format_metric_value(-12.345, "pct") == "-12,35 %"
    assert format_metric_value(3.0, "int") == "3"
    assert format_metric_value(2.25, "days") == "2,2"
    assert format_metric_value(None, "ratio") == "–"
    assert format_metric_value(float("nan"), "ratio") == "–"


def test_outcome_without_equity() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-02", periods=3),
            "equity": [100.0, 101.0, 102.0],
            "cash": 100.0,
            "exposure": 0.0,
            "drawdown": 0.0,
        }
    )
    metrics = compute_metrics(equity, None, Timeframe.D1, initial_capital=100.0)
    assert metrics.num_trades == 0
    result = replace(_empty_result(), equity=equity.iloc[:0], final_equity=100.0)
    outcome = BacktestOutcome(result, None, "X", benchmark=BARS["SPY"].set_index("date")["close"])
    assert outcome.symbols == []
    assert outcome.benchmark_equity() is None


def _empty_result() -> BacktestResult:
    return BacktestResult(
        strategy_id="x",
        strategy_version="1",
        params={},
        symbol="X",
        timeframe=Timeframe.D1,
        start=dt.date(2020, 1, 1),
        end=dt.date(2020, 2, 1),
        settings={},
        initial_capital=100.0,
        final_equity=100.0,
        trades=pd.DataFrame(columns=list(TRADE_COLUMNS)),
        equity=pd.DataFrame(columns=list(EQUITY_COLUMNS)),
        orders=pd.DataFrame(columns=list(ORDER_COLUMNS)),
    )
