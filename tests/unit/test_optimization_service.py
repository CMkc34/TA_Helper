"""Tests für den OptimizationService (G-07): Raster, Walk-Forward, Monte Carlo, Abbruch."""

from __future__ import annotations

import datetime as dt
import threading
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.engine import BacktestError
from trading_helfer.backtest.monte_carlo import MonteCarloMethod
from trading_helfer.backtest.optimize import Objective, OptimizationError, ParamRange
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.services.backtest_service import BacktestService, UniverseKind
from trading_helfer.services.optimization_service import (
    OptimizationCancelledError,
    OptimizationMode,
    OptimizationRequest,
    OptimizationService,
)
from trading_helfer.strategies.base import BarContext, Order, Param, Params, Sizing, Strategy
from trading_helfer.strategies.registry import StrategyInfo, StrategyRegistry


def _bars(close: np.ndarray, start: str = "2014-01-02") -> pd.DataFrame:
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


def _sawtooth(cycles: int, half: int = 7) -> np.ndarray:
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


class Idle(Cross):
    """Handelt nie: Monte Carlo hat keine Trades."""

    id = "test_idle"

    def on_bar(self, ctx: BarContext) -> list[Order]:
        return []


BARS = {
    "AAA": _bars(_sawtooth(110)),  # ca. 2014–2019 (6 Jahre)
    "MSFT": _bars(_sawtooth(110) * 2),
}


def _loader(symbol: str) -> pd.DataFrame:
    return BARS.get(symbol, pd.DataFrame(columns=["date", "open", "high", "low", "close"]))


def _registry() -> StrategyRegistry:
    registry = StrategyRegistry(None, include_builtin=False)
    for cls in (Cross, Idle):
        registry._strategies[cls.id] = StrategyInfo(cls.id, cls, "test")
    return registry


@pytest.fixture
def service(tmp_path: Path) -> OptimizationService:
    settings = Settings(data_dir=tmp_path / "data")
    backtest = BacktestService(
        settings, _registry(), bars_loader=_loader, symbols_loader=lambda: sorted(BARS)
    )
    return OptimizationService(backtest)


def _request(**kw: object) -> OptimizationRequest:
    base: dict[str, object] = {
        "strategy_id": "test_cross",
        "ranges": {"fast": ParamRange(2, 3, 1), "slow": ParamRange(5, 8, 3)},
        "universe": "AAA",
        "timeframe": Timeframe.D1,
        "start": dt.date(2014, 1, 1),
        "end": dt.date(2016, 12, 31),
        "mc_simulations": 200,
        "workers": 1,
    }
    base.update(kw)
    return OptimizationRequest(**base)  # type: ignore[arg-type]


def test_strategy_helpers(service: OptimizationService) -> None:
    assert [i.id for i in service.strategies()] == ["test_cross", "test_idle"]
    assert set(service.strategy_params("test_cross")) == {"fast", "slow"}
    assert service.strategy_timeframes("test_cross") == [Timeframe.D1, Timeframe.W1]
    assert service.strategy_description("test_cross") == "SMA-Kreuzung für Tests."
    assert service.settings is service.backtest_service.settings
    assert service.count_combinations("test_cross", {"fast": ParamRange(1, 5, 1)}) == 5
    assert service.count_combinations("test_cross", {}) == 1
    with pytest.raises(OptimizationError):
        service.count_combinations("test_cross", {"fast": ParamRange(5, 1, 1)})
    service.reload_strategies()


def test_grid_run(service: OptimizationService) -> None:
    progress: list[tuple[int, str]] = []
    outcome = service.run(_request(), progress=lambda p, m: progress.append((p, m)))
    result = outcome.optimization
    assert len(result.table) == 4
    assert result.param_names == ["fast", "slow"]
    assert result.best_params is not None
    assert outcome.walk_forward is None
    assert outcome.best_result is not None
    assert outcome.best_result.params == result.best_params
    assert outcome.monte_carlo is not None
    assert outcome.monte_carlo.n_simulations == 200
    assert outcome.monte_carlo.n_trades == len(outcome.best_result.trades)
    percents = [p for p, _m in progress]
    assert percents == sorted(percents)
    assert percents[-1] == 100
    assert any("Optimierung: 4/4" in m for _p, m in progress)


def test_grid_run_is_deterministic(service: OptimizationService) -> None:
    a = service.run(_request())
    b = service.run(_request())
    pd.testing.assert_frame_equal(a.optimization.table, b.optimization.table)
    assert a.monte_carlo is not None
    assert b.monte_carlo is not None
    assert a.monte_carlo.fingerprint() == b.monte_carlo.fingerprint()


def test_inputs_are_passed_to_optimize(
    service: OptimizationService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_helfer.services import optimization_service as module

    seen: dict[str, object] = {}
    real = module.optimize

    def spy(*args: object, **kwargs: object) -> object:
        seen["args"] = args
        seen.update(kwargs)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "optimize", spy)
    service.run(
        _request(
            objective=Objective.CAGR,
            capital=5000.0,
            fee_per_order=2.0,
            slippage_pct=0.1,
            mc_simulations=0,
        )
    )
    args = seen["args"]
    assert isinstance(args, tuple)
    assert args[1] == {"fast": [2, 3], "slow": [5, 8]}
    assert args[2] == "AAA"
    assert args[3] is Timeframe.D1
    assert args[6].default_capital == 5000.0  # type: ignore[attr-defined]
    assert seen["objective"] is Objective.CAGR
    assert seen["universe"] is None
    costs = seen["costs"]
    assert costs.fee_per_order == 2.0  # type: ignore[attr-defined]
    assert costs.slippage_pct == 0.1  # type: ignore[attr-defined]


def test_monte_carlo_disabled_and_without_trades(service: OptimizationService) -> None:
    assert service.run(_request(mc_simulations=0)).monte_carlo is None
    outcome = service.run(_request(strategy_id="test_idle"))
    assert outcome.monte_carlo is None
    assert outcome.optimization.best_params is None or outcome.best_result is not None
    assert any("Monte Carlo" in line or "Kombination" in line for line in outcome.log)


def test_bootstrap_method(service: OptimizationService) -> None:
    outcome = service.run(_request(mc_method=MonteCarloMethod.BOOTSTRAP, seed=7))
    assert outcome.monte_carlo is not None
    assert outcome.monte_carlo.method is MonteCarloMethod.BOOTSTRAP
    assert outcome.monte_carlo.seed == 7


def test_walk_forward_run(service: OptimizationService) -> None:
    outcome = service.run(
        _request(
            mode=OptimizationMode.WALK_FORWARD,
            start=dt.date(2014, 1, 1),
            end=dt.date(2019, 12, 31),
            train_years=2,
            test_years=1,
        )
    )
    wf = outcome.walk_forward
    assert wf is not None
    assert len(wf.windows) == 4
    assert wf.train_years == 2
    assert outcome.optimization is wf.windows[-1].optimization
    assert not wf.equity.empty
    assert outcome.monte_carlo is not None
    total = sum(len(w.test_result.trades) for w in wf.windows)
    assert outcome.monte_carlo.n_trades == total


def test_index_universe(service: OptimizationService) -> None:
    outcome = service.run(
        _request(universe="ndx100", universe_kind=UniverseKind.INDEX, max_positions=2)
    )
    assert outcome.best_result is not None
    assert set(outcome.best_result.trades["symbol"]) <= {"MSFT"}


def test_errors(service: OptimizationService) -> None:
    with pytest.raises(BacktestError, match="Startkapital"):
        service.run(_request(capital=0.0))
    with pytest.raises(BacktestError, match="Symbol"):
        service.run(_request(universe=" "))
    with pytest.raises(BacktestError, match="keine Kurse"):
        service.run(_request(universe="ZZZ"))
    with pytest.raises(BacktestError, match="Unbekannter Index"):
        service.run(_request(universe="DAX", universe_kind=UniverseKind.INDEX))
    with pytest.raises(OptimizationError):
        service.run(_request(ranges={"gibt_es_nicht": None}))


def test_cancel_before_start(service: OptimizationService) -> None:
    with pytest.raises(OptimizationCancelledError):
        service.run(_request(), should_cancel=lambda: True)


@pytest.mark.parametrize("workers", [1, 2])
def test_cancel_during_grid(service: OptimizationService, workers: int) -> None:
    event = threading.Event()
    done: list[int] = []

    def progress(percent: int, message: str) -> None:
        if "/" in message:
            done.append(percent)
            event.set()  # nach der ersten Kombination abbrechen

    request = _request(ranges={"fast": ParamRange(1, 6, 1), "slow": ParamRange(5, 10, 1)})
    with pytest.raises(OptimizationCancelledError, match="abgebrochen"):
        service.run(
            request.__class__(**{**request.__dict__, "workers": workers}),
            progress=progress,
            should_cancel=event.is_set,
        )
    assert len(done) == 1


def test_cancel_during_walk_forward(service: OptimizationService) -> None:
    calls = {"n": 0}

    def should_cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(OptimizationCancelledError):
        service.run(
            _request(
                mode=OptimizationMode.WALK_FORWARD,
                end=dt.date(2019, 12, 31),
                train_years=2,
            ),
            should_cancel=should_cancel,
        )
