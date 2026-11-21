"""Tests der Monte-Carlo-Analyse (B-09)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import run_backtest
from trading_helfer.backtest.monte_carlo import (
    DEFAULT_SIMULATIONS,
    MonteCarloError,
    MonteCarloMethod,
    Percentiles,
    monte_carlo,
    monte_carlo_for_result,
    trade_returns,
)
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.strategies.base import BarContext, Order, Sizing, Strategy

RETURNS = [0.10, -0.05, 0.20, -0.15, 0.03, -0.08, 0.12, 0.01, -0.02, 0.07]


@pytest.mark.parametrize("method", list(MonteCarloMethod))
def test_deterministic_with_seed(method: MonteCarloMethod) -> None:
    a = monte_carlo(RETURNS, 1000.0, method=method, seed=7)
    b = monte_carlo(RETURNS, 1000.0, method=method, seed=7)
    assert a.fingerprint() == b.fingerprint()
    assert a.final_equity.tobytes() == b.final_equity.tobytes()
    assert a.max_drawdown_pct.tobytes() == b.max_drawdown_pct.tobytes()
    assert a.summary() == b.summary()


def test_different_seed_changes_result() -> None:
    a = monte_carlo(RETURNS, 1000.0, method="bootstrap", seed=1)
    b = monte_carlo(RETURNS, 1000.0, method="bootstrap", seed=2)
    assert a.fingerprint() != b.fingerprint()


@pytest.mark.parametrize("method", list(MonteCarloMethod))
def test_identical_trades_collapse_percentiles(method: MonteCarloMethod) -> None:
    mc = monte_carlo([0.05] * 8, 1000.0, method=method, seed=3)
    fe = mc.final_equity_percentiles
    dd = mc.max_drawdown_percentiles
    assert fe.p5 == fe.p50 == fe.p95 == pytest.approx(1000.0 * 1.05**8)
    assert dd.p5 == dd.p50 == dd.p95 == 0.0

    losing = monte_carlo([-0.1] * 4, 1000.0, method=method, seed=3)
    dd = losing.max_drawdown_percentiles
    assert dd.p5 == dd.p50 == dd.p95 == pytest.approx((0.9**4 - 1) * 100)
    assert losing.loss_probability_pct == 100.0


def test_defaults_and_shapes() -> None:
    mc = monte_carlo(RETURNS, 1000.0)
    assert mc.n_simulations == DEFAULT_SIMULATIONS == 1000
    assert mc.method is MonteCarloMethod.SHUFFLE
    assert mc.final_equity.shape == (1000,)
    assert mc.max_drawdown_pct.shape == (1000,)
    assert mc.n_trades == len(RETURNS)
    assert np.all(mc.max_drawdown_pct <= 0)


def test_shuffle_keeps_final_equity_and_varies_drawdown() -> None:
    mc = monte_carlo(RETURNS, 1000.0, method=MonteCarloMethod.SHUFFLE, seed=5)
    expected = 1000.0 * float(np.prod(1 + np.array(RETURNS)))
    np.testing.assert_allclose(mc.final_equity, expected, rtol=1e-12)
    assert mc.original_final_equity == pytest.approx(expected)
    dd = mc.max_drawdown_percentiles
    assert dd.p5 < dd.p50 < dd.p95 <= 0
    assert mc.max_drawdown_pct.min() <= mc.original_max_drawdown_pct


def test_bootstrap_varies_final_equity() -> None:
    mc = monte_carlo(RETURNS, 1000.0, method="bootstrap", seed=5)
    fe = mc.final_equity_percentiles
    assert fe.p5 < fe.p50 < fe.p95
    tr = mc.total_return_percentiles
    assert tr.p50 == pytest.approx((fe.p50 / 1000.0 - 1) * 100)
    assert 0 < mc.loss_probability_pct < 100


def test_original_path_drawdown() -> None:
    mc = monte_carlo([0.1, -0.5, 0.2], 1000.0, n_simulations=5)
    # 1000 → 1100 → 550 → 660: Drawdown vom Höchststand 1100 auf 550 = −50 %
    assert mc.original_max_drawdown_pct == pytest.approx(-50.0)
    assert mc.original_final_equity == pytest.approx(660.0)


def test_total_loss_is_floored_at_zero() -> None:
    mc = monte_carlo([0.5, -1.5], 1000.0, n_simulations=10, method="bootstrap", seed=0)
    assert np.all(mc.final_equity >= 0)
    assert mc.original_final_equity == 0.0
    assert mc.original_max_drawdown_pct == -100.0


def test_percentiles_linear() -> None:
    p = Percentiles.of(np.arange(101, dtype=float))
    assert (p.p5, p.p50, p.p95) == (5.0, 50.0, 95.0)


def test_summary_is_json() -> None:
    mc = monte_carlo(RETURNS, 1000.0, n_simulations=50)
    data = json.loads(json.dumps(mc.summary()))
    assert data["method"] == "shuffle"
    assert set(data["final_equity"]) == {"p5", "p50", "p95"}
    assert data["n_simulations"] == 50


@pytest.mark.parametrize(
    ("kwargs", "returns", "match"),
    [
        ({}, [], "Ohne abgeschlossene Trades"),
        ({}, [0.1, float("nan")], "ungültige Werte"),
        ({"n_simulations": 0}, [0.1], "Anzahl der Simulationen"),
        ({"n_simulations": True}, [0.1], "Anzahl der Simulationen"),
        ({"method": "jackknife"}, [0.1], "Unbekanntes Verfahren"),
    ],
)
def test_invalid_input(kwargs: dict[str, object], returns: list[float], match: str) -> None:
    with pytest.raises(MonteCarloError, match=match):
        monte_carlo(returns, 1000.0, **kwargs)  # type: ignore[arg-type]


def test_invalid_capital() -> None:
    with pytest.raises(MonteCarloError, match="Startkapital"):
        monte_carlo([0.1], 0.0)
    with pytest.raises(MonteCarloError, match="Startkapital"):
        trade_returns(pd.DataFrame({"pnl": [1.0]}), -5.0)


def test_trade_returns_account_based_and_sorted() -> None:
    trades = pd.DataFrame(
        {
            "exit_date": pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"]),
            "pnl": [-60.0, 100.0, 50.0],
        }
    )
    r = trade_returns(trades, 1000.0)
    # Reihenfolge nach Ausstieg: +100 (1000), +50 (1100), −60 (1150)
    np.testing.assert_allclose(r, [0.1, 50 / 1100, -60 / 1150])
    assert 1000.0 * np.prod(1 + r) == pytest.approx(1090.0)


def test_trade_returns_errors() -> None:
    with pytest.raises(MonteCarloError, match="pnl"):
        trade_returns(pd.DataFrame({"x": [1.0]}), 1000.0)
    with pytest.raises(MonteCarloError, match="NaN"):
        trade_returns(pd.DataFrame({"pnl": [np.nan]}), 1000.0)
    with pytest.raises(MonteCarloError, match="null"):
        trade_returns(pd.DataFrame({"pnl": [-1000.0, 5.0]}), 1000.0)


class _InAndOut(Strategy):
    id = "test_in_and_out"
    name = "Rein und raus"
    version = "1.0.0"
    description = "Kauft alle fünf Kerzen und verkauft zwei Kerzen später."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1]

    def __init__(self) -> None:
        self._bars = 0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        self._bars += 1
        if ctx.position is None and self._bars % 5 == 1:
            return [ctx.buy(sizing=Sizing.percent_equity(50))]
        if ctx.position is not None and self._bars % 5 == 3:
            return ctx.exit(tag="Zeit")
        return []


def test_monte_carlo_for_engine_result(tmp_path: Path) -> None:
    n = 400
    t = np.arange(n)
    close = 100.0 + 10.0 * np.sin(t / 7.0) + 0.05 * t
    opens = np.concatenate([[close[0]], close[:-1]])
    bars = pd.DataFrame(
        {
            "date": pd.bdate_range("2015-01-05", periods=n),
            "open": opens,
            "high": np.maximum(opens, close) * 1.01,
            "low": np.minimum(opens, close) * 0.99,
            "close": close,
            "volume": 1000,
        }
    )
    result = run_backtest(
        _InAndOut(),
        None,
        "TEST",
        Timeframe.W1,
        "2015-01-05",
        "2016-12-31",
        Settings(data_dir=tmp_path),
        bars=bars,
        costs=CostModel(fee_per_order=1.0, slippage_pct=0.0),
    )
    assert len(result.trades) >= 5
    mc = monte_carlo_for_result(result, seed=11)
    closed_equity = result.initial_capital + float(result.trades["pnl"].sum())
    assert mc.original_final_equity == pytest.approx(closed_equity)
    np.testing.assert_allclose(mc.final_equity, closed_equity, rtol=1e-9)
    again = monte_carlo_for_result(result, seed=11)
    assert again.fingerprint() == mc.fingerprint()
    boot = monte_carlo_for_result(result, method="bootstrap", n_simulations=200, seed=11)
    assert boot.n_simulations == 200
