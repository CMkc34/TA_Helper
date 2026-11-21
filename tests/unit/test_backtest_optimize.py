"""Tests der Parameter-Optimierung (B-08)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest import optimize as optimize_module
from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import BacktestError, run_backtest
from trading_helfer.backtest.metrics import compute_metrics
from trading_helfer.backtest.optimize import (
    METRIC_COLUMNS,
    Objective,
    OptimizationError,
    ParamRange,
    build_grid,
    count_combinations,
    grid_search,
    objective_value,
    optimize,
    robustness_scores,
)
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    ParamError,
    Params,
    Sizing,
    Strategy,
)
from trading_helfer.strategies.builtin.sma_cross import SmaCross

NO_COSTS = CostModel(fee_per_order=0.0, slippage_pct=0.0)
FIXTURE = Path(__file__).parents[1] / "fixtures" / "strategies" / "SYNTH_long_daily.csv"


def _bars(close: np.ndarray | list[float], start: str = "2020-01-02") -> pd.DataFrame:
    """Tageskurse an Werktagen; Open = Vortages-Close, High/Low ±1 %."""
    close_arr = np.asarray(close, dtype=float)
    opens = np.concatenate([[close_arr[0]], close_arr[:-1]])
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(close_arr)),
            "open": opens,
            "high": np.maximum(opens, close_arr) * 1.01,
            "low": np.minimum(opens, close_arr) * 0.99,
            "close": close_arr,
            "volume": 1000,
        }
    )


def _tent(peak: int = 50, length: int = 101) -> np.ndarray:
    """Steigt bis Kerze ``peak`` von 100 auf 200 und fällt danach wieder."""
    up = np.linspace(100.0, 200.0, peak + 1)
    down = np.linspace(200.0, 100.0, length - peak)[1:]
    return np.concatenate([up, down])


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", fee_per_order=0.0, slippage_pct=0.0)


class HoldBars(Strategy):
    """Kauft auf der ersten Kerze und verkauft, wenn ``hold`` Kerzen gesehen wurden."""

    id = "test_hold_bars"
    name = "Halten über n Kerzen"
    version = "1.0.0"
    description = "Testet die Optimierung: bestes hold = Spitze der Zeltkurve."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        "hold": Param(10, 1, 500, 5, "Haltedauer in Kerzen"),
        "fail_above": Param(1000, 1, 1000, 1, "Fehler ab diesem hold (Test)"),
    }

    def on_bar(self, ctx: BarContext) -> list[Order]:
        p = self.current_params
        if p.hold >= p.fail_above:
            raise RuntimeError("absichtlicher Fehler")
        t = len(ctx.bars) - 1
        if t == 0:
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        if ctx.position is not None and t >= p.hold:
            return ctx.exit()
        return []

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        self.current_params = p
        return pd.DataFrame(index=bars.index)


class Never(Strategy):
    id = "test_never"
    name = "Handelt nie"
    version = "1.0.0"
    description = "Keine Trades; Sharpe nicht definiert."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1]
    params: ClassVar[dict[str, Param]] = {"x": Param(1, 1, 3, 1, "egal")}

    def on_bar(self, ctx: BarContext) -> list[Order]:
        return []


class Mixed(Strategy):
    id = "test_mixed"
    name = "Gemischte Parameter"
    version = "1.0.0"
    description = "Parameter aller Typen."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1]
    params: ClassVar[dict[str, Param]] = {
        "n": Param(5, 2, 10, 2, "ganzzahlig"),
        "f": Param(0.5, 0.1, 0.5, 0.1, "Kommazahl"),
        "flag": Param(False, description="Schalter"),
        "mode": Param("a", description="Auswahl", choices=("a", "b", "c")),
        "free": Param(3, description="ohne Grenzen"),
        "nostep": Param(0.5, 0.0, 1.0, description="ohne Schritt"),
    }

    def on_bar(self, ctx: BarContext) -> list[Order]:
        return []


# ---------------------------------------------------------------------------
# Raster
# ---------------------------------------------------------------------------


def test_build_grid_from_param_definitions() -> None:
    grid = build_grid(SmaCross)
    assert grid["fast"] == list(range(2, 51))
    assert grid["slow"] == list(range(10, 201, 5))
    assert all(isinstance(v, int) for v in grid["slow"])
    assert count_combinations(grid) == 49 * 39


def test_build_grid_overrides_and_fixed_defaults() -> None:
    grid = build_grid(Mixed, {"n": None, "f": None, "flag": None, "mode": None})
    assert grid["n"] == [2, 4, 6, 8, 10]
    assert grid["f"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert grid["flag"] == [False, True]
    assert grid["mode"] == ["a", "b", "c"]
    assert grid["free"] == [3]
    assert grid["nostep"] == [0.5]
    assert list(grid) == list(Mixed.params)

    grid = build_grid(Mixed, {"n": ParamRange(2, 7, 2), "free": [9, 1, 9, 4], "mode": ["c"]})
    assert grid["n"] == [2, 4, 6]
    assert grid["free"] == [1, 4, 9]
    assert grid["mode"] == ["c"]


@pytest.mark.parametrize(
    ("ranges", "error", "text"),
    [
        ({"unknown": None}, OptimizationError, "Unbekannte"),
        ({"free": None}, OptimizationError, "kein min/max"),
        ({"nostep": None}, OptimizationError, "keine Schrittweite"),
        ({"flag": ParamRange(0, 1, 1)}, OptimizationError, "nur für Zahlen"),
        ({"n": []}, OptimizationError, "keine Werte"),
        ({"n": "abc"}, OptimizationError, "Werteliste"),
        ({"n": ParamRange(2, 30, 2)}, ParamError, "Maximum"),
        ({"n": ParamRange(5, 2, 1)}, OptimizationError, "größer als max"),
        ({"n": ParamRange(2, 5, 0)}, OptimizationError, "Schrittweite"),
    ],
)
def test_build_grid_rejects_invalid_ranges(
    ranges: dict[str, Any], error: type[Exception], text: str
) -> None:
    with pytest.raises(error, match=text):
        build_grid(Mixed, ranges)


def test_param_range_float_steps_are_exact() -> None:
    assert ParamRange(0.1, 0.3, 0.1).values() == [0.1, 0.2, 0.3]
    assert ParamRange(1, 2, 0.4).values() == [1.0, 1.4, 1.8]


# ---------------------------------------------------------------------------
# Raster-Suche mit synthetischer Zielfunktion
# ---------------------------------------------------------------------------


def test_grid_search_finds_known_optimum() -> None:
    grid = {"a": list(range(7)), "b": [round(0.1 * i, 1) for i in range(11)]}

    def bowl(p: dict[str, Any]) -> float:
        return 10.0 - (p["a"] - 4) ** 2 - 20.0 * (p["b"] - 0.3) ** 2

    calls: list[tuple[int, int]] = []
    table = grid_search(bowl, grid, progress=lambda d, t: calls.append((d, t)))
    best = table.attrs["best"]
    assert table.loc[best, "a"] == 4
    assert table.loc[best, "b"] == pytest.approx(0.3)
    assert table.loc[best, "objective"] == pytest.approx(10.0)
    assert table.loc[best, "robustness"] == 1.0
    assert len(table) == 77
    assert calls[-1] == (77, 77)


def test_grid_search_first_wins_on_tie_and_skips_nan() -> None:
    values = {0: None, 1: 2.0, 2: 2.0, 3: math.inf}
    table = grid_search(lambda p: values[p["x"]], {"x": [0, 1, 2, 3]})
    assert table.attrs["best"] == 1
    assert math.isnan(table.loc[0, "objective"])
    assert math.isnan(table.loc[3, "objective"])

    table = grid_search(lambda p: None, {"x": [0, 1]})
    assert table.attrs["best"] is None


def test_robustness_counts_positive_neighbours() -> None:
    assert robustness_scores({"x": [1, 2, 3]}, [1.0, -1.0, 1.0]) == [0.0, 1.0, 0.0]
    grid = {"a": [1, 2, 3], "b": [1, 2]}
    values = [1.0, -1.0, 1.0, math.nan, 1.0, 1.0]
    scores = robustness_scores(grid, values)
    # (a=2, b=1) hat Nachbarn (1,1),(1,2),(2,2),(3,1),(3,2): positiv sind 3 von 5
    # ((1,2) ist negativ, (2,2) nicht bewertbar).
    assert scores[2] == pytest.approx(3 / 5)
    # (a=1, b=1) hat Nachbarn (1,2),(2,1),(2,2): positiv nur (2,1).
    assert scores[0] == pytest.approx(1 / 3)
    single = robustness_scores({"a": [1], "b": [2]}, [5.0])
    assert math.isnan(single[0])


def test_robustness_low_for_isolated_spike() -> None:
    grid = {"a": list(range(5)), "b": list(range(5))}

    def spike(p: dict[str, Any]) -> float:
        return 100.0 if (p["a"], p["b"]) == (2, 2) else -1.0

    table = grid_search(spike, grid)
    assert table.loc[table.attrs["best"], "robustness"] == 0.0


def test_large_grid_warns(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(optimize_module, "MAX_COMBINATIONS", 5)
    with caplog.at_level(logging.WARNING, logger=optimize_module.__name__):
        grid_search(lambda p: 1.0, {"x": list(range(6))})
    assert "überschreitet" in caplog.text


# ---------------------------------------------------------------------------
# Optimierung mit Backtests
# ---------------------------------------------------------------------------


def test_optimize_finds_peak_of_tent(settings: Settings) -> None:
    bars = _bars(_tent(peak=50))
    grid = build_grid(HoldBars, {"hold": ParamRange(10, 90, 10)})
    calls: list[tuple[int, int]] = []
    result = optimize(
        HoldBars,
        grid,
        "TENT",
        Timeframe.D1,
        "2020-01-01",
        "2020-12-31",
        settings,
        objective=Objective.CAGR,
        bars=bars,
        workers=1,
        progress=lambda d, t: calls.append((d, t)),
    )
    assert result.best_params == {"hold": 50, "fail_above": 1000}
    assert result.best_value is not None
    assert result.best_value > 0
    assert result.best_robustness == 1.0
    assert list(result.table.columns) == [
        "hold",
        "fail_above",
        "objective",
        *METRIC_COLUMNS,
        "robustness",
        "error",
    ]
    assert calls == [(i, 9) for i in range(1, 10)]
    assert result.objective is Objective.CAGR
    assert result.param_names == ["hold", "fail_above"]
    assert not result.log


def test_optimize_records_failing_combinations(settings: Settings) -> None:
    grid = {"hold": [10, 50], "fail_above": [40]}
    result = optimize(
        HoldBars,
        grid,
        "TENT",
        Timeframe.D1,
        "2020-01-01",
        "2020-12-31",
        settings,
        objective="cagr",
        bars=_bars(_tent()),
        workers=1,
    )
    assert result.table["error"].iloc[0] == ""
    assert "absichtlicher Fehler" in result.table["error"].iloc[1]
    assert math.isnan(result.table["objective"].iloc[1])
    assert result.best_params == {"hold": 10, "fail_above": 40}
    assert "1 Kombination(en) mit Fehler" in result.log[0]


def test_optimize_without_valid_objective(settings: Settings) -> None:
    result = optimize(
        Never,
        build_grid(Never),
        "X",
        Timeframe.D1,
        "2020-01-01",
        "2020-12-31",
        settings,
        objective=Objective.SHARPE,
        bars=_bars(np.linspace(100, 110, 60)),
        workers=1,
    )
    assert result.best_index is None
    assert result.best_params is None
    assert result.best_value is None
    assert result.best_robustness is None
    assert "Keine Kombination" in result.log[-1]


def test_optimize_threads_match_sequential(settings: Settings) -> None:
    grid = build_grid(HoldBars, {"hold": ParamRange(10, 90, 10)})
    kwargs: dict[str, Any] = {"objective": Objective.MAR, "bars": _bars(_tent())}
    one = optimize(HoldBars, grid, "T", Timeframe.D1, "2020-01-01", "2020-12-31", settings,
                   workers=1, **kwargs)  # fmt: skip
    many = optimize(HoldBars(), grid, "T", Timeframe.D1, "2020-01-01", "2020-12-31", settings,
                    workers=3, **kwargs)  # fmt: skip
    pd.testing.assert_frame_equal(one.table, many.table)


@pytest.fixture(scope="module")
def synth_daily() -> pd.DataFrame:
    frame = pd.read_csv(FIXTURE, parse_dates=["date"])
    return frame[frame["date"] < "2004-01-01"].reset_index(drop=True)


def test_optimize_processes_match_sequential(settings: Settings, synth_daily: pd.DataFrame) -> None:
    grid = build_grid(SmaCross, {"fast": [5, 10, 15], "slow": [20, 30, 40]})
    args = (SmaCross, grid, "SYNTH", Timeframe.W1, "1997-01-01", "2003-12-31", settings)
    sequential = optimize(*args, bars=synth_daily, workers=1)
    parallel = optimize(*args, bars=synth_daily, workers=2)
    pd.testing.assert_frame_equal(sequential.table, parallel.table)
    assert sequential.best_params is not None


def test_optimize_falls_back_to_threads(
    settings: Settings, synth_daily: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("kein Prozess")

    monkeypatch.setattr(optimize_module, "ProcessPoolExecutor", broken)
    grid = build_grid(SmaCross, {"fast": [5, 10, 15], "slow": [20, 30, 40]})
    args = (SmaCross, grid, "SYNTH", Timeframe.W1, "1997-01-01", "2003-12-31", settings)
    result = optimize(*args, bars=synth_daily, workers=2)
    assert result.table["error"].eq("").all()


def test_optimize_portfolio_universe(settings: Settings) -> None:
    bars = {"AAA": _bars(_tent(peak=50)), "BBB": _bars(_tent(peak=30))}
    result = optimize(
        HoldBars,
        {"hold": [20, 30, 50, 70]},
        None,
        Timeframe.D1,
        "2020-01-01",
        "2020-12-31",
        settings,
        objective=Objective.CAGR,
        universe=iter(["BBB", "AAA"]),
        bars=bars,
        max_positions=2,
        workers=1,
    )
    assert result.best_params is not None
    assert result.best_params["hold"] in (30, 50)
    assert result.table["error"].eq("").all()


def test_optimize_portfolio_loads_universe_from_database(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_collect(symbols: list[str], bars: Any, s: Any, end: Any, log: Any) -> dict[str, Any]:
        return {sym: _bars(_tent()) for sym in symbols}

    monkeypatch.setattr(optimize_module, "_collect_bars", fake_collect)
    result = optimize(
        HoldBars,
        {"hold": [20, 50]},
        None,
        Timeframe.D1,
        "2020-01-01",
        "2020-12-31",
        settings,
        objective=Objective.CAGR,
        universe=["AAA"],
        workers=1,
    )
    assert result.best_params == {"hold": 50, "fail_above": 1000}


def test_optimize_loads_symbol_from_database(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(optimize_module, "_load_symbol_daily", lambda s, sym, end: _bars(_tent()))
    result = optimize(
        HoldBars, {"hold": [20, 50]}, "TENT", Timeframe.D1, "2020-01-01", "2020-12-31",
        settings, objective="cagr", workers=1,
    )  # fmt: skip
    assert result.best_params == {"hold": 50, "fail_above": 1000}


@pytest.mark.parametrize(
    ("kwargs", "error", "text"),
    [
        ({"symbol": None}, OptimizationError, "genau eines"),
        ({"universe": ["A"]}, OptimizationError, "genau eines"),
        ({"grid": {"nope": [1]}}, OptimizationError, "Unbekannte"),
        ({"grid": {"hold": []}}, OptimizationError, "keine Werte"),
        ({"bars": {"T": None}}, OptimizationError, "DataFrame"),
        ({"symbol": None, "universe": ["T"]}, OptimizationError, "je Symbol"),
        ({"start": "2021-01-01"}, BacktestError, "nach dem Ende"),
        ({"start": "kein Datum"}, BacktestError, "kein gültiges Datum"),
        ({"objective": "unbekannt"}, ValueError, "unbekannt"),
    ],
)
def test_optimize_rejects_invalid_input(
    settings: Settings, kwargs: dict[str, Any], error: type[Exception], text: str
) -> None:
    call: dict[str, Any] = {
        "grid": {"hold": [10]},
        "symbol": "T",
        "start": "2020-01-01",
        "bars": _bars(_tent()),
    }
    call.update(kwargs)
    with pytest.raises(error, match=text):
        optimize(
            HoldBars,
            call.pop("grid"),
            call.pop("symbol"),
            Timeframe.D1,
            call.pop("start"),
            "2020-12-31",
            settings,
            workers=1,
            **call,
        )


def test_heatmap_holds_other_params_at_best(settings: Settings) -> None:
    grid = {"hold": [10, 30, 50], "fail_above": [100, 200]}
    result = optimize(
        HoldBars, grid, "T", Timeframe.D1, "2020-01-01", "2020-12-31", settings,
        objective="cagr", bars=_bars(_tent()), workers=1,
    )  # fmt: skip
    heat = result.heatmap("hold", "fail_above")
    assert list(heat.index) == [100, 200]
    assert list(heat.columns) == [10, 30, 50]
    assert heat.loc[100, 50] == pytest.approx(result.best_value)
    with pytest.raises(OptimizationError, match="Unbekannter"):
        result.heatmap("hold", "nope")
    with pytest.raises(OptimizationError, match="zwei verschiedene"):
        result.heatmap("hold", "hold")
    assert Objective.CALMAR.label.startswith("Calmar")


# ---------------------------------------------------------------------------
# Zielgrößen
# ---------------------------------------------------------------------------


class AlwaysLong(Strategy):
    id = "test_always_long"
    name = "Immer investiert"
    version = "1.0.0"
    description = "Kauft auf der ersten Kerze."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1, Timeframe.W1]

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and len(ctx.bars) == 1:
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        return []


def test_calmar_uses_last_three_years_mar_whole_period(settings: Settings) -> None:
    # Absturz im ersten Jahr, danach stetiger Anstieg über fünf Jahre.
    crash = np.linspace(100.0, 50.0, 252)
    steps = np.arange(5 * 252)
    rise = np.linspace(50.0, 200.0, 5 * 252) * (1 + 0.02 * np.sin(steps / 10))
    result = run_backtest(
        AlwaysLong(), None, "X", Timeframe.D1, "2010-01-01", "2030-12-31", settings,
        bars=_bars(np.concatenate([crash, rise]), start="2010-01-04"), costs=NO_COSTS,
    )  # fmt: skip
    full = compute_metrics(result.equity, result.trades, Timeframe.D1,
                           initial_capital=result.initial_capital)  # fmt: skip
    mar = objective_value(result, Objective.MAR)
    calmar = objective_value(result, "calmar")
    assert mar == pytest.approx(full.calmar)
    assert calmar is not None
    assert mar is not None
    assert calmar > mar
    sharpe = objective_value(result, Objective.SHARPE, risk_free_rate_pct=0.0)
    assert sharpe == pytest.approx(full.sharpe)
    assert objective_value(result, Objective.CAGR) == pytest.approx(full.cagr_pct)


def test_calmar_equals_mar_for_short_curves(settings: Settings) -> None:
    result = run_backtest(
        AlwaysLong(), None, "X", Timeframe.D1, "2020-01-01", "2030-12-31", settings,
        bars=_bars(_tent(length=300)), costs=NO_COSTS,
    )  # fmt: skip
    assert objective_value(result, "calmar") == objective_value(result, "mar")
