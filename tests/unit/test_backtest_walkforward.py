"""Tests der Walk-Forward-Analyse (B-08)."""

from __future__ import annotations

import itertools
from datetime import date, timedelta
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest import walkforward as wf_module
from trading_helfer.backtest.engine import EQUITY_COLUMNS, TRADE_COLUMNS, BacktestError
from trading_helfer.backtest.optimize import Objective, OptimizationError
from trading_helfer.backtest.walkforward import (
    WINDOW_COLUMN,
    walk_forward,
    walk_forward_windows,
)
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    Params,
    Sizing,
    Strategy,
)


def _daily(start: str, end: str) -> pd.DataFrame:
    """Werktägliche Kurse mit Trend und Wellen (deterministisch)."""
    dates = pd.bdate_range(start, end)
    n = np.arange(len(dates))
    close = 100.0 * np.exp(0.0003 * n) * (1 + 0.1 * np.sin(n / 40.0))
    opens = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": np.maximum(opens, close) * 1.01,
            "low": np.minimum(opens, close) * 0.99,
            "close": close,
            "volume": 1000,
        }
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", fee_per_order=0.0, slippage_pct=0.0)


SEEN: list[pd.Timestamp] = []


class SpySma(Strategy):
    """SMA-Kreuzung, die das letzte Datum jeder gesehenen Kursreihe protokolliert."""

    id = "test_spy_sma"
    name = "Spion"
    version = "1.0.0"
    description = "Merkt sich, bis zu welchem Datum Kurse übergeben wurden."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        "fast": Param(4, 2, 20, 2, "schnell"),
        "slow": Param(12, 6, 40, 6, "langsam"),
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        SEEN.append(pd.Timestamp(bars["date"].max()))
        close = bars["close"]
        return pd.DataFrame(
            {"fast": close.rolling(p.fast).mean(), "slow": close.rolling(p.slow).mean()},
            index=bars.index,
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and ctx.crossed_above("fast", "slow"):
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        if ctx.position is not None and ctx.crossed_below("fast", "slow"):
            return ctx.exit()
        return []


class Never(Strategy):
    id = "test_never"
    name = "Handelt nie"
    version = "1.0.0"
    description = "Keine Trades."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {"x": Param(2, 1, 3, 1, "egal")}

    def on_bar(self, ctx: BarContext) -> list[Order]:
        return []


# ---------------------------------------------------------------------------
# Fenster
# ---------------------------------------------------------------------------


def test_windows_are_contiguous_and_disjoint() -> None:
    spans = walk_forward_windows("2000-01-01", "2010-06-30", train_years=5, test_years=1)
    assert [s.index for s in spans] == [1, 2, 3, 4, 5, 6]
    assert spans[0].train_start == date(2000, 1, 1)
    assert spans[0].train_end == date(2004, 12, 31)
    assert spans[0].test_start == date(2005, 1, 1)
    assert spans[0].test_end == date(2005, 12, 31)
    for s in spans:
        assert s.train_end + timedelta(days=1) == s.test_start
        assert s.train_start <= s.train_end < s.test_start <= s.test_end
        assert s.train_start.year + 5 == s.test_start.year
    for prev, nxt in itertools.pairwise(spans):
        assert prev.test_end + timedelta(days=1) == nxt.test_start
        assert nxt.train_start == date(prev.train_start.year + 1, 1, 1)
    # Der letzte Test endet beim Ende des Zeitraums (verkürzt).
    assert spans[-1].test_start == date(2010, 1, 1)
    assert spans[-1].test_end == date(2010, 6, 30)


def test_windows_anchored_and_longer_tests() -> None:
    spans = walk_forward_windows(
        "2000-03-15", "2011-12-31", train_years=3, test_years=2, anchored=True
    )
    assert all(s.train_start == date(2000, 3, 15) for s in spans)
    assert [s.test_start for s in spans] == [
        date(2003, 3, 15),
        date(2005, 3, 15),
        date(2007, 3, 15),
        date(2009, 3, 15),
        date(2011, 3, 15),
    ]
    assert spans[0].test_end == date(2005, 3, 14)
    assert spans[-1].test_end == date(2011, 12, 31)


@pytest.mark.parametrize(
    ("args", "text"),
    [
        ({"train_years": 0}, "mindestens 1 Jahr"),
        ({"test_years": 0}, "mindestens 1 Jahr"),
        ({"end": "2004-12-31"}, "kürzer als"),
    ],
)
def test_windows_reject_invalid(args: dict[str, Any], text: str) -> None:
    call: dict[str, Any] = {"start": "2000-01-01", "end": "2010-12-31"}
    call.update(args)
    with pytest.raises(OptimizationError, match=text):
        walk_forward_windows(call.pop("start"), call.pop("end"), **call)


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------


def test_walk_forward_never_trains_on_test_data(settings: Settings) -> None:
    SEEN.clear()
    grid = {"fast": [4, 8], "slow": [12, 24]}
    calls: list[tuple[int, int]] = []
    result = walk_forward(
        SpySma,
        grid,
        "SYN",
        Timeframe.W1,
        "2000-01-01",
        "2008-12-31",
        settings,
        train_years=5,
        test_years=1,
        objective=Objective.CAGR,
        bars=_daily("1999-01-01", "2010-12-31"),
        workers=1,
        progress=lambda d, t: calls.append((d, t)),
    )
    assert len(result.windows) == 4
    combos = 4
    # Je Fenster: vier Trainingsläufe, dann ein Testlauf (sequenziell, workers=1).
    assert len(SEEN) == len(result.windows) * (combos + 1)
    for k, window in enumerate(result.windows):
        span = window.span
        block = SEEN[k * (combos + 1) : (k + 1) * (combos + 1)]
        for seen in block[:combos]:
            assert seen <= pd.Timestamp(span.train_end)
            assert seen < pd.Timestamp(span.test_start)
        assert pd.Timestamp(span.train_end) < block[-1] <= pd.Timestamp(span.test_end)
        # Gewählte Parameter stammen aus dem Raster; Test handelt nur im Testzeitraum.
        assert window.params["fast"] in grid["fast"]
        assert window.params["slow"] in grid["slow"]
        test_equity = window.test_result.equity
        assert pd.to_datetime(test_equity["date"]).min() >= pd.Timestamp(span.test_start)
        assert pd.to_datetime(test_equity["date"]).max() <= pd.Timestamp(span.test_end)
        assert window.optimization.best_params == window.params
        assert window.train_value == window.optimization.best_value
    assert calls[-1] == (4 * 5, 4 * 5)
    assert [c[0] for c in calls] == sorted(c[0] for c in calls)


def test_walk_forward_composes_out_of_sample_equity(settings: Settings) -> None:
    result = walk_forward(
        SpySma,
        {"fast": [4, 8], "slow": [12, 24]},
        "SYN",
        Timeframe.W1,
        "2000-01-01",
        "2008-12-31",
        settings,
        bars=_daily("1999-01-01", "2010-12-31"),
        workers=1,
    )
    equity = result.equity
    assert list(equity.columns) == list(EQUITY_COLUMNS)
    dates = pd.to_datetime(equity["date"])
    assert dates.is_monotonic_increasing
    assert dates.is_unique
    assert dates.min() >= pd.Timestamp(result.windows[0].span.test_start)
    assert dates.max() <= pd.Timestamp("2008-12-31")
    # Endstand = Startkapital × Produkt der Fensterrenditen.
    growth = np.prod([w.test_result.final_equity / w.test_result.initial_capital
                      for w in result.windows])  # fmt: skip
    assert result.final_equity == pytest.approx(settings.default_capital * growth)
    assert (equity["drawdown"] <= 0).all()
    assert equity["drawdown"].iloc[0] == 0.0
    # Trades aller Fenster mit Fensternummer.
    assert list(result.trades.columns) == [*TRADE_COLUMNS, WINDOW_COLUMN]
    assert set(result.trades[WINDOW_COLUMN]) <= {1, 2, 3, 4}
    table = result.params_table()
    assert list(table.columns) == [
        WINDOW_COLUMN,
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "fast",
        "slow",
        "train_value",
        "test_value",
        "robustness",
    ]
    assert len(table) == 4
    metrics = result.metrics
    assert metrics.end_equity == pytest.approx(result.final_equity)
    assert result.objective is Objective.SHARPE
    assert result.train_years == 5
    assert result.test_years == 1
    assert not result.anchored


def test_walk_forward_is_deterministic(settings: Settings) -> None:
    kwargs: dict[str, Any] = {
        "bars": _daily("1999-01-01", "2008-12-31"),
        "objective": "mar",
        "train_years": 4,
        "test_years": 2,
    }
    a = walk_forward(SpySma, {"fast": [4, 8]}, "S", Timeframe.W1, "2000-01-01", "2008-12-31",
                     settings, workers=1, **kwargs)  # fmt: skip
    b = walk_forward(SpySma(), {"fast": [4, 8]}, "S", Timeframe.W1, "2000-01-01",
                     "2008-12-31", settings, workers=2, **kwargs)  # fmt: skip
    pd.testing.assert_frame_equal(a.equity, b.equity)
    pd.testing.assert_frame_equal(a.params_table(), b.params_table())


def test_walk_forward_falls_back_to_defaults(settings: Settings) -> None:
    result = walk_forward(
        Never,
        {"x": [1, 3]},
        "S",
        Timeframe.W1,
        "2000-01-01",
        "2007-12-31",
        settings,
        objective=Objective.SHARPE,
        bars=_daily("2000-01-01", "2007-12-31"),
        workers=1,
    )
    assert all(w.params == {"x": 2} for w in result.windows)
    assert all(w.train_value is None for w in result.windows)
    assert all(w.test_value is None for w in result.windows)
    assert "Standardparameter" in result.log[0]
    assert len(result.trades) == 0
    assert list(result.trades.columns) == [*TRADE_COLUMNS, WINDOW_COLUMN]
    assert result.final_equity == pytest.approx(settings.default_capital)


def test_walk_forward_without_test_bars_names_window(settings: Settings) -> None:
    with pytest.raises(BacktestError, match="Fenster 1 \\(Test\\): Keine Kerzen"):
        walk_forward(
            Never,
            {"x": [1]},
            "S",
            Timeframe.W1,
            "2000-01-01",
            "2006-12-31",
            settings,
            objective="cagr",
            bars=_daily("2000-01-01", "2004-12-31"),
            workers=1,
        )


def test_walk_forward_test_error_names_window(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(ctx: Any, params: Any) -> Any:
        raise BacktestError("keine Kurse")

    monkeypatch.setattr(wf_module, "_run", boom)
    with pytest.raises(BacktestError, match="Fenster 1 \\(Test\\): keine Kurse"):
        walk_forward(Never, {"x": [1]}, "S", Timeframe.W1, "2000-01-01", "2006-12-31",
                     settings, bars=_daily("2000-01-01", "2006-12-31"), workers=1)  # fmt: skip


def test_walk_forward_portfolio(settings: Settings) -> None:
    bars = {"AAA": _daily("1999-01-01", "2007-12-31"), "BBB": _daily("1999-06-01", "2007-12-31")}
    result = walk_forward(
        SpySma,
        {"fast": [4, 8]},
        None,
        Timeframe.W1,
        "2000-01-01",
        "2007-12-31",
        settings,
        universe=["AAA", "BBB"],
        max_positions=2,
        bars=bars,
        workers=1,
    )
    assert len(result.windows) == 3
    assert len(result.equity) > 0
    assert {"AAA", "BBB"} >= set(result.trades["symbol"])
