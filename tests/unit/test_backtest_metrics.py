"""Tests der Performance-Kennzahlen (B-05).

Referenzwerte sind von Hand gerechnet für eine kurze Monats-Equity-Kurve:
Startkapital 1000, Equity 1100 → 990 → 1188 (Renditen +10 %, −10 %, +20 %).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import run_backtest
from trading_helfer.backtest.metrics import (
    PERIODS_PER_YEAR,
    YEAR_COLUMN,
    MetricsError,
    benchmark_comparison,
    compute_metrics,
    metrics_for_result,
    monthly_returns,
    periods_per_year,
)
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.strategies.base import BarContext, Order, Sizing, Strategy

DATES = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])


def _equity(values: list[float], exposure: list[float] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"date": pd.date_range("2020-01-31", periods=len(values), freq="ME"), "equity": values}
    )
    if exposure is not None:
        frame["exposure"] = exposure
    return frame


@pytest.fixture
def equity() -> pd.DataFrame:
    return pd.DataFrame(
        {"date": DATES, "equity": [1100.0, 990.0, 1188.0], "exposure": [1.0, 0.5, 0.0]}
    )


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pnl": [50.0, -20.0, 30.0, -10.0],
            "pnl_pct": [5.0, -2.0, 3.0, -1.0],
            "entry_date": pd.to_datetime(["2020-01-02", "2020-01-10", "2020-02-03", "2020-03-02"]),
            "exit_date": pd.to_datetime(["2020-01-12", "2020-01-30", "2020-02-13", "2020-03-12"]),
        }
    )


@pytest.fixture
def benchmark() -> pd.Series:
    # Buy and Hold: +5 %, −10 %, +10 %
    return pd.Series(
        [100.0, 105.0, 94.5, 103.95],
        index=pd.to_datetime(["2019-12-31", "2020-01-31", "2020-02-29", "2020-03-31"]),
    )


# ---------------------------------------------------------------------------
# Handgerechnete Werte
# ---------------------------------------------------------------------------


def test_return_and_risk_metrics_match_hand_calculation(
    equity: pd.DataFrame, trades: pd.DataFrame
) -> None:
    m = compute_metrics(equity, trades, Timeframe.MN1, initial_capital=1000.0)

    assert m.periods_per_year == 12
    assert m.periods == 3
    assert m.years == pytest.approx(0.25)
    assert m.start_equity == 1000.0
    assert m.end_equity == 1188.0
    assert m.total_return_pct == pytest.approx(18.8)
    # CAGR = 1.188^(1/0.25) − 1 = 1.188^4 − 1
    assert m.cagr_pct == pytest.approx(99.1891, abs=1e-4)
    # Mittel 0.0666667, Stichproben-Std 0.1527525 → × √12
    assert m.volatility_pct == pytest.approx(52.9150, abs=1e-4)
    # Sharpe = 0.0666667 × √12 / 0.1527525
    assert m.sharpe == pytest.approx(1.511858, abs=1e-6)
    # Downside = √(0.01 / 3) = 0.0577350 → Sortino = 0.0666667 × √12 / 0.0577350 = 4
    assert m.sortino == pytest.approx(4.0)
    # Max Drawdown 990 / 1100 − 1 = −10 %; Calmar = 0.991891 / 0.1
    assert m.max_drawdown_pct == pytest.approx(-10.0)
    assert m.calmar == pytest.approx(9.91891, abs=1e-5)
    # Höchststand 31.01., Erholung 31.03. (2020 ist Schaltjahr): 2 Kerzen, 60 Tage
    assert m.max_drawdown_duration_bars == 2
    assert m.max_drawdown_duration_days == 60
    # Exposure 1.0, 0.5, 0.0
    assert m.exposure_pct == pytest.approx(50.0)
    assert m.time_in_market_pct == pytest.approx(200.0 / 3.0)
    assert m.benchmark is None


def test_trade_metrics_match_hand_calculation(equity: pd.DataFrame, trades: pd.DataFrame) -> None:
    m = compute_metrics(equity, trades, Timeframe.MN1, initial_capital=1000.0)

    assert m.num_trades == 4
    assert m.win_rate_pct == pytest.approx(50.0)
    assert m.profit_factor == pytest.approx(80.0 / 30.0)
    assert m.avg_win == pytest.approx(40.0)
    assert m.avg_loss == pytest.approx(-15.0)
    assert m.avg_win_pct == pytest.approx(4.0)
    assert m.avg_loss_pct == pytest.approx(-1.5)
    assert m.expectancy == pytest.approx(12.5)
    assert m.expectancy_pct == pytest.approx(1.25)
    # Haltedauer 10, 20, 10, 10 Tage
    assert m.avg_holding_days == pytest.approx(12.5)


def test_risk_free_rate_lowers_sharpe(equity: pd.DataFrame) -> None:
    m = compute_metrics(equity, None, Timeframe.MN1, initial_capital=1000.0, risk_free_rate_pct=12)
    rf = 1.12 ** (1 / 12) - 1  # 0.0094888 je Monat
    excess = np.array([0.1, -0.1, 0.2]) - rf
    assert m.risk_free_rate_pct == 12.0
    assert m.sharpe == pytest.approx(
        excess.mean() * math.sqrt(12) / np.std(excess, ddof=1), abs=1e-12
    )
    assert m.sharpe == pytest.approx(1.296672, abs=1e-6)
    downside = math.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    assert m.sortino == pytest.approx(excess.mean() * math.sqrt(12) / downside)
    # Volatilität hängt nicht vom Zins ab
    assert m.volatility_pct == pytest.approx(52.9150, abs=1e-4)


def test_risk_free_rate_from_settings_is_default_zero() -> None:
    assert Settings().risk_free_rate_pct == 0.0


def test_benchmark_comparison_matches_hand_calculation(
    equity: pd.DataFrame, benchmark: pd.Series
) -> None:
    m = compute_metrics(equity, None, Timeframe.MN1, initial_capital=1000.0, benchmark=benchmark)
    b = m.benchmark
    assert b is not None
    assert b.total_return_pct == pytest.approx(3.95)
    assert b.cagr_pct == pytest.approx((1.0395**4 - 1) * 100)
    assert b.excess_return_pct == pytest.approx(18.8 - 3.95)
    # cov(s, b) = 0.0158333, var(b) = 0.0108333 → Beta = 19/13
    assert b.beta == pytest.approx(19 / 13)
    # Alpha = (0.0666667 − 19/13 × 0.0166667) × 12
    assert b.alpha_pct == pytest.approx(50.7692, abs=1e-4)
    assert b.correlation == pytest.approx(0.995871, abs=1e-6)


def test_benchmark_without_earlier_price_starts_at_first_date(equity: pd.DataFrame) -> None:
    bench = pd.Series([100.0, 90.0, 99.0], index=DATES)
    b = compute_metrics(
        equity, None, Timeframe.MN1, initial_capital=1000.0, benchmark=bench
    ).benchmark
    assert b is not None
    assert b.total_return_pct == pytest.approx(-1.0)


def test_benchmark_daily_prices_are_aligned_to_bar_dates(equity: pd.DataFrame) -> None:
    days = pd.bdate_range("2019-12-02", "2020-04-15")
    bench = pd.Series(np.linspace(100.0, 120.0, len(days)), index=days)
    b = compute_metrics(
        equity, None, Timeframe.MN1, initial_capital=1000.0, benchmark=bench
    ).benchmark
    assert b is not None
    start = bench.loc[:"2020-01-30"].iloc[-1]
    end = bench.loc[:"2020-03-31"].iloc[-1]
    assert b.total_return_pct == pytest.approx((end / start - 1) * 100)


def test_benchmark_flat_has_no_beta() -> None:
    returns = np.array([0.1, -0.1, 0.2])
    bench = pd.Series([50.0, 50.0, 50.0], index=DATES)
    b = benchmark_comparison(returns, DATES, bench, Timeframe.MN1)
    assert b.beta is None
    assert b.alpha_pct is None
    assert b.correlation is None
    assert b.total_return_pct == 0.0


def test_benchmark_errors() -> None:
    returns = np.array([0.1, -0.1, 0.2])
    with pytest.raises(MetricsError, match="leer"):
        benchmark_comparison(returns, DATES, pd.Series(dtype=float), Timeframe.MN1)
    with pytest.raises(MetricsError, match="positiv"):
        benchmark_comparison(returns, DATES, pd.Series([1.0, -1.0, 2.0], index=DATES), "1M")
    late = pd.Series([1.0], index=pd.to_datetime(["2020-03-01"]))
    with pytest.raises(MetricsError, match="keinen Kurs am 2020-01-31"):
        benchmark_comparison(returns, DATES, late, Timeframe.MN1)
    with pytest.raises(MetricsError, match="Länge"):
        benchmark_comparison(returns[:2], DATES, late, Timeframe.MN1)
    with pytest.raises(MetricsError, match="Keine Renditen"):
        benchmark_comparison(np.array([]), DATES[:0], late, Timeframe.MN1)


# ---------------------------------------------------------------------------
# Annualisierung
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [(Timeframe.D1, 252), (Timeframe.D3, 84), (Timeframe.W1, 52), (Timeframe.MN1, 12)],
)
def test_periods_per_year(timeframe: Timeframe, expected: int) -> None:
    assert periods_per_year(timeframe) == expected
    assert periods_per_year(timeframe.value) == expected
    assert PERIODS_PER_YEAR[timeframe] == expected


def test_unknown_timeframe() -> None:
    with pytest.raises(MetricsError, match="Zeitrahmen"):
        periods_per_year("2W")


def test_annualization_scales_with_timeframe(equity: pd.DataFrame) -> None:
    monthly = compute_metrics(equity, None, Timeframe.MN1, initial_capital=1000.0)
    weekly = compute_metrics(equity, None, Timeframe.W1, initial_capital=1000.0)
    assert weekly.years == pytest.approx(3 / 52)
    assert weekly.cagr_pct == pytest.approx((1.188 ** (52 / 3) - 1) * 100)
    assert weekly.volatility_pct is not None
    assert monthly.volatility_pct is not None
    assert weekly.volatility_pct / monthly.volatility_pct == pytest.approx(math.sqrt(52 / 12))
    assert weekly.sharpe is not None
    assert monthly.sharpe is not None
    assert weekly.sharpe / monthly.sharpe == pytest.approx(math.sqrt(52 / 12))


# ---------------------------------------------------------------------------
# Grenzfälle
# ---------------------------------------------------------------------------


def test_no_trades_and_flat_equity() -> None:
    eq = _equity([1000.0, 1000.0, 1000.0], exposure=[0.0, 0.0, 0.0])
    m = compute_metrics(eq, pd.DataFrame(columns=["pnl"]), Timeframe.W1, initial_capital=1000.0)
    assert m.num_trades == 0
    assert m.win_rate_pct is None
    assert m.profit_factor is None
    assert m.avg_win is None
    assert m.avg_loss is None
    assert m.expectancy is None
    assert m.expectancy_pct is None
    assert m.avg_holding_days is None
    assert m.total_return_pct == 0.0
    assert m.cagr_pct == 0.0
    assert m.volatility_pct == 0.0
    assert m.sharpe is None
    assert m.sortino is None
    assert m.calmar is None
    assert m.max_drawdown_pct == 0.0
    assert m.max_drawdown_duration_bars == 0
    assert m.exposure_pct == 0.0
    assert m.time_in_market_pct == 0.0


def test_only_losses() -> None:
    eq = _equity([950.0, 900.0, 850.0])
    trades = pd.DataFrame({"pnl": [-50.0, -50.0, -50.0], "pnl_pct": [-5.0, -5.3, -5.6]})
    m = compute_metrics(eq, trades, Timeframe.MN1, initial_capital=1000.0)
    assert m.num_trades == 3
    assert m.win_rate_pct == 0.0
    assert m.profit_factor == 0.0
    assert m.avg_win is None
    assert m.avg_win_pct is None
    assert m.avg_loss == pytest.approx(-50.0)
    assert m.avg_loss_pct == pytest.approx(-5.3)
    assert m.expectancy == pytest.approx(-50.0)
    assert m.avg_holding_days is None  # keine Datumsspalten
    assert m.total_return_pct == pytest.approx(-15.0)
    assert m.cagr_pct == pytest.approx((0.85**4 - 1) * 100)
    assert m.max_drawdown_pct == pytest.approx(-15.0)
    # vom Startkapital an unter Wasser, bis zum Ende nicht erholt
    assert m.max_drawdown_duration_bars == 3
    assert (
        m.max_drawdown_duration_days
        == (pd.Timestamp("2020-03-31") - pd.Timestamp("2020-01-31")).days
    )
    assert m.sharpe is not None
    assert m.sharpe < 0
    assert m.sortino is not None
    assert m.sortino < 0
    assert m.calmar is not None
    assert m.calmar < 0
    assert m.exposure_pct is None
    assert m.time_in_market_pct is None


def test_only_wins_has_no_profit_factor() -> None:
    trades = pd.DataFrame({"pnl": [10.0, 0.0]})
    m = compute_metrics(_equity([1010.0, 1010.0]), trades, "1M", initial_capital=1000.0)
    assert m.profit_factor is None
    assert m.win_rate_pct == 50.0  # 0 zählt nicht als Gewinn
    assert m.avg_loss is None
    assert m.avg_win_pct is None  # keine pnl_pct-Spalte
    assert m.expectancy_pct is None


def test_total_loss_gives_cagr_minus_100() -> None:
    m = compute_metrics(_equity([500.0, 0.0, 0.0]), None, Timeframe.MN1, initial_capital=1000.0)
    assert m.cagr_pct == -100.0
    assert m.max_drawdown_pct == -100.0
    assert m.total_return_pct == -100.0


def test_single_bar_has_no_volatility() -> None:
    m = compute_metrics(_equity([1100.0]), None, Timeframe.MN1, initial_capital=1000.0)
    assert m.periods == 1
    assert m.volatility_pct is None
    assert m.sharpe is None
    assert m.sortino is None  # keine negative Rendite
    assert m.total_return_pct == pytest.approx(10.0)


def test_without_initial_capital_starts_at_first_value() -> None:
    m = compute_metrics(_equity([1000.0, 1100.0]), None, Timeframe.MN1)
    assert m.periods == 1
    assert m.start_equity == 1000.0
    assert m.total_return_pct == pytest.approx(10.0)


def test_longest_drawdown_is_reported() -> None:
    # Phase 1: 2 Kerzen (1100 → 1000 → 1100); Phase 2: 3 Kerzen, am Ende nicht erholt
    eq = _equity([1100.0, 1000.0, 1100.0, 1200.0, 1150.0, 1100.0, 1190.0])
    m = compute_metrics(eq, None, Timeframe.MN1, initial_capital=1000.0)
    assert m.max_drawdown_duration_bars == 3
    assert m.max_drawdown_pct == pytest.approx((1000 / 1100 - 1) * 100)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"date": DATES}), "Spalten"),
        (pd.DataFrame({"date": [], "equity": []}), "leer"),
        (pd.DataFrame({"date": DATES, "equity": [1.0, np.nan, 1.0]}), "ungültige"),
        (pd.DataFrame({"date": DATES[::-1], "equity": [1.0, 1.0, 1.0]}), "sortiert"),
    ],
)
def test_invalid_equity(frame: pd.DataFrame, message: str) -> None:
    with pytest.raises(MetricsError, match=message):
        compute_metrics(frame, None, Timeframe.MN1)


def test_invalid_inputs(equity: pd.DataFrame) -> None:
    with pytest.raises(MetricsError, match="Startkapital"):
        compute_metrics(equity, None, Timeframe.MN1, initial_capital=0.0)
    with pytest.raises(MetricsError, match="Zins"):
        compute_metrics(equity, None, Timeframe.MN1, risk_free_rate_pct=-100.0)
    with pytest.raises(MetricsError, match="pnl"):
        compute_metrics(equity, pd.DataFrame({"x": [1]}), Timeframe.MN1)


def test_to_dict_is_json_serializable(
    equity: pd.DataFrame, trades: pd.DataFrame, benchmark: pd.Series
) -> None:
    m = compute_metrics(equity, trades, Timeframe.MN1, initial_capital=1000.0, benchmark=benchmark)
    data = m.to_dict()
    assert data["timeframe"] == "1M"
    assert isinstance(data["benchmark"], dict)
    assert json.loads(json.dumps(data))["num_trades"] == 4
    empty = compute_metrics(_equity([1000.0]), None, Timeframe.MN1).to_dict()
    assert json.loads(json.dumps(empty))["sharpe"] is None


def test_metrics_are_deterministic(
    equity: pd.DataFrame, trades: pd.DataFrame, benchmark: pd.Series
) -> None:
    a = compute_metrics(equity, trades, Timeframe.MN1, initial_capital=1000.0, benchmark=benchmark)
    b = compute_metrics(equity, trades, Timeframe.MN1, initial_capital=1000.0, benchmark=benchmark)
    assert json.dumps(a.to_dict()) == json.dumps(b.to_dict())


# ---------------------------------------------------------------------------
# Monatsrenditen
# ---------------------------------------------------------------------------


def test_monthly_returns_table(equity: pd.DataFrame) -> None:
    table = monthly_returns(equity, initial_capital=1000.0)
    assert list(table.index) == [2020]
    assert list(table.columns) == [*range(1, 13), YEAR_COLUMN]
    assert table.loc[2020, 1] == pytest.approx(10.0)
    assert table.loc[2020, 2] == pytest.approx(-10.0)
    assert table.loc[2020, 3] == pytest.approx(20.0)
    assert table.loc[2020, 4:12].isna().all()
    assert table.loc[2020, YEAR_COLUMN] == pytest.approx(18.8)


def test_monthly_returns_from_weekly_bars_across_years() -> None:
    dates = pd.to_datetime(["2020-12-18", "2020-12-31", "2021-01-15", "2021-01-29", "2021-03-05"])
    eq = pd.DataFrame({"date": dates, "equity": [990.0, 1050.0, 1000.0, 1155.0, 1039.5]})
    table = monthly_returns(eq)  # ohne Startkapital: Basis ist der erste Wert 990
    assert list(table.index) == [2020, 2021]
    assert table.loc[2020, 12] == pytest.approx((1050 / 990 - 1) * 100)
    assert table.loc[2021, 1] == pytest.approx(10.0)
    assert math.isnan(table.loc[2021, 2])
    assert table.loc[2021, 3] == pytest.approx(-10.0)
    assert table.loc[2021, YEAR_COLUMN] == pytest.approx(-1.0)
    assert table.loc[2020, YEAR_COLUMN] == pytest.approx((1050 / 990 - 1) * 100)


# ---------------------------------------------------------------------------
# Zusammenspiel mit der Engine
# ---------------------------------------------------------------------------


class _BuyOnce(Strategy):
    id = "test_metrics_buy_once"
    name = "Einmal kaufen"
    version = "1.0.0"
    description = "Kauft auf der ersten Kerze, verkauft auf der zehnten."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1]

    def __init__(self) -> None:
        self._bars = 0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        self._bars += 1
        if self._bars == 1:
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        if self._bars == 10 and ctx.position is not None:
            return ctx.exit(tag="Ende")
        return []


def test_metrics_for_engine_result(tmp_path: Path) -> None:
    close = 100.0 * 1.002 ** np.arange(200)
    opens = np.concatenate([[close[0]], close[:-1]])
    bars = pd.DataFrame(
        {
            "date": pd.bdate_range("2021-01-04", periods=len(close)),
            "open": opens,
            "high": np.maximum(opens, close) * 1.01,
            "low": np.minimum(opens, close) * 0.99,
            "close": close,
            "volume": 1000,
        }
    )
    result = run_backtest(
        _BuyOnce(),
        None,
        "TEST",
        Timeframe.W1,
        "2021-01-04",
        "2021-12-31",
        Settings(data_dir=tmp_path),
        bars=bars,
        costs=CostModel(fee_per_order=1.0, slippage_pct=0.0),
    )
    benchmark = pd.Series(close, index=bars["date"])
    m = metrics_for_result(result, benchmark=benchmark)
    assert m.timeframe is Timeframe.W1
    assert m.periods == len(result.equity)
    assert m.total_return_pct == pytest.approx(result.total_return_pct)
    assert m.num_trades == len(result.trades) == 1
    assert m.expectancy == pytest.approx(float(result.trades["pnl"].iloc[0]))
    assert m.max_drawdown_pct <= float(result.equity["drawdown"].min()) * 100 + 1e-9
    assert m.time_in_market_pct is not None
    assert 0 < m.time_in_market_pct < 100
    assert m.benchmark is not None
    assert m.benchmark.total_return_pct > m.total_return_pct  # Benchmark durchgehend investiert
