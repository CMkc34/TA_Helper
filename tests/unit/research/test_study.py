"""Tests des festen Studienablaufs (R-01)."""

from __future__ import annotations

import dataclasses
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.data.storage import MarketStore
from trading_helfer.data.universe import IndexChange, IndexHistory
from trading_helfer.research.report import summary_dict
from trading_helfer.research.study import (
    CRITERIA,
    EQUAL_WEIGHT,
    IN_SAMPLE_END,
    MARKET_PHASES,
    OUT_OF_SAMPLE_START,
    MarketPhase,
    StudyConfig,
    StudyError,
    StudyResult,
    Thresholds,
    Verdict,
    buy_and_hold_curve,
    equal_weight_curve,
    evaluate_criteria,
    load_study_bars,
    neighborhood_grid,
    parse_timeframe,
    phase_stats,
    run_study,
    slice_metrics,
)
from trading_helfer.strategies.base import BarContext, Order, Param, Params, Sizing, Strategy
from trading_helfer.strategies.builtin.donchian_breakout import DonchianBreakout
from trading_helfer.strategies.builtin.sma_cross import SmaCross

# ---------------------------------------------------------------------------
# Hilfen
# ---------------------------------------------------------------------------


def _weekly(closes: list[float], first_monday: str = "1999-12-27") -> pd.DataFrame:
    """Tageskurse, deren Wochen-Close der Reihe nach ``closes`` ist (Mo–Fr gleich)."""
    rows = []
    monday = pd.Timestamp(first_monday)
    for week, close in enumerate(closes):
        for day in range(5):
            d = monday + pd.Timedelta(days=7 * week + day)
            rows.append((d, close, close, close, close, 100))
    frame = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    # 2000-01-17 ist Feiertag (MLK); Kurse an Nicht-Handelstagen stören die Wochen nicht.
    return frame


SEEN: list[pd.Timestamp] = []


class SpySma(SmaCross):
    """SMA-Kreuzung, die sich das letzte Datum jeder übergebenen Kursreihe merkt."""

    id = "test_spy_sma"

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        SEEN.append(pd.Timestamp(bars["date"].max()))
        return super().indicators(bars, p)


class HoldNoParams(Strategy):
    """Kauft einmal und hält; ohne Parameter (keine Nachbarschaft)."""

    id = "test_hold"
    name = "Halten"
    version = "1.0.0"
    description = "Kauft auf der ersten Kerze."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {}

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        return pd.DataFrame(index=bars.index)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None:
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        return []


# ---------------------------------------------------------------------------
# Parameter-Nachbarschaft
# ---------------------------------------------------------------------------


def test_neighborhood_grid_one_step_around_chosen_values() -> None:
    grid = neighborhood_grid(SmaCross, {"fast": 10, "slow": 40})
    assert grid == {"fast": [9, 10, 11], "slow": [35, 40, 45]}
    assert all(isinstance(v, int) for v in grid["fast"])


def test_neighborhood_grid_respects_bounds_and_floats() -> None:
    assert neighborhood_grid(SmaCross, {"fast": 2, "slow": 200}) == {
        "fast": [2, 3],
        "slow": [195, 200],
    }
    grid = neighborhood_grid(DonchianBreakout, {"period": 20, "atr_mult": 3.0})
    assert grid["atr_mult"] == [2.5, 3.0, 3.5]
    assert grid["period"] == [19, 20, 21]


def test_neighborhood_grid_keeps_params_without_step_fixed() -> None:
    class Mixed(HoldNoParams):
        params: ClassVar[dict[str, Param]] = {
            "flag": Param(True),
            "mode": Param("a", choices=("a", "b")),
            "level": Param(1.5, 0.0, 3.0),
        }

    assert neighborhood_grid(Mixed, {"flag": False, "mode": "b", "level": 2.0}) == {
        "flag": [False],
        "mode": ["b"],
        "level": [2.0],
    }


# ---------------------------------------------------------------------------
# Baselines und Ausschnitte
# ---------------------------------------------------------------------------


def test_buy_and_hold_curve_starts_at_last_close_before_start() -> None:
    daily = _weekly([100.0, 100.0, 110.0, 121.0])
    curve = buy_and_hold_curve(daily, Timeframe.W1, date(2000, 1, 10), date(2000, 1, 21), 1000.0)
    assert list(curve["date"].dt.date) == [date(2000, 1, 14), date(2000, 1, 21)]
    assert curve["equity"].tolist() == pytest.approx([1100.0, 1210.0])


def test_buy_and_hold_curve_without_earlier_close_enters_at_first_close() -> None:
    daily = _weekly([100.0, 100.0, 110.0, 121.0])
    curve = buy_and_hold_curve(daily, Timeframe.W1, date(1999, 1, 1), date(2000, 1, 14), 1000.0)
    assert curve["equity"].tolist() == pytest.approx([1000.0, 1000.0, 1100.0])
    empty = buy_and_hold_curve(daily.iloc[:0], Timeframe.W1, date(2000, 1, 1), date(2000, 2, 1), 1)
    assert empty.empty


def test_equal_weight_curve_averages_member_returns() -> None:
    daily = {"A": _weekly([100.0, 100.0, 110.0, 121.0]), "B": _weekly([50.0] * 4)}
    start, end = date(2000, 1, 10), date(2000, 1, 21)
    both = equal_weight_curve(daily, Timeframe.W1, start, end, 1000.0)
    assert both["equity"].tolist() == pytest.approx([1050.0, 1102.5])
    only_a = equal_weight_curve(daily, Timeframe.W1, start, end, 1000.0, lambda _d: {"A"})
    assert only_a["equity"].tolist() == pytest.approx([1100.0, 1210.0])
    everyone = equal_weight_curve(daily, Timeframe.W1, start, end, 1000.0, lambda _d: None)
    assert everyone["equity"].tolist() == pytest.approx([1050.0, 1102.5])
    nobody = equal_weight_curve(daily, Timeframe.W1, start, end, 1000.0, lambda _d: set())
    assert nobody["equity"].tolist() == pytest.approx([1000.0, 1000.0])


def test_equal_weight_curve_membership_uses_previous_bar() -> None:
    daily = {"A": _weekly([100.0, 100.0, 110.0, 121.0]), "B": _weekly([50.0] * 4)}

    def members(day: date) -> set[str]:
        return {"A"} if day < date(2000, 1, 14) else {"B"}

    curve = equal_weight_curve(
        daily, Timeframe.W1, date(2000, 1, 10), date(2000, 1, 21), 1000.0, members
    )
    # Woche bis 14.01.: Mitglieder vom 07.01. = A (+10 %); Woche bis 21.01.: B (0 %).
    assert curve["equity"].tolist() == pytest.approx([1100.0, 1100.0])


def test_equal_weight_curve_without_bars_is_empty() -> None:
    curve = equal_weight_curve(
        {"A": _weekly([1.0]).iloc[:0]}, Timeframe.W1, date(2000, 1, 1), date(2000, 2, 1), 1.0
    )
    assert curve.empty
    assert list(curve.columns) == ["date", "equity"]


def _curve(values: list[float], start: str = "2000-01-07") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="W-FRI")
    return pd.DataFrame({"date": dates, "equity": values})


def test_phase_stats_return_and_drawdown_per_phase() -> None:
    curve = _curve([100.0, 120.0, 90.0, 108.0, 150.0])
    phases = (
        MarketPhase("A", date(2000, 1, 10), date(2000, 1, 28)),
        MarketPhase("leer", date(2010, 1, 1), date(2010, 12, 31)),
    )
    table = phase_stats({"S": curve}, phases, 100.0)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["phase"] == "A"
    assert row["series"] == "S"
    # Basis 100 (07.01.), dann 120, 90, 108 → Rendite +8 %, Drawdown 90/120 − 1 = −25 %.
    assert row["return_pct"] == pytest.approx(8.0)
    assert row["max_drawdown_pct"] == pytest.approx(-25.0)


def test_phase_stats_uses_initial_capital_when_phase_starts_first() -> None:
    table = phase_stats(
        {"S": _curve([90.0, 99.0])},
        (MarketPhase("früh", date(1999, 1, 1), date(2000, 12, 31)),),
        100,
    )
    assert table.iloc[0]["return_pct"] == pytest.approx(-1.0)
    assert table.iloc[0]["max_drawdown_pct"] == pytest.approx(-10.0)


def test_slice_metrics_rebases_on_value_before_start() -> None:
    curve = _curve([100.0, 110.0, 121.0])
    m = slice_metrics(curve, Timeframe.W1, date(2000, 1, 10), date(2000, 1, 31), 100.0)
    assert m is not None
    assert m.start_equity == pytest.approx(100.0)
    assert m.total_return_pct == pytest.approx(21.0)
    assert slice_metrics(curve, Timeframe.W1, date(2005, 1, 1), date(2005, 2, 1), 100.0) is None


# ---------------------------------------------------------------------------
# Kriterien
# ---------------------------------------------------------------------------


def _metrics(study: StudyResult, **changes: Any) -> Any:
    return dataclasses.replace(study.in_sample.metrics, **changes)


def test_evaluate_criteria_all_passed(full_study: StudyResult) -> None:
    criteria = evaluate_criteria(
        _metrics(full_study, sharpe=1.0),
        _metrics(full_study, sharpe=0.7),
        _metrics(full_study, max_drawdown_pct=-20.0, num_trades=100),
        _metrics(full_study, max_drawdown_pct=-50.0),
        0.6,
        _metrics(full_study, total_return_pct=0.1),
    )
    assert [c.key for c in criteria] == ["sharpe", "drawdown", "trades", "robustness", "costs"]
    assert [c.passed for c in criteria] == [True] * 5
    assert criteria[0].value == "OOS 0,70 / IS 1,00"
    assert criteria[1].value == "-20,0 % gegenüber -50,0 %"
    assert criteria[3].value == "60 %"


def test_evaluate_criteria_failures(full_study: StudyResult) -> None:
    criteria = evaluate_criteria(
        _metrics(full_study, sharpe=1.0),
        _metrics(full_study, sharpe=0.69),
        _metrics(full_study, max_drawdown_pct=-50.0, num_trades=99),
        _metrics(full_study, max_drawdown_pct=-50.0),
        0.59,
        _metrics(full_study, total_return_pct=0.0),
    )
    assert [c.passed for c in criteria] == [False] * 5


def test_evaluate_criteria_open_and_non_positive_in_sample(full_study: StudyResult) -> None:
    open_ = evaluate_criteria(
        _metrics(full_study, sharpe=1.0),
        None,
        _metrics(full_study, num_trades=500),
        None,
        None,
        _metrics(full_study, total_return_pct=5.0),
    )
    assert [c.passed for c in open_] == [None, None, True, None, True]
    negative = evaluate_criteria(
        _metrics(full_study, sharpe=-0.5),
        _metrics(full_study, sharpe=2.0),
        _metrics(full_study),
        None,
        1.0,
        _metrics(full_study),
    )
    assert negative[0].passed is False
    assert "IS nicht positiv" in negative[0].value


def test_evaluate_criteria_custom_thresholds(full_study: StudyResult) -> None:
    criteria = evaluate_criteria(
        _metrics(full_study, sharpe=1.0),
        _metrics(full_study, sharpe=0.5),
        _metrics(full_study, num_trades=10),
        None,
        0.3,
        _metrics(full_study, total_return_pct=1.0),
        Thresholds(sharpe_ratio=0.5, min_trades=10, min_robustness=0.3, cost_factor=3.0),
    )
    assert [c.passed for c in criteria] == [True, None, True, True, True]
    assert criteria[4].name == "Positiv nach 3-fachen Kosten"


def test_verdict_rules(full_study: StudyResult) -> None:
    def with_states(*states: bool | None) -> StudyResult:
        crit = [
            dataclasses.replace(c, passed=s)
            for c, s in zip(full_study.criteria, states, strict=True)
        ]
        return dataclasses.replace(full_study, criteria=crit)

    assert with_states(True, True, True, True, True).verdict is Verdict.PASSED
    assert with_states(None, True, True, True, True).verdict is Verdict.OPEN
    assert with_states(None, False, True, True, True).verdict is Verdict.FAILED


def test_default_thresholds_match_issue() -> None:
    assert Thresholds(0.7, 100, 0.6, 2.0) == CRITERIA
    assert [p.name for p in MARKET_PHASES] == [
        "Dotcom-Crash 2000–2002",
        "Finanzkrise 2008–2009",
        "Corona 2020",
        "Zinswende 2022",
    ]
    assert date(2015, 12, 31) == IN_SAMPLE_END
    assert date(2016, 1, 1) == OUT_OF_SAMPLE_START


# ---------------------------------------------------------------------------
# Ganze Studie
# ---------------------------------------------------------------------------


def test_full_study_periods_and_parts(full_study: StudyResult) -> None:
    s = full_study
    assert s.data_end == date(2023, 12, 29)
    assert s.in_sample.start == date(2000, 1, 1)
    assert s.in_sample.end == date(2015, 12, 31)
    assert s.in_sample.result.equity["date"].max() <= pd.Timestamp("2015-12-31")
    assert s.out_of_sample is not None
    assert s.out_of_sample.start == date(2016, 1, 1)
    assert s.out_of_sample.result.equity["date"].min() >= pd.Timestamp("2016-01-01")
    assert s.full.end == date(2023, 12, 29)
    assert set(s.in_sample.baselines) == {"SPY", "QQQ", EQUAL_WEIGHT}
    assert list(s.curves) == ["Strategie", "Buy and Hold SPY", "Buy and Hold QQQ", EQUAL_WEIGHT]
    assert s.walk_forward is not None
    assert len(s.walk_forward.windows) == 19  # Tests 2005 … 2023
    assert s.robustness.source == "Parameter-Nachbarschaft"
    assert s.robustness.optimization is not None
    assert len(s.robustness.optimization.table) == 9
    assert s.monte_carlo is not None
    assert s.monte_carlo.n_simulations == 200
    assert s.monte_carlo.n_trades == len(s.full.result.trades)
    assert sorted(set(s.phases["phase"])) == sorted(p.name for p in MARKET_PHASES)
    assert len(s.phases) == 16
    assert s.full.metrics.num_trades == len(s.full.result.trades)


def test_full_study_cost_stress_doubles_costs(full_study: StudyResult) -> None:
    stress = full_study.cost_stress.result
    normal = full_study.full.result
    assert stress.settings["fee_per_order"] == pytest.approx(2 * normal.settings["fee_per_order"])
    assert stress.settings["slippage_pct"] == pytest.approx(2 * normal.settings["slippage_pct"])
    assert stress.final_equity < normal.final_equity
    assert full_study.criteria[4].value.endswith("%")


def test_full_study_criteria_match_measurements(full_study: StudyResult) -> None:
    s = full_study
    by_key = {c.key: c for c in s.criteria}
    assert by_key["trades"].passed is (s.full.metrics.num_trades >= 100)
    assert by_key["drawdown"].passed is (
        s.full.metrics.max_drawdown_pct > s.full.baselines["SPY"].max_drawdown_pct
    )
    assert by_key["costs"].passed is (s.cost_stress.metrics.total_return_pct > 0)
    assert s.robustness.value is not None
    assert by_key["robustness"].passed is (s.robustness.value >= 0.6)
    assert s.verdict in (Verdict.PASSED, Verdict.FAILED)


def test_study_is_deterministic(
    full_study: StudyResult, bars: dict[str, pd.DataFrame], settings: Settings
) -> None:
    again = run_study(
        SmaCross(),
        {"fast": 10, "slow": 40},
        Timeframe.W1,
        settings,
        symbol="AAA",
        bars=bars,
        config=StudyConfig(workers=1, mc_simulations=200),
    )
    assert summary_dict(again) == summary_dict(full_study)
    assert again.full.result.fingerprint() == full_study.full.result.fingerprint()
    assert again.monte_carlo is not None
    assert full_study.monte_carlo is not None
    assert again.monte_carlo.fingerprint() == full_study.monte_carlo.fingerprint()


def test_in_sample_only_never_sees_later_prices(
    bars: dict[str, pd.DataFrame], settings: Settings
) -> None:
    SEEN.clear()
    steps: list[str] = []
    study = run_study(
        SpySma,
        None,
        "1W",
        settings,
        symbol="AAA",
        bars=bars,
        config=StudyConfig(include_out_of_sample=False, workers=1, mc_simulations=50),
        progress=steps.append,
    )
    assert SEEN
    assert max(SEEN) <= pd.Timestamp("2015-12-31")
    assert study.out_of_sample is None
    assert study.data_end == date(2015, 12, 31)
    assert study.full.result is study.in_sample.result
    assert study.criteria[0].passed is None
    assert study.verdict is not Verdict.PASSED
    assert any("Out-of-Sample nicht ausgewertet" in line for line in study.log)
    assert "Out-of-Sample-Backtest" not in steps
    assert steps[0] == "Kurse laden"
    assert study.walk_forward is not None
    assert study.walk_forward.windows[-1].span.test_end <= date(2015, 12, 31)


def test_portfolio_study_with_index_history(
    bars: dict[str, pd.DataFrame], settings: Settings
) -> None:
    history = IndexHistory(
        index_code="TEST",
        current=frozenset({"AAA", "CCC"}),
        changes=(IndexChange(date(2012, 1, 3), frozenset({"CCC"}), frozenset({"BBB"})),),
    )
    config = StudyConfig(
        in_sample_start=date(2008, 1, 1),
        train_years=3,
        workers=1,
        mc_simulations=50,
        max_positions=2,
    )
    study = run_study(
        SmaCross,
        {"fast": 5, "slow": 20},
        "W1",
        settings,
        universe=history,
        bars=bars,
        config=config,
    )
    assert study.universe == "TEST"
    assert study.in_sample.start == date(2008, 1, 1)
    traded = set(study.full.result.trades["symbol"])
    assert traded <= {"AAA", "BBB", "CCC"}
    assert EQUAL_WEIGHT in study.curves
    assert study.walk_forward is not None
    assert study.walk_forward.windows[0].span.test_start == date(2011, 1, 1)


def test_portfolio_study_with_symbol_list(
    bars: dict[str, pd.DataFrame], settings: Settings
) -> None:
    config = StudyConfig(in_sample_start=date(2010, 1, 1), train_years=3, workers=1)
    study = run_study(
        SmaCross, None, "W1", settings, universe=["BBB", "AAA"], bars=bars, config=config
    )
    assert study.universe == "Liste: AAA, BBB"
    assert study.out_of_sample is not None


def test_robustness_falls_back_to_walk_forward_windows(
    bars: dict[str, pd.DataFrame], settings: Settings
) -> None:
    config = StudyConfig(in_sample_start=date(2010, 1, 1), train_years=2, workers=1)
    study = run_study(HoldNoParams, None, "W1", settings, symbol="SPY", bars=bars, config=config)
    assert study.robustness.source == "Walk-Forward-Fenster"
    assert study.walk_forward is not None
    values = [w.test_value for w in study.walk_forward.windows if w.test_value is not None]
    expected = sum(v > 0 for v in values) / len(values)
    assert study.robustness.value == pytest.approx(expected)


def test_missing_benchmarks_leave_drawdown_open(
    bars: dict[str, pd.DataFrame], settings: Settings
) -> None:
    only = {"AAA": bars["AAA"]}
    config = StudyConfig(in_sample_start=date(2010, 1, 1), train_years=2, workers=1)
    study = run_study(SmaCross, None, "W1", settings, symbol="AAA", bars=only, config=config)
    assert study.criteria[1].passed is None
    assert any("Keine Kurse für den Benchmark SPY" in line for line in study.log)
    assert any("Keine Kurse für den Benchmark QQQ" in line for line in study.log)
    assert list(study.curves) == ["Strategie", EQUAL_WEIGHT]
    assert study.verdict is not Verdict.PASSED


def test_end_date_and_missing_out_of_sample(
    bars: dict[str, pd.DataFrame], settings: Settings
) -> None:
    config = StudyConfig(in_sample_start=date(2010, 1, 1), train_years=2, end=date(2015, 6, 30))
    study = run_study(SmaCross, None, "W1", settings, symbol="AAA", bars=bars, config=config)
    assert study.out_of_sample is None
    assert study.data_end == date(2015, 6, 30)
    assert study.in_sample.end == date(2015, 6, 30)
    assert study.in_sample.result.equity["date"].max() == pd.Timestamp("2015-06-26")
    assert any("Out-of-Sample entfällt" in line for line in study.log)
    assert any("vor dem Ende des In-Sample-Zeitraums" in line for line in study.log)


def test_walk_forward_failure_is_logged(bars: dict[str, pd.DataFrame], settings: Settings) -> None:
    config = StudyConfig(
        in_sample_start=date(2014, 1, 1), train_years=20, workers=1, mc_simulations=10
    )
    study = run_study(SmaCross, None, "W1", settings, symbol="AAA", bars=bars, config=config)
    assert study.walk_forward is None
    assert any(line.startswith("Walk-Forward nicht möglich") for line in study.log)


def test_custom_walk_forward_grid(bars: dict[str, pd.DataFrame], settings: Settings) -> None:
    config = StudyConfig(in_sample_start=date(2010, 1, 1), train_years=3, workers=1)
    study = run_study(
        SmaCross,
        {"fast": 10, "slow": 40},
        "W1",
        settings,
        symbol="AAA",
        bars=bars,
        config=config,
        grid={"fast": [5, 20]},
    )
    assert study.walk_forward is not None
    assert {w.params["fast"] for w in study.walk_forward.windows} <= {5, 20}
    assert {w.params["slow"] for w in study.walk_forward.windows} == {40}


def test_monte_carlo_without_trades_is_logged(
    bars: dict[str, pd.DataFrame], settings: Settings
) -> None:
    flat = bars["AAA"].assign(open=10.0, high=10.0, low=10.0, close=10.0)
    config = StudyConfig(in_sample_start=date(2010, 1, 1), train_years=2, workers=1)
    study = run_study(
        SmaCross, None, "W1", settings, symbol="AAA", bars={"AAA": flat}, config=config
    )
    assert study.monte_carlo is None
    assert any(line.startswith("Monte-Carlo nicht möglich") for line in study.log)
    assert study.robustness.value is None or study.robustness.value == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "genau eines"),
        ({"symbol": "AAA", "universe": ["AAA"]}, "genau eines"),
        ({"symbol": "ZZZ"}, "Keine Kurse für die Werte"),
        ({"universe": []}, "keine Symbole"),
    ],
)
def test_invalid_inputs(
    kwargs: dict[str, Any], message: str, bars: dict[str, pd.DataFrame], settings: Settings
) -> None:
    with pytest.raises(StudyError, match=message):
        run_study(SmaCross, None, "W1", settings, bars=bars, **kwargs)


def test_invalid_periods_and_old_data(bars: dict[str, pd.DataFrame], settings: Settings) -> None:
    bad = StudyConfig(in_sample_end=date(2017, 1, 1))
    with pytest.raises(StudyError, match="Zeiträume"):
        run_study(SmaCross, None, "W1", settings, symbol="AAA", bars=bars, config=bad)
    old = {"AAA": bars["AAA"][bars["AAA"]["date"] < "1998-01-01"]}
    with pytest.raises(StudyError, match="vor dem Beginn"):
        run_study(SmaCross, None, "W1", settings, symbol="AAA", bars=old)


def test_parse_timeframe() -> None:
    assert parse_timeframe("W1") is Timeframe.W1
    assert parse_timeframe("1w") is Timeframe.W1
    assert parse_timeframe("d3") is Timeframe.D3
    assert parse_timeframe(Timeframe.MN1) is Timeframe.MN1
    with pytest.raises(StudyError, match="Unbekannter Zeitrahmen"):
        parse_timeframe("2W")


# ---------------------------------------------------------------------------
# Kurse aus der Datenbank
# ---------------------------------------------------------------------------


def test_load_study_bars_reads_universe_and_benchmarks(
    tmp_path: Path, bars: dict[str, pd.DataFrame]
) -> None:
    settings = Settings(data_dir=tmp_path)
    with pytest.raises(StudyError, match="existiert nicht"):
        load_study_bars(settings, "AAA", None)
    with MarketStore(settings.database_path) as store:
        for sym in ("AAA", "SPY"):
            store.upsert_bars(sym, bars[sym].iloc[:300])
    loaded = load_study_bars(settings, "AAA", None, StudyConfig(end=date(1995, 3, 31)))
    assert sorted(loaded) == ["AAA", "SPY"]
    assert loaded["AAA"]["date"].max() <= pd.Timestamp("1995-03-31")
    listed = load_study_bars(settings, None, ["AAA", "BBB"])
    assert sorted(listed) == ["AAA", "SPY"]
    assert np.allclose(listed["AAA"]["close"].to_numpy(), bars["AAA"]["close"].iloc[:300])


def test_load_study_bars_skips_excluded_symbols(
    tmp_path: Path, bars: dict[str, pd.DataFrame]
) -> None:
    settings = Settings(data_dir=tmp_path)
    with MarketStore(settings.database_path) as store:
        for sym in ("AAA", "BBB", "SPY"):
            store.upsert_bars(sym, bars[sym].iloc[:300] if sym in bars else bars["AAA"].iloc[:300])
        store.set_exclusion("BBB", "Fremdkurse", "auto")
        store.set_exclusion("SPY", "von Hand", "manuell")  # Benchmarks bleiben
    log: list[str] = []
    listed = load_study_bars(settings, None, ["AAA", "BBB"], log=log)
    assert sorted(listed) == ["AAA", "SPY"]
    assert len(log) == 1
    assert "BBB" in log[0]
    assert sorted(load_study_bars(settings, "BBB", None)) == ["BBB", "SPY"]


def test_load_study_bars_for_index_code(
    tmp_path: Path, bars: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_helfer.research import study as study_module

    history = IndexHistory("TEST", frozenset({"AAA"}), ())
    monkeypatch.setattr(study_module, "load_history", lambda code: history)
    settings = Settings(data_dir=tmp_path)
    with MarketStore(settings.database_path) as store:
        store.upsert_bars("AAA", bars["AAA"].iloc[:50])
    loaded = load_study_bars(settings, None, "TEST")
    assert list(loaded) == ["AAA"]
