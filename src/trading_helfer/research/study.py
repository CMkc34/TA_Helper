"""Fester Studienablauf für Kandidaten-Strategien (R-01).

Ziel ist eine robuste 3D/1W-Strategie, die über Marktphasen hinweg den Benchmark
risikobereinigt schlägt. Damit kein Ergebnis durch Überanpassung zustande kommt, prüft
:func:`run_study` jeden Kandidaten gleich:

1. **In-Sample** 2000–2015: Backtest mit den gewählten Parametern.
2. **Walk-Forward** 5/1 Jahre (rollierend) ab 2000 über den ausgewerteten Zeitraum. Ohne
   eigenes Raster wird die Parameter-Nachbarschaft (siehe 3.) optimiert.
3. **Robustheit**: Raster-Optimierung im In-Sample über die Nachbarschaft der gewählten
   Parameter (jeder Zahlenparameter einen ``step`` darunter, gleich, einen darüber).
   Robustheitsmaß = Anteil der Nachbarn mit positivem Sharpe
   (:func:`~trading_helfer.backtest.optimize.robustness_scores`). Hat die Strategie keine
   Nachbarn (keine Zahlenparameter mit ``step``), gilt der Anteil der Walk-Forward-Fenster
   mit positivem Test-Sharpe.
4. **Gesamtzeitraum** 2000 bis Datenende: Grundlage für Trades, Max Drawdown,
   Monte-Carlo (Umordnung der Trades, Seed fest) und Marktphasen.
5. **Kosten-Stresstest**: Gesamtzeitraum mit doppelten Kosten (Gebühr und Slippage).
6. **Marktphasen** 2000–02, 2008–09, 2020, 2022: Rendite und Max Drawdown je Phase.
7. **Out-of-Sample** 2016 bis Datenende: erst ganz am Ende. Mit
   ``include_out_of_sample=False`` werden alle Kurse nach dem In-Sample-Ende vor jeder
   Berechnung abgeschnitten; so kann eine Strategie entwickelt werden, ohne die
   Out-of-Sample-Daten je zu sehen.

Baselines (ohne Kosten, gleicher Zeitrahmen): Buy and Hold SPY und QQQ sowie das
gleichgewichtete Universum (je Kerze Durchschnitt der Renditen aller Werte, die zu Beginn
der Kerze Indexmitglied waren; bei einem einzelnen Wert dessen Buy and Hold).

Bewertungskriterien (verbindlich für R-02 bis R-04, siehe :data:`CRITERIA`):

- Out-of-Sample-Sharpe ≥ 0,7 × In-Sample-Sharpe (In-Sample-Sharpe muss positiv sein)
- Max Drawdown im Gesamtzeitraum kleiner als der von Buy and Hold SPY
- mindestens 100 abgeschlossene Trades im Gesamtzeitraum
- Robustheitsmaß ≥ 60 %
- Gesamtrendite nach doppelten Kosten positiv

Beispiel::

    study = run_study(SmaCross, {"fast": 10, "slow": 40}, Timeframe.W1, settings,
                      universe="SP500")
    print(study.verdict, [c.passed for c in study.criteria])
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date as Date  # noqa: N812
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import BacktestError, BacktestResult, run_backtest
from trading_helfer.backtest.metrics import MetricsError, PerformanceMetrics, compute_metrics
from trading_helfer.backtest.monte_carlo import (
    MonteCarloError,
    MonteCarloMethod,
    MonteCarloResult,
    monte_carlo_for_result,
)
from trading_helfer.backtest.optimize import (
    Objective,
    OptimizationError,
    OptimizationResult,
    optimize,
)
from trading_helfer.backtest.portfolio import (
    DEFAULT_MAX_POSITIONS,
    Universe,
    exclusion_note,
    run_portfolio_backtest,
)
from trading_helfer.backtest.walkforward import WalkForwardResult, walk_forward
from trading_helfer.core.bars import DATE_COLUMN
from trading_helfer.core.types import Timeframe
from trading_helfer.data.resample import PARTIAL_COLUMN, resample
from trading_helfer.data.universe import SURVIVORSHIP_PREFIX, IndexHistory, load_history
from trading_helfer.strategies.base import Strategy

if TYPE_CHECKING:
    from trading_helfer.core.config import Settings

__all__ = [
    "BENCHMARKS",
    "CRITERIA",
    "EQUAL_WEIGHT",
    "IN_SAMPLE_END",
    "IN_SAMPLE_START",
    "MARKET_PHASES",
    "OUT_OF_SAMPLE_START",
    "PRIMARY_BENCHMARK",
    "Criterion",
    "MarketPhase",
    "PeriodResult",
    "Robustness",
    "StudyConfig",
    "StudyError",
    "StudyResult",
    "Thresholds",
    "Verdict",
    "buy_and_hold_curve",
    "equal_weight_curve",
    "evaluate_criteria",
    "load_study_bars",
    "neighborhood_grid",
    "parse_timeframe",
    "phase_stats",
    "run_study",
    "slice_metrics",
]

IN_SAMPLE_START: Final = Date(2000, 1, 1)
"""Beginn des In-Sample-Zeitraums (und des Gesamtzeitraums)."""

IN_SAMPLE_END: Final = Date(2015, 12, 31)
"""Ende des In-Sample-Zeitraums."""

OUT_OF_SAMPLE_START: Final = Date(2016, 1, 1)
"""Beginn des Out-of-Sample-Zeitraums (bis zum Datenende)."""

PRIMARY_BENCHMARK: Final = "SPY"
"""Benchmark für das Drawdown-Kriterium."""

BENCHMARKS: Final[tuple[str, ...]] = ("SPY", "QQQ")
"""Buy-and-Hold-Baselines."""

EQUAL_WEIGHT: Final = "Universum gleichgewichtet"
"""Name der Baseline „gleichgewichtetes Universum“."""

STRATEGY_LABEL: Final = "Strategie"
"""Name der Strategie in Tabellen."""

ProgressCallback = Callable[[str], None]
"""Fortschritt: Beschreibung des aktuellen Schritts (Deutsch)."""


class StudyError(ValueError):
    """Studie kann nicht ausgeführt werden; Text auf Deutsch."""


class Verdict(StrEnum):
    """Gesamturteil einer Studie."""

    PASSED = "bestanden"
    FAILED = "nicht bestanden"
    OPEN = "offen"


@dataclass(frozen=True)
class MarketPhase:
    """Marktphase für die Auswertung je Phase (Grenzen eingeschlossen).

    Attributes:
        name: Bezeichnung (Deutsch).
        start: Erster Tag.
        end: Letzter Tag.
    """

    name: str
    start: Date
    end: Date


MARKET_PHASES: Final[tuple[MarketPhase, ...]] = (
    MarketPhase("Dotcom-Crash 2000–2002", Date(2000, 1, 1), Date(2002, 12, 31)),
    MarketPhase("Finanzkrise 2008–2009", Date(2008, 1, 1), Date(2009, 12, 31)),
    MarketPhase("Corona 2020", Date(2020, 1, 1), Date(2020, 12, 31)),
    MarketPhase("Zinswende 2022", Date(2022, 1, 1), Date(2022, 12, 31)),
)
"""Marktphasen laut R-01."""


@dataclass(frozen=True)
class Thresholds:
    """Grenzwerte der Bewertungskriterien.

    Attributes:
        sharpe_ratio: Out-of-Sample-Sharpe muss mindestens dieses Vielfache des
            In-Sample-Sharpe erreichen.
        min_trades: Mindestanzahl abgeschlossener Trades im Gesamtzeitraum.
        min_robustness: Mindestwert des Robustheitsmaßes (0 bis 1).
        cost_factor: Faktor für Gebühren und Slippage im Kosten-Stresstest.
    """

    sharpe_ratio: float = 0.7
    min_trades: int = 100
    min_robustness: float = 0.6
    cost_factor: float = 2.0


CRITERIA: Final = Thresholds()
"""Verbindliche Grenzwerte für R-02 bis R-04."""


@dataclass(frozen=True)
class StudyConfig:
    """Fester Rahmen einer Studie. Die Standardwerte sind für R-02 bis R-04 verbindlich.

    Attributes:
        in_sample_start: Beginn In-Sample und Gesamtzeitraum.
        in_sample_end: Ende In-Sample.
        out_of_sample_start: Beginn Out-of-Sample.
        end: Letzter Datentag; ``None`` = letzter Tag mit Kursen des Universums.
        include_out_of_sample: ``False`` schneidet alle Kurse nach ``in_sample_end`` ab.
        train_years: Walk-Forward-Training in Jahren.
        test_years: Walk-Forward-Test in Jahren.
        mc_simulations: Anzahl der Monte-Carlo-Simulationen.
        mc_seed: Seed der Monte-Carlo-Simulation.
        phases: Marktphasen.
        benchmarks: Buy-and-Hold-Baselines (Symbole).
        thresholds: Grenzwerte der Kriterien.
        max_positions: Maximale Positionen im Portfolio-Backtest.
        workers: Prozesse bzw. Threads für Optimierung und Portfolio (``None`` = CPU-Kerne).
    """

    in_sample_start: Date = IN_SAMPLE_START
    in_sample_end: Date = IN_SAMPLE_END
    out_of_sample_start: Date = OUT_OF_SAMPLE_START
    end: Date | None = None
    include_out_of_sample: bool = True
    train_years: int = 5
    test_years: int = 1
    mc_simulations: int = 1000
    mc_seed: int = 42
    phases: tuple[MarketPhase, ...] = MARKET_PHASES
    benchmarks: tuple[str, ...] = BENCHMARKS
    thresholds: Thresholds = CRITERIA
    max_positions: int = DEFAULT_MAX_POSITIONS
    workers: int | None = None


@dataclass(frozen=True)
class PeriodResult:
    """Backtest eines Zeitraums mit Kennzahlen.

    Attributes:
        name: Bezeichnung (z. B. „In-Sample“).
        start: Erster Handelstag.
        end: Letzter Datentag.
        result: Backtest-Ergebnis.
        metrics: Kennzahlen der Strategie.
        baselines: Kennzahlen der Baselines im selben Zeitraum (Name → Kennzahlen).
    """

    name: str
    start: Date
    end: Date
    result: BacktestResult
    metrics: PerformanceMetrics
    baselines: dict[str, PerformanceMetrics] = field(default_factory=dict)


@dataclass(frozen=True)
class Robustness:
    """Robustheitsmaß der gewählten Parameter.

    Attributes:
        value: Anteil (0 bis 1) oder ``None``, wenn nicht bestimmbar.
        source: Herkunft („Parameter-Nachbarschaft“ oder „Walk-Forward-Fenster“).
        optimization: In-Sample-Optimierung über die Nachbarschaft (``None`` bei Fehler).
    """

    value: float | None
    source: str
    optimization: OptimizationResult | None


@dataclass(frozen=True)
class Criterion:
    """Ergebnis eines Bewertungskriteriums.

    Attributes:
        key: Kurzname (``sharpe``, ``drawdown``, ``trades``, ``robustness``, ``costs``).
        name: Beschreibung des Kriteriums (Deutsch).
        value: Gemessener Wert als Text.
        passed: ``True``/``False`` oder ``None``, wenn (noch) nicht prüfbar.
    """

    key: str
    name: str
    value: str
    passed: bool | None


@dataclass(frozen=True)
class StudyResult:
    """Ergebnis einer Studie (Grundlage für :func:`~trading_helfer.research.report.write_report`).

    Attributes:
        strategy_id: ID der Strategie.
        strategy_name: Name der Strategie.
        strategy_version: Version der Strategie.
        params: Gewählte Parameter.
        timeframe: Zeitrahmen.
        universe: Name des Universums oder Symbol.
        config: Rahmen der Studie.
        costs: Kostenmodell des Normalfalls.
        initial_capital: Startkapital.
        data_end: Letzter ausgewerteter Datentag.
        in_sample: In-Sample-Backtest.
        out_of_sample: Out-of-Sample-Backtest oder ``None``, wenn nicht ausgewertet.
        full: Gesamtzeitraum.
        cost_stress: Gesamtzeitraum mit erhöhten Kosten.
        walk_forward: Walk-Forward-Analyse oder ``None`` (Grund im Protokoll).
        robustness: Robustheitsmaß.
        monte_carlo: Monte-Carlo-Umordnung der Trades des Gesamtzeitraums oder ``None``.
        curves: Equity-Kurven des Gesamtzeitraums (Strategie und Baselines; Spalten
            ``date``, ``equity``).
        phases: Rendite und Max Drawdown je Marktphase und Reihe.
        criteria: Bewertungskriterien.
        log: Hinweise, auf Deutsch.
    """

    strategy_id: str
    strategy_name: str
    strategy_version: str
    params: dict[str, Any]
    timeframe: Timeframe
    universe: str
    config: StudyConfig
    costs: CostModel
    initial_capital: float
    data_end: Date
    in_sample: PeriodResult
    out_of_sample: PeriodResult | None
    full: PeriodResult
    cost_stress: PeriodResult
    walk_forward: WalkForwardResult | None
    robustness: Robustness
    monte_carlo: MonteCarloResult | None
    curves: dict[str, pd.DataFrame]
    phases: pd.DataFrame
    criteria: list[Criterion]
    log: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict:
        """Gesamturteil: bestanden nur, wenn alle Kriterien erfüllt sind."""
        states = [c.passed for c in self.criteria]
        if any(s is False for s in states):
            return Verdict.FAILED
        if any(s is None for s in states):
            return Verdict.OPEN
        return Verdict.PASSED


# ---------------------------------------------------------------------------
# Öffentliche Funktion
# ---------------------------------------------------------------------------


def run_study(
    strategy: Strategy | type[Strategy],
    params: Mapping[str, Any] | None,
    timeframe: Timeframe | str,
    settings: Settings,
    *,
    symbol: str | None = None,
    universe: Universe | None = None,
    bars: Mapping[str, pd.DataFrame] | None = None,
    config: StudyConfig | None = None,
    grid: Mapping[str, Sequence[Any]] | None = None,
    progress: ProgressCallback | None = None,
) -> StudyResult:
    """Prüft eine Strategie nach dem festen Studienablauf.

    Args:
        strategy: Strategie (Instanz oder Klasse).
        params: Gewählte Parameter; fehlende erhalten den Standardwert.
        timeframe: Zeitrahmen (3D oder 1W empfohlen).
        settings: Einstellungen (Kapital, Kosten, risikofreier Zins, Datenbank).
        symbol: Einzelner Wert; genau eines von ``symbol`` und ``universe``.
        universe: Index-Code, Index-Historie oder Symbolliste für einen Portfolio-Backtest.
        bars: Bereinigte Tageskurse je Symbol (inklusive Benchmarks) statt Laden aus der
            Datenbank.
        config: Rahmen der Studie; ``None`` = verbindlicher Standard.
        grid: Raster für die Walk-Forward-Optimierung; ``None`` = Parameter-Nachbarschaft.
        progress: Fortschritt je Schritt.

    Returns:
        Alle Ergebnisse und die Bewertungskriterien.

    Raises:
        StudyError: Ungültige Angaben oder keine Kurse.
        BacktestError: Fehler im Backtest (z. B. Strategie unterstützt den Zeitrahmen nicht).
    """
    cfg = config if config is not None else StudyConfig()
    template = strategy() if isinstance(strategy, type) else strategy
    tf = parse_timeframe(timeframe)
    if (symbol is None) == (universe is None):
        raise StudyError("Bitte genau eines angeben: ein Symbol oder ein Universum.")
    if not cfg.in_sample_start <= cfg.in_sample_end < cfg.out_of_sample_start:
        raise StudyError("Zeiträume ungültig: In-Sample muss vor Out-of-Sample liegen.")
    chosen = template.make_params(params).to_dict()
    say = progress if progress is not None else _quiet
    log: list[str] = []

    say("Kurse laden")
    daily = (
        dict(bars)
        if bars is not None
        else load_study_bars(settings, symbol, universe, cfg, log=log)
    )
    daily = _cut(daily, cfg)
    universe_name, strategy_symbols, members = _universe_parts(symbol, universe, cfg)
    data_end = _data_end(daily, strategy_symbols, cfg)
    if data_end < cfg.in_sample_end and cfg.include_out_of_sample:
        log.append(f"Kurse enden am {data_end}, vor dem Ende des In-Sample-Zeitraums.")
    oos_ready = cfg.include_out_of_sample and data_end >= cfg.out_of_sample_start
    if not cfg.include_out_of_sample:
        log.append(
            f"Out-of-Sample nicht ausgewertet: alle Kurse nach {cfg.in_sample_end} "
            "wurden vor der Berechnung abgeschnitten."
        )
    elif not oos_ready:
        log.append(f"Keine Kurse ab {cfg.out_of_sample_start}; Out-of-Sample entfällt.")
    full_end = data_end if oos_ready else min(data_end, cfg.in_sample_end)

    costs = CostModel.from_settings(settings)
    runner = _Runner(template, chosen, tf, settings, symbol, universe, daily, cfg)
    rf = settings.risk_free_rate_pct

    say("Baselines berechnen")
    curves = _baseline_curves(
        daily, tf, cfg, strategy_symbols, members, full_end, settings.default_capital, log
    )

    say("In-Sample-Backtest")
    is_end = min(cfg.in_sample_end, full_end)
    in_sample = runner.period("In-Sample", cfg.in_sample_start, is_end, costs, curves, rf)

    say("Robustheit (Parameter-Nachbarschaft)")
    neighborhood = neighborhood_grid(template, chosen)
    optimization = runner.optimize(neighborhood, cfg.in_sample_start, is_end, log)

    say("Walk-Forward")
    wf = runner.walk_forward(
        grid if grid is not None else neighborhood, cfg.in_sample_start, full_end, log
    )
    robustness = _robustness(optimization, chosen, wf)

    if full_end == is_end:
        full = PeriodResult(
            "Gesamtzeitraum",
            in_sample.start,
            in_sample.end,
            in_sample.result,
            in_sample.metrics,
            in_sample.baselines,
        )
    else:
        say("Backtest Gesamtzeitraum")
        full = runner.period("Gesamtzeitraum", cfg.in_sample_start, full_end, costs, curves, rf)

    say("Kosten-Stresstest")
    factor = cfg.thresholds.cost_factor
    stressed = CostModel(
        fee_per_order=costs.fee_per_order * factor,
        fee_pct=costs.fee_pct * factor,
        slippage_pct=min(costs.slippage_pct * factor, 99.0),
        borrow_fee_pct=costs.borrow_fee_pct * factor,
    )
    stress = runner.period(
        f"Kosten × {factor:g}", cfg.in_sample_start, full_end, stressed, curves, rf
    )

    say("Monte-Carlo")
    mc: MonteCarloResult | None = None
    try:
        mc = monte_carlo_for_result(
            full.result,
            n_simulations=cfg.mc_simulations,
            method=MonteCarloMethod.SHUFFLE,
            seed=cfg.mc_seed,
        )
    except MonteCarloError as exc:
        log.append(f"Monte-Carlo nicht möglich: {exc}")

    all_curves = {STRATEGY_LABEL: full.result.equity.loc[:, [DATE_COLUMN, "equity"]], **curves}
    phases = phase_stats(all_curves, cfg.phases, full.result.initial_capital)

    out_of_sample: PeriodResult | None = None
    if oos_ready:
        say("Out-of-Sample-Backtest")
        out_of_sample = runner.period(
            "Out-of-Sample", cfg.out_of_sample_start, full_end, costs, curves, rf
        )

    primary = full.baselines.get(PRIMARY_BENCHMARK)
    criteria = evaluate_criteria(
        in_sample.metrics,
        out_of_sample.metrics if out_of_sample is not None else None,
        full.metrics,
        primary,
        robustness.value,
        stress.metrics,
        cfg.thresholds,
    )
    log.extend(line for line in full.result.log if line.startswith(SURVIVORSHIP_PREFIX))
    return StudyResult(
        strategy_id=template.id,
        strategy_name=template.name,
        strategy_version=template.version,
        params=chosen,
        timeframe=tf,
        universe=universe_name,
        config=cfg,
        costs=costs,
        initial_capital=float(settings.default_capital),
        data_end=full_end,
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        full=full,
        cost_stress=stress,
        walk_forward=wf,
        robustness=robustness,
        monte_carlo=mc,
        curves=all_curves,
        phases=phases,
        criteria=criteria,
        log=log,
    )


def _quiet(_text: str) -> None:
    return None


def parse_timeframe(value: Timeframe | str) -> Timeframe:
    """Zeitrahmen aus Name (``W1``) oder Wert (``1W``).

    Raises:
        StudyError: Unbekannter Zeitrahmen.
    """
    if isinstance(value, Timeframe):
        return value
    text = str(value).strip().upper()
    for tf in Timeframe:
        if text in (tf.name, tf.value.upper()):
            return tf
    known = ", ".join(f"{tf.name}/{tf.value}" for tf in Timeframe)
    raise StudyError(f"Unbekannter Zeitrahmen {value!r}. Bekannt: {known}.")


# ---------------------------------------------------------------------------
# Kriterien
# ---------------------------------------------------------------------------


def evaluate_criteria(
    in_sample: PerformanceMetrics,
    out_of_sample: PerformanceMetrics | None,
    full: PerformanceMetrics,
    benchmark: PerformanceMetrics | None,
    robustness: float | None,
    cost_stress: PerformanceMetrics,
    thresholds: Thresholds = CRITERIA,
) -> list[Criterion]:
    """Prüft die fünf Bewertungskriterien.

    Args:
        in_sample: Kennzahlen In-Sample.
        out_of_sample: Kennzahlen Out-of-Sample oder ``None`` (Kriterium dann offen).
        full: Kennzahlen des Gesamtzeitraums.
        benchmark: Kennzahlen von Buy and Hold des Haupt-Benchmarks im Gesamtzeitraum.
        robustness: Robustheitsmaß (0 bis 1) oder ``None``.
        cost_stress: Kennzahlen des Gesamtzeitraums mit erhöhten Kosten.
        thresholds: Grenzwerte.

    Returns:
        Kriterien in fester Reihenfolge: Sharpe, Drawdown, Trades, Robustheit, Kosten.
    """
    t = thresholds
    criteria: list[Criterion] = []

    name = f"Out-of-Sample-Sharpe ≥ {_num(t.sharpe_ratio, 1)} × In-Sample-Sharpe"
    is_sharpe = in_sample.sharpe
    if out_of_sample is None:
        criteria.append(Criterion("sharpe", name, "Out-of-Sample noch nicht ausgewertet", None))
    else:
        oos_sharpe = out_of_sample.sharpe
        value = f"OOS {_num(oos_sharpe, 2)} / IS {_num(is_sharpe, 2)}"
        if is_sharpe is None or is_sharpe <= 0 or oos_sharpe is None:
            criteria.append(Criterion("sharpe", name, value + " (IS nicht positiv)", False))
        else:
            ok = oos_sharpe >= t.sharpe_ratio * is_sharpe
            criteria.append(Criterion("sharpe", name, value, ok))

    name = f"Max Drawdown kleiner als Buy and Hold {PRIMARY_BENCHMARK}"
    if benchmark is None:
        criteria.append(Criterion("drawdown", name, f"keine Kurse für {PRIMARY_BENCHMARK}", None))
    else:
        value = (
            f"{_num(full.max_drawdown_pct, 1)} % gegenüber {_num(benchmark.max_drawdown_pct, 1)} %"
        )
        ok = full.max_drawdown_pct > benchmark.max_drawdown_pct
        criteria.append(Criterion("drawdown", name, value, ok))

    criteria.append(
        Criterion(
            "trades",
            f"Mindestens {t.min_trades} Trades",
            str(full.num_trades),
            full.num_trades >= t.min_trades,
        )
    )

    name = f"Robustheitsmaß ≥ {_num(t.min_robustness * 100, 0)} %"
    if robustness is None:
        criteria.append(Criterion("robustness", name, "nicht bestimmbar", None))
    else:
        criteria.append(
            Criterion(
                "robustness",
                name,
                f"{_num(robustness * 100, 0)} %",
                robustness >= t.min_robustness,
            )
        )

    criteria.append(
        Criterion(
            "costs",
            f"Positiv nach {_num(t.cost_factor, 0)}-fachen Kosten",
            f"{_num(cost_stress.total_return_pct, 1)} %",
            cost_stress.total_return_pct > 0,
        )
    )
    return criteria


def _num(value: float | None, digits: int) -> str:
    """Zahl mit deutschem Dezimalkomma; ``None`` = „–“."""
    if value is None or not math.isfinite(value):
        return "–"
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


# ---------------------------------------------------------------------------
# Parameter-Nachbarschaft und Robustheit
# ---------------------------------------------------------------------------


def neighborhood_grid(
    strategy: Strategy | type[Strategy], params: Mapping[str, Any]
) -> dict[str, list[Any]]:
    """Raster aus den gewählten Parametern und je einem ``step`` darunter und darüber.

    Nur Zahlenparameter mit ``step`` werden variiert (innerhalb ``min``/``max``); alle
    anderen bleiben fest.

    Args:
        strategy: Strategie (bestimmt die Parameterdefinitionen).
        params: Gewählte, geprüfte Parameter.

    Returns:
        Raster Name → aufsteigende Werte.
    """
    spec = strategy.params
    grid: dict[str, list[Any]] = {}
    for name, param in spec.items():
        value = params.get(name, param.default)
        if param.kind not in ("int", "float") or param.step is None:
            grid[name] = [value]
            continue
        values: list[Any] = []
        for candidate in (value - param.step, value, value + param.step):
            number = round(float(candidate), 10)
            if param.min is not None and number < param.min:
                continue
            if param.max is not None and number > param.max:
                continue
            normalized = int(number) if param.kind == "int" else number
            if normalized not in values:
                values.append(normalized)
        grid[name] = sorted(values)
    return grid


def _robustness(
    optimization: OptimizationResult | None,
    chosen: Mapping[str, Any],
    wf: WalkForwardResult | None,
) -> Robustness:
    if optimization is not None:
        table = optimization.table
        mask = np.ones(len(table), dtype=bool)
        for name in optimization.grid:
            mask &= (table[name] == chosen[name]).to_numpy()
        rows = np.flatnonzero(mask)
        if rows.size:
            value = float(table["robustness"].iloc[int(rows[0])])
            if not math.isnan(value):
                return Robustness(value, "Parameter-Nachbarschaft", optimization)
    if wf is not None:
        values = [w.test_value for w in wf.windows if w.test_value is not None]
        if values:
            share = sum(1 for v in values if v > 0) / len(values)
            return Robustness(share, "Walk-Forward-Fenster", optimization)
    return Robustness(None, "nicht bestimmbar", optimization)


# ---------------------------------------------------------------------------
# Baselines und Auswertung je Zeitraum
# ---------------------------------------------------------------------------


def buy_and_hold_curve(
    daily: pd.DataFrame,
    timeframe: Timeframe,
    start: Date,
    end: Date,
    capital: float,
) -> pd.DataFrame:
    """Buy and Hold ohne Kosten auf abgeschlossenen Kerzen des Zeitrahmens.

    Ausgangskurs ist der letzte Close vor ``start`` (sonst der erste Close im Zeitraum).

    Args:
        daily: Bereinigte Tageskurse.
        timeframe: Zeitrahmen.
        start: Erster Tag.
        end: Letzter Tag.
        capital: Kapital zu Beginn.

    Returns:
        Spalten ``date`` und ``equity`` je Kerze in ``[start, end]``; leer ohne Kurse.
    """
    closes = _closes(daily, timeframe, end)
    return _curve_from_returns(closes.pct_change(fill_method=None), closes, start, capital)


def equal_weight_curve(
    daily: Mapping[str, pd.DataFrame],
    timeframe: Timeframe,
    start: Date,
    end: Date,
    capital: float,
    members: Callable[[Date], frozenset[str] | set[str] | None] | None = None,
) -> pd.DataFrame:
    """Gleichgewichtetes Universum ohne Kosten, je Kerze neu gewichtet.

    Rendite je Kerze = Durchschnitt der Renditen aller Werte, die am Ende der vorigen
    Kerze Mitglied waren und für beide Kerzen einen Close haben. Ohne solchen Wert ist
    die Rendite 0.

    Args:
        daily: Bereinigte Tageskurse je Symbol.
        timeframe: Zeitrahmen.
        start: Erster Tag.
        end: Letzter Tag.
        capital: Kapital zu Beginn.
        members: Mitglieder an einem Tag; ``None`` oder Rückgabe ``None`` = alle.

    Returns:
        Spalten ``date`` und ``equity`` je Kerze in ``[start, end]``.
    """
    series = {s: _closes(daily[s], timeframe, end) for s in sorted(daily)}
    series = {s: c for s, c in series.items() if len(c)}
    if not series:
        return pd.DataFrame({DATE_COLUMN: pd.Series(dtype="datetime64[ns]"), "equity": []})
    closes = pd.DataFrame(series).sort_index()
    returns = closes.pct_change(fill_method=None)
    if members is not None:
        mask = np.zeros(returns.shape, dtype=bool)
        columns = list(returns.columns)
        index = list(returns.index)
        for i in range(1, len(index)):
            allowed = members(index[i - 1].date())
            if allowed is None:
                mask[i, :] = True
            else:
                mask[i, :] = [c in allowed for c in columns]
        returns = returns.where(mask)
    average = returns.mean(axis=1, skipna=True)
    return _curve_from_returns(average, closes, start, capital)


def _closes(daily: pd.DataFrame, timeframe: Timeframe, end: Date) -> pd.Series:
    if daily.empty:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(daily[DATE_COLUMN])
    frame = resample(daily.loc[(dates <= pd.Timestamp(end)).to_numpy()], timeframe)
    if len(frame) and bool(frame[PARTIAL_COLUMN].iloc[-1]):
        frame = frame.iloc[:-1]
    index = pd.DatetimeIndex(pd.to_datetime(frame[DATE_COLUMN]))
    return pd.Series(frame["close"].to_numpy(dtype=float), index=index)


def _curve_from_returns(
    returns: pd.Series, closes: pd.Series | pd.DataFrame, start: Date, capital: float
) -> pd.DataFrame:
    index = pd.DatetimeIndex(returns.index)
    in_range = index >= pd.Timestamp(start)
    r = returns.to_numpy(dtype=float)[in_range]
    if r.size and not (index < pd.Timestamp(start)).any():
        r[0] = 0.0  # kein Kurs vor dem Start: Einstieg zum ersten Close
    r = np.nan_to_num(r, nan=0.0)
    equity = capital * np.cumprod(1.0 + r)
    return pd.DataFrame({DATE_COLUMN: index[in_range], "equity": equity})


def slice_metrics(
    curve: pd.DataFrame,
    timeframe: Timeframe,
    start: Date,
    end: Date,
    initial_capital: float,
    *,
    risk_free_rate_pct: float = 0.0,
) -> PerformanceMetrics | None:
    """Kennzahlen eines Ausschnitts ``[start, end]`` einer Equity-Kurve.

    Ausgangswert ist der letzte Wert vor ``start`` (sonst ``initial_capital``).

    Args:
        curve: Spalten ``date`` und ``equity``.
        timeframe: Zeitrahmen.
        start: Erster Tag.
        end: Letzter Tag.
        initial_capital: Kapital vor dem ersten Wert der Kurve.
        risk_free_rate_pct: Risikofreier Zins p. a. in Prozent.

    Returns:
        Kennzahlen oder ``None``, wenn der Ausschnitt leer ist.
    """
    part, base = _segment(curve, start, end, initial_capital)
    if part.empty or base <= 0:
        return None
    try:
        return compute_metrics(
            part, None, timeframe, initial_capital=base, risk_free_rate_pct=risk_free_rate_pct
        )
    except MetricsError:  # pragma: no cover - leere Ausschnitte sind oben abgefangen
        return None


def _segment(
    curve: pd.DataFrame, start: Date, end: Date, initial_capital: float
) -> tuple[pd.DataFrame, float]:
    dates = pd.to_datetime(curve[DATE_COLUMN])
    before = curve.loc[(dates < pd.Timestamp(start)).to_numpy()]
    inside = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    part = curve.loc[inside.to_numpy(), [DATE_COLUMN, "equity"]].reset_index(drop=True)
    base = float(before["equity"].iloc[-1]) if len(before) else float(initial_capital)
    return part, base


def phase_stats(
    curves: Mapping[str, pd.DataFrame],
    phases: Sequence[MarketPhase],
    initial_capital: float,
) -> pd.DataFrame:
    """Rendite und Max Drawdown je Marktphase und Reihe.

    Args:
        curves: Equity-Kurven (Name → Spalten ``date``, ``equity``).
        phases: Marktphasen.
        initial_capital: Ausgangswert, wenn eine Phase vor der Kurve beginnt.

    Returns:
        Spalten ``phase``, ``series``, ``return_pct``, ``max_drawdown_pct``; Phasen ohne
        Werte fehlen.
    """
    rows: list[dict[str, Any]] = []
    for phase in phases:
        for name, curve in curves.items():
            part, base = _segment(curve, phase.start, phase.end, initial_capital)
            if part.empty or base <= 0:
                continue
            values = np.concatenate([[base], part["equity"].to_numpy(dtype=float)])
            peak = np.maximum.accumulate(values)
            rows.append(
                {
                    "phase": phase.name,
                    "series": name,
                    "return_pct": (values[-1] / base - 1.0) * 100.0,
                    "max_drawdown_pct": float(np.min(values / peak - 1.0)) * 100.0,
                }
            )
    return pd.DataFrame(rows, columns=["phase", "series", "return_pct", "max_drawdown_pct"])


def _baseline_curves(
    daily: Mapping[str, pd.DataFrame],
    timeframe: Timeframe,
    cfg: StudyConfig,
    strategy_symbols: Sequence[str] | None,
    members: Callable[[Date], frozenset[str] | set[str] | None] | None,
    end: Date,
    capital: float,
    log: list[str],
) -> dict[str, pd.DataFrame]:
    curves: dict[str, pd.DataFrame] = {}
    for bench in cfg.benchmarks:
        frame = daily.get(bench)
        curve = (
            buy_and_hold_curve(frame, timeframe, cfg.in_sample_start, end, capital)
            if frame is not None
            else None
        )
        if curve is None or curve.empty:
            log.append(f"Keine Kurse für den Benchmark {bench}; Baseline entfällt.")
            continue
        curves[f"Buy and Hold {bench}"] = curve
    pool = {
        s: f
        for s, f in daily.items()
        if (strategy_symbols is None and s not in cfg.benchmarks)
        or (strategy_symbols is not None and s in strategy_symbols)
    }
    if pool:
        ew = equal_weight_curve(pool, timeframe, cfg.in_sample_start, end, capital, members)
        if not ew.empty:
            curves[EQUAL_WEIGHT] = ew
    return curves


# ---------------------------------------------------------------------------
# Kurse und Universum
# ---------------------------------------------------------------------------


def load_study_bars(
    settings: Settings,
    symbol: str | None,
    universe: Universe | None,
    config: StudyConfig | None = None,
    *,
    log: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Lädt bereinigte Tageskurse aller Werte der Studie und der Benchmarks.

    Werte des Universums, die wegen verdächtiger Kurse ausgeschlossen sind (Tabelle
    ``symbol_exclusions``), fehlen; ein einzeln gewählter Wert und die Benchmarks
    werden immer geladen.

    Args:
        settings: Einstellungen (Datenbank).
        symbol: Einzelner Wert oder ``None``.
        universe: Universum oder ``None``.
        config: Rahmen (Benchmarks, Zeitraum).
        log: Nimmt den Hinweis auf ausgeschlossene Werte auf.

    Returns:
        Symbol → Tageskurse; Werte ohne Kurse fehlen.

    Raises:
        StudyError: Die Datenbank existiert nicht.
    """
    from trading_helfer.data.storage import MarketStore

    cfg = config if config is not None else StudyConfig()
    path = Path(settings.database_path)
    if not path.exists():
        raise StudyError(f"Datenbank {path} existiert nicht; bitte zuerst Daten laden.")
    _name, symbols, _members = _universe_parts(symbol, universe, cfg)
    wanted = sorted({*(symbols or []), *cfg.benchmarks})
    result: dict[str, pd.DataFrame] = {}
    with MarketStore(path, read_only=True) as store:
        excluded = set(store.excluded_symbols()) if symbol is None else set()
        excluded -= set(cfg.benchmarks)
        for sym in wanted:
            if sym in excluded:
                continue
            frame = store.load_bars(sym, end=cfg.end, adjusted=True)
            if len(frame):
                result[sym] = frame
    skipped = sorted(excluded & set(wanted))
    if skipped and log is not None:
        log.append(exclusion_note(skipped))
    return result


def _universe_parts(
    symbol: str | None, universe: Universe | None, cfg: StudyConfig
) -> tuple[str, list[str] | None, Callable[[Date], frozenset[str] | set[str] | None] | None]:
    """Name, Symbole der Strategie und Mitgliedschaftsfunktion (``None`` = alle)."""
    if symbol is not None:
        return symbol, [symbol], None
    history: IndexHistory
    if isinstance(universe, str):
        history = load_history(universe)
    elif isinstance(universe, IndexHistory):
        history = universe
    else:
        assert universe is not None
        symbols = sorted({str(s) for s in universe})
        if not symbols:
            raise StudyError("Das Universum enthält keine Symbole.")
        name = "Liste: " + ", ".join(symbols[:10]) + (" …" if len(symbols) > 10 else "")
        return name, symbols, None
    last = cfg.end if cfg.end is not None else Date.today()
    symbols = sorted(history.all_members(cfg.in_sample_start, max(last, cfg.in_sample_start)))
    return history.index_code, symbols, history.members_on


def _cut(daily: Mapping[str, pd.DataFrame], cfg: StudyConfig) -> dict[str, pd.DataFrame]:
    last = cfg.end
    if not cfg.include_out_of_sample:
        last = cfg.in_sample_end if last is None else min(last, cfg.in_sample_end)
    if last is None:
        return dict(daily)
    result: dict[str, pd.DataFrame] = {}
    for sym, frame in daily.items():
        dates = pd.to_datetime(frame[DATE_COLUMN])
        part = frame.loc[(dates <= pd.Timestamp(last)).to_numpy()].reset_index(drop=True)
        if len(part):
            result[sym] = part
    return result


def _data_end(
    daily: Mapping[str, pd.DataFrame], symbols: Sequence[str] | None, cfg: StudyConfig
) -> Date:
    relevant = [
        f
        for s, f in daily.items()
        if (symbols is None and s not in cfg.benchmarks) or (symbols is not None and s in symbols)
    ]
    if not relevant:
        raise StudyError("Keine Kurse für die Werte der Studie.")
    last = max(pd.Timestamp(pd.to_datetime(f[DATE_COLUMN]).max()) for f in relevant)
    end = last.date()
    if cfg.end is not None:
        end = min(end, cfg.end)
    if end < cfg.in_sample_start:
        raise StudyError(f"Kurse enden am {end}, vor dem Beginn {cfg.in_sample_start}.")
    return end


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------


class _Runner:
    """Führt die Backtests einer Studie mit gemeinsamen Kursen aus."""

    def __init__(
        self,
        strategy: Strategy,
        params: dict[str, Any],
        timeframe: Timeframe,
        settings: Settings,
        symbol: str | None,
        universe: Universe | None,
        daily: dict[str, pd.DataFrame],
        cfg: StudyConfig,
    ) -> None:
        self.strategy = strategy
        self.params = params
        self.timeframe = timeframe
        self.settings = settings
        self.symbol = symbol
        self.universe = universe
        self.daily = daily
        self.cfg = cfg

    @property
    def bars(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        if self.symbol is not None:
            if self.symbol not in self.daily:
                raise StudyError(f"Keine Kurse für {self.symbol}.")
            return self.daily[self.symbol]
        return self.daily

    def _market(self) -> pd.DataFrame | None:
        symbol = self.strategy.market_symbol
        return self.daily.get(symbol) if symbol else None

    def backtest(self, start: Date, end: Date, costs: CostModel) -> BacktestResult:
        bars = self.bars
        if self.symbol is not None:
            assert isinstance(bars, pd.DataFrame)
            return run_backtest(
                self.strategy,
                self.params,
                self.symbol,
                self.timeframe,
                start,
                end,
                self.settings,
                bars=bars,
                costs=costs,
                market=self._market(),
            )
        assert self.universe is not None
        assert isinstance(bars, dict)
        return run_portfolio_backtest(
            self.strategy,
            self.params,
            self.universe,
            self.timeframe,
            start,
            end,
            self.settings,
            max_positions=self.cfg.max_positions,
            bars=bars,
            costs=costs,
            workers=self.cfg.workers,
        )

    def period(
        self,
        name: str,
        start: Date,
        end: Date,
        costs: CostModel,
        curves: Mapping[str, pd.DataFrame],
        rf: float,
    ) -> PeriodResult:
        result = self.backtest(start, end, costs)
        if result.equity.empty:
            raise StudyError(f"{name}: keine Kerzen im Zeitraum {start} bis {end}.")
        metrics = compute_metrics(
            result.equity,
            result.trades,
            self.timeframe,
            initial_capital=result.initial_capital,
            risk_free_rate_pct=rf,
        )
        baselines: dict[str, PerformanceMetrics] = {}
        for label, curve in curves.items():
            m = slice_metrics(
                curve, self.timeframe, start, end, result.initial_capital, risk_free_rate_pct=rf
            )
            if m is not None:
                baselines[label.removeprefix("Buy and Hold ")] = m
        return PeriodResult(name, start, end, result, metrics, baselines)

    def optimize(
        self, grid: Mapping[str, Sequence[Any]], start: Date, end: Date, log: list[str]
    ) -> OptimizationResult | None:
        try:
            return optimize(
                self.strategy,
                grid,
                self.symbol,
                self.timeframe,
                start,
                end,
                self.settings,
                objective=Objective.SHARPE,
                universe=self.universe,
                max_positions=self.cfg.max_positions,
                bars=self.bars,
                workers=self.cfg.workers,
            )
        except (OptimizationError, BacktestError) as exc:
            log.append(f"Robustheits-Optimierung nicht möglich: {exc}")
            return None

    def walk_forward(
        self, grid: Mapping[str, Sequence[Any]], start: Date, end: Date, log: list[str]
    ) -> WalkForwardResult | None:
        try:
            return walk_forward(
                self.strategy,
                grid,
                self.symbol,
                self.timeframe,
                start,
                end,
                self.settings,
                train_years=self.cfg.train_years,
                test_years=self.cfg.test_years,
                objective=Objective.SHARPE,
                universe=self.universe,
                max_positions=self.cfg.max_positions,
                bars=self.bars,
                workers=self.cfg.workers,
            )
        except (OptimizationError, BacktestError) as exc:
            log.append(f"Walk-Forward nicht möglich: {exc}")
            return None
