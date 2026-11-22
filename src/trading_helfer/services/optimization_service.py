"""Fassade für den Optimierungs-Tab: Parameter-Raster, Walk-Forward und Monte Carlo.

Das Modul importiert kein Qt. Es verbindet die Strategie-Registry und die Kurse des
:class:`~trading_helfer.services.backtest_service.BacktestService` mit der
Raster-Optimierung (:func:`~trading_helfer.backtest.optimize.optimize`), der
Walk-Forward-Analyse (:func:`~trading_helfer.backtest.walkforward.walk_forward`) und der
Monte-Carlo-Umordnung der Trades (:func:`~trading_helfer.backtest.monte_carlo.monte_carlo`).

**Abbruch**: :meth:`OptimizationService.run` fragt nach jeder bewerteten Kombination
``should_cancel``. Liefert es ``True``, werden wartende Kombinationen gestrichen, laufende
noch beendet und :class:`OptimizationCancelledError` ausgelöst.

**Monte Carlo**: Beim Raster wird die beste Kombination im ganzen Zeitraum noch einmal
gerechnet und deren Trades umgeordnet; bei Walk-Forward die Trades aller Testzeiträume
(Renditen je Fenster auf dessen Startkapital bezogen).

Beispiel::

    service = OptimizationService(BacktestService(settings))
    request = OptimizationRequest("trend_sma_cross", {"fast": ParamRange(5, 20, 5)},
                                  "AAPL", Timeframe.W1, dt.date(2000, 1, 1),
                                  dt.date(2024, 12, 31))
    outcome = service.run(request)
    print(outcome.optimization.best_params, outcome.monte_carlo.summary())
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import numpy as np
import pandas as pd

from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import BacktestError, BacktestResult
from trading_helfer.backtest.monte_carlo import (
    DEFAULT_SEED,
    DEFAULT_SIMULATIONS,
    MonteCarloError,
    MonteCarloMethod,
    MonteCarloResult,
    monte_carlo,
    trade_returns,
)
from trading_helfer.backtest.optimize import (
    Objective,
    OptimizationResult,
    RangeSpec,
    build_grid,
    count_combinations,
    optimize,
)
from trading_helfer.backtest.portfolio import DEFAULT_MAX_POSITIONS
from trading_helfer.backtest.walkforward import WalkForwardResult, walk_forward
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.services.backtest_service import (
    BacktestRequest,
    BacktestService,
    UniverseKind,
)
from trading_helfer.strategies.base import Param
from trading_helfer.strategies.registry import StrategyInfo

__all__ = [
    "OptimizationCancelledError",
    "OptimizationMode",
    "OptimizationOutcome",
    "OptimizationRequest",
    "OptimizationService",
]

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, str], None]
"""Fortschritt in Prozent (0–100) und kurze Meldung."""

CancelFn = Callable[[], bool]
"""Liefert ``True``, wenn der Lauf abgebrochen werden soll."""

_GRID_SHARE: Final = 90
"""Anteil der Raster-Bewertung am Fortschritt in Prozent (Rest: Bestlauf, Monte Carlo)."""


class OptimizationCancelledError(Exception):
    """Der Lauf wurde auf Wunsch abgebrochen."""

    def __init__(self) -> None:
        super().__init__("Optimierung abgebrochen.")


class OptimizationMode(StrEnum):
    """Art der Optimierung."""

    GRID = "grid"
    """Raster über den ganzen Zeitraum."""
    WALK_FORWARD = "walk_forward"
    """Rollierend im Training optimieren, außerhalb der Stichprobe testen."""


@dataclass(frozen=True)
class OptimizationRequest:
    """Eingaben des Optimierungs-Formulars.

    Attributes:
        strategy_id: ID der Strategie.
        ranges: Zu variierende Parameter (``ParamRange``, Werteliste oder ``None`` für
            den Bereich aus der Definition); nicht genannte bleiben auf dem Standardwert.
        universe: Symbol oder Index-Code.
        timeframe: Zeitrahmen.
        start: Erster Tag des Zeitraums.
        end: Letzter Tag des Zeitraums.
        universe_kind: Einzelwert oder Index.
        objective: Zielgröße.
        mode: Raster oder Walk-Forward.
        train_years: Trainingsdauer je Fenster (Walk-Forward).
        test_years: Testdauer und Schrittweite (Walk-Forward).
        anchored: Training beginnt immer bei ``start`` (Walk-Forward).
        capital: Startkapital in USD.
        fee_per_order: Fixe Gebühr je Order in USD.
        fee_pct: Prozentuale Gebühr.
        slippage_pct: Slippage in Prozent.
        max_positions: Höchstzahl Positionen (nur Index).
        mc_simulations: Anzahl Monte-Carlo-Simulationen; 0 = ohne Monte Carlo.
        mc_method: Umordnen (``shuffle``) oder Ziehen mit Zurücklegen (``bootstrap``).
        seed: Seed für Monte Carlo.
        workers: Prozesse bzw. Threads; ``None`` = Anzahl CPU-Kerne.
    """

    strategy_id: str
    ranges: Mapping[str, RangeSpec]
    universe: str
    timeframe: Timeframe
    start: dt.date
    end: dt.date
    universe_kind: UniverseKind = UniverseKind.SYMBOL
    objective: Objective = Objective.SHARPE
    mode: OptimizationMode = OptimizationMode.GRID
    train_years: int = 5
    test_years: int = 1
    anchored: bool = False
    capital: float = 1000.0
    fee_per_order: float = 1.0
    fee_pct: float = 0.0
    slippage_pct: float = 0.05
    max_positions: int = DEFAULT_MAX_POSITIONS
    mc_simulations: int = DEFAULT_SIMULATIONS
    mc_method: MonteCarloMethod = MonteCarloMethod.SHUFFLE
    seed: int = DEFAULT_SEED
    workers: int | None = None

    def costs(self) -> CostModel:
        """Kostenmodell aus Gebühren und Slippage."""
        return CostModel(
            fee_per_order=self.fee_per_order,
            fee_pct=self.fee_pct,
            slippage_pct=self.slippage_pct,
        )


@dataclass(frozen=True)
class OptimizationOutcome:
    """Ergebnis für die Anzeige.

    Attributes:
        request: Eingaben des Laufs.
        optimization: Raster-Ergebnis (bei Walk-Forward: Training des letzten Fensters).
        walk_forward: Walk-Forward-Ergebnis oder ``None`` beim Raster.
        best_result: Backtest der besten Kombination über den ganzen Zeitraum (Raster).
        monte_carlo: Monte-Carlo-Ergebnis oder ``None`` (keine Trades oder abgeschaltet).
        log: Hinweise auf Deutsch.
    """

    request: OptimizationRequest
    optimization: OptimizationResult
    walk_forward: WalkForwardResult | None = None
    best_result: BacktestResult | None = None
    monte_carlo: MonteCarloResult | None = None
    log: list[str] = field(default_factory=list)


class OptimizationService:
    """Verbindet den Optimierungs-Tab mit Registry, Kursen, Optimierung und Monte Carlo.

    Args:
        backtest_service: Liefert Einstellungen, Strategien und Kurse.
    """

    def __init__(self, backtest_service: BacktestService) -> None:
        self.backtest_service = backtest_service

    @property
    def settings(self) -> Settings:
        """Einstellungen des Backtest-Service."""
        return self.backtest_service.settings

    # -- Strategien und Raster -------------------------------------------------------

    def strategies(self) -> list[StrategyInfo]:
        """Alle geladenen Strategien in Ladereihenfolge."""
        return self.backtest_service.strategies()

    def strategy_params(self, strategy_id: str) -> dict[str, Param]:
        """Parameterbeschreibungen der Strategie ``strategy_id``."""
        return self.backtest_service.strategy_params(strategy_id)

    def strategy_timeframes(self, strategy_id: str) -> list[Timeframe]:
        """Zeitrahmen, die die Strategie unterstützt."""
        return self.backtest_service.strategy_timeframes(strategy_id)

    def strategy_description(self, strategy_id: str) -> str:
        """Beschreibung der Strategie ``strategy_id``."""
        return str(self.backtest_service.registry.get(strategy_id).description)

    def reload_strategies(self) -> None:
        """Liest die Plugins neu ein."""
        self.backtest_service.reload_strategies()

    def count_combinations(self, strategy_id: str, ranges: Mapping[str, RangeSpec]) -> int:
        """Anzahl der Kombinationen des Rasters.

        Raises:
            OptimizationError: Ungültiger Bereich.
            trading_helfer.strategies.base.ParamError: Wert außerhalb der Grenzen.
        """
        cls = self.backtest_service.registry.get(strategy_id)
        return count_combinations(build_grid(cls, ranges))

    # -- Lauf ------------------------------------------------------------------------

    def run(
        self,
        request: OptimizationRequest,
        progress: ProgressFn | None = None,
        should_cancel: CancelFn | None = None,
    ) -> OptimizationOutcome:
        """Führt Raster-Optimierung oder Walk-Forward und danach Monte Carlo aus.

        Args:
            request: Formulareingaben.
            progress: Optionaler Fortschritts-Rückruf.
            should_cancel: Wird nach jeder Kombination gefragt; ``True`` bricht ab.

        Raises:
            OptimizationCancelledError: Abbruch über ``should_cancel``.
            OptimizationError: Ungültiges Raster oder Zeitraum.
            BacktestError: Ungültige Eingaben oder keine Kurse.
        """
        report = progress if progress is not None else (lambda _p, _m: None)
        cancelled = should_cancel if should_cancel is not None else (lambda: False)
        if request.capital <= 0:
            raise BacktestError("Das Startkapital muss größer als 0 sein.")
        cls = self.backtest_service.registry.get(request.strategy_id)
        grid = build_grid(cls, request.ranges)
        total = count_combinations(grid)
        bars = self._bars(request, report)
        if cancelled():
            raise OptimizationCancelledError()
        settings = self.settings.model_copy(update={"default_capital": request.capital})
        symbol, universe = self._target(request)
        label = "Walk-Forward" if request.mode is OptimizationMode.WALK_FORWARD else "Optimierung"

        def on_step(done: int, steps: int) -> None:
            if cancelled():
                raise OptimizationCancelledError()
            percent = 30 + int((_GRID_SHARE - 30) * done / max(steps, 1))
            report(percent, f"{label}: {done}/{steps}")

        report(30, f"{label}: {total} Kombination(en)")
        common: dict[str, Any] = {
            "objective": request.objective,
            "universe": universe,
            "max_positions": request.max_positions,
            "bars": bars,
            "costs": request.costs(),
            "workers": request.workers,
            "progress": on_step,
        }
        if request.mode is OptimizationMode.WALK_FORWARD:
            wf = walk_forward(
                cls,
                grid,
                symbol,
                request.timeframe,
                request.start,
                request.end,
                settings,
                train_years=request.train_years,
                test_years=request.test_years,
                anchored=request.anchored,
                **common,
            )
            report(_GRID_SHARE, "Monte Carlo")
            wf_mc, wf_log = self._monte_carlo_wf(wf, request)
            report(100, "Fertig")
            return OptimizationOutcome(
                request=request,
                optimization=wf.windows[-1].optimization,
                walk_forward=wf,
                monte_carlo=wf_mc,
                log=[*wf.log, *wf_log],
            )
        result = optimize(
            cls, grid, symbol, request.timeframe, request.start, request.end, settings, **common
        )
        log = list(result.log)
        best: BacktestResult | None = None
        mc: MonteCarloResult | None = None
        if result.best_params is not None:
            report(_GRID_SHARE, "Backtest der besten Kombination")
            best = self._best_backtest(request, result.best_params)
            report(95, "Monte Carlo")
            mc, mc_log = self._monte_carlo([best.trades], request)
            log.extend(mc_log)
        report(100, "Fertig")
        return OptimizationOutcome(
            request=request, optimization=result, best_result=best, monte_carlo=mc, log=log
        )

    def _bars(
        self, request: OptimizationRequest, report: ProgressFn
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        if request.universe_kind is UniverseKind.INDEX:
            return self.backtest_service.index_bars(
                request.universe, request.start, request.end, report
            )
        symbol = request.universe.strip().upper()
        if not symbol:
            raise BacktestError("Bitte ein Symbol angeben.")
        report(5, f"Lade {symbol}")
        bars = self.backtest_service.load_bars(symbol)
        if bars.empty:
            raise BacktestError(f"Für {symbol} sind keine Kurse gespeichert.")
        return bars

    @staticmethod
    def _target(request: OptimizationRequest) -> tuple[str | None, str | None]:
        code = request.universe.strip().upper()
        if request.universe_kind is UniverseKind.INDEX:
            return None, code
        return code, None

    def _best_backtest(
        self, request: OptimizationRequest, params: dict[str, Any]
    ) -> BacktestResult:
        outcome = self.backtest_service.run(
            BacktestRequest(
                strategy_id=request.strategy_id,
                params=params,
                universe=request.universe,
                timeframe=request.timeframe,
                start=request.start,
                end=request.end,
                universe_kind=request.universe_kind,
                capital=request.capital,
                fee_per_order=request.fee_per_order,
                fee_pct=request.fee_pct,
                slippage_pct=request.slippage_pct,
                benchmark="",
                save=False,
                max_positions=request.max_positions,
            )
        )
        return outcome.result

    def _monte_carlo_wf(
        self, wf: WalkForwardResult, request: OptimizationRequest
    ) -> tuple[MonteCarloResult | None, list[str]]:
        return self._monte_carlo([w.test_result.trades for w in wf.windows], request)

    @staticmethod
    def _monte_carlo(
        trade_lists: list[pd.DataFrame], request: OptimizationRequest
    ) -> tuple[MonteCarloResult | None, list[str]]:
        """Monte Carlo über die Trades; jede Liste bezieht sich auf das Startkapital."""
        if request.mc_simulations <= 0:
            return None, []
        try:
            parts = [
                trade_returns(trades, request.capital) for trades in trade_lists if not trades.empty
            ]
            returns = np.concatenate(parts) if parts else np.zeros(0)
            if returns.size == 0:
                return None, ["Monte Carlo: keine abgeschlossenen Trades."]
            result = monte_carlo(
                returns,
                request.capital,
                n_simulations=request.mc_simulations,
                method=request.mc_method,
                seed=request.seed,
            )
        except MonteCarloError as exc:
            return None, [f"Monte Carlo nicht möglich: {exc}"]
        return result, []
