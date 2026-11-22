"""Parameter-Optimierung über ein Raster (Grid-Search).

Ablauf:

1. :func:`build_grid` erzeugt aus den Parameterdefinitionen der Strategie
   (``Param.min``/``max``/``step``) oder aus eigenen Bereichen ein Raster. Nicht genannte
   Parameter bleiben auf dem Standardwert.
2. :func:`optimize` führt für jede Kombination einen Backtest aus (ein Wert mit
   :func:`~trading_helfer.backtest.engine.run_backtest` oder ein Universum mit
   :func:`~trading_helfer.backtest.portfolio.run_portfolio_backtest`) und bewertet ihn mit
   der gewählten Zielgröße (:class:`Objective`).
3. Die beste Kombination hat den höchsten Zielwert; bei Gleichstand gewinnt die erste in
   Rasterreihenfolge. Kombinationen ohne Zielwert (z. B. Sharpe ohne Schwankung) oder
   mit Fehler zählen nicht.

Mehr als :data:`MAX_COMBINATIONS` Kombinationen sind erlaubt, erzeugen aber eine
Warnung. Die Backtests laufen in mehreren Prozessen, wenn sich die Strategie in einem
neuen Prozess laden lässt (Windows startet Prozesse neu), sonst in Threads. Der Weg
ändert das Ergebnis nicht: gleiche Daten und Parameter ergeben dieselbe Tabelle.

Robustheit: Für jede Kombination der Anteil ihrer Nachbarn im Raster (jeder Parameter
höchstens einen Rasterschritt entfernt, ohne die Kombination selbst) mit positivem
Zielwert. Ein hoher Wert zeigt ein breites Plateau statt einer einzelnen Spitze.

Beispiel::

    grid = build_grid(SmaCross, {"fast": ParamRange(5, 20, 5), "slow": None})
    result = optimize(SmaCross, grid, "AAPL", Timeframe.W1, "2000-01-01", "2024-12-31",
                      settings, objective=Objective.SHARPE)
    print(result.best_params, result.best_value, result.best_robustness)
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
import os
import pickle
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import date as Date  # noqa: N812
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd

from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import (
    BacktestError,
    BacktestResult,
    DateInput,
    run_backtest,
)
from trading_helfer.backtest.engine import _load_daily as _load_symbol_daily
from trading_helfer.backtest.metrics import PerformanceMetrics, compute_metrics
from trading_helfer.backtest.portfolio import (
    DEFAULT_MAX_POSITIONS,
    Universe,
    _collect_bars,
    _process_safe,
    _resolve_universe,
    run_portfolio_backtest,
)
from trading_helfer.core.types import Timeframe
from trading_helfer.data.universe import IndexHistory
from trading_helfer.strategies.base import Param, ParamError, Strategy

if TYPE_CHECKING:
    from trading_helfer.core.config import Settings

__all__ = [
    "CALMAR_YEARS",
    "MAX_COMBINATIONS",
    "METRIC_COLUMNS",
    "Objective",
    "OptimizationError",
    "OptimizationResult",
    "ParamGrid",
    "ParamRange",
    "ProgressCallback",
    "build_grid",
    "count_combinations",
    "grid_search",
    "objective_value",
    "optimize",
    "robustness_scores",
]

logger = logging.getLogger(__name__)

MAX_COMBINATIONS: Final = 10_000
"""Ab dieser Anzahl Kombinationen warnt die Optimierung (sie läuft trotzdem)."""

CALMAR_YEARS: Final = 3
"""Calmar Ratio bezieht sich auf die letzten drei Jahre der Equity-Kurve."""

METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "total_return_pct",
    "cagr_pct",
    "sharpe",
    "calmar",
    "mar",
    "max_drawdown_pct",
    "num_trades",
)
"""Kennzahlen, die die Ergebnistabelle je Kombination enthält."""

_PROCESS_MIN_COMBINATIONS: Final = 8
"""Ab dieser Anzahl Kombinationen lohnt sich der Start von Prozessen."""

ParamGrid = dict[str, list[Any]]
"""Raster: Parametername → Werte in Rasterreihenfolge (geordnet)."""

ProgressCallback = Callable[[int, int], None]
"""Fortschritt: ``(fertig, gesamt)``; wird im aufrufenden Prozess aufgerufen."""


class OptimizationError(ValueError):
    """Optimierung konnte nicht ausgeführt werden; Text auf Deutsch."""


class Objective(StrEnum):
    """Zielgröße der Optimierung (größer ist besser).

    - ``SHARPE``: Sharpe Ratio (annualisiert, risikofreier Zins aus den Einstellungen).
    - ``CAGR``: jährliche Wachstumsrate in Prozent.
    - ``CALMAR``: CAGR / |Max Drawdown| der letzten :data:`CALMAR_YEARS` Jahre
      (klassische Definition; bei kürzerer Kurve über die ganze Laufzeit).
    - ``MAR``: CAGR / |Max Drawdown| über die ganze Laufzeit.
    """

    SHARPE = "sharpe"
    CAGR = "cagr"
    CALMAR = "calmar"
    MAR = "mar"

    @property
    def label(self) -> str:
        """Anzeigename auf Deutsch."""
        return _OBJECTIVE_LABELS[self]


_OBJECTIVE_LABELS: Final[Mapping[Objective, str]] = {
    Objective.SHARPE: "Sharpe Ratio",
    Objective.CAGR: "CAGR (%)",
    Objective.CALMAR: "Calmar Ratio (3 Jahre)",
    Objective.MAR: "MAR Ratio",
}


@dataclass(frozen=True)
class ParamRange:
    """Bereich eines Parameters für das Raster (beide Grenzen eingeschlossen).

    Attributes:
        min: Erster Wert.
        max: Letzter Wert (eingeschlossen, wenn er auf dem Raster liegt).
        step: Schrittweite (> 0).
    """

    min: float
    max: float
    step: float

    def values(self) -> list[float]:
        """Rasterwerte von ``min`` bis ``max`` in Schritten von ``step``.

        Raises:
            OptimizationError: ``step`` ≤ 0 oder ``min`` > ``max``.
        """
        if not self.step > 0:
            raise OptimizationError(f"Schrittweite muss größer als 0 sein, erhalten: {self.step}.")
        if self.min > self.max:
            raise OptimizationError(f"min ({self.min}) ist größer als max ({self.max}).")
        count = math.floor((self.max - self.min) / self.step + 1e-9) + 1
        return [round(self.min + i * self.step, 10) for i in range(count)]


RangeSpec = ParamRange | Sequence[Any] | None
"""Bereich eines Parameters: :class:`ParamRange`, feste Werteliste oder ``None`` (aus
``Param.min``/``max``/``step``)."""


# ---------------------------------------------------------------------------
# Raster
# ---------------------------------------------------------------------------


def build_grid(
    strategy: Strategy | type[Strategy],
    ranges: Mapping[str, RangeSpec] | None = None,
) -> ParamGrid:
    """Erzeugt das Parameter-Raster einer Strategie.

    Args:
        strategy: Strategie (Instanz oder Klasse).
        ranges: Zu variierende Parameter. ``ParamRange`` = eigener Bereich, Liste = feste
            Werte (Zahlen werden sortiert), ``None`` = Bereich aus der Parameterdefinition
            (``bool``: falsch/wahr, Auswahl: alle Auswahlen). Nicht genannte Parameter
            bleiben auf dem Standardwert. ``ranges=None`` variiert alle Parameter.

    Returns:
        Raster mit allen Parametern der Strategie in Definitionsreihenfolge.

    Raises:
        OptimizationError: Unbekannter Parameter, leerer Bereich oder Bereich ohne
            Grenzen in der Definition.
        ParamError: Ein Rasterwert ist für den Parameter nicht erlaubt.
    """
    spec = strategy.params
    if ranges is None:
        ranges = dict.fromkeys(spec)
    unknown = sorted(set(ranges) - set(spec))
    if unknown:
        raise OptimizationError(f"Unbekannte(r) Parameter: {', '.join(unknown)}.")
    grid: ParamGrid = {}
    for name, param in spec.items():
        if name not in ranges:
            grid[name] = [param.default]
            continue
        raw = _range_values(name, param, ranges[name])
        values = _dedupe([param.validate(v, name) for v in raw])
        if not values:
            raise OptimizationError(f"Parameter {name}: Bereich enthält keine Werte.")
        grid[name] = values
    return grid


def count_combinations(grid: Mapping[str, Sequence[Any]]) -> int:
    """Anzahl der Kombinationen eines Rasters (Produkt der Wertanzahlen)."""
    return math.prod(len(v) for v in grid.values())


def _range_values(name: str, param: Param, spec: RangeSpec) -> list[Any]:
    if isinstance(spec, ParamRange):
        if param.kind not in ("int", "float"):
            raise OptimizationError(f"Parameter {name}: Bereich nur für Zahlen möglich.")
        values: list[Any] = spec.values()
        return [int(v) if param.kind == "int" and float(v).is_integer() else v for v in values]
    if spec is None:
        if param.kind == "bool":
            return [False, True]
        if param.kind == "choice":
            assert param.choices is not None
            return list(param.choices)
        if param.min is None or param.max is None:
            raise OptimizationError(
                f"Parameter {name} hat kein min/max; bitte Bereich oder Werte angeben."
            )
        step = param.step if param.step is not None else (1 if param.kind == "int" else None)
        if step is None:
            raise OptimizationError(
                f"Parameter {name} hat keine Schrittweite; bitte Bereich oder Werte angeben."
            )
        return _range_values(name, param, ParamRange(param.min, param.max, step))
    if isinstance(spec, str) or not isinstance(spec, Iterable):
        raise OptimizationError(f"Parameter {name}: Bereich oder Werteliste erwartet.")
    values = list(spec)
    if param.kind in ("int", "float"):
        values = sorted(values)
    return values


def _dedupe(values: list[Any]) -> list[Any]:
    seen: list[Any] = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def _combinations(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    names = list(grid)
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*grid.values())]


# ---------------------------------------------------------------------------
# Zielgröße
# ---------------------------------------------------------------------------


def objective_value(
    result: BacktestResult,
    objective: Objective | str,
    *,
    risk_free_rate_pct: float = 0.0,
) -> float | None:
    """Zielwert eines Backtest-Ergebnisses.

    Args:
        result: Ergebnis eines Backtests.
        objective: Zielgröße.
        risk_free_rate_pct: Risikofreier Zins p. a. in Prozent (für Sharpe).

    Returns:
        Zielwert oder ``None``, wenn er nicht definiert ist (z. B. keine Schwankung).
    """
    value = _objective_values(result, risk_free_rate_pct)[Objective(objective).value]
    return None if value is None else float(value)


def _objective_values(result: BacktestResult, rf_pct: float) -> dict[str, Any]:
    """Alle Spalten aus :data:`METRIC_COLUMNS` (``sharpe``/``cagr``/``calmar``/``mar``)."""
    if len(result.equity) == 0:
        return dict.fromkeys((*METRIC_COLUMNS, "cagr"))
    m = compute_metrics(
        result.equity,
        result.trades,
        result.timeframe,
        initial_capital=result.initial_capital,
        risk_free_rate_pct=rf_pct,
    )
    return {
        "total_return_pct": m.total_return_pct,
        "cagr_pct": m.cagr_pct,
        "cagr": m.cagr_pct,
        "sharpe": m.sharpe,
        "calmar": _calmar(result, m),
        "mar": m.calmar,
        "max_drawdown_pct": m.max_drawdown_pct,
        "num_trades": m.num_trades,
    }


def _calmar(result: BacktestResult, full: PerformanceMetrics) -> float | None:
    """CAGR / |Max Drawdown| der letzten :data:`CALMAR_YEARS` Jahre."""
    equity = result.equity
    dates = pd.to_datetime(equity["date"])
    cutoff = dates.iloc[-1] - pd.DateOffset(years=CALMAR_YEARS)
    before = np.flatnonzero((dates <= cutoff).to_numpy())
    if len(before) == 0:
        return full.calmar
    first = int(before[-1])
    window = equity.iloc[first:].reset_index(drop=True)
    if len(window) < 2:
        return full.calmar
    m = compute_metrics(window, None, result.timeframe)
    return m.calmar


# ---------------------------------------------------------------------------
# Raster-Suche (allgemein)
# ---------------------------------------------------------------------------


def grid_search(
    evaluate: Callable[[dict[str, Any]], float | None],
    grid: Mapping[str, Sequence[Any]],
    *,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Bewertet jede Kombination eines Rasters mit einer Funktion (ohne Parallelisierung).

    Args:
        evaluate: Zielfunktion; ``None`` oder NaN = nicht bewertbar.
        grid: Raster.
        progress: Fortschritt ``(fertig, gesamt)``.

    Returns:
        Tabelle in Rasterreihenfolge mit den Parameterspalten, ``objective`` und
        ``robustness``. ``df.attrs["best"]`` ist der Zeilenindex der besten Kombination
        oder ``None``.
    """
    combos = _combinations(grid)
    _warn_size(len(combos), [])
    values: list[float] = []
    for i, combo in enumerate(combos, start=1):
        values.append(_as_float(evaluate(dict(combo))))
        if progress is not None:
            progress(i, len(combos))
    table = pd.DataFrame(combos, columns=list(grid))
    table["objective"] = values
    table["robustness"] = robustness_scores(grid, values)
    table.attrs["best"] = _best_index(values)
    return table


def robustness_scores(grid: Mapping[str, Sequence[Any]], values: Sequence[float]) -> list[float]:
    """Anteil der Nachbarn mit positivem Zielwert je Kombination.

    Nachbarn sind alle Kombinationen, deren Rasterindex in jedem Parameter höchstens um
    eins abweicht (ohne die Kombination selbst). Parameter mit einem einzigen Wert
    spielen keine Rolle.

    Args:
        grid: Raster (bestimmt Form und Reihenfolge).
        values: Zielwerte in Rasterreihenfolge (NaN = nicht bewertbar, zählt nicht als
            positiv).

    Returns:
        Anteile zwischen 0 und 1 in Rasterreihenfolge; NaN, wenn es keine Nachbarn gibt.
    """
    shape = tuple(len(v) for v in grid.values())
    arr = np.asarray(values, dtype=float).reshape(shape) if shape else np.asarray(values)
    positive = np.nan_to_num(arr, nan=-1.0) > 0
    pos_count = np.zeros(arr.shape, dtype=float)
    total = np.zeros(arr.shape, dtype=float)
    for offset in itertools.product((-1, 0, 1), repeat=len(shape)):
        if not any(offset):
            continue
        src, dst = [], []
        for off, size in zip(offset, shape, strict=True):
            if off == 0:
                src.append(slice(None))
                dst.append(slice(None))
            elif off == 1:
                src.append(slice(1, size))
                dst.append(slice(0, size - 1))
            else:
                src.append(slice(0, size - 1))
                dst.append(slice(1, size))
        pos_count[tuple(dst)] += positive[tuple(src)]
        total[tuple(dst)] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(total > 0, pos_count / np.where(total > 0, total, 1), np.nan)
    return [float(v) for v in scores.reshape(-1)]


def _as_float(value: Any) -> float:
    if value is None:
        return math.nan
    number = float(value)
    return number if math.isfinite(number) else math.nan


def _best_index(values: Sequence[float]) -> int | None:
    best: int | None = None
    for i, v in enumerate(values):
        if not math.isnan(v) and (best is None or v > values[best]):
            best = i
    return best


def _warn_size(count: int, log: list[str]) -> None:
    if count > MAX_COMBINATIONS:
        text = (
            f"Raster mit {count:,} Kombinationen überschreitet die empfohlene Grenze von "
            f"{MAX_COMBINATIONS:,}; die Optimierung kann sehr lange dauern."
        ).replace(",", ".")
        logger.warning(text)
        log.append(text)


# ---------------------------------------------------------------------------
# Optimierung mit Backtests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationResult:
    """Ergebnis einer Optimierung.

    Attributes:
        strategy_id: ID der Strategie.
        objective: Zielgröße.
        grid: Verwendetes Raster.
        table: Eine Zeile je Kombination in Rasterreihenfolge: Parameterspalten,
            ``objective``, :data:`METRIC_COLUMNS`, ``robustness`` und ``error``
            (Fehlertext oder leer).
        best_index: Zeile der besten Kombination oder ``None``, wenn keine bewertbar war.
        log: Hinweise (z. B. Warnung bei großem Raster), auf Deutsch.
    """

    strategy_id: str
    objective: Objective
    grid: ParamGrid
    table: pd.DataFrame
    best_index: int | None
    log: list[str] = field(default_factory=list)

    @property
    def param_names(self) -> list[str]:
        """Parameternamen in Rasterreihenfolge."""
        return list(self.grid)

    @property
    def best_params(self) -> dict[str, Any] | None:
        """Parameter der besten Kombination oder ``None``."""
        if self.best_index is None:
            return None
        row = self.table.iloc[self.best_index]
        return {name: _plain(row[name]) for name in self.grid}

    @property
    def best_value(self) -> float | None:
        """Zielwert der besten Kombination oder ``None``."""
        if self.best_index is None:
            return None
        return float(self.table["objective"].iloc[self.best_index])

    @property
    def best_robustness(self) -> float | None:
        """Robustheit der besten Kombination (Anteil positiver Nachbarn) oder ``None``."""
        if self.best_index is None:
            return None
        value = float(self.table["robustness"].iloc[self.best_index])
        return None if math.isnan(value) else value

    def heatmap(self, x: str, y: str) -> pd.DataFrame:
        """Zielwerte als Matrix für eine Heatmap über zwei Parameter.

        Weitere Parameter werden auf den Wert der besten Kombination festgehalten (ohne
        beste Kombination auf ihren ersten Rasterwert).

        Args:
            x: Parameter der Spalten.
            y: Parameter der Zeilen.

        Returns:
            DataFrame mit ``y``-Werten als Index und ``x``-Werten als Spalten.

        Raises:
            OptimizationError: Unbekannter Parameter oder ``x == y``.
        """
        for name in (x, y):
            if name not in self.grid:
                raise OptimizationError(f"Unbekannter Parameter: {name!r}.")
        if x == y:
            raise OptimizationError("Für die Heatmap zwei verschiedene Parameter wählen.")
        fixed = self.best_params or {n: v[0] for n, v in self.grid.items()}
        mask = np.ones(len(self.table), dtype=bool)
        for name in self.grid:
            if name not in (x, y):
                mask &= (self.table[name] == fixed[name]).to_numpy()
        sub = self.table[mask]
        matrix = sub.pivot(index=y, columns=x, values="objective")
        return matrix.reindex(index=self.grid[y], columns=self.grid[x])


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class _Context:
    """Alles, was ein Arbeitsprozess für einen Backtest braucht (wird gepickelt)."""

    strategy: Strategy
    symbol: str | None
    universe: Universe | None
    timeframe: Timeframe
    start: Date
    end: Date
    settings: Settings
    bars: Any
    costs: CostModel | None
    max_positions: int


_WORKER_CONTEXT: _Context | None = None


def _init_worker(ctx: _Context) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = ctx


def _evaluate_in_worker(combo: dict[str, Any]) -> dict[str, Any]:
    assert _WORKER_CONTEXT is not None
    return _evaluate(_WORKER_CONTEXT, combo)


def _evaluate(ctx: _Context, combo: dict[str, Any]) -> dict[str, Any]:
    """Backtest einer Kombination; liefert Kennzahlen oder einen Fehlertext."""
    try:
        result = _run(ctx, combo)
        values = _objective_values(result, ctx.settings.risk_free_rate_pct)
        values["error"] = ""
    except (BacktestError, ParamError) as exc:
        values = dict.fromkeys((*METRIC_COLUMNS, "cagr"))
        values["error"] = str(exc)
    return values


def _run(ctx: _Context, combo: Mapping[str, Any]) -> BacktestResult:
    # Eigene Kopie je Lauf: Strategien dürfen Zustand halten, Threads teilen sonst Objekte.
    strategy = copy.deepcopy(ctx.strategy)
    if ctx.symbol is not None:
        return run_backtest(
            strategy,
            combo,
            ctx.symbol,
            ctx.timeframe,
            ctx.start,
            ctx.end,
            ctx.settings,
            bars=ctx.bars,
            costs=ctx.costs,
        )
    assert ctx.universe is not None
    return run_portfolio_backtest(
        strategy,
        combo,
        ctx.universe,
        ctx.timeframe,
        ctx.start,
        ctx.end,
        ctx.settings,
        max_positions=ctx.max_positions,
        bars=ctx.bars,
        costs=ctx.costs,
        workers=1,
    )


def optimize(
    strategy: Strategy | type[Strategy],
    grid: Mapping[str, Sequence[Any]],
    symbol: str | None,
    timeframe: Timeframe,
    start: DateInput,
    end: DateInput,
    settings: Settings,
    *,
    objective: Objective | str = Objective.SHARPE,
    universe: Universe | None = None,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    bars: pd.DataFrame | Mapping[str, pd.DataFrame] | None = None,
    costs: CostModel | None = None,
    workers: int | None = None,
    progress: ProgressCallback | None = None,
) -> OptimizationResult:
    """Optimiert die Parameter einer Strategie über ein Raster.

    Args:
        strategy: Strategie (Instanz oder Klasse).
        grid: Raster, z. B. aus :func:`build_grid`. Fehlende Parameter erhalten den
            Standardwert.
        symbol: Wertpapier für einen Einzelwert-Backtest; ``None`` mit ``universe``.
        timeframe: Zeitrahmen.
        start: Erste Kerze, auf der gehandelt werden darf (einschließlich).
        end: Letzter Tag der Daten (einschließlich).
        settings: Einstellungen (Kapital, Kosten, risikofreier Zins, Datenbank).
        objective: Zielgröße (Sharpe, CAGR, Calmar oder MAR).
        universe: Universum für einen Portfolio-Backtest statt ``symbol``.
        max_positions: Maximale Anzahl Positionen im Portfolio-Backtest.
        bars: Bereinigte Tageskurse (Einzelwert: ein DataFrame; Universum: je Symbol)
            statt Laden aus der Datenbank.
        costs: Kostenmodell statt der Kosten aus ``settings``.
        workers: Anzahl Prozesse bzw. Threads; ``None`` = Anzahl CPU-Kerne, ``1`` = ohne
            Parallelisierung.
        progress: Fortschritt ``(fertig, gesamt)`` nach jeder Kombination.

    Returns:
        Tabelle aller Kombinationen mit Zielwert, Kennzahlen und Robustheit.

    Raises:
        OptimizationError: Leeres Raster, unbekannter Parameter oder weder bzw. beide von
            ``symbol`` und ``universe``.
        BacktestError: Ungültiger Zeitraum oder keine Kurse.
    """
    template = strategy() if isinstance(strategy, type) else strategy
    goal = Objective(objective)
    full_grid = _complete_grid(template, grid)
    combos = _combinations(full_grid)
    log: list[str] = []
    _warn_size(len(combos), log)
    ctx = _make_context(
        template, symbol, universe, timeframe, start, end, settings, bars, costs, max_positions
    )
    rows = _evaluate_all(ctx, combos, workers, progress)
    table = pd.DataFrame(combos, columns=list(full_grid))
    metrics = pd.DataFrame(rows)
    table["objective"] = [_as_float(r[goal.value]) for r in rows]
    for column in METRIC_COLUMNS:
        table[column] = metrics[column].to_numpy()
    values = table["objective"].tolist()
    table["robustness"] = robustness_scores(full_grid, values)
    table["error"] = metrics["error"].to_numpy()
    failed = int((table["error"] != "").sum())
    if failed:
        first = table["error"][table["error"] != ""].iloc[0]
        log.append(f"{failed} Kombination(en) mit Fehler, z. B.: {first}")
    best = _best_index(values)
    if best is None:
        log.append("Keine Kombination hat einen gültigen Zielwert.")
    return OptimizationResult(template.id, goal, full_grid, table, best, log)


def _complete_grid(template: Strategy, grid: Mapping[str, Sequence[Any]]) -> ParamGrid:
    unknown = sorted(set(grid) - set(template.params))
    if unknown:
        raise OptimizationError(f"Unbekannte(r) Parameter: {', '.join(unknown)}.")
    full: ParamGrid = {}
    for name, param in template.params.items():
        values = list(grid[name]) if name in grid else [param.default]
        if not values:
            raise OptimizationError(f"Parameter {name}: Bereich enthält keine Werte.")
        full[name] = values
    return full


def _make_context(
    template: Strategy,
    symbol: str | None,
    universe: Universe | None,
    timeframe: Timeframe,
    start: DateInput,
    end: DateInput,
    settings: Settings,
    bars: pd.DataFrame | Mapping[str, pd.DataFrame] | None,
    costs: CostModel | None,
    max_positions: int,
) -> _Context:
    if (symbol is None) == (universe is None):
        raise OptimizationError("Bitte genau eines angeben: ein Symbol oder ein Universum.")
    start_day, end_day = _to_day(start, "start"), _to_day(end, "end")
    if start_day > end_day:
        raise BacktestError(f"Beginn {start_day} liegt nach dem Ende {end_day}.")
    if universe is not None and not isinstance(universe, str | IndexHistory):
        universe = sorted({str(s) for s in universe})
    loaded = _preload(symbol, universe, start_day, end_day, settings, bars)
    return _Context(
        template,
        symbol,
        universe,
        Timeframe(timeframe),
        start_day,
        end_day,
        settings,
        loaded,
        costs,
        max_positions,
    )


def _preload(
    symbol: str | None,
    universe: Universe | None,
    start: Date,
    end: Date,
    settings: Settings,
    bars: pd.DataFrame | Mapping[str, pd.DataFrame] | None,
) -> Any:
    """Lädt die Kurse einmal, damit nicht jede Kombination die Datenbank liest."""
    if symbol is not None:
        if bars is None:
            return _load_symbol_daily(settings, symbol, end)
        if not isinstance(bars, pd.DataFrame):
            raise OptimizationError("Für ein Symbol bars als DataFrame übergeben.")
        return bars
    if bars is not None:
        if isinstance(bars, pd.DataFrame):
            raise OptimizationError("Für ein Universum bars je Symbol übergeben.")
        return dict(bars)
    assert universe is not None
    _, candidates, _ = _resolve_universe(universe, start, end, [])
    return _collect_bars(candidates, None, settings, end, [])


def _to_day(value: DateInput, name: str) -> Date:
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise BacktestError(f"{name}: kein gültiges Datum: {value!r}.") from exc
    if pd.isna(ts):
        raise BacktestError(f"{name}: kein gültiges Datum: {value!r}.")
    return ts.date()


def _evaluate_all(
    ctx: _Context,
    combos: list[dict[str, Any]],
    workers: int | None,
    progress: ProgressCallback | None,
) -> list[dict[str, Any]]:
    """Bewertet alle Kombinationen; Ergebnisliste in Rasterreihenfolge."""
    count = workers if workers is not None else (os.cpu_count() or 1)
    count = max(1, min(count, len(combos)))
    total = len(combos)
    if count == 1:
        rows = []
        for i, combo in enumerate(combos, start=1):
            rows.append(_evaluate(ctx, combo))
            if progress is not None:
                progress(i, total)
        return rows
    if total >= _PROCESS_MIN_COMBINATIONS and _process_safe(ctx.strategy):
        try:
            with ProcessPoolExecutor(
                max_workers=count, initializer=_init_worker, initargs=(ctx,)
            ) as pool:
                return _collect(pool, _evaluate_in_worker, combos, progress)
        except (
            BrokenProcessPool,
            ImportError,
            AttributeError,
            TypeError,
            pickle.PicklingError,
        ) as exc:
            logger.warning("Prozess-Pool nicht verfügbar (%s); Optimierung in Threads.", exc)
    with ThreadPoolExecutor(max_workers=count) as pool:
        return _collect(pool, lambda combo: _evaluate(ctx, combo), combos, progress)


def _collect(
    pool: Executor,
    func: Callable[[dict[str, Any]], dict[str, Any]],
    combos: list[dict[str, Any]],
    progress: ProgressCallback | None,
) -> list[dict[str, Any]]:
    futures = {pool.submit(func, combo): i for i, combo in enumerate(combos)}
    rows: list[dict[str, Any] | None] = [None] * len(combos)
    try:
        for done, future in enumerate(as_completed(futures), start=1):
            rows[futures[future]] = future.result()
            if progress is not None:
                progress(done, len(combos))
    except BaseException:
        # Abbruch (z. B. aus dem Fortschritts-Rückruf): wartende Kombinationen streichen,
        # damit das Beenden des Pools nicht auf das ganze Raster wartet.
        for future in futures:
            future.cancel()
        raise
    return [r for r in rows if r is not None]
