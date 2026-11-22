"""Fassade für den Backtest-Tab: Läufe starten, speichern, laden, vergleichen, exportieren.

Das Modul importiert kein Qt. Es verbindet die Strategie-Registry, die Backtest-Engine
(:func:`~trading_helfer.backtest.engine.run_backtest`), die Kennzahlen und den
Ergebnisspeicher (:class:`~trading_helfer.backtest.results.BacktestRunStore`) mit der
Oberfläche. Jede Methode öffnet die Datenbank selbst und schließt sie wieder, damit sie
in einem Hintergrund-Thread laufen kann.

**Universum**: Ein Index wird mit dem Portfolio-Backtest
(:func:`~trading_helfer.backtest.portfolio.run_portfolio_backtest`) gerechnet: gemeinsames
Kapital, höchstens ``max_positions`` Positionen, Käufe nur für Werte, die am Tag ``t``
Indexmitglied sind.

Beispiel::

    service = BacktestService(settings)
    request = BacktestRequest("trend_sma_cross", {"fast": 10}, "AAPL", Timeframe.W1,
                              dt.date(2000, 1, 1), dt.date(2024, 12, 31))
    outcome = service.run(request)
    print(outcome.metrics.total_return_pct, outcome.run_id)
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import (
    TRADE_COLUMNS,
    BacktestError,
    BacktestResult,
    run_backtest,
)
from trading_helfer.backtest.metrics import (
    MetricsError,
    PerformanceMetrics,
    metrics_for_result,
    monthly_returns,
)
from trading_helfer.backtest.portfolio import DEFAULT_MAX_POSITIONS, run_portfolio_backtest
from trading_helfer.backtest.results import (
    BacktestRunStore,
    RunFilter,
    RunSummary,
    strategy_source_hash,
)
from trading_helfer.core.config import Settings
from trading_helfer.core.markers import Marker, MarkerKind
from trading_helfer.core.types import Side, Timeframe
from trading_helfer.data import universe as index_universe
from trading_helfer.data.storage import MarketStore
from trading_helfer.data.update import BENCHMARKS
from trading_helfer.strategies.base import Param
from trading_helfer.strategies.registry import StrategyInfo, StrategyRegistry

__all__ = [
    "COMPARE_METRICS",
    "DEFAULT_BENCHMARK",
    "INDEX_LABELS",
    "METRIC_TILES",
    "BacktestOutcome",
    "BacktestRequest",
    "BacktestService",
    "BarsLoader",
    "UniverseKind",
    "compare_runs",
    "format_metric_value",
    "trade_markers",
    "trades_to_csv",
]

logger = logging.getLogger(__name__)

BarsLoader = Callable[[str], pd.DataFrame]
"""Liefert bereinigte Tageskurse eines Symbols (leer, wenn es keine gibt)."""

ProgressFn = Callable[[int, str], None]
"""Fortschritt in Prozent (0–100) und kurze Meldung."""

DEFAULT_BENCHMARK: Final = BENCHMARKS[0]
"""Standard-Benchmark für Equity-Vergleich und Kennzahlen (SPY)."""

INDEX_LABELS: Final[dict[str, str]] = {"SP500": "S&P 500", "NDX100": "Nasdaq-100"}
"""Indizes, die als Universum zur Wahl stehen, mit Anzeigenamen."""

METRIC_TILES: Final[tuple[tuple[str, str, str], ...]] = (
    ("total_return_pct", "Gesamtrendite", "pct"),
    ("cagr_pct", "CAGR", "pct"),
    ("max_drawdown_pct", "Max. Drawdown", "pct"),
    ("sharpe", "Sharpe", "ratio"),
    ("sortino", "Sortino", "ratio"),
    ("calmar", "Calmar", "ratio"),
    ("volatility_pct", "Volatilität", "pct"),
    ("num_trades", "Trades", "int"),
    ("win_rate_pct", "Trefferquote", "pct"),
    ("profit_factor", "Profit Factor", "ratio"),
    ("expectancy", "Erwartungswert/Trade", "money"),
    ("avg_holding_days", "Haltedauer (Tage)", "days"),
    ("exposure_pct", "Exposure", "pct"),
    ("end_equity", "Endkapital", "money"),
    ("benchmark.total_return_pct", "Benchmark-Rendite", "pct"),
    ("benchmark.excess_return_pct", "Überrendite", "pct"),
)
"""Kennzahlen-Kacheln: Schlüssel (``benchmark.`` für den Vergleich), Titel, Format."""

COMPARE_METRICS: Final[tuple[tuple[str, str, str], ...]] = (
    *METRIC_TILES[:2],
    ("max_drawdown_duration_days", "Max. Drawdown-Dauer (Tage)", "int"),
    *METRIC_TILES[2:],
    ("avg_win_pct", "Ø Gewinn", "pct"),
    ("avg_loss_pct", "Ø Verlust", "pct"),
    ("benchmark.alpha_pct", "Alpha p. a.", "pct"),
    ("benchmark.beta", "Beta", "ratio"),
)
"""Zeilen der Vergleichstabelle zweier Läufe."""

CSV_SEPARATOR: Final = ";"
"""Trennzeichen beim CSV-Export (Excel mit deutscher Ländereinstellung)."""


class UniverseKind(StrEnum):
    """Art des Universums eines Laufs."""

    SYMBOL = "symbol"
    """Ein einzelner Wert."""
    INDEX = "index"
    """Alle Werte, die im Zeitraum Mitglied des Index waren."""


@dataclass(frozen=True)
class BacktestRequest:
    """Eingaben des Backtest-Formulars.

    Attributes:
        strategy_id: ID der Strategie.
        params: Parameterwerte (fehlende erhalten den Standardwert).
        universe: Symbol (z. B. ``"AAPL"``) oder Index-Code (``"SP500"``, ``"NDX100"``).
        timeframe: Zeitrahmen.
        start: Erste Kerze, auf der gehandelt werden darf.
        end: Letzter Tag der Daten.
        universe_kind: Einzelwert oder Index.
        capital: Startkapital in USD.
        fee_per_order: Fixe Gebühr je Order in USD.
        fee_pct: Prozentuale Gebühr auf den Orderwert.
        slippage_pct: Slippage in Prozent.
        benchmark: Symbol des Benchmarks; leer = ohne Benchmark.
        save: Lauf in der Datenbank speichern.
        max_positions: Höchstzahl gleichzeitiger Positionen (nur Index).
    """

    strategy_id: str
    params: dict[str, Any]
    universe: str
    timeframe: Timeframe
    start: dt.date
    end: dt.date
    universe_kind: UniverseKind = UniverseKind.SYMBOL
    capital: float = 1000.0
    fee_per_order: float = 1.0
    fee_pct: float = 0.0
    slippage_pct: float = 0.05
    benchmark: str = DEFAULT_BENCHMARK
    save: bool = True
    max_positions: int = DEFAULT_MAX_POSITIONS

    def costs(self) -> CostModel:
        """Kostenmodell aus Gebühren und Slippage."""
        return CostModel(
            fee_per_order=self.fee_per_order,
            fee_pct=self.fee_pct,
            slippage_pct=self.slippage_pct,
        )


@dataclass(frozen=True)
class BacktestOutcome:
    """Ergebnis für die Anzeige: Lauf, Kennzahlen, Benchmark und Monatsrenditen.

    Attributes:
        result: Ergebnis der Engine (bei einem Index: zusammengefasst).
        metrics: Kennzahlen oder ``None``, wenn sie nicht berechenbar sind.
        universe: Symbol oder Index-Code.
        run_id: ID des gespeicherten Laufs oder ``None``, wenn nicht gespeichert.
        benchmark_symbol: Symbol des Benchmarks (leer = ohne).
        benchmark: Schlusskurse des Benchmarks mit Datumsindex oder ``None``.
        monthly: Monatsrenditen (Jahre × Monate) oder ``None``.
    """

    result: BacktestResult
    metrics: PerformanceMetrics | None
    universe: str
    run_id: str | None = None
    benchmark_symbol: str = ""
    benchmark: pd.Series | None = None
    monthly: pd.DataFrame | None = field(default=None)

    @property
    def symbols(self) -> list[str]:
        """Symbole mit Trades, alphabetisch."""
        if self.result.trades.empty:
            return []
        return sorted({str(s) for s in self.result.trades["symbol"]})

    def benchmark_equity(self) -> pd.Series | None:
        """Buy and Hold des Benchmarks auf das Startkapital skaliert, an der Equity-Kurve.

        Returns:
            Werte je Datum der Equity-Kurve oder ``None`` ohne Benchmark oder Kurse.
        """
        bench = self.benchmark
        equity = self.result.equity
        if bench is None or bench.empty or equity.empty:
            return None
        dates = pd.DatetimeIndex(pd.to_datetime(equity["date"]))
        series = bench.sort_index()
        series.index = pd.DatetimeIndex(pd.to_datetime(series.index))
        aligned = series.reindex(series.index.union(dates)).ffill().reindex(dates)
        before = series.loc[series.index < dates[0]]
        base = float(before.iloc[-1]) if len(before) else float(aligned.iloc[0])
        if not np.isfinite(base) or base <= 0:
            return None
        return aligned / base * self.result.initial_capital


def format_metric_value(value: Any, fmt: str) -> str:
    """Formatiert eine Kennzahl deutsch (Komma, Tausenderpunkt); ``None`` wird „–“.

    Args:
        value: Zahl oder ``None``.
        fmt: ``pct``, ``ratio``, ``int``, ``money`` oder ``days``.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    number = float(value)
    if fmt == "pct":
        text = f"{number:,.2f} %"
    elif fmt == "int":
        text = f"{round(number):,d}"
    elif fmt == "money":
        text = f"{number:,.2f} $"
    elif fmt == "days":
        text = f"{number:,.1f}"
    else:
        text = f"{number:,.2f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def metric_value(metrics: PerformanceMetrics | None, key: str) -> Any:
    """Wert der Kennzahl ``key`` (``benchmark.<feld>`` für den Benchmark-Vergleich)."""
    if metrics is None:
        return None
    if key.startswith("benchmark."):
        bench = metrics.benchmark
        return None if bench is None else getattr(bench, key.split(".", 1)[1])
    return getattr(metrics, key)


def compare_runs(
    a: PerformanceMetrics | None, b: PerformanceMetrics | None
) -> list[tuple[str, str, str, str]]:
    """Vergleichstabelle zweier Läufe.

    Returns:
        Je Kennzahl aus :data:`COMPARE_METRICS`: Titel, Wert A, Wert B und Differenz
        B − A (leer, wenn ein Wert fehlt).
    """
    rows: list[tuple[str, str, str, str]] = []
    for key, title, fmt in COMPARE_METRICS:
        va, vb = metric_value(a, key), metric_value(b, key)
        diff = ""
        if va is not None and vb is not None:
            delta = float(vb) - float(va)
            diff = format_metric_value(delta, fmt)
            if delta > 0:
                diff = "+" + diff
        rows.append((title, format_metric_value(va, fmt), format_metric_value(vb, fmt), diff))
    return rows


def trade_markers(trades: pd.DataFrame, symbol: str | None = None) -> list[Marker]:
    """Ein- und Ausstiegsmarker der Trades für den Chart.

    Args:
        trades: Trade-Liste (Spalten :data:`TRADE_COLUMNS`).
        symbol: Nur Trades dieses Symbols; ``None`` = alle.

    Returns:
        Je Trade ein Einstieg (Long: ``BUY``, Short: ``SELL``) und ein ``EXIT``.
    """
    markers: list[Marker] = []
    records: list[dict[Any, Any]] = trades.to_dict("records")
    for row in records:
        if symbol is not None and str(row["symbol"]) != symbol:
            continue
        side = str(row["side"])
        entry_kind = MarkerKind.SELL if side == Side.SHORT.value else MarkerKind.BUY
        label = "Short" if entry_kind is MarkerKind.SELL else "Kauf"
        qty = format_metric_value(row["qty"], "ratio")
        markers.append(
            Marker(
                pd.Timestamp(row["entry_date"]).date(),
                entry_kind,
                float(row["entry_price"]),
                f"{label} {qty} Stk.",
            )
        )
        exit_date = row["exit_date"]
        if exit_date is not None and not pd.isna(exit_date):
            pnl = format_metric_value(row["pnl"], "money")
            markers.append(
                Marker(
                    pd.Timestamp(exit_date).date(),
                    MarkerKind.EXIT,
                    float(row["exit_price"]),
                    f"Ausstieg ({pnl})",
                )
            )
    return markers


def trades_to_csv(trades: pd.DataFrame, path: Path | str) -> int:
    """Schreibt die Trade-Liste als CSV (Semikolon, Dezimalkomma, UTF-8 mit BOM).

    Args:
        trades: Trade-Liste.
        path: Zieldatei.

    Returns:
        Anzahl geschriebener Trades.
    """
    frame = trades.reindex(columns=list(TRADE_COLUMNS)).copy()
    for column in ("entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
    frame.to_csv(
        path,
        sep=CSV_SEPARATOR,
        decimal=",",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    return len(frame)


class BacktestService:
    """Verbindet den Backtest-Tab mit Registry, Engine und Ergebnisspeicher.

    Args:
        settings: Einstellungen (Datenbank, Plugin-Ordner, Standardkosten).
        registry: Strategie-Registry; Standard lädt eingebaute Strategien und Plugins.
        bars_loader: Liefert bereinigte Tageskurse; Standard liest aus der Datenbank.
        symbols_loader: Liefert alle Symbole mit Kursen; Standard liest aus der Datenbank.
    """

    def __init__(
        self,
        settings: Settings,
        registry: StrategyRegistry | None = None,
        bars_loader: BarsLoader | None = None,
        symbols_loader: Callable[[], list[str]] | None = None,
    ) -> None:
        self.settings = settings
        if registry is None:
            registry = StrategyRegistry(settings.plugins_path)
            report = registry.load()
            for error in report.errors:
                logger.warning("Strategie nicht geladen: %s", error)
        self.registry = registry
        self._bars_loader = bars_loader if bars_loader is not None else self._load_from_store
        self._symbols_loader = (
            symbols_loader if symbols_loader is not None else self._symbols_from_store
        )

    # -- Strategien ------------------------------------------------------------------

    def strategies(self) -> list[StrategyInfo]:
        """Alle geladenen Strategien in Ladereihenfolge."""
        return self.registry.list_all()

    def reload_strategies(self) -> None:
        """Liest die Plugins neu ein (z. B. nach „Strategie hinzufügen“)."""
        report = self.registry.reload()
        for error in report.errors:
            logger.warning("Strategie nicht geladen: %s", error)

    def strategy_params(self, strategy_id: str) -> dict[str, Param]:
        """Parameterbeschreibungen der Strategie ``strategy_id``."""
        return dict(self.registry.get(strategy_id).params)

    def strategy_timeframes(self, strategy_id: str) -> list[Timeframe]:
        """Zeitrahmen, die die Strategie unterstützt."""
        return list(self.registry.get(strategy_id).timeframes)

    def default_request(self, strategy_id: str, universe: str = "") -> BacktestRequest:
        """Formular-Vorbelegung mit Standardparametern und Kosten aus den Einstellungen."""
        cls = self.registry.get(strategy_id)
        timeframe = cls.timeframes[0] if cls.timeframes else Timeframe.W1
        return BacktestRequest(
            strategy_id=strategy_id,
            params={name: p.default for name, p in cls.params.items()},
            universe=universe,
            timeframe=timeframe,
            start=self.settings.start_date,
            end=dt.date.today(),
            capital=self.settings.default_capital,
            fee_per_order=self.settings.fee_per_order,
            slippage_pct=self.settings.slippage_pct,
        )

    # -- Lauf ------------------------------------------------------------------------

    def run(self, request: BacktestRequest, progress: ProgressFn | None = None) -> BacktestOutcome:
        """Führt den Backtest aus, berechnet Kennzahlen und speichert den Lauf.

        Args:
            request: Formulareingaben.
            progress: Optionaler Fortschritts-Rückruf.

        Raises:
            BacktestError: Ungültige Eingaben oder keine Kurse.
            trading_helfer.strategies.registry.StrategyNotFoundError: Unbekannte Strategie.
            trading_helfer.strategies.base.ParamError: Ungültige Parameter.
        """
        report = progress if progress is not None else (lambda _p, _m: None)
        if request.capital <= 0:
            raise BacktestError("Das Startkapital muss größer als 0 sein.")
        cls = self.registry.get(request.strategy_id)
        cls.make_params(request.params)  # Fehler früh und auf Deutsch melden
        costs = request.costs()
        if request.universe_kind is UniverseKind.INDEX:
            result = self._run_index(request, costs, report)
        else:
            symbol = request.universe.strip().upper()
            if not symbol:
                raise BacktestError("Bitte ein Symbol angeben.")
            report(5, f"Lade {symbol}")
            bars = self._bars_loader(symbol)
            if bars.empty:
                raise BacktestError(f"Für {symbol} sind keine Kurse gespeichert.")
            report(20, f"Backtest {symbol}")
            result = self._run_one(cls, request, symbol, request.capital, costs, bars)
        result = replace(result, settings={**result.settings, "benchmark": request.benchmark})
        report(90, "Kennzahlen")
        benchmark = self._benchmark(request.benchmark)
        metrics = _safe_metrics(result, self.settings.risk_free_rate_pct, benchmark)
        universe = (
            result.symbol
            if request.universe_kind is UniverseKind.SYMBOL
            else request.universe.upper()
        )
        run_id: str | None = None
        if request.save:
            report(95, "Speichern")
            run_id = self._save(result, metrics, universe, strategy_source_hash(cls))
        report(100, "Fertig")
        return BacktestOutcome(
            result=result,
            metrics=metrics,
            universe=universe,
            run_id=run_id,
            benchmark_symbol=request.benchmark,
            benchmark=benchmark,
            monthly=_safe_monthly(result),
        )

    def _run_one(
        self,
        cls: type[Any],
        request: BacktestRequest,
        symbol: str,
        capital: float,
        costs: CostModel,
        bars: pd.DataFrame,
    ) -> BacktestResult:
        settings = self.settings.model_copy(update={"default_capital": capital})
        return run_backtest(
            cls,
            request.params,
            symbol,
            request.timeframe,
            request.start,
            request.end,
            settings,
            bars=bars,
            costs=costs,
        )

    def load_bars(self, symbol: str) -> pd.DataFrame:
        """Bereinigte Tageskurse von ``symbol`` (leer, wenn es keine gibt)."""
        return self._bars_loader(symbol)

    def index_bars(
        self,
        code: str,
        start: dt.date,
        end: dt.date,
        progress: ProgressFn | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Kurse aller Werte, die im Zeitraum Mitglied des Index ``code`` waren.

        Args:
            code: Index-Code (``"SP500"``, ``"NDX100"``).
            start: Erster Tag des Zeitraums.
            end: Letzter Tag des Zeitraums.
            progress: Fortschritt 5–30 % beim Laden.

        Raises:
            BacktestError: Unbekannter Index oder keine Kurse im Zeitraum.
        """
        report = progress if progress is not None else (lambda _p, _m: None)
        code = code.strip().upper()
        try:
            members = index_universe.all_members(code, start, end)
        except index_universe.UniverseError as exc:
            raise BacktestError(str(exc)) from exc
        available = set(self._symbols_loader())
        symbols = sorted(members & available)
        if not symbols:
            raise BacktestError(
                f"Für {INDEX_LABELS.get(code, code)} sind im Zeitraum keine Kurse gespeichert."
            )
        bars: dict[str, pd.DataFrame] = {}
        for i, symbol in enumerate(symbols):
            report(5 + int(25 * i / len(symbols)), f"Lade {symbol} ({i + 1}/{len(symbols)})")
            frame = self._bars_loader(symbol)
            if not frame.empty:
                bars[symbol] = frame
        return bars

    def _run_index(
        self, request: BacktestRequest, costs: CostModel, report: ProgressFn
    ) -> BacktestResult:
        code = request.universe.strip().upper()
        bars = self.index_bars(code, request.start, request.end, report)
        report(30, f"Portfolio-Backtest über {len(bars)} Werte")
        settings = self.settings.model_copy(update={"default_capital": request.capital})
        return run_portfolio_backtest(
            self.registry.get(request.strategy_id),
            request.params,
            code,
            request.timeframe,
            request.start,
            request.end,
            settings,
            max_positions=request.max_positions,
            bars=bars,
            costs=costs,
        )

    def _benchmark(self, symbol: str) -> pd.Series | None:
        if not symbol:
            return None
        try:
            bars = self._bars_loader(symbol)
        except Exception:  # fehlender Benchmark ist kein Grund, den Lauf abzubrechen
            logger.warning("Benchmark %s konnte nicht geladen werden", symbol, exc_info=True)
            return None
        if bars.empty:
            return None
        return pd.Series(
            bars["close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(bars["date"])),
            name=symbol,
        )

    # -- Gespeicherte Läufe ----------------------------------------------------------

    def list_runs(self, limit: int | None = None) -> list[RunSummary]:
        """Gespeicherte Läufe, neueste zuerst (leer ohne Datenbank)."""
        path = Path(self.settings.database_path)
        if not path.exists():
            return []
        with MarketStore(path) as store:
            return BacktestRunStore(store).list_runs(RunFilter(limit=limit))

    def load_run(self, run_id: str) -> BacktestOutcome:
        """Lädt einen gespeicherten Lauf samt Benchmark und Monatsrenditen.

        Raises:
            trading_helfer.backtest.results.RunNotFoundError: Unbekannte ID.
        """
        with MarketStore(self.settings.database_path) as store:
            stored = BacktestRunStore(store).load_run(run_id)
        bench_symbol = str(stored.summary.settings.get("benchmark", DEFAULT_BENCHMARK))
        benchmark = self._benchmark(bench_symbol)
        return BacktestOutcome(
            result=stored.result,
            metrics=stored.metrics,
            universe=stored.summary.universe,
            run_id=stored.run_id,
            benchmark_symbol=bench_symbol,
            benchmark=benchmark,
            monthly=_safe_monthly(stored.result),
        )

    def delete_run(self, run_id: str) -> bool:
        """Löscht einen gespeicherten Lauf; ``True``, wenn es ihn gab."""
        with MarketStore(self.settings.database_path) as store:
            return BacktestRunStore(store).delete_run(run_id)

    def export_trades(self, outcome: BacktestOutcome, path: Path | str) -> int:
        """Exportiert die Trades eines Ergebnisses als CSV; gibt die Anzahl zurück."""
        return trades_to_csv(outcome.result.trades, path)

    def _save(
        self,
        result: BacktestResult,
        metrics: PerformanceMetrics | None,
        universe: str,
        source_hash: str | None,
    ) -> str:
        with MarketStore(self.settings.database_path) as store:
            return BacktestRunStore(store).save_run(
                result, metrics=metrics, source_hash=source_hash, universe=universe
            )

    # -- Standard-Loader -------------------------------------------------------------

    def _load_from_store(self, symbol: str) -> pd.DataFrame:
        path = Path(self.settings.database_path)
        if not path.exists():
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        with MarketStore(path, read_only=True) as store:
            return store.load_bars(symbol, adjusted=True)

    def _symbols_from_store(self) -> list[str]:
        path = Path(self.settings.database_path)
        if not path.exists():
            return []
        with MarketStore(path, read_only=True) as store:
            return store.symbols()


def _safe_metrics(
    result: BacktestResult, risk_free_rate_pct: float, benchmark: pd.Series | None
) -> PerformanceMetrics | None:
    try:
        return metrics_for_result(
            result, risk_free_rate_pct=risk_free_rate_pct, benchmark=benchmark
        )
    except MetricsError:
        if benchmark is None:
            return None
    try:  # Benchmark ohne Kurs im Zeitraum: Kennzahlen ohne Vergleich
        return metrics_for_result(result, risk_free_rate_pct=risk_free_rate_pct)
    except MetricsError:
        return None


def _safe_monthly(result: BacktestResult) -> pd.DataFrame | None:
    try:
        return monthly_returns(result.equity, initial_capital=result.initial_capital)
    except MetricsError:
        return None
