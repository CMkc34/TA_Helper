"""Performance-Kennzahlen eines Backtests.

Berechnet aus Equity-Kurve und Trade-Liste alle Kennzahlen aus der Spezifikation
(Abschnitt „Backtest-Engine“), eine Monatsrenditen-Tabelle und den Vergleich mit
Buy and Hold eines Benchmarks (Alpha, Beta, Korrelation).

Konventionen:

- Renditen je Kerze: ``r_t = E_t / E_{t-1} − 1``; die erste Rendite bezieht sich auf
  das Startkapital (Kapital vor der ersten Kerze). Ohne Startkapital beginnt die Reihe
  mit dem ersten Equity-Wert.
- Annualisierung nach Zeitrahmen mit :data:`PERIODS_PER_YEAR` (1D 252, 3D 84, 1W 52,
  1M 12 Kerzen). Die Laufzeit in Jahren ist ``Anzahl Renditen / Kerzen pro Jahr``.
- Risikofreier Zins ``rf`` in Prozent p. a. (Standard 0), je Kerze
  ``(1 + rf)^(1/P) − 1``.
- Volatilität und Sharpe nutzen die Stichproben-Standardabweichung (``ddof=1``);
  Sortino die Downside-Abweichung ``sqrt(mean(min(r − rf, 0)²))``.
- Prozentwerte enden auf ``_pct``; nicht definierte Werte (z. B. Profit Factor ohne
  Verlust-Trades) sind ``None``, damit das Ergebnis als JSON gespeichert werden kann.

Beispiel::

    result = run_backtest(...)
    m = metrics_for_result(result, risk_free_rate_pct=2.0, benchmark=spy_close)
    print(m.cagr_pct, m.sharpe, m.max_drawdown_pct)
    table = monthly_returns(result.equity, initial_capital=result.initial_capital)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, TypedDict

import numpy as np
import pandas as pd

from trading_helfer.backtest.engine import BARS_PER_YEAR
from trading_helfer.core.types import Timeframe

if TYPE_CHECKING:
    from trading_helfer.backtest.engine import BacktestResult

__all__ = [
    "PERIODS_PER_YEAR",
    "YEAR_COLUMN",
    "BenchmarkComparison",
    "MetricsError",
    "PerformanceMetrics",
    "benchmark_comparison",
    "compute_metrics",
    "metrics_for_result",
    "monthly_returns",
    "periods_per_year",
]

PERIODS_PER_YEAR: Final[Mapping[Timeframe, int]] = BARS_PER_YEAR
"""Kerzen pro Jahr je Zeitrahmen (1D 252, 3D 84, 1W 52, 1M 12)."""

YEAR_COLUMN: Final = "Jahr"
"""Spalte der Monatsrenditen-Tabelle mit der Jahresrendite."""

_DATE: Final = "date"
_EQUITY: Final = "equity"
_EXPOSURE: Final = "exposure"


class MetricsError(ValueError):
    """Kennzahlen können nicht berechnet werden; Text auf Deutsch."""


@dataclass(frozen=True)
class BenchmarkComparison:
    """Vergleich der Strategie mit Buy and Hold eines Benchmarks.

    Attributes:
        total_return_pct: Gesamtrendite Buy and Hold des Benchmarks im selben Zeitraum.
        cagr_pct: Jährliche Wachstumsrate des Benchmarks.
        excess_return_pct: Gesamtrendite der Strategie minus der des Benchmarks
            (Prozentpunkte).
        alpha_pct: Jensens Alpha p. a. in Prozent (über dem risikofreien Zins).
        beta: Beta der Strategie gegenüber dem Benchmark.
        correlation: Korrelation der Renditen je Kerze.
    """

    total_return_pct: float
    cagr_pct: float | None
    excess_return_pct: float
    alpha_pct: float | None
    beta: float | None
    correlation: float | None


@dataclass(frozen=True)
class PerformanceMetrics:
    """Alle Performance-Kennzahlen eines Backtests.

    Attributes:
        timeframe: Zeitrahmen der Equity-Kurve.
        periods_per_year: Kerzen pro Jahr für die Annualisierung.
        risk_free_rate_pct: Verwendeter risikofreier Zins p. a. in Prozent.
        start_equity: Kapital zu Beginn.
        end_equity: Kapital am Ende.
        periods: Anzahl der Renditen (Kerzen).
        years: Laufzeit in Jahren (``periods / periods_per_year``).
        total_return_pct: Gesamtrendite.
        cagr_pct: Jährliche Wachstumsrate.
        volatility_pct: Annualisierte Volatilität der Renditen.
        sharpe: Sharpe Ratio (annualisiert).
        sortino: Sortino Ratio (annualisiert).
        calmar: CAGR / |Max Drawdown|.
        max_drawdown_pct: Größter Rückgang vom Höchststand (0 oder negativ).
        max_drawdown_duration_bars: Längste Dauer unter einem Höchststand in Kerzen
            (vom Höchststand bis zur Erholung bzw. bis zum Ende).
        max_drawdown_duration_days: Dieselbe Dauer in Kalendertagen.
        num_trades: Anzahl abgeschlossener Trades.
        win_rate_pct: Anteil der Trades mit Gewinn (``pnl > 0``).
        profit_factor: Summe der Gewinne / Summe der Verluste.
        avg_win: Durchschnittlicher Gewinn je Gewinn-Trade in USD.
        avg_loss: Durchschnittlicher Verlust je Verlust-Trade in USD (negativ).
        avg_win_pct: Durchschnittlicher Gewinn je Gewinn-Trade in Prozent.
        avg_loss_pct: Durchschnittlicher Verlust je Verlust-Trade in Prozent (negativ).
        expectancy: Erwartungswert je Trade in USD (Durchschnitt aller ``pnl``).
        expectancy_pct: Erwartungswert je Trade in Prozent.
        avg_holding_days: Durchschnittliche Haltedauer in Kalendertagen.
        exposure_pct: Durchschnittlicher Anteil des Kapitals in Positionen.
        time_in_market_pct: Anteil der Kerzen mit offener Position.
        benchmark: Vergleich mit Buy and Hold des Benchmarks oder ``None``.
    """

    timeframe: Timeframe
    periods_per_year: int
    risk_free_rate_pct: float
    start_equity: float
    end_equity: float
    periods: int
    years: float
    total_return_pct: float
    cagr_pct: float | None
    volatility_pct: float | None
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    max_drawdown_pct: float
    max_drawdown_duration_bars: int
    max_drawdown_duration_days: int
    num_trades: int
    win_rate_pct: float | None
    profit_factor: float | None
    avg_win: float | None
    avg_loss: float | None
    avg_win_pct: float | None
    avg_loss_pct: float | None
    expectancy: float | None
    expectancy_pct: float | None
    avg_holding_days: float | None
    exposure_pct: float | None
    time_in_market_pct: float | None
    benchmark: BenchmarkComparison | None = None

    def to_dict(self) -> dict[str, object]:
        """Kennzahlen als JSON-taugliches Dictionary (für ``metrics_json``)."""
        data = asdict(self)
        data["timeframe"] = self.timeframe.value
        return data


# ---------------------------------------------------------------------------
# Öffentliche Funktionen
# ---------------------------------------------------------------------------


def periods_per_year(timeframe: Timeframe | str) -> int:
    """Kerzen pro Jahr für die Annualisierung.

    Args:
        timeframe: Zeitrahmen (``1D``, ``3D``, ``1W`` oder ``1M``).

    Returns:
        252, 84, 52 bzw. 12.

    Raises:
        MetricsError: Unbekannter Zeitrahmen.
    """
    try:
        return PERIODS_PER_YEAR[Timeframe(timeframe)]
    except ValueError as exc:
        raise MetricsError(f"Unbekannter Zeitrahmen: {timeframe!r}.") from exc


def compute_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame | None,
    timeframe: Timeframe | str,
    *,
    initial_capital: float | None = None,
    risk_free_rate_pct: float = 0.0,
    benchmark: pd.Series | None = None,
) -> PerformanceMetrics:
    """Berechnet alle Performance-Kennzahlen.

    Args:
        equity: Equity-Kurve mit Spalten ``date`` und ``equity``, optional ``exposure``
            (Positionswert / Kapital), eine Zeile je Kerze, aufsteigend.
        trades: Abgeschlossene Trades mit ``pnl``, ``pnl_pct``, ``entry_date`` und
            ``exit_date``; ``None`` oder leer = keine Trades.
        timeframe: Zeitrahmen der Kerzen (bestimmt die Annualisierung).
        initial_capital: Kapital vor der ersten Kerze; ``None`` = erster Equity-Wert.
        risk_free_rate_pct: Risikofreier Zins p. a. in Prozent (Standard 0).
        benchmark: Schlusskurse des Benchmarks mit Datumsindex für den Vergleich mit
            Buy and Hold; ``None`` = kein Vergleich.

    Returns:
        Alle Kennzahlen.

    Raises:
        MetricsError: Leere oder ungültige Equity-Kurve.
    """
    ppy = periods_per_year(timeframe)
    dates, values, start_equity = _equity_series(equity, initial_capital)
    returns = _returns(values)
    n = len(returns)
    years = n / ppy
    end_equity = float(values[-1])
    total = end_equity / start_equity - 1.0
    cagr = _cagr(start_equity, end_equity, years)
    rf_period = _rf_per_period(risk_free_rate_pct, ppy)
    excess = returns - rf_period

    volatility = _std(returns) * math.sqrt(ppy) if n >= 2 else None
    sharpe = _ratio(float(np.mean(excess)) * math.sqrt(ppy), _std(excess)) if n >= 2 else None
    downside = math.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2))) if n else 0.0
    sortino = _ratio(float(np.mean(excess)) * math.sqrt(ppy), downside) if n else None

    max_dd, dd_bars, dd_days = _drawdown(values, dates)
    calmar = _ratio(cagr, abs(max_dd)) if cagr is not None else None

    trade_stats = _trade_stats(trades)
    exposure_pct, time_in_market_pct = _exposure(equity)

    comparison = None
    if benchmark is not None:
        comparison = benchmark_comparison(
            returns,
            dates[1:],
            benchmark,
            timeframe,
            risk_free_rate_pct=risk_free_rate_pct,
        )

    return PerformanceMetrics(
        timeframe=Timeframe(timeframe),
        periods_per_year=ppy,
        risk_free_rate_pct=float(risk_free_rate_pct),
        start_equity=start_equity,
        end_equity=end_equity,
        periods=n,
        years=years,
        total_return_pct=total * 100.0,
        cagr_pct=_pct(cagr),
        volatility_pct=_pct(volatility),
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown_pct=max_dd * 100.0,
        max_drawdown_duration_bars=dd_bars,
        max_drawdown_duration_days=dd_days,
        exposure_pct=exposure_pct,
        time_in_market_pct=time_in_market_pct,
        benchmark=comparison,
        **trade_stats,
    )


def metrics_for_result(
    result: BacktestResult,
    *,
    risk_free_rate_pct: float = 0.0,
    benchmark: pd.Series | None = None,
) -> PerformanceMetrics:
    """Kennzahlen für ein Ergebnis von :func:`~trading_helfer.backtest.engine.run_backtest`.

    Args:
        result: Backtest-Ergebnis (Equity-Kurve, Trades, Startkapital, Zeitrahmen).
        risk_free_rate_pct: Risikofreier Zins p. a. in Prozent (Standard 0).
        benchmark: Schlusskurse des Benchmarks mit Datumsindex oder ``None``.

    Returns:
        Alle Kennzahlen.
    """
    return compute_metrics(
        result.equity,
        result.trades,
        result.timeframe,
        initial_capital=result.initial_capital,
        risk_free_rate_pct=risk_free_rate_pct,
        benchmark=benchmark,
    )


def monthly_returns(equity: pd.DataFrame, *, initial_capital: float | None = None) -> pd.DataFrame:
    """Monatsrenditen-Tabelle (Jahre × Monate) in Prozent.

    Die Rendite eines Monats ist ``E(Monatsende) / E(Vormonatsende) − 1``; der erste Monat
    bezieht sich auf ``initial_capital`` (bzw. den ersten Equity-Wert). Monate ohne
    Kerze (z. B. bei 3D-Lücken oder vor dem Start) bleiben leer (NaN); ihre Entwicklung
    steckt im nächsten Monat mit Kerze.

    Args:
        equity: Equity-Kurve mit Spalten ``date`` und ``equity``.
        initial_capital: Kapital vor der ersten Kerze; ``None`` = erster Equity-Wert.

    Returns:
        DataFrame mit Index Jahr (int), Spalten 1–12 (Monate) und
        :data:`YEAR_COLUMN` (Jahresrendite, aufgezinst aus den Monaten).

    Raises:
        MetricsError: Leere oder ungültige Equity-Kurve.
    """
    _, _, start_equity = _equity_series(equity, initial_capital)
    index = pd.DatetimeIndex(pd.to_datetime(equity[_DATE]))
    series = pd.Series(equity[_EQUITY].to_numpy(dtype=float), index=index)
    month_end = series.groupby([index.year, index.month]).last()
    prev = np.concatenate([[start_equity], month_end.to_numpy(dtype=float)[:-1]])
    rets = (month_end.to_numpy(dtype=float) / prev - 1.0) * 100.0
    years = sorted({int(y) for y, _ in month_end.index})
    table = pd.DataFrame(
        np.nan, index=pd.Index(years, name="year"), columns=[*range(1, 13), YEAR_COLUMN]
    )
    for (year, month), value in zip(month_end.index, rets, strict=True):
        table.loc[int(year), int(month)] = value
    for year in years:
        row = table.loc[year, list(range(1, 13))].dropna().to_numpy(dtype=float)
        table.loc[year, YEAR_COLUMN] = (float(np.prod(1.0 + row / 100.0)) - 1.0) * 100.0
    return table


def benchmark_comparison(
    returns: np.ndarray | pd.Series,
    dates: pd.DatetimeIndex | pd.Series,
    benchmark: pd.Series,
    timeframe: Timeframe | str,
    *,
    risk_free_rate_pct: float = 0.0,
) -> BenchmarkComparison:
    """Vergleicht Strategie-Renditen mit Buy and Hold des Benchmarks.

    Der Benchmark wird auf die Daten der Equity-Kurve ausgerichtet (letzter bekannter
    Kurs je Datum). Der Ausgangskurs ist der letzte Kurs vor dem ersten Datum; gibt es
    keinen, der Kurs am ersten Datum (erste Benchmark-Rendite dann 0).

    Args:
        returns: Strategie-Renditen je Kerze (Dezimalbrüche), gleiche Länge wie ``dates``.
        dates: Datum je Rendite (Datum der Kerze, an deren Ende die Rendite feststeht).
        benchmark: Schlusskurse des Benchmarks mit Datumsindex.
        timeframe: Zeitrahmen (bestimmt die Annualisierung).
        risk_free_rate_pct: Risikofreier Zins p. a. in Prozent.

    Returns:
        Gesamtrendite, CAGR, Überrendite, Alpha, Beta und Korrelation.

    Raises:
        MetricsError: Benchmark leer, ohne Kurs im Zeitraum oder Längen passen nicht.
    """
    ppy = periods_per_year(timeframe)
    strat = np.asarray(returns, dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    if len(strat) != len(index):
        raise MetricsError("Renditen und Daten haben unterschiedliche Länge.")
    if len(strat) == 0:
        raise MetricsError("Keine Renditen für den Benchmark-Vergleich.")
    bench = _clean_benchmark(benchmark)
    aligned = bench.reindex(bench.index.union(index)).ffill().reindex(index)
    if aligned.isna().any():
        raise MetricsError(
            f"Benchmark hat keinen Kurs am {index[int(np.argmax(aligned.isna()))].date()}."
        )
    before = bench.loc[bench.index < index[0]]
    base = float(before.iloc[-1]) if len(before) else float(aligned.iloc[0])
    prices = np.concatenate([[base], aligned.to_numpy(dtype=float)])
    bench_ret = prices[1:] / prices[:-1] - 1.0

    total = prices[-1] / base - 1.0
    years = len(bench_ret) / ppy
    cagr = _cagr(1.0, 1.0 + total, years)
    strat_total = float(np.prod(1.0 + strat)) - 1.0

    rf = _rf_per_period(risk_free_rate_pct, ppy)
    beta = alpha = corr = None
    if len(strat) >= 2:
        xs, xb = strat - rf, bench_ret - rf
        var_b = float(np.var(xb, ddof=1))
        if var_b > 0:
            beta = float(np.cov(xs, xb, ddof=1)[0, 1]) / var_b
            alpha = (float(np.mean(xs)) - beta * float(np.mean(xb))) * ppy * 100.0
        if var_b > 0 and _std(strat) > 0:
            corr = float(np.corrcoef(strat, bench_ret)[0, 1])
    return BenchmarkComparison(
        total_return_pct=total * 100.0,
        cagr_pct=_pct(cagr),
        excess_return_pct=(strat_total - total) * 100.0,
        alpha_pct=alpha,
        beta=beta,
        correlation=corr,
    )


# ---------------------------------------------------------------------------
# Interna
# ---------------------------------------------------------------------------


def _equity_series(
    equity: pd.DataFrame, initial_capital: float | None
) -> tuple[pd.DatetimeIndex, np.ndarray, float]:
    """Daten und Equity-Werte, vorne um das Startkapital ergänzt.

    Returns:
        ``(dates, values, start)``; ``values[0]`` ist das Startkapital, ``dates[0]``
        dafür gleich dem ersten Datum der Kurve; mit Startkapital ist ``values`` eins
        länger als die Kurve, sonst gleich lang.
    """
    if _EQUITY not in equity.columns or _DATE not in equity.columns:
        raise MetricsError("Equity-Kurve braucht die Spalten 'date' und 'equity'.")
    if equity.empty:
        raise MetricsError("Equity-Kurve ist leer.")
    curve = equity[_EQUITY].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(equity[_DATE]))
    if not np.all(np.isfinite(curve)):
        raise MetricsError("Equity-Kurve enthält ungültige Werte (NaN oder unendlich).")
    if not dates.is_monotonic_increasing:
        raise MetricsError("Equity-Kurve ist nicht aufsteigend nach Datum sortiert.")
    if initial_capital is None:
        start = float(curve[0])
        values = curve
        all_dates = dates
    else:
        start = float(initial_capital)
        values = np.concatenate([[start], curve])
        all_dates = pd.DatetimeIndex(dates[:1].append(dates))
    if start <= 0:
        raise MetricsError(f"Startkapital muss positiv sein, erhalten: {start}.")
    return all_dates, values, start


def _returns(values: np.ndarray) -> np.ndarray:
    prev = values[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, values[1:] / np.where(prev > 0, prev, 1.0) - 1.0, -1.0)
    return np.asarray(rets, dtype=float)


def _cagr(start: float, end: float, years: float) -> float | None:
    if years <= 0:
        return None
    if end <= 0:
        return -1.0
    return float((end / start) ** (1.0 / years) - 1.0)


def _rf_per_period(rate_pct: float, ppy: int) -> float:
    if rate_pct <= -100:
        raise MetricsError(f"Risikofreier Zins muss größer als -100 % sein: {rate_pct}.")
    return float((1.0 + rate_pct / 100.0) ** (1.0 / ppy) - 1.0)


def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) >= 2 else 0.0


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 1e-15 or not math.isfinite(denominator):
        return None
    return numerator / denominator


def _pct(value: float | None) -> float | None:
    return None if value is None else value * 100.0


def _drawdown(values: np.ndarray, dates: pd.DatetimeIndex) -> tuple[float, int, int]:
    """Max Drawdown (Dezimalbruch, ≤ 0) und längste Unterwasser-Phase in Kerzen und Tagen."""
    peak = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, values / np.where(peak > 0, peak, 1.0) - 1.0, 0.0)
    max_dd = float(dd.min()) if len(dd) else 0.0
    best_bars = best_days = 0
    peak_idx = 0
    n = len(values)

    def phase(end: int) -> None:
        nonlocal best_bars, best_days
        if end - peak_idx > best_bars:
            best_bars = end - peak_idx
            best_days = int((dates[end] - dates[peak_idx]).days)

    for i in range(1, n):
        if values[i] >= values[peak_idx]:
            if i > peak_idx + 1:  # dazwischen lag die Kurve unter dem Höchststand
                phase(i)
            peak_idx = i
    if peak_idx < n - 1:  # am Ende noch nicht erholt
        phase(n - 1)
    return max_dd, best_bars, best_days


class _TradeStats(TypedDict):
    num_trades: int
    win_rate_pct: float | None
    profit_factor: float | None
    avg_win: float | None
    avg_loss: float | None
    avg_win_pct: float | None
    avg_loss_pct: float | None
    expectancy: float | None
    expectancy_pct: float | None
    avg_holding_days: float | None


def _trade_stats(trades: pd.DataFrame | None) -> _TradeStats:
    empty: _TradeStats = {
        "num_trades": 0,
        "win_rate_pct": None,
        "profit_factor": None,
        "avg_win": None,
        "avg_loss": None,
        "avg_win_pct": None,
        "avg_loss_pct": None,
        "expectancy": None,
        "expectancy_pct": None,
        "avg_holding_days": None,
    }
    if trades is None or trades.empty:
        return empty
    if "pnl" not in trades.columns:
        raise MetricsError("Trade-Liste braucht die Spalte 'pnl'.")
    pnl = trades["pnl"].to_numpy(dtype=float)
    pnl_pct = (
        trades["pnl_pct"].to_numpy(dtype=float)
        if "pnl_pct" in trades.columns
        else np.full(len(pnl), np.nan)
    )
    wins, losses = pnl > 0, pnl < 0
    gross_win = float(pnl[wins].sum())
    gross_loss = float(-pnl[losses].sum())
    holding = None
    if {"entry_date", "exit_date"} <= set(trades.columns):
        delta = pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])
        holding = float(delta.dt.days.mean())
    return {
        "num_trades": len(pnl),
        "win_rate_pct": float(wins.mean()) * 100.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else None,
        "avg_win": float(pnl[wins].mean()) if wins.any() else None,
        "avg_loss": float(pnl[losses].mean()) if losses.any() else None,
        "avg_win_pct": _nanmean(pnl_pct[wins]),
        "avg_loss_pct": _nanmean(pnl_pct[losses]),
        "expectancy": float(pnl.mean()),
        "expectancy_pct": _nanmean(pnl_pct),
        "avg_holding_days": holding,
    }


def _nanmean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def _exposure(equity: pd.DataFrame) -> tuple[float | None, float | None]:
    if _EXPOSURE not in equity.columns:
        return None, None
    exp = equity[_EXPOSURE].to_numpy(dtype=float)
    exp = exp[np.isfinite(exp)]
    if not exp.size:
        return None, None
    return float(np.abs(exp).mean()) * 100.0, float((np.abs(exp) > 1e-12).mean()) * 100.0


def _clean_benchmark(benchmark: pd.Series) -> pd.Series:
    if benchmark.empty:
        raise MetricsError("Benchmark-Kurse sind leer.")
    series = pd.Series(
        benchmark.to_numpy(dtype=float), index=pd.DatetimeIndex(pd.to_datetime(benchmark.index))
    )
    series = series[np.isfinite(series.to_numpy())]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty or bool((series <= 0).any()):
        raise MetricsError("Benchmark-Kurse müssen positiv sein.")
    return series
