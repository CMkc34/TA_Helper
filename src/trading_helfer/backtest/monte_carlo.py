"""Monte-Carlo-Analyse der Trades eines Backtests.

Die Reihenfolge der Trades ist zum Teil Zufall. Die Analyse erzeugt deshalb viele
alternative Trade-Folgen und zeigt, wie stark Endkapital und Max Drawdown davon abhängen.

Verfahren (:class:`MonteCarloMethod`):

- ``SHUFFLE`` (Umordnung): jede Simulation ist eine zufällige Permutation aller Trades.
  Weil die Renditen multiplikativ verkettet werden, ist das Endkapital in jeder
  Simulation gleich (bis auf Rundung); nur der Verlauf und damit der Max Drawdown ändern sich.
- ``BOOTSTRAP``: jede Simulation zieht ebenso viele Trades mit Zurücklegen. Endkapital
  und Max Drawdown streuen.

Konventionen:

- Grundlage sind die Renditen je Trade auf das Konto: ``r_k = pnl_k / E_{k-1}`` mit
  ``E_0 = Startkapital`` und ``E_k = E_{k-1} + pnl_k`` (Trades in Reihenfolge des
  Ausstiegs). Verkettet ergeben sie genau das Kapital aus abgeschlossenen Trades, auch
  wenn nicht das ganze Kapital investiert war.
- Ein Pfad beginnt beim Startkapital; ``E_k = E_{k-1} · (1 + r_k)``, nach unten bei 0
  begrenzt. Max Drawdown in Prozent (0 oder negativ) bezieht sich auf den bisherigen
  Höchststand inklusive Startkapital.
- Perzentile 5/50/95 mit linearer Interpolation (``numpy.percentile``).
- Zufall nur über ``numpy.random.default_rng(seed)``: gleiche Eingaben und gleicher
  Seed ergeben byte-gleiche Ergebnisse.

Beispiel::

    result = run_backtest(...)
    mc = monte_carlo_for_result(result, n_simulations=1000, seed=42)
    print(mc.final_equity_percentiles.p5, mc.max_drawdown_percentiles.p50)
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from trading_helfer.backtest.engine import BacktestResult

__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_SIMULATIONS",
    "PERCENTILES",
    "MonteCarloError",
    "MonteCarloMethod",
    "MonteCarloResult",
    "Percentiles",
    "monte_carlo",
    "monte_carlo_for_result",
    "trade_returns",
]

DEFAULT_SIMULATIONS: Final = 1000
"""Standardanzahl der Simulationen."""

DEFAULT_SEED: Final = 42
"""Standard-Seed des Zufallsgenerators."""

PERCENTILES: Final[tuple[float, float, float]] = (5.0, 50.0, 95.0)
"""Ausgewiesene Perzentile."""

FloatArray = npt.NDArray[np.float64]


class MonteCarloError(ValueError):
    """Monte-Carlo-Analyse nicht möglich; Text auf Deutsch."""


class MonteCarloMethod(StrEnum):
    """Verfahren zur Erzeugung alternativer Trade-Folgen."""

    SHUFFLE = "shuffle"
    """Zufällige Umordnung aller Trades (ohne Zurücklegen)."""

    BOOTSTRAP = "bootstrap"
    """Ziehen mit Zurücklegen."""


@dataclass(frozen=True)
class Percentiles:
    """5., 50. und 95. Perzentil einer Verteilung.

    Attributes:
        p5: 5-%-Perzentil (ungünstiger Fall).
        p50: Median.
        p95: 95-%-Perzentil (günstiger Fall).
    """

    p5: float
    p50: float
    p95: float

    @classmethod
    def of(cls, values: FloatArray) -> Percentiles:
        """Perzentile einer nicht leeren Wertereihe."""
        p5, p50, p95 = np.percentile(values, PERCENTILES)
        return cls(float(p5), float(p50), float(p95))


@dataclass(frozen=True, eq=False)
class MonteCarloResult:
    """Ergebnis einer Monte-Carlo-Analyse.

    Attributes:
        method: Verwendetes Verfahren.
        n_simulations: Anzahl der Simulationen.
        n_trades: Anzahl der Trades je Simulation.
        seed: Seed des Zufallsgenerators.
        initial_capital: Startkapital jedes Pfades.
        trade_returns: Renditen je Trade (Dezimal), Grundlage der Simulation.
        final_equity: Endkapital je Simulation (Länge ``n_simulations``).
        max_drawdown_pct: Max Drawdown je Simulation in Prozent (0 oder negativ).
        original_final_equity: Endkapital der tatsächlichen Trade-Folge.
        original_max_drawdown_pct: Max Drawdown der tatsächlichen Trade-Folge.
    """

    method: MonteCarloMethod
    n_simulations: int
    n_trades: int
    seed: int
    initial_capital: float
    trade_returns: FloatArray
    final_equity: FloatArray
    max_drawdown_pct: FloatArray
    original_final_equity: float
    original_max_drawdown_pct: float

    @property
    def final_equity_percentiles(self) -> Percentiles:
        """5/50/95-Perzentile des Endkapitals."""
        return Percentiles.of(self.final_equity)

    @property
    def total_return_percentiles(self) -> Percentiles:
        """5/50/95-Perzentile der Gesamtrendite in Prozent."""
        return Percentiles.of((self.final_equity / self.initial_capital - 1.0) * 100.0)

    @property
    def max_drawdown_percentiles(self) -> Percentiles:
        """5/50/95-Perzentile des Max Drawdowns in Prozent.

        Weil Drawdowns negativ sind, ist ``p5`` der tiefe (ungünstige) Drawdown.
        """
        return Percentiles.of(self.max_drawdown_pct)

    @property
    def loss_probability_pct(self) -> float:
        """Anteil der Simulationen mit Endkapital unter dem Startkapital in Prozent."""
        return float(np.mean(self.final_equity < self.initial_capital) * 100.0)

    def summary(self) -> dict[str, Any]:
        """Kennzahlen als JSON-fähiges Wörterbuch (ohne die Einzelwerte)."""

        def pct(p: Percentiles) -> dict[str, float]:
            return {"p5": p.p5, "p50": p.p50, "p95": p.p95}

        return {
            "method": self.method.value,
            "n_simulations": self.n_simulations,
            "n_trades": self.n_trades,
            "seed": self.seed,
            "initial_capital": self.initial_capital,
            "original_final_equity": self.original_final_equity,
            "original_max_drawdown_pct": self.original_max_drawdown_pct,
            "final_equity": pct(self.final_equity_percentiles),
            "total_return_pct": pct(self.total_return_percentiles),
            "max_drawdown_pct": pct(self.max_drawdown_percentiles),
            "loss_probability_pct": self.loss_probability_pct,
        }

    def fingerprint(self) -> str:
        """SHA-256 über alle Ergebnisdaten; gleiche Eingaben ergeben denselben Wert."""
        h = hashlib.sha256()
        header = (
            self.method.value,
            self.n_simulations,
            self.n_trades,
            self.seed,
            repr(self.initial_capital),
            repr(self.original_final_equity),
            repr(self.original_max_drawdown_pct),
        )
        h.update(repr(header).encode())
        for arr in (self.trade_returns, self.final_equity, self.max_drawdown_pct):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()


def trade_returns(trades: pd.DataFrame, initial_capital: float) -> FloatArray:
    """Renditen je Trade bezogen auf das Konto.

    ``r_k = pnl_k / E_{k-1}`` mit ``E_0 = initial_capital``; die Trades werden nach
    ``exit_date`` sortiert (stabil), falls die Spalte vorhanden ist.

    Args:
        trades: Abgeschlossene Trades mit Spalte ``pnl`` (USD).
        initial_capital: Startkapital (> 0).

    Returns:
        Renditen als Dezimalzahlen (0.05 = +5 %).

    Raises:
        MonteCarloError: Spalte ``pnl`` fehlt, Werte ungültig oder das Konto ist vor einem
            Trade nicht mehr positiv.
    """
    _check_capital(initial_capital)
    if "pnl" not in trades.columns:
        raise MonteCarloError("Die Trade-Liste enthält keine Spalte „pnl“.")
    ordered = trades
    if "exit_date" in trades.columns and len(trades) > 1:
        ordered = trades.assign(_exit=pd.to_datetime(trades["exit_date"])).sort_values(
            "_exit", kind="stable"
        )
    pnl = ordered["pnl"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(pnl)):
        raise MonteCarloError("Die Trade-Liste enthält ungültige Gewinne/Verluste (NaN).")
    before = initial_capital + np.concatenate([[0.0], np.cumsum(pnl)[:-1]])
    if np.any(before <= 0):
        raise MonteCarloError("Das Kapital ist vor einem Trade auf null oder darunter gefallen.")
    return pnl / before


def monte_carlo(
    returns: Sequence[float] | FloatArray | pd.Series,
    initial_capital: float,
    *,
    n_simulations: int = DEFAULT_SIMULATIONS,
    method: MonteCarloMethod | str = MonteCarloMethod.SHUFFLE,
    seed: int = DEFAULT_SEED,
) -> MonteCarloResult:
    """Monte-Carlo-Simulation über Trade-Renditen.

    Args:
        returns: Renditen je Trade (Dezimal) in tatsächlicher Reihenfolge.
        initial_capital: Startkapital jedes Pfades (> 0).
        n_simulations: Anzahl der Simulationen (≥ 1, Standard 1.000).
        method: ``shuffle`` (Umordnung) oder ``bootstrap`` (Ziehen mit Zurücklegen).
        seed: Seed für ``numpy.random.default_rng``.

    Returns:
        Verteilungen von Endkapital und Max Drawdown samt Perzentilen.

    Raises:
        MonteCarloError: Keine Trades, ungültige Renditen, Kapital oder Anzahl.
    """
    _check_capital(initial_capital)
    try:
        mc_method = MonteCarloMethod(method)
    except ValueError:
        raise MonteCarloError(
            f"Unbekanntes Verfahren „{method}“; erlaubt sind „shuffle“ und „bootstrap“."
        ) from None
    if isinstance(n_simulations, bool) or not isinstance(n_simulations, int) or n_simulations < 1:
        raise MonteCarloError("Die Anzahl der Simulationen muss eine ganze Zahl ≥ 1 sein.")
    r = np.asarray(returns, dtype=np.float64).ravel()
    if r.size == 0:
        raise MonteCarloError("Ohne abgeschlossene Trades ist keine Monte-Carlo-Analyse möglich.")
    if not np.all(np.isfinite(r)):
        raise MonteCarloError("Die Trade-Renditen enthalten ungültige Werte (NaN oder ∞).")

    rng = np.random.default_rng(seed)
    n = r.size
    if mc_method is MonteCarloMethod.SHUFFLE:
        samples = rng.permuted(np.broadcast_to(r, (n_simulations, n)), axis=1)
    else:
        samples = r[rng.integers(0, n, size=(n_simulations, n))]

    final, mdd = _paths(samples, initial_capital)
    orig_final, orig_mdd = _paths(r[np.newaxis, :], initial_capital)
    return MonteCarloResult(
        method=mc_method,
        n_simulations=n_simulations,
        n_trades=n,
        seed=seed,
        initial_capital=float(initial_capital),
        trade_returns=r.copy(),
        final_equity=final,
        max_drawdown_pct=mdd,
        original_final_equity=float(orig_final[0]),
        original_max_drawdown_pct=float(orig_mdd[0]),
    )


def monte_carlo_for_result(
    result: BacktestResult,
    *,
    n_simulations: int = DEFAULT_SIMULATIONS,
    method: MonteCarloMethod | str = MonteCarloMethod.SHUFFLE,
    seed: int = DEFAULT_SEED,
) -> MonteCarloResult:
    """Monte-Carlo-Analyse für ein Ergebnis von :func:`.engine.run_backtest`.

    Args:
        result: Backtest-Ergebnis (Trades und Startkapital).
        n_simulations: Anzahl der Simulationen (Standard 1.000).
        method: ``shuffle`` oder ``bootstrap``.
        seed: Seed des Zufallsgenerators.

    Returns:
        Ergebnis der Simulation.
    """
    returns = trade_returns(result.trades, result.initial_capital)
    return monte_carlo(
        returns,
        result.initial_capital,
        n_simulations=n_simulations,
        method=method,
        seed=seed,
    )


def _check_capital(initial_capital: float) -> None:
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise MonteCarloError("Das Startkapital muss größer als null sein.")


def _paths(samples: FloatArray, initial_capital: float) -> tuple[FloatArray, FloatArray]:
    """Endkapital und Max Drawdown (Prozent) je Zeile von ``samples``."""
    growth = np.cumprod(np.maximum(1.0 + samples, 0.0), axis=1)
    start = np.ones((samples.shape[0], 1))
    equity = initial_capital * np.concatenate([start, growth], axis=1)
    peak = np.maximum.accumulate(equity, axis=1)
    drawdown = equity / peak - 1.0
    return equity[:, -1].copy(), drawdown.min(axis=1) * 100.0
