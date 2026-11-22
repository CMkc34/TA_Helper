"""Kandidat R-04: Pullback im Aufwärtstrend auf 3-Tages-Kerzen.

Idee: Werte in einem langfristigen Aufwärtstrend (Close über der SMA 200 Tage, auf
3D umgerechnet etwa 67 Kerzen) kehren nach einem kurzen Rücksetzer oft zum Trend
zurück. Gekauft wird der Rücksetzer, verkauft wird die Erholung. Ein Zeit-Stop und ein
fester ATR-Stop begrenzen Haltedauer und Verlust.

Regeln (Entscheidung auf dem Close der Kerze t, Ausführung am Open von t+1):

- **Einstieg**: Mitglied des Universums, Close über ``trend_sma`` und Rückgang:
  ``trigger="rsi"`` → RSI(2) unter ``rsi_entry``; ``trigger="closes"`` → mindestens
  ``down_closes`` Closes in Folge tiefer als der jeweils vorherige.
- **Positionsgröße**: 10 % des Kapitals (passt zu höchstens 10 Positionen). Gibt es mehr
  Signale als freie Plätze, gewinnen die Werte mit dem tiefsten RSI(2).
- **Stop**: Verkaufs-Stop ``atr_mult`` × ATR(14) unter dem Einstiegskurs, gesetzt auf der
  Einstiegskerze und danach unverändert.
- **Erholung**: Verkauf, sobald der RSI(2) über ``rsi_exit`` steigt.
- **Zeit-Stop**: Verkauf nach ``max_bars`` Kerzen, wenn bis dahin weder Erholung noch Stop.

Für die Studie (``research/run_study.py``) variiert die Robustheitsprüfung nur die
Parameter mit Schrittweite (``trend_period``, ``rsi_entry``, ``atr_mult``); ``rsi_exit``
und ``max_bars`` sind feste Regeln aus dem Issue und lassen sich mit ``--param`` ändern.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pandas as pd

from trading_helfer.core.types import OrderType, Timeframe
from trading_helfer.indicators.momentum import rsi
from trading_helfer.indicators.trend import sma
from trading_helfer.indicators.volatility import atr
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    Params,
    PlotSpec,
    Sizing,
    Strategy,
)

POSITION_PERCENT = 10.0
"""Anteil des Kapitals je Position in Prozent (10 Positionen à 10 %)."""

ATR_PERIOD = 14
"""Periode der ATR für den Stop."""


def down_close_streak(close: pd.Series) -> pd.Series:
    """Anzahl der Closes in Folge, die jeweils unter dem vorherigen Close liegen.

    Args:
        close: Schlusskurse.

    Returns:
        Serie gleicher Länge; 0, wenn der Close nicht unter dem vorherigen liegt.
    """
    values = close.to_numpy(dtype=float)
    streak = np.zeros(len(values), dtype=float)
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            streak[i] = streak[i - 1] + 1.0
    return pd.Series(streak, index=close.index)


def bars_held(ctx: BarContext) -> int:
    """Kerzen seit der Einstiegskerze (0 auf der Einstiegskerze selbst).

    Args:
        ctx: Kontext mit offener Position.

    Returns:
        Anzahl der Kerzen nach der Einstiegskerze; 0 ohne Position.
    """
    if ctx.position is None:
        return 0
    dates = ctx.bars.dates
    entry = np.datetime64(ctx.position.entry_date, "D").astype(dates.dtype)
    first = int(np.searchsorted(dates, entry, side="left"))
    return max(0, len(dates) - 1 - first)


class PullbackTrend(Strategy):
    """Kauft kurze Rücksetzer im Aufwärtstrend, verkauft die Erholung (R-04)."""

    id = "pullback_trend"
    name = "Pullback im Trend"
    version = "1.0.0"
    description = (
        "Pullback im Aufwärtstrend: Kauf, wenn der Close über der SMA 67 (≈ SMA 200 Tage "
        "auf 3-Tages-Kerzen) liegt und der RSI(2) unter 10 fällt (alternativ 3 tiefere "
        "Closes in Folge). Verkauf bei Erholung (RSI(2) über 70) oder nach 10 Kerzen; "
        "Stop 2 × ATR(14) unter dem Einstieg. 10 % je Position. Für 3D gedacht. Nur Long."
    )
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D3, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        "trend_period": Param(67, 20, 150, 10, "Periode der Trend-SMA in Kerzen"),
        "trigger": Param(
            "rsi", choices=("rsi", "closes"), description="Rückgang: RSI(2) oder tiefere Closes"
        ),
        "rsi_entry": Param(10.0, 2.0, 40.0, 5.0, "Kauf, wenn der RSI(2) darunter liegt"),
        "down_closes": Param(3, 2, 6, None, "Tiefere Closes in Folge (trigger=closes)"),
        "rsi_exit": Param(70.0, 50.0, 95.0, None, "Erholung: Verkauf, wenn der RSI(2) darüber"),
        "max_bars": Param(10, 2, 30, None, "Zeit-Stop: Verkauf nach so vielen Kerzen"),
        "atr_mult": Param(2.0, 1.0, 5.0, 0.5, "Abstand des Stops in ATR(14) unter dem Einstieg"),
    }

    def __init__(self) -> None:
        """Legt die Strategie mit Standardparametern an."""
        self._p = self.make_params()

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """Trend-SMA, RSI(2), Serie tieferer Closes, ATR(14) und Einstiegssignal."""
        self._p = p
        trend = sma(bars, p.trend_period)
        rsi2 = rsi(bars, 2)
        streak = down_close_streak(bars["close"])
        pulled_back = rsi2 < p.rsi_entry if p.trigger == "rsi" else streak >= p.down_closes
        setup = (bars["close"] > trend) & pulled_back
        return pd.DataFrame(
            {
                "trend_sma": trend,
                "rsi2": rsi2,
                "down_streak": streak,
                "atr": atr(bars, ATR_PERIOD),
                "setup": setup.astype(float),
                "strength": 100.0 - rsi2,
                "entry_level": float(p.rsi_entry),
                "exit_level": float(p.rsi_exit),
            },
            index=bars.index,
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Einstieg beim Rücksetzer; danach Stop, Erholung oder Zeit-Stop."""
        position = ctx.position
        if position is None:
            if ctx.is_member and ctx.value("setup") > 0:
                return [
                    ctx.buy(
                        sizing=Sizing.percent_equity(POSITION_PERCENT),
                        tag="Rücksetzer im Aufwärtstrend",
                        strength=ctx.value("strength"),
                    )
                ]
            return []
        orders: list[Order] = []
        if position.entry_date == ctx.date:
            distance = float(self._p.atr_mult) * ctx.value("atr")
            stop = position.entry_price - distance
            if math.isfinite(stop) and stop > 0:
                orders += ctx.exit(type=OrderType.STOP, stop_price=stop, tag="ATR-Stop")
        if ctx.value("rsi2") > float(self._p.rsi_exit):
            orders += ctx.exit(tag="Erholung (RSI(2))")
        elif bars_held(ctx) >= int(self._p.max_bars):
            orders += ctx.exit(tag="Zeit-Stop")
        return orders

    def plot(self) -> list[PlotSpec]:
        """Trend-SMA im Kurs-Chart, RSI(2) mit Schwellen in eigenem Panel."""
        return [
            PlotSpec("trend_sma", label="SMA Trend", color="#ff7f0e"),
            PlotSpec("rsi2", pane="separate", label="RSI(2)", color="#9467bd"),
            PlotSpec("entry_level", pane="separate", label="Kaufschwelle", color="#2ca02c"),
            PlotSpec("exit_level", pane="separate", label="Erholung", color="#d62728"),
        ]
