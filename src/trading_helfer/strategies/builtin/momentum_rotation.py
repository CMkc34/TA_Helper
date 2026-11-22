"""Kandidat R-03: Momentum-Rotation auf Wochenkerzen.

Jede Woche werden alle Indexmitglieder nach ihrer Rendite über ``lookback`` Kerzen ohne
die jüngsten ``skip`` Kerzen gereiht (RS-Rang, 26 Wochen ohne letzte Woche). Gekauft
werden die besten ``top_n`` Werte, solange der Markt (SPY) über seiner SMA 40W liegt.
Verkauft wird, sobald der RS-Rang eines gehaltenen Werts unter ``exit_rank`` fällt oder
der Wert nicht mehr im Index ist.

Die Rangfolge rechnet die Portfolio-Engine über ``rank_columns`` (Querschnitt je Termin,
nur Indexmitglieder). Gibt es mehr Kaufsignale als freie Plätze, entscheidet die
Momentum-Rendite (Spalte ``strength``).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from trading_helfer.core.types import Timeframe
from trading_helfer.indicators.trend import sma
from trading_helfer.strategies.base import (
    POSITION_SUFFIX,
    RANK_SUFFIX,
    BarContext,
    Order,
    Param,
    Params,
    PlotSpec,
    Sizing,
    Strategy,
)

__all__ = ["MARKET_ABOVE_COLUMN", "MOMENTUM_COLUMN", "MomentumRotation"]

MOMENTUM_COLUMN = "momentum"
"""Rendite über ``lookback`` Kerzen ohne die jüngsten ``skip`` Kerzen (Bruch)."""

MARKET_ABOVE_COLUMN = "market_above_sma"
"""1, wenn der Markt-Close über seiner SMA liegt, sonst 0 (von der Engine angehängt)."""


class MomentumRotation(Strategy):
    """Wöchentliche Rotation in die stärksten Werte des Index.

    Ablauf je Kerze und Wert:

    - Ohne Position: Kauf, wenn der Wert unter den besten ``top_n`` steht, der RS-Rang
      mindestens ``exit_rank`` ist (sonst würde sofort wieder verkauft) und der
      Marktfilter erfüllt ist. Größe: ``100 / top_n`` Prozent des Kapitals.
    - Mit Position: Verkauf, wenn der RS-Rang unter ``exit_rank`` fällt oder fehlt
      (Wert nicht mehr im Index oder ohne Kurs). Der Marktfilter schließt keine
      Positionen, er sperrt nur neue Käufe; mit ``market_exit`` wird dagegen alles
      verkauft, sobald der Markt unter seiner SMA schließt.

    Fehlen die Marktkurse (z. B. ohne Datenbank), gilt der Marktfilter als erfüllt.
    """

    id = "momentum_rotation"
    name = "Momentum-Rotation"
    version = "1.1.0"
    description = (
        "Rotation: Jede Woche Rangliste aller Indexmitglieder nach der 26-Wochen-Rendite "
        "ohne die letzte Woche (RS-Rang). Kauf der 10 stärksten Werte, solange SPY über "
        "seiner SMA 40W liegt; Verkauf, wenn der RS-Rang unter 30 fällt. Für den "
        "Portfolio-Backtest über einen Index mit 10 Positionen gedacht (1W). Nur Long."
    )
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        "lookback": Param(26, 8, 52, 4, "Kerzen für die Momentum-Rendite"),
        "skip": Param(1, 0, 4, None, "Jüngste Kerzen, die nicht zur Rendite zählen"),
        "top_n": Param(10, 1, 50, None, "Kauf nur unter den besten N Werten"),
        "exit_rank": Param(30.0, 0.0, 90.0, 10.0, "Verkauf, wenn der RS-Rang darunter fällt"),
        "market_sma": Param(40, 10, 80, 5, "Periode der Markt-SMA (Marktfilter)"),
        "market_filter": Param(True, description="Käufe nur, wenn der Markt über der SMA liegt"),
        "market_exit": Param(
            False, description="Alle Positionen verkaufen, wenn der Markt unter die SMA fällt"
        ),
    }
    rank_columns: ClassVar[tuple[str, ...]] = (MOMENTUM_COLUMN,)
    market_symbol: ClassVar[str | None] = "SPY"

    def __init__(self) -> None:
        """Legt die Strategie mit Standardparametern an."""
        # Als dict, damit sich die Strategie kopieren und pickeln lässt.
        self._p: dict[str, Any] = self.make_params().to_dict()

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """Momentum-Rendite je Kerze; dient zugleich als Signalstärke."""
        self._p = p.to_dict()
        close = bars["close"].astype("float64")
        recent = close.shift(int(p.skip))
        past = close.shift(int(p.skip) + int(p.lookback))
        momentum = (recent / past.where(past > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame({MOMENTUM_COLUMN: momentum, "strength": momentum}, index=bars.index)

    def market_indicators(self, market: pd.DataFrame, p: Params) -> pd.DataFrame:
        """Marktfilter: 1, wenn der Markt-Close über seiner SMA liegt, sonst 0."""
        average = sma(market, int(p.market_sma))
        close = market["close"].astype("float64").to_numpy()
        mean = average.to_numpy(dtype="float64")
        above = np.where(np.isnan(mean), np.nan, (close > mean).astype("float64"))
        return pd.DataFrame({"above_sma": above}, index=market.index)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Kauf der stärksten Werte, Verkauf bei schwachem RS-Rang."""
        p = self._p
        rank = self._number(ctx, MOMENTUM_COLUMN + RANK_SUFFIX)
        if ctx.position is not None:
            if p["market_exit"] and self._market_down(ctx):
                return ctx.exit(tag="Markt unter SMA")
            if math.isnan(rank) or rank < float(p["exit_rank"]):
                return ctx.exit(tag="RS-Rang zu schwach")
            return []
        place = self._number(ctx, MOMENTUM_COLUMN + POSITION_SUFFIX)
        if math.isnan(place) or place > int(p["top_n"]) or rank < float(p["exit_rank"]):
            return []
        if p["market_filter"] and not self._market_ok(ctx):
            return []
        momentum = self._number(ctx, MOMENTUM_COLUMN)
        return [
            ctx.buy(
                sizing=Sizing.percent_equity(100.0 / int(p["top_n"])),
                tag=f"Momentum Platz {int(place)}",
                strength=momentum,
            )
        ]

    def plot(self) -> list[PlotSpec]:
        """Momentum-Rendite in eigenem Panel (der RS-Rang entsteht erst im Portfolio)."""
        return [PlotSpec(MOMENTUM_COLUMN, pane="separate", label="Momentum", color="#1f77b4")]

    @staticmethod
    def _number(ctx: BarContext, column: str) -> float:
        if column not in ctx.ind:
            return math.nan
        value = ctx.ind[column]
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    @staticmethod
    def _market_down(ctx: BarContext) -> bool:
        """``True`` nur, wenn der Markt sicher unter der SMA liegt (nicht im Anlauf)."""
        if MARKET_ABOVE_COLUMN not in ctx.ind:
            return False
        value = MomentumRotation._number(ctx, MARKET_ABOVE_COLUMN)
        return not math.isnan(value) and value < 0.5

    @staticmethod
    def _market_ok(ctx: BarContext) -> bool:
        if MARKET_ABOVE_COLUMN not in ctx.ind:
            return True  # keine Marktkurse: Filter nicht anwendbar
        value = MomentumRotation._number(ctx, MARKET_ABOVE_COLUMN)
        return not math.isnan(value) and value > 0.5
