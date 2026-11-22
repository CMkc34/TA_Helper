"""R-02: Wochen-Trendfolge – Ausbruch mit Marktfilter und ATR-Trailing-Stop.

Kandidat aus Phase 5 (Strategie-Entwicklung), geprüft mit dem Forschungsrahmen aus R-01
(``research/run_study.py``). Bericht und Bewertung: ``research/kandidaten/R-02_trendfolge_1w.md``.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from trading_helfer.core.types import OrderType, Timeframe
from trading_helfer.indicators.trend import sma
from trading_helfer.indicators.volatility import atr, donchian
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    Param,
    Params,
    PlotSpec,
    Sizing,
    Strategy,
)


class TrendBreakoutMarketFilter(Strategy):
    """Kauft Ausbrüche über das Donchian-Hoch, aber nur bei Markt über seiner SMA.

    Einstieg: Close über dem höchsten Hoch der vorherigen ``period`` Kerzen und Close
    über der eigenen SMA ``trend``, während SPY über seiner SMA ``market_sma`` liegt.
    Bei mehr Signalen als freien Plätzen gewinnt die höhere Rendite über
    ``momentum`` Kerzen. Ausstieg nur über einen Trailing Stop im Abstand von
    ``atr_mult`` ATR(14), der auf der Einstiegskerze gesetzt und danach von der Engine
    nachgezogen wird.
    """

    id = "trend_breakout_mf"
    name = "Trendfolge-Ausbruch mit Marktfilter"
    version = "1.0.0"
    description = (
        "Trendfolge für Wochenkerzen (R-02): Kauf, wenn der Close über das höchste Hoch "
        "der vorherigen 20 Kerzen steigt und über der eigenen SMA 40 liegt, aber nur, "
        "solange SPY über seiner SMA 40W schließt (Marktfilter). Rangfolge im Portfolio "
        "nach 26-Wochen-Rendite, je Position 10 % des Kapitals (max. 10 Positionen). "
        "Ausstieg über einen Trailing Stop im Abstand von 3 × ATR(14). Nur Long."
    )
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1, Timeframe.D3]
    market_symbol: ClassVar[str | None] = "SPY"
    # Nur die drei Kernparameter haben eine Schrittweite: Robustheit und Walk-Forward
    # prüfen damit 3 × 3 × 3 = 27 Nachbarn statt 729.
    params: ClassVar[dict[str, Param]] = {
        "period": Param(20, 5, 60, 5, "Kerzen für das Ausbruchshoch (Donchian-Kanal)"),
        "trend": Param(40, 10, 100, None, "Periode der Trend-SMA des Werts"),
        "market_sma": Param(40, 10, 80, 10, "Periode der SMA des Markts (SPY)"),
        "atr_mult": Param(3.0, 1.5, 5.0, 0.5, "Abstand des Trailing Stops in ATR(14)"),
        "momentum": Param(26, 4, 52, None, "Kerzen für die Rendite zur Rangfolge"),
        "position_pct": Param(10.0, 5.0, 100.0, None, "Anteil am Kapital je Position in %"),
    }

    def __init__(self) -> None:
        """Legt die Strategie mit Standardparametern an."""
        self._atr_mult = 3.0
        self._position_pct = 10.0

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """Donchian-Hoch der Vorkerzen, Trend-SMA, Rendite zur Rangfolge und ATR(14)."""
        # Nur Zahlen merken: die Engine kopiert Strategien, Params lassen sich nicht kopieren.
        self._atr_mult = float(p.atr_mult)
        self._position_pct = float(p.position_pct)
        close = bars["close"]
        return pd.DataFrame(
            {
                "dc_upper": donchian(bars, p.period)["upper"].shift(1),
                "trend": sma(bars, p.trend),
                "momentum": close / close.shift(p.momentum) - 1.0,
                "atr": atr(bars, 14),
            },
            index=bars.index,
        )

    def market_indicators(self, market: pd.DataFrame, p: Params) -> pd.DataFrame:
        """SMA des Markts; erscheint als ``market_sma``."""
        return pd.DataFrame({"sma": sma(market, p.market_sma)}, index=market.index)

    def _market_ok(self, ctx: BarContext) -> bool:
        if "market_sma" not in ctx.ind:
            return True  # ohne Marktkurse (z. B. Probelauf im Validator) kein Filter
        return ctx.value("market_close") > ctx.value("market_sma")

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Einstieg beim Ausbruch im Aufwärtsmarkt, Trailing Stop nach der Ausführung."""
        position = ctx.position
        if position is None:
            if not ctx.is_member or not self._market_ok(ctx):
                return []
            close = ctx.value("close")
            if close > ctx.value("dc_upper") and close > ctx.value("trend"):
                return [
                    ctx.buy(
                        sizing=Sizing.percent_equity(self._position_pct),
                        strength=ctx.value("momentum"),
                        tag="Ausbruch",
                    )
                ]
            return []
        if position.entry_date == ctx.date:
            return ctx.exit(
                type=OrderType.TRAILING_STOP,
                trail_atr=self._atr_mult,
                tag="ATR-Trailing-Stop",
            )
        return []

    def plot(self) -> list[PlotSpec]:
        """Donchian-Hoch und Trend-SMA im Kurs-Chart, ATR in eigenem Panel."""
        return [
            PlotSpec("dc_upper", label="Donchian oben", color="#2ca02c"),
            PlotSpec("trend", label="Trend-SMA", color="#1f77b4"),
            PlotSpec("atr", pane="separate", label="ATR(14)", color="#7f7f7f"),
        ]
