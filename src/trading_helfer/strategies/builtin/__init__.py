"""Mitgelieferte Beispielstrategien.

- :class:`SmaCross` (``trend_sma_cross``): SMA 10/40-Kreuzung auf 1W.
- :class:`DonchianBreakout` (``trend_donchian_breakout``): Ausbruch über das
  20-Kerzen-Hoch mit ATR-Trailing-Stop auf 1W.
- :class:`Rsi2MeanReversion` (``mr_rsi2_trend``): RSI(2) < 10 über der SMA 200, Ausstieg
  bei RSI > 70, auf 3D.
"""

from trading_helfer.strategies.builtin.donchian_breakout import DonchianBreakout
from trading_helfer.strategies.builtin.rsi2_mean_reversion import Rsi2MeanReversion
from trading_helfer.strategies.builtin.sma_cross import SmaCross

__all__ = ["BUILTIN_STRATEGIES", "DonchianBreakout", "Rsi2MeanReversion", "SmaCross"]

BUILTIN_STRATEGIES = (SmaCross, DonchianBreakout, Rsi2MeanReversion)
"""Alle mitgelieferten Strategien in Ladereihenfolge."""
