"""Momentum-Indikatoren: RSI, Stochastik, ROC, Williams %R, CCI.

Die Berechnungen folgen den Konventionen von TA-Lib (Wilder-Glättung beim RSI,
SMA-Glättung bei der Stochastik, mittlere absolute Abweichung beim CCI), damit
die Werte mit den Referenzwerten in ``tests/fixtures/indicators/`` übereinstimmen.
Ist die Spanne bzw. Abweichung null (flache Kurse), liefern Stochastik, Williams %R
und CCI wie TA-Lib den Wert ``0``, der RSI den neutralen Wert ``50``. Die Anlaufphase ist ``NaN``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from trading_helfer.indicators.registry import indicator
from trading_helfer.indicators.trend import (
    FloatArray,
    _check_period,
    _column,
    _series,
    _sma_array,
)

__all__ = ["cci", "roc", "rsi", "stochastic", "williams_r"]


def _rolling_max(x: FloatArray, period: int) -> FloatArray:
    out = np.full(len(x), np.nan)
    if len(x) >= period:
        out[period - 1 :] = sliding_window_view(x, period).max(axis=1)
    return out


def _rolling_min(x: FloatArray, period: int) -> FloatArray:
    out = np.full(len(x), np.nan)
    if len(x) >= period:
        out[period - 1 :] = sliding_window_view(x, period).min(axis=1)
    return out


def _safe_ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """``numerator / denominator``, ``0`` bei Nenner null, NaN bleibt NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denominator == 0, 0.0, numerator / denominator)
    out[np.isnan(denominator) | np.isnan(numerator)] = np.nan
    return np.asarray(out, dtype=np.float64)


@indicator("rsi", params={"period": 14, "source": "close"}, pane="separate", outputs=["rsi"])
def rsi(df: pd.DataFrame, period: int = 14, source: str = "close") -> pd.Series[float]:
    """Relative Strength Index nach Wilder.

    Startwerte sind die mittleren Gewinne und Verluste der ersten ``period``
    Veränderungen, danach ``avg_t = (avg_{t−1} · (period − 1) + x_t) / period``.

    Args:
        df: Kursreihe.
        period: Glättungsperiode.
        source: Spalte, auf die der Indikator angewendet wird.

    Returns:
        Serie ``rsi`` im Bereich 0 … 100 (``50`` ohne Kursbewegung); die ersten
        ``period`` Werte sind NaN.
    """
    _check_period(period)
    x = _column(df, source)
    out = np.full(len(x), np.nan)
    if len(x) > period:
        diff = np.diff(x)
        gains = np.where(diff > 0, diff, 0.0)
        losses = np.where(diff < 0, -diff, 0.0)
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))
        for i in range(period, len(x)):
            if i > period:
                avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
            total = avg_gain + avg_loss
            out[i] = 50.0 if total == 0 else 100.0 * avg_gain / total
    return _series(out, df, "rsi")


@indicator(
    "stoch",
    params={"k_period": 14, "k_smooth": 3, "d_period": 3},
    pane="separate",
    outputs=["k", "d"],
)
def stochastic(
    df: pd.DataFrame, k_period: int = 14, k_smooth: int = 3, d_period: int = 3
) -> pd.DataFrame:
    """Langsame Stochastik (Slow Stochastic, wie ``talib.STOCH`` mit SMA).

    ``fast_k = 100 · (close − tiefstes Tief) / (höchstes Hoch − tiefstes Tief)`` über
    ``k_period`` Kerzen, ``k = SMA(fast_k, k_smooth)``, ``d = SMA(k, d_period)``.

    Args:
        df: Kursreihe mit ``high``, ``low``, ``close``.
        k_period: Betrachtungszeitraum für Hoch und Tief.
        k_smooth: Glättung der %K-Linie.
        d_period: Periode der Signallinie %D.

    Returns:
        DataFrame mit ``k`` (ab Index ``k_period + k_smooth − 2``) und ``d``
        (ab Index ``k_period + k_smooth + d_period − 3``).
    """
    for value, name in ((k_period, "k_period"), (k_smooth, "k_smooth"), (d_period, "d_period")):
        _check_period(value, name)
    high, low, close = _column(df, "high"), _column(df, "low"), _column(df, "close")
    hh, ll = _rolling_max(high, k_period), _rolling_min(low, k_period)
    fast_k = 100.0 * _safe_ratio(close - ll, hh - ll)
    k = _shifted_sma(fast_k, k_smooth, k_period - 1)
    d = _shifted_sma(k, d_period, k_period + k_smooth - 2)
    return pd.DataFrame({"k": k, "d": d}, index=df.index)


def _shifted_sma(x: FloatArray, period: int, start: int) -> FloatArray:
    """SMA über ``x[start:]`` (davor NaN), Ergebnis wieder in voller Länge."""
    out = np.full(len(x), np.nan)
    if start < len(x):
        out[start:] = _sma_array(x[start:], period)
    return out


@indicator("roc", params={"period": 10, "source": "close"}, pane="separate", outputs=["roc"])
def roc(df: pd.DataFrame, period: int = 10, source: str = "close") -> pd.Series[float]:
    """Rate of Change in Prozent: ``100 · (x_t / x_{t−period} − 1)``.

    Args:
        df: Kursreihe.
        period: Abstand in Kerzen.
        source: Spalte, auf die der Indikator angewendet wird.

    Returns:
        Serie ``roc``; die ersten ``period`` Werte sind NaN.
    """
    _check_period(period)
    x = _column(df, source)
    out = np.full(len(x), np.nan)
    if len(x) > period:
        prev = x[:-period]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[period:] = np.where(prev == 0, 0.0, 100.0 * (x[period:] / prev - 1.0))
    return _series(out, df, "roc")


@indicator("willr", params={"period": 14}, pane="separate", outputs=["willr"])
def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series[float]:
    """Williams %R: ``−100 · (höchstes Hoch − close) / (höchstes Hoch − tiefstes Tief)``.

    Args:
        df: Kursreihe mit ``high``, ``low``, ``close``.
        period: Betrachtungszeitraum.

    Returns:
        Serie ``willr`` im Bereich −100 … 0; die ersten ``period − 1`` Werte sind NaN.
    """
    _check_period(period)
    high, low, close = _column(df, "high"), _column(df, "low"), _column(df, "close")
    hh, ll = _rolling_max(high, period), _rolling_min(low, period)
    return _series(-100.0 * _safe_ratio(hh - close, hh - ll), df, "willr")


@indicator("cci", params={"period": 20}, pane="separate", outputs=["cci"])
def cci(df: pd.DataFrame, period: int = 20) -> pd.Series[float]:
    """Commodity Channel Index nach Lambert.

    Typischer Preis ``tp = (high + low + close) / 3``,
    ``cci = (tp − SMA(tp)) / (0,015 · mittlere absolute Abweichung von SMA(tp))``.

    Args:
        df: Kursreihe mit ``high``, ``low``, ``close``.
        period: Betrachtungszeitraum.

    Returns:
        Serie ``cci``; die ersten ``period − 1`` Werte sind NaN.
    """
    _check_period(period)
    tp = (_column(df, "high") + _column(df, "low") + _column(df, "close")) / 3.0
    out = np.full(len(tp), np.nan)
    if len(tp) >= period:
        windows = sliding_window_view(tp, period)
        mean = windows.mean(axis=1)
        mean_dev = np.abs(windows - mean[:, None]).mean(axis=1)
        out[period - 1 :] = _safe_ratio(tp[period - 1 :] - mean, 0.015 * mean_dev)
    return _series(out, df, "cci")
