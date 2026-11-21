"""Volumen-Indikatoren: OBV, gleitender VWAP, Chaikin Money Flow, Volumen-SMA.

OBV folgt der Konvention von TA-Lib (Startwert ist das Volumen der ersten Kerze),
damit die Werte mit den Referenzwerten in ``tests/fixtures/indicators/`` übereinstimmen.
Die Anlaufphase ist ``NaN``; der Wert bei ``t`` hängt nur von Kerzen bis ``t`` ab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from trading_helfer.indicators.registry import indicator
from trading_helfer.indicators.trend import FloatArray, _check_period, _column, _series

__all__ = ["chaikin_money_flow", "obv", "volume_sma", "vwap"]


def _rolling_sum(x: FloatArray, period: int) -> FloatArray:
    """Gleitende Summe über ``period`` Werte; die ersten ``period − 1`` Werte sind NaN."""
    out = np.full(len(x), np.nan)
    if len(x) >= period:
        out[period - 1 :] = sliding_window_view(x, period).sum(axis=1)
    return out


def _safe_ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """``numerator / denominator``; NaN, wo der Nenner 0 ist."""
    out = np.full(len(numerator), np.nan)
    valid = denominator != 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


@indicator("obv", params={}, pane="separate", outputs=["obv"])
def obv(df: pd.DataFrame) -> pd.Series[float]:
    """On-Balance-Volume: kumuliertes Volumen mit dem Vorzeichen der Kursänderung.

    Bei steigendem Schlusskurs wird das Volumen addiert, bei fallendem abgezogen, bei
    unverändertem bleibt der Wert gleich. Startwert ist das Volumen der ersten Kerze.

    Args:
        df: Kursreihe mit ``close`` und ``volume``.

    Returns:
        Serie ``obv`` ohne Anlaufphase.
    """
    close = _column(df, "close")
    volume = _column(df, "volume")
    signed = np.sign(np.diff(close)) * volume[1:]
    values = np.cumsum(np.concatenate((volume[:1], signed)))
    return _series(values, df, "obv")


@indicator("vwap", params={"period": 20}, pane="price", outputs=["vwap"])
def vwap(df: pd.DataFrame, period: int = 20) -> pd.Series[float]:
    """Gleitender VWAP: volumengewichteter typischer Kurs ``(H + L + C) / 3``.

    Args:
        df: Kursreihe mit ``high``, ``low``, ``close``, ``volume``.
        period: Anzahl Kerzen im Fenster.

    Returns:
        Serie ``vwap``; die ersten ``period − 1`` Werte sind NaN, ebenso Fenster ohne
        Volumen.
    """
    _check_period(period)
    typical = (_column(df, "high") + _column(df, "low") + _column(df, "close")) / 3.0
    volume = _column(df, "volume")
    values = _safe_ratio(_rolling_sum(typical * volume, period), _rolling_sum(volume, period))
    return _series(values, df, "vwap")


@indicator("cmf", params={"period": 20}, pane="separate", outputs=["cmf"])
def chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series[float]:
    """Chaikin Money Flow: Summe des Money-Flow-Volumens geteilt durch die Volumensumme.

    Der Money-Flow-Multiplikator ist ``((C − L) − (H − C)) / (H − L)`` und 0, wenn
    ``H = L``. Das Ergebnis liegt zwischen −1 und 1.

    Args:
        df: Kursreihe mit ``high``, ``low``, ``close``, ``volume``.
        period: Anzahl Kerzen im Fenster.

    Returns:
        Serie ``cmf``; die ersten ``period − 1`` Werte sind NaN, ebenso Fenster ohne
        Volumen.
    """
    _check_period(period)
    high, low, close = _column(df, "high"), _column(df, "low"), _column(df, "close")
    volume = _column(df, "volume")
    multiplier = np.nan_to_num(_safe_ratio((close - low) - (high - close), high - low))
    values = _safe_ratio(_rolling_sum(multiplier * volume, period), _rolling_sum(volume, period))
    return _series(values, df, "cmf")


@indicator("volume_sma", params={"period": 20}, pane="separate", outputs=["volume_sma"])
def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series[float]:
    """Einfacher gleitender Durchschnitt des Volumens.

    Args:
        df: Kursreihe mit ``volume``.
        period: Anzahl Kerzen im Durchschnitt.

    Returns:
        Serie ``volume_sma``; die ersten ``period − 1`` Werte sind NaN.
    """
    _check_period(period)
    values = _rolling_sum(_column(df, "volume"), period) / period
    return _series(values, df, "volume_sma")
