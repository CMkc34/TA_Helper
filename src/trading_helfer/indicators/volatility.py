"""Volatilitäts-Indikatoren: ATR, Bollinger Bänder, Keltner-, Donchian-Kanäle, HV.

Die Berechnungen folgen den Konventionen von TA-Lib (Wilder-ATR, Bollinger mit
Standardabweichung der Grundgesamtheit), damit die Werte mit den Referenzwerten in
``tests/fixtures/indicators/`` übereinstimmen. Die Anlaufphase ist ``NaN``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from trading_helfer.indicators.momentum import _rolling_max, _rolling_min
from trading_helfer.indicators.registry import IndicatorError, indicator
from trading_helfer.indicators.trend import (
    _check_period,
    _column,
    _ema_array,
    _series,
    atr_wilder,
)

__all__ = ["atr", "bollinger", "donchian", "historical_volatility", "keltner"]


def _check_positive(value: float, name: str) -> None:
    if not value > 0:
        raise IndicatorError(f"Parameter {name!r} muss größer als 0 sein, erhalten: {value!r}.")


@indicator("atr", params={"period": 14}, pane="separate", outputs=["atr"])
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series[float]:
    """Average True Range nach Wilder (siehe :func:`trend.atr_wilder`).

    Args:
        df: Kursreihe mit ``high``, ``low``, ``close``.
        period: Glättungsperiode.

    Returns:
        Serie ``atr``; die ersten ``period`` Werte sind NaN.
    """
    return atr_wilder(df, period)


@indicator(
    "bbands",
    params={"period": 20, "num_std": 2.0, "source": "close"},
    pane="price",
    outputs=["upper", "middle", "lower"],
)
def bollinger(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0, source: str = "close"
) -> pd.DataFrame:
    """Bollinger Bänder: ``SMA ± num_std · Standardabweichung`` (Grundgesamtheit, wie TA-Lib).

    Args:
        df: Kursreihe.
        period: Anzahl Kerzen für Durchschnitt und Standardabweichung.
        num_std: Bandabstand in Standardabweichungen.
        source: Spalte, auf die der Indikator angewendet wird.

    Returns:
        DataFrame mit ``upper``, ``middle``, ``lower``; die ersten ``period − 1``
        Werte sind NaN.
    """
    _check_period(period)
    _check_positive(num_std, "num_std")
    x = _column(df, source)
    middle = np.full(len(x), np.nan)
    std = np.full(len(x), np.nan)
    if len(x) >= period:
        windows = sliding_window_view(x, period)
        middle[period - 1 :] = windows.mean(axis=1)
        std[period - 1 :] = windows.std(axis=1, ddof=0)
    return pd.DataFrame(
        {"upper": middle + num_std * std, "middle": middle, "lower": middle - num_std * std},
        index=df.index,
    )


@indicator(
    "keltner",
    params={"period": 20, "multiplier": 2.0, "atr_period": 10},
    pane="price",
    outputs=["upper", "middle", "lower"],
)
def keltner(
    df: pd.DataFrame, period: int = 20, multiplier: float = 2.0, atr_period: int = 10
) -> pd.DataFrame:
    """Keltner-Kanäle: ``EMA(close, period) ± multiplier · ATR(atr_period)``.

    Mittellinie ist die EMA des Schlusskurses (SMA-Startwert), der Bandabstand
    die Wilder-ATR. Standard wie bei StockCharts: EMA 20, ATR 10, Faktor 2.

    Args:
        df: Kursreihe mit ``high``, ``low``, ``close``.
        period: Periode der EMA.
        multiplier: ATR-Vielfaches für den Bandabstand.
        atr_period: Periode der ATR.

    Returns:
        DataFrame mit ``upper``, ``middle`` (ab Index ``period − 1``) und ``lower``;
        die Bänder beginnen ab ``max(period − 1, atr_period)``.
    """
    _check_period(period)
    _check_period(atr_period, "atr_period")
    _check_positive(multiplier, "multiplier")
    middle = _ema_array(_column(df, "close"), period)
    width = multiplier * atr_wilder(df, atr_period).to_numpy()
    return pd.DataFrame(
        {"upper": middle + width, "middle": middle, "lower": middle - width}, index=df.index
    )


@indicator(
    "donchian",
    params={"period": 20},
    pane="price",
    outputs=["upper", "middle", "lower"],
)
def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian-Kanäle: höchstes Hoch und tiefstes Tief der letzten ``period`` Kerzen.

    Die aktuelle Kerze zählt mit; für Ausbruchssignale wird der Kanal der Vorkerze
    verglichen (``upper.shift(1)``).

    Args:
        df: Kursreihe mit ``high`` und ``low``.
        period: Anzahl Kerzen.

    Returns:
        DataFrame mit ``upper``, ``middle`` (Mittelwert) und ``lower``; die ersten
        ``period − 1`` Werte sind NaN.
    """
    _check_period(period)
    upper = _rolling_max(_column(df, "high"), period)
    lower = _rolling_min(_column(df, "low"), period)
    return pd.DataFrame(
        {"upper": upper, "middle": (upper + lower) / 2.0, "lower": lower}, index=df.index
    )


@indicator(
    "hv",
    params={"period": 20, "annualization": 252, "source": "close"},
    pane="separate",
    outputs=["hv"],
)
def historical_volatility(
    df: pd.DataFrame, period: int = 20, annualization: int = 252, source: str = "close"
) -> pd.Series[float]:
    """Historische Volatilität: annualisierte Standardabweichung der Log-Renditen.

    ``hv = std(ln(x_t / x_{t−1}), period Renditen, Stichprobe) · √annualization``,
    als Dezimalzahl (``0.25`` = 25 % p. a.). Für Wochenkerzen passt
    ``annualization = 52``.

    Args:
        df: Kursreihe.
        period: Anzahl Renditen (mindestens 2).
        annualization: Kerzen pro Jahr.
        source: Spalte, auf die der Indikator angewendet wird.

    Returns:
        Serie ``hv``; die ersten ``period`` Werte sind NaN.
    """
    _check_period(period)
    if period < 2:
        raise IndicatorError("Parameter 'period' muss mindestens 2 sein.")
    _check_period(annualization, "annualization")
    x = _column(df, source)
    out = np.full(len(x), np.nan)
    if len(x) > period:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.log(x[1:] / x[:-1])
        out[period:] = sliding_window_view(log_ret, period).std(axis=1, ddof=1)
        out[period:] *= math.sqrt(annualization)
    return _series(out, df, "hv")
