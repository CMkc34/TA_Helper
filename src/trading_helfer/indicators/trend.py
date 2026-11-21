"""Trend-Indikatoren: SMA, EMA, WMA, MACD, ADX/DMI, Ichimoku, Parabolic SAR, Supertrend.

Die Berechnungen folgen den Konventionen von TA-Lib (gleitende Durchschnitte mit
SMA-Startwert, Wilder-Glättung für ADX/DMI und ATR), damit die Werte mit den
Referenzwerten in ``tests/fixtures/indicators/`` übereinstimmen. Alle Funktionen
erwarten bereinigte Kurse und liefern ein Ergebnis mit dem Index der Eingabe;
die Anlaufphase ist ``NaN``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from trading_helfer.indicators.registry import IndicatorError, indicator

__all__ = [
    "adx",
    "atr_wilder",
    "ema",
    "ichimoku",
    "macd",
    "parabolic_sar",
    "sma",
    "supertrend",
    "wma",
]

FloatArray = npt.NDArray[np.float64]


def _column(df: pd.DataFrame, name: str) -> FloatArray:
    """Spalte ``name`` als ``float64``-Array; Fehler, wenn sie fehlt."""
    if name not in df.columns:
        raise IndicatorError(f"Spalte {name!r} fehlt in den Kursdaten.")
    return df[name].to_numpy(dtype=np.float64, copy=True)


def _check_period(period: int, name: str = "period") -> None:
    if not isinstance(period, int | np.integer) or period < 1:
        raise IndicatorError(
            f"Parameter {name!r} muss eine ganze Zahl ≥ 1 sein, erhalten: {period!r}."
        )


def _series(values: FloatArray, df: pd.DataFrame, name: str) -> pd.Series[float]:
    return pd.Series(values, index=df.index, name=name, dtype="float64")


def _sma_array(x: FloatArray, period: int) -> FloatArray:
    """Einfacher gleitender Durchschnitt; die ersten ``period - 1`` Werte sind NaN."""
    out = np.full(len(x), np.nan)
    if len(x) >= period:
        csum = np.cumsum(np.insert(x, 0, 0.0))
        out[period - 1 :] = (csum[period:] - csum[:-period]) / period
    return out


def _ema_array(x: FloatArray, period: int, start: int = 0) -> FloatArray:
    """Exponentieller Durchschnitt ab Index ``start``, Startwert = SMA der ersten Werte.

    Der erste Wert steht bei ``start + period - 1``; davor ist das Ergebnis NaN.
    """
    out = np.full(len(x), np.nan)
    first = start + period - 1
    if first >= len(x):
        return out
    k = 2.0 / (period + 1)
    value = float(np.mean(x[start : first + 1]))
    out[first] = value
    for i in range(first + 1, len(x)):
        value = (x[i] - value) * k + value
        out[i] = value
    return out


def _true_range(high: FloatArray, low: FloatArray, close: FloatArray) -> FloatArray:
    """True Range; der erste Wert ist NaN (kein Vortages-Schluss)."""
    tr = np.full(len(high), np.nan)
    if len(high) > 1:
        prev_close = close[:-1]
        tr[1:] = np.maximum.reduce(
            [high[1:] - low[1:], np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)]
        )
    return tr


@indicator("sma", params={"period": 20, "source": "close"}, pane="price", outputs=["sma"])
def sma(df: pd.DataFrame, period: int = 20, source: str = "close") -> pd.Series[float]:
    """Einfacher gleitender Durchschnitt (Simple Moving Average).

    Args:
        df: Kursreihe.
        period: Anzahl Kerzen.
        source: Spalte, auf die der Durchschnitt angewendet wird.

    Returns:
        Serie ``sma``; die ersten ``period - 1`` Werte sind NaN.
    """
    _check_period(period)
    return _series(_sma_array(_column(df, source), period), df, "sma")


@indicator("ema", params={"period": 20, "source": "close"}, pane="price", outputs=["ema"])
def ema(df: pd.DataFrame, period: int = 20, source: str = "close") -> pd.Series[float]:
    """Exponentieller gleitender Durchschnitt mit Glättungsfaktor ``2 / (period + 1)``.

    Startwert ist der SMA der ersten ``period`` Werte (wie TA-Lib).

    Args:
        df: Kursreihe.
        period: Anzahl Kerzen.
        source: Spalte, auf die der Durchschnitt angewendet wird.

    Returns:
        Serie ``ema``; die ersten ``period - 1`` Werte sind NaN.
    """
    _check_period(period)
    return _series(_ema_array(_column(df, source), period), df, "ema")


@indicator("wma", params={"period": 20, "source": "close"}, pane="price", outputs=["wma"])
def wma(df: pd.DataFrame, period: int = 20, source: str = "close") -> pd.Series[float]:
    """Linear gewichteter gleitender Durchschnitt (jüngste Kerze hat Gewicht ``period``).

    Args:
        df: Kursreihe.
        period: Anzahl Kerzen.
        source: Spalte, auf die der Durchschnitt angewendet wird.

    Returns:
        Serie ``wma``; die ersten ``period - 1`` Werte sind NaN.
    """
    _check_period(period)
    x = _column(df, source)
    out = np.full(len(x), np.nan)
    if len(x) >= period:
        weights = np.arange(period, 0, -1, dtype=np.float64)  # Faltung: jüngster Wert zuerst
        out[period - 1 :] = np.convolve(x, weights, mode="valid") / weights.sum()
    return _series(out, df, "wma")


@indicator(
    "macd",
    params={"fast": 12, "slow": 26, "signal": 9, "source": "close"},
    pane="separate",
    outputs=["macd", "signal", "hist"],
)
def macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, source: str = "close"
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    ``macd = EMA(fast) − EMA(slow)``, ``signal = EMA(macd, signal)``,
    ``hist = macd − signal``. Wie bei TA-Lib beginnen beide EMAs am selben Index
    (``slow − 1``); die schnelle EMA startet dafür mit dem SMA der ``fast`` Werte davor.

    Args:
        df: Kursreihe.
        fast: Periode der schnellen EMA.
        slow: Periode der langsamen EMA (größer als ``fast``).
        signal: Periode der Signallinie.
        source: Spalte, auf die der Indikator angewendet wird.

    Returns:
        DataFrame mit ``macd`` (ab Index ``slow − 1``), ``signal`` und ``hist``
        (ab Index ``slow + signal − 2``).
    """
    for value, name in ((fast, "fast"), (slow, "slow"), (signal, "signal")):
        _check_period(value, name)
    if fast >= slow:
        raise IndicatorError("Parameter 'fast' muss kleiner als 'slow' sein.")
    x = _column(df, source)
    line = _ema_array(x, fast, start=slow - fast) - _ema_array(x, slow)
    sig = _ema_array(line, signal, start=slow - 1)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig}, index=df.index)


def _wilder_dm_tr(
    high: FloatArray, low: FloatArray, close: FloatArray, period: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Wilder-geglättete +DM, −DM und True Range wie TA-Lib (erster Wert bei ``period``)."""
    n = len(high)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    if n > 1:
        up = high[1:] - high[:-1]
        down = low[:-1] - low[1:]
        plus_dm[1:] = np.where((up > 0) & (up > down), up, 0.0)
        minus_dm[1:] = np.where((down > 0) & (down > up), down, 0.0)
    tr = _true_range(high, low, close)

    smoothed = [np.full(n, np.nan) for _ in range(3)]
    if n <= period:
        return smoothed[0], smoothed[1], smoothed[2]
    for out, raw in zip(smoothed, (plus_dm, minus_dm, tr), strict=True):
        value = float(np.sum(raw[1:period]))  # TA-Lib: period − 1 Werte als Startsumme
        for i in range(period, n):
            value = value - value / period + raw[i]
            out[i] = value
    return smoothed[0], smoothed[1], smoothed[2]


@indicator(
    "adx",
    params={"period": 14},
    pane="separate",
    outputs=["adx", "plus_di", "minus_di"],
)
def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index mit +DI und −DI (Directional Movement nach Wilder).

    Args:
        df: Kursreihe mit ``high``, ``low``, ``close``.
        period: Glättungsperiode.

    Returns:
        DataFrame mit ``plus_di`` und ``minus_di`` (ab Index ``period``) sowie
        ``adx`` (ab Index ``2 · period − 1``).
    """
    _check_period(period)
    high, low, close = _column(df, "high"), _column(df, "low"), _column(df, "close")
    pdm, mdm, tr = _wilder_dm_tr(high, low, close, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(tr == 0, 0.0, 100.0 * pdm / tr)
        minus_di = np.where(tr == 0, 0.0, 100.0 * mdm / tr)
        di_sum = plus_di + minus_di
        dx = np.where(di_sum == 0, 0.0, 100.0 * np.abs(plus_di - minus_di) / di_sum)
    plus_di[np.isnan(tr)] = np.nan
    minus_di[np.isnan(tr)] = np.nan
    dx[np.isnan(tr)] = np.nan

    n = len(high)
    adx_values = np.full(n, np.nan)
    first = 2 * period - 1
    if n > first:
        value = float(np.mean(dx[period : first + 1]))
        adx_values[first] = value
        for i in range(first + 1, n):
            value = (value * (period - 1) + dx[i]) / period
            adx_values[i] = value
    return pd.DataFrame(
        {"adx": adx_values, "plus_di": plus_di, "minus_di": minus_di}, index=df.index
    )


def _midpoint(high: FloatArray, low: FloatArray, period: int) -> FloatArray:
    """(Höchstes Hoch + tiefstes Tief) / 2 über ``period`` Kerzen."""
    hh = pd.Series(high).rolling(period, min_periods=period).max().to_numpy()
    ll = pd.Series(low).rolling(period, min_periods=period).min().to_numpy()
    return np.asarray((hh + ll) / 2.0, dtype=np.float64)


def _shift_forward(x: FloatArray, periods: int) -> FloatArray:
    out = np.full(len(x), np.nan)
    if periods < len(x):
        out[periods:] = x[: len(x) - periods]
    return out


@indicator(
    "ichimoku",
    params={"tenkan": 9, "kijun": 26, "senkou": 52},
    pane="price",
    outputs=["tenkan", "kijun", "senkou_a", "senkou_b"],
)
def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
    """Ichimoku Kinko Hyo.

    Die Wolke (``senkou_a``, ``senkou_b``) ist um ``kijun`` Kerzen nach vorn versetzt:
    der Wert bei ``t`` wurde bei ``t − kijun`` berechnet. Die Chikou-Linie
    (Schlusskurs ``kijun`` Kerzen zurückversetzt) wird nicht als Spalte geliefert,
    weil sie Daten nach ``t`` bräuchte; der Chart zeichnet sie aus dem Schlusskurs.

    Args:
        df: Kursreihe mit ``high`` und ``low``.
        tenkan: Periode der Umkehrlinie.
        kijun: Periode der Basislinie und Versatz der Wolke.
        senkou: Periode der Wolkengrenze B.

    Returns:
        DataFrame mit ``tenkan``, ``kijun``, ``senkou_a``, ``senkou_b``.
    """
    for value, name in ((tenkan, "tenkan"), (kijun, "kijun"), (senkou, "senkou")):
        _check_period(value, name)
    high, low = _column(df, "high"), _column(df, "low")
    tenkan_line = _midpoint(high, low, tenkan)
    kijun_line = _midpoint(high, low, kijun)
    span_a = _shift_forward((tenkan_line + kijun_line) / 2.0, kijun)
    span_b = _shift_forward(_midpoint(high, low, senkou), kijun)
    return pd.DataFrame(
        {"tenkan": tenkan_line, "kijun": kijun_line, "senkou_a": span_a, "senkou_b": span_b},
        index=df.index,
    )


@indicator(
    "psar",
    params={"acceleration": 0.02, "maximum": 0.2},
    pane="price",
    outputs=["psar"],
)
def parabolic_sar(
    df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2
) -> pd.Series[float]:
    """Parabolic SAR nach Wilder (Algorithmus wie TA-Lib).

    Die Startrichtung ergibt sich aus der ersten Kerze: Ist das −DM der zweiten Kerze
    positiv, beginnt der SAR im Abwärtstrend, sonst im Aufwärtstrend.

    Args:
        df: Kursreihe mit ``high`` und ``low``.
        acceleration: Start- und Schrittweite des Beschleunigungsfaktors.
        maximum: Obergrenze des Beschleunigungsfaktors.

    Returns:
        Serie ``psar``; der erste Wert ist NaN.
    """
    if not (acceleration > 0 and maximum > 0):
        raise IndicatorError("Parameter 'acceleration' und 'maximum' müssen größer als 0 sein.")
    high, low = _column(df, "high"), _column(df, "low")
    n = len(high)
    out = np.full(n, np.nan)
    if n < 2:
        return _series(out, df, "psar")

    af = min(acceleration, maximum)
    up_move = high[1] - high[0]
    down_move = low[0] - low[1]
    is_long = not (down_move > 0 and up_move < down_move)
    if is_long:
        ep, sar = high[1], low[0]
    else:
        ep, sar = low[1], high[0]
    new_high, new_low = high[1], low[1]  # wie TA-Lib: erste Iteration ohne Vortag

    for i in range(1, n):
        prev_high, prev_low = new_high, new_low
        new_high, new_low = high[i], low[i]
        if is_long:
            if new_low <= sar:  # Wechsel in den Abwärtstrend
                is_long = False
                sar = max(ep, prev_high, new_high)
                out[i] = sar
                af = min(acceleration, maximum)
                ep = new_low
                sar = max(sar + af * (ep - sar), prev_high, new_high)
            else:
                out[i] = sar
                if new_high > ep:
                    ep = new_high
                    af = min(af + acceleration, maximum)
                sar = min(sar + af * (ep - sar), prev_low, new_low)
        elif new_high >= sar:  # Wechsel in den Aufwärtstrend
            is_long = True
            sar = min(ep, prev_low, new_low)
            out[i] = sar
            af = min(acceleration, maximum)
            ep = new_high
            sar = min(sar + af * (ep - sar), prev_low, new_low)
        else:
            out[i] = sar
            if new_low < ep:
                ep = new_low
                af = min(af + acceleration, maximum)
            sar = max(sar + af * (ep - sar), prev_high, new_high)
    return _series(out, df, "psar")


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series[float]:
    """Average True Range nach Wilder (wie TA-Lib).

    Erster Wert bei Index ``period`` als Mittel der True Ranges 1 … ``period``,
    danach ``ATR_t = (ATR_{t−1} · (period − 1) + TR_t) / period``.

    Args:
        df: Kursreihe mit ``high``, ``low``, ``close``.
        period: Glättungsperiode.

    Returns:
        Serie ``atr``; die ersten ``period`` Werte sind NaN.
    """
    _check_period(period)
    tr = _true_range(_column(df, "high"), _column(df, "low"), _column(df, "close"))
    out = np.full(len(tr), np.nan)
    if len(tr) > period:
        value = float(np.mean(tr[1 : period + 1]))
        out[period] = value
        for i in range(period + 1, len(tr)):
            value = (value * (period - 1) + tr[i]) / period
            out[i] = value
    return _series(out, df, "atr")


@indicator(
    "supertrend",
    params={"period": 10, "multiplier": 3.0},
    pane="price",
    outputs=["supertrend", "direction"],
)
def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend auf Basis der Wilder-ATR.

    Bänder: ``(high + low) / 2 ± multiplier · ATR``. Das untere Band steigt nur
    (außer der Vortages-Schluss lag darunter), das obere fällt nur (außer der
    Vortages-Schluss lag darüber). Startrichtung ist abwärts; ein Schluss über dem
    oberen Band dreht auf aufwärts, ein Schluss unter dem unteren Band auf abwärts.

    Args:
        df: Kursreihe mit ``high``, ``low``, ``close``.
        period: ATR-Periode.
        multiplier: ATR-Vielfaches für den Bandabstand.

    Returns:
        DataFrame mit ``supertrend`` (aktives Band) und ``direction``
        (``1.0`` aufwärts, ``-1.0`` abwärts), beide ab Index ``period``.
    """
    _check_period(period)
    if not multiplier > 0:
        raise IndicatorError("Parameter 'multiplier' muss größer als 0 sein.")
    high, low, close = _column(df, "high"), _column(df, "low"), _column(df, "close")
    atr = atr_wilder(df, period).to_numpy()
    mid = (high + low) / 2.0
    basic_upper = mid + multiplier * atr
    basic_lower = mid - multiplier * atr

    n = len(close)
    line = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if n > period:
        upper, lower, trend = basic_upper[period], basic_lower[period], -1.0
        line[period], direction[period] = upper, trend
        for i in range(period + 1, n):
            prev_close = close[i - 1]
            upper = basic_upper[i] if basic_upper[i] < upper or prev_close > upper else upper
            lower = basic_lower[i] if basic_lower[i] > lower or prev_close < lower else lower
            if trend < 0:
                trend = 1.0 if close[i] > upper else -1.0
            else:
                trend = -1.0 if close[i] < lower else 1.0
            line[i] = lower if trend > 0 else upper
            direction[i] = trend
    return pd.DataFrame({"supertrend": line, "direction": direction}, index=df.index)
