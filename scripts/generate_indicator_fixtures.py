"""Erzeugt synthetische Kursdaten und Referenzwerte für die Indikator-Tests.

Einmalig ausgeführt; die Ergebnisse liegen in ``tests/fixtures/indicators/``.
TA-Lib ist keine Abhängigkeit des Projekts, nur dieses Skripts::

    uv run --no-project --with TA-Lib --with pandas python scripts/generate_indicator_fixtures.py

Ichimoku und Supertrend gibt es in TA-Lib nicht; ihre Referenz ist eine einfache,
unabhängige Schleifen-Implementierung (Supertrend auf Basis von ``talib.ATR``).
Ebenso Keltner (``talib.EMA`` ± ``talib.ATR``), Donchian (``talib.MAX``/``MIN``) und
historische Volatilität (Stichproben-Standardabweichung der Log-Renditen).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import talib

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "indicators"
N = 400


def synthetic_bars(seed: int = 16) -> pd.DataFrame:
    """Aufwärtstrend, Abwärtstrend, Sägezahn, Gap und Seitwärtsphase mit Rauschen."""
    rng = np.random.default_rng(seed)
    close = np.empty(N)
    price = 100.0
    for i in range(N):
        if i < 100:
            drift = 0.004  # Aufwärtstrend
        elif i < 180:
            drift = -0.005  # Abwärtstrend
        elif i < 280:
            drift = 0.03 if (i // 10) % 2 == 0 else -0.03  # Sägezahn
            drift *= 0.5
        else:
            drift = 0.0005  # Seitwärts
        if i == 300:
            price *= 1.12  # Gap nach oben
        if i == 350:
            price *= 0.9  # Gap nach unten
        price *= math.exp(drift + rng.normal(0, 0.01))
        close[i] = price
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.004, N))
    open_[300] = close[300] * 0.995
    open_[350] = close[350] * 1.005
    spread_hi = np.abs(rng.normal(0, 0.008, N))
    spread_lo = np.abs(rng.normal(0, 0.008, N))
    high = np.maximum(open_, close) * (1 + spread_hi)
    low = np.minimum(open_, close) * (1 - spread_lo)
    volume = rng.integers(100_000, 1_000_000, N)
    dates = pd.bdate_range("2020-01-01", periods=N)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_.round(4),
            "high": high.round(4),
            "low": low.round(4),
            "close": close.round(4),
            "volume": volume,
        }
    )


def midpoint(high: np.ndarray, low: np.ndarray, i: int, period: int) -> float:
    if i < period - 1:
        return math.nan
    return (max(high[i - period + 1 : i + 1]) + min(low[i - period + 1 : i + 1])) / 2


def ichimoku_ref(df: pd.DataFrame) -> dict[str, np.ndarray]:
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    tenkan = np.array([midpoint(high, low, i, 9) for i in range(N)])
    kijun = np.array([midpoint(high, low, i, 26) for i in range(N)])
    span_b_raw = np.array([midpoint(high, low, i, 52) for i in range(N)])
    span_a = np.full(N, math.nan)
    span_b = np.full(N, math.nan)
    for i in range(26, N):
        span_a[i] = (tenkan[i - 26] + kijun[i - 26]) / 2
        span_b[i] = span_b_raw[i - 26]
    return {"tenkan": tenkan, "kijun": kijun, "senkou_a": span_a, "senkou_b": span_b}


def supertrend_ref(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> dict[str, np.ndarray]:
    high, low, close = (df[c].to_numpy() for c in ("high", "low", "close"))
    atr = talib.ATR(high, low, close, timeperiod=period)
    st = np.full(N, math.nan)
    direction = np.full(N, math.nan)
    final_up = final_lo = math.nan
    for i in range(N):
        if math.isnan(atr[i]):
            continue
        up = (high[i] + low[i]) / 2 + mult * atr[i]
        lo = (high[i] + low[i]) / 2 - mult * atr[i]
        if math.isnan(final_up):
            final_up, final_lo, d = up, lo, -1.0
        else:
            final_up = up if (up < final_up or close[i - 1] > final_up) else final_up
            final_lo = lo if (lo > final_lo or close[i - 1] < final_lo) else final_lo
            prev_d = direction[i - 1]
            if prev_d < 0:
                d = 1.0 if close[i] > final_up else -1.0
            else:
                d = -1.0 if close[i] < final_lo else 1.0
        direction[i] = d
        st[i] = final_lo if d > 0 else final_up
    return {"supertrend": st, "direction": direction}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bars = synthetic_bars()
    high, low, close = (bars[c].to_numpy(dtype=float) for c in ("high", "low", "close"))

    macd, signal, hist = talib.MACD(close, 12, 26, 9)
    # TA-Lib gibt die MACD-Linie erst ab dem Start der Signallinie aus; die Linie selbst
    # ist ab Index 25 definiert (schnelle EMA ab Index 14 gestartet, wie intern in TA-Lib).
    macd_line = np.full(N, math.nan)
    macd_line[14:] = talib.EMA(close[14:], 12)
    macd_line = macd_line - talib.EMA(close, 26)
    assert np.allclose(macd_line[33:], macd[33:], atol=1e-12)

    ref = pd.DataFrame(
        {
            "sma_20": talib.SMA(close, 20),
            "ema_20": talib.EMA(close, 20),
            "wma_20": talib.WMA(close, 20),
            "macd_macd": macd_line,
            "macd_signal": signal,
            "macd_hist": hist,
            "adx_adx": talib.ADX(high, low, close, 14),
            "adx_plus_di": talib.PLUS_DI(high, low, close, 14),
            "adx_minus_di": talib.MINUS_DI(high, low, close, 14),
            "psar_psar": talib.SAR(high, low, 0.02, 0.2),
        }
    )
    for key, values in ichimoku_ref(bars).items():
        ref[f"ichimoku_{key}"] = values
    for key, values in supertrend_ref(bars).items():
        ref[f"supertrend_{key}"] = values

    bars.to_csv(OUT / "bars.csv", index=False, date_format="%Y-%m-%d")
    ref.insert(0, "date", bars["date"].dt.strftime("%Y-%m-%d"))
    ref.to_csv(OUT / "trend_reference.csv", index=False, float_format="%.10f")

    mv = momentum_volatility_ref(bars)
    mv.insert(0, "date", bars["date"].dt.strftime("%Y-%m-%d"))
    mv.to_csv(OUT / "momentum_volatility_reference.csv", index=False, float_format="%.10f")


def historical_volatility_ref(close: np.ndarray, period: int = 20) -> np.ndarray:
    out = np.full(N, math.nan)
    log_ret = [math.log(close[i] / close[i - 1]) for i in range(1, N)]
    for i in range(period, N):
        window = log_ret[i - period : i]
        mean = sum(window) / period
        var = sum((r - mean) ** 2 for r in window) / (period - 1)
        out[i] = math.sqrt(var) * math.sqrt(252)
    return out


def momentum_volatility_ref(bars: pd.DataFrame) -> pd.DataFrame:
    high, low, close = (bars[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    # TA-Lib gibt %K erst ab dem Start von %D aus; %K selbst ist zwei Kerzen früher
    # definiert, deshalb hier aus STOCHF (%K roh) mit SMA(3) nachgebildet.
    slow_k, slow_d = talib.STOCH(high, low, close, 14, 3, 0, 3, 0)
    fast_k, _ = talib.STOCHF(high, low, close, 14, 1, 0)
    k_line = np.full(N, math.nan)
    k_line[13:] = talib.SMA(fast_k[13:], 3)
    assert np.allclose(k_line[17:], slow_k[17:], atol=1e-12)
    upper, middle, lower = talib.BBANDS(close, 20, 2.0, 2.0, 0)
    ema20 = talib.EMA(close, 20)
    atr10 = talib.ATR(high, low, close, 10)
    dc_upper, dc_lower = talib.MAX(high, 20), talib.MIN(low, 20)
    return pd.DataFrame(
        {
            "rsi_rsi": talib.RSI(close, 14),
            "stoch_k": k_line,
            "stoch_d": slow_d,
            "roc_roc": talib.ROC(close, 10),
            "willr_willr": talib.WILLR(high, low, close, 14),
            "cci_cci": talib.CCI(high, low, close, 20),
            "atr_atr": talib.ATR(high, low, close, 14),
            "bbands_upper": upper,
            "bbands_middle": middle,
            "bbands_lower": lower,
            "keltner_upper": ema20 + 2.0 * atr10,
            "keltner_middle": ema20,
            "keltner_lower": ema20 - 2.0 * atr10,
            "donchian_upper": dc_upper,
            "donchian_middle": (dc_upper + dc_lower) / 2,
            "donchian_lower": dc_lower,
            "hv_hv": historical_volatility_ref(close),
        }
    )


if __name__ == "__main__":
    main()
