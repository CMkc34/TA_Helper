"""Black-Scholes-Merton-Bewertung europäischer Optionen mit stetiger Dividendenrendite.

Konventionen aller Funktionen:

* ``spot`` und ``strike`` in Kurswährung, beide > 0.
* ``time`` ist die Restlaufzeit in Jahren (z. B. ``30 / 365``).
* ``rate`` (risikoloser Zins) und ``dividend_yield`` sind stetige Jahresraten
  (``0.05`` = 5 %).
* ``volatility`` ist die annualisierte Volatilität (``0.2`` = 20 %).

Die Griechen sind die unskalierten partiellen Ableitungen des Preises: Theta je Jahr
(für Theta je Kalendertag durch 365 teilen), Vega je 1,00 Volatilität und Rho je
1,00 Zins (für „je Prozentpunkt“ durch 100 teilen).

Mit :func:`historical_volatility` lassen sich Optionen auch ohne historische
Optionsdaten näherungsweise bewerten, etwa für einen genäherten Options-Backtest.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

__all__ = [
    "Greeks",
    "OptionType",
    "bs_greeks",
    "bs_price",
    "historical_volatility",
    "implied_volatility",
]

_SQRT_2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


class OptionType(StrEnum):
    """Art der Option."""

    CALL = "CALL"
    PUT = "PUT"


@dataclass(frozen=True, slots=True)
class Greeks:
    """Preis und Griechen einer Option.

    Theta ist je Jahr, Vega je 1,00 Volatilität und Rho je 1,00 Zins angegeben.
    """

    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT_2)


def _norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _validate(spot: float, strike: float, time: float, volatility: float) -> None:
    if not spot > 0:
        raise ValueError(f"Kurs muss größer als 0 sein, ist {spot}.")
    if not strike > 0:
        raise ValueError(f"Basispreis muss größer als 0 sein, ist {strike}.")
    if not time >= 0:
        raise ValueError(f"Restlaufzeit darf nicht negativ sein, ist {time}.")
    if not volatility >= 0:
        raise ValueError(f"Volatilität darf nicht negativ sein, ist {volatility}.")


def _d1_d2(
    spot: float, strike: float, time: float, rate: float, volatility: float, dividend_yield: float
) -> tuple[float, float]:
    vol_sqrt_t = volatility * math.sqrt(time)
    d1 = (
        math.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility * volatility) * time
    ) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    time: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Preis einer europäischen Option nach Black-Scholes-Merton.

    Bei ``time == 0`` oder ``volatility == 0`` gilt der deterministische Grenzwert
    ``max(±(S·e^(-qT) − K·e^(-rT)), 0)``; bei ``time == 0`` ist das der innere Wert.

    Raises:
        ValueError: bei Kurs oder Basispreis ≤ 0 oder negativer Laufzeit/Volatilität.
    """
    _validate(spot, strike, time, volatility)
    fwd_spot = spot * math.exp(-dividend_yield * time)
    disc_strike = strike * math.exp(-rate * time)
    sign = 1.0 if option_type is OptionType.CALL else -1.0
    if time == 0 or volatility == 0:
        return max(sign * (fwd_spot - disc_strike), 0.0)
    d1, d2 = _d1_d2(spot, strike, time, rate, volatility, dividend_yield)
    return sign * (fwd_spot * _norm_cdf(sign * d1) - disc_strike * _norm_cdf(sign * d2))


def bs_greeks(
    option_type: OptionType,
    spot: float,
    strike: float,
    time: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Preis, Delta, Gamma, Theta, Vega und Rho nach Black-Scholes-Merton.

    Raises:
        ValueError: bei ungültigen Eingaben oder wenn Laufzeit oder Volatilität 0 sind
            (dort sind die Griechen nicht definiert).
    """
    _validate(spot, strike, time, volatility)
    if time == 0 or volatility == 0:
        raise ValueError("Griechen brauchen eine Restlaufzeit und Volatilität größer als 0.")
    d1, d2 = _d1_d2(spot, strike, time, rate, volatility, dividend_yield)
    sqrt_t = math.sqrt(time)
    q_disc = math.exp(-dividend_yield * time)
    r_disc = math.exp(-rate * time)
    fwd_spot = spot * q_disc
    disc_strike = strike * r_disc
    pdf_d1 = _norm_pdf(d1)

    gamma = q_disc * pdf_d1 / (spot * volatility * sqrt_t)
    vega = fwd_spot * pdf_d1 * sqrt_t
    decay = -fwd_spot * pdf_d1 * volatility / (2.0 * sqrt_t)

    sign = 1.0 if option_type is OptionType.CALL else -1.0
    n_d1 = _norm_cdf(sign * d1)
    n_d2 = _norm_cdf(sign * d2)
    price = sign * (fwd_spot * n_d1 - disc_strike * n_d2)
    delta = sign * q_disc * n_d1
    theta = decay + sign * (dividend_yield * fwd_spot * n_d1 - rate * disc_strike * n_d2)
    rho = sign * strike * time * r_disc * n_d2
    return Greeks(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def _brentq(
    func: Callable[[float], float],
    lower: float,
    upper: float,
    xtol: float,
    rtol: float,
    max_iter: int,
) -> float:
    """Nullstelle von ``func`` in ``[lower, upper]`` nach dem Brent-Verfahren."""
    x_pre, x_cur = lower, upper
    f_pre, f_cur = func(x_pre), func(x_cur)
    if f_pre == 0:
        return x_pre
    if f_cur == 0:
        return x_cur
    if (f_pre < 0) == (f_cur < 0):
        raise ValueError("Brent-Verfahren: Intervall enthält keinen Vorzeichenwechsel.")
    x_blk = f_blk = s_pre = s_cur = 0.0
    for _ in range(max_iter):
        if f_pre != 0 and f_cur != 0 and (f_pre < 0) != (f_cur < 0):
            x_blk, f_blk = x_pre, f_pre
            s_pre = s_cur = x_cur - x_pre
        if abs(f_blk) < abs(f_cur):
            x_pre, x_cur, x_blk = x_cur, x_blk, x_cur
            f_pre, f_cur, f_blk = f_cur, f_blk, f_cur
        tol = 0.5 * (xtol + rtol * abs(x_cur))
        s_bis = 0.5 * (x_blk - x_cur)
        if f_cur == 0 or abs(s_bis) < tol:
            return x_cur
        if abs(s_pre) > tol and abs(f_cur) < abs(f_pre):
            if x_pre == x_blk:
                # Sekantenschritt
                s_try = -f_cur * (x_cur - x_pre) / (f_cur - f_pre)
            else:
                # inverse quadratische Interpolation
                d_pre = (f_pre - f_cur) / (x_pre - x_cur)
                d_blk = (f_blk - f_cur) / (x_blk - x_cur)
                s_try = -f_cur * (f_blk * d_blk - f_pre * d_pre) / (d_blk * d_pre * (f_blk - f_pre))
            if 2 * abs(s_try) < min(abs(s_pre), 3 * abs(s_bis) - tol):
                s_pre, s_cur = s_cur, s_try
            else:
                s_pre = s_cur = s_bis
        else:
            s_pre = s_cur = s_bis
        x_pre, f_pre = x_cur, f_cur
        x_cur += s_cur if abs(s_cur) > tol else math.copysign(tol, s_bis)
        f_cur = func(x_cur)
    raise ValueError(f"Brent-Verfahren: keine Konvergenz nach {max_iter} Schritten.")


def implied_volatility(
    price: float,
    option_type: OptionType,
    spot: float,
    strike: float,
    time: float,
    rate: float,
    dividend_yield: float = 0.0,
    *,
    lower: float = 1e-6,
    upper: float = 5.0,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Implizite Volatilität zu einem Marktpreis, bestimmt mit dem Brent-Verfahren.

    Gesucht wird im Intervall ``[lower, upper]``.

    Raises:
        ValueError: bei ungültigen Eingaben, einem Preis außerhalb der Arbitragegrenzen
            oder wenn die Volatilität nicht im Suchintervall liegt.
    """
    _validate(spot, strike, time, 0.0)
    if not time > 0:
        raise ValueError("Implizite Volatilität braucht eine Restlaufzeit größer als 0.")
    if not 0 < lower < upper:
        raise ValueError(f"Ungültiges Suchintervall [{lower}, {upper}].")
    fwd_spot = spot * math.exp(-dividend_yield * time)
    disc_strike = strike * math.exp(-rate * time)
    if option_type is OptionType.CALL:
        floor, cap = max(fwd_spot - disc_strike, 0.0), fwd_spot
    else:
        floor, cap = max(disc_strike - fwd_spot, 0.0), disc_strike
    if not floor < price < cap:
        raise ValueError(
            f"Preis {price} liegt außerhalb der Arbitragegrenzen ({floor:.6f}, {cap:.6f})."
        )

    def objective(sigma: float) -> float:
        return bs_price(option_type, spot, strike, time, rate, sigma, dividend_yield) - price

    if objective(lower) > 0 or objective(upper) < 0:
        raise ValueError(f"Implizite Volatilität liegt nicht im Suchintervall [{lower}, {upper}].")
    return _brentq(
        objective, lower, upper, xtol=tol, rtol=4 * np.finfo(float).eps, max_iter=max_iter
    )


def historical_volatility(closes: Iterable[float], periods_per_year: float = 252.0) -> float:
    """Annualisierte historische Volatilität aus Schlusskursen.

    Standardabweichung (Stichprobe, ``ddof=1``) der logarithmischen Renditen mal
    ``sqrt(periods_per_year)``. Für Wochenkerzen ``periods_per_year=52`` angeben.

    Raises:
        ValueError: bei weniger als drei Kursen oder Kursen ≤ 0.
    """
    values = np.asarray(list(closes), dtype=float)
    if values.size < 3:
        raise ValueError("Für die historische Volatilität sind mindestens drei Kurse nötig.")
    if not np.all(values > 0):
        raise ValueError("Kurse müssen größer als 0 sein.")
    if not periods_per_year > 0:
        raise ValueError(f"Perioden je Jahr müssen größer als 0 sein, ist {periods_per_year}.")
    log_returns = np.diff(np.log(values))
    return float(np.std(log_returns, ddof=1) * math.sqrt(periods_per_year))
