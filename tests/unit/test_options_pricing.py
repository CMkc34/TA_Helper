"""Black-Scholes-Merton-Bewertung, Griechen und implizite Volatilität."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_helfer.options import (
    Greeks,
    OptionType,
    bs_greeks,
    bs_price,
    historical_volatility,
    implied_volatility,
)
from trading_helfer.options.pricing import _brentq

CALL, PUT = OptionType.CALL, OptionType.PUT
TOL = 1e-6

# Lehrbuchbeispiele. Die Sollwerte sind mit 40 Stellen (mpmath) berechnet; die gedruckten
# Lehrbuchwerte stimmen damit auf die dort angegebenen Stellen überein.
# Felder: (Typ, S, K, T, r, sigma, q, Preis, Delta, Gamma, Theta/Jahr, Vega, Rho)
TEXTBOOK = [
    # Hull, Options, Futures and Other Derivatives, Bsp. 15.6: c = 4,76; p = 0,81
    (CALL, 42, 40, 0.5, 0.1, 0.2, 0.0,
     4.75942239287153, 0.779131290942669, 0.0499626704059119, -4.55909219459263,
     8.81341505960285, 13.9820459133603),
    (PUT, 42, 40, 0.5, 0.1, 0.2, 0.0,
     0.808599372900094, -0.220868709057331, 0.0499626704059119, -0.75417449658977,
     8.81341505960285, -5.042542576654),
    # Hull, Kap. „The Greek Letters“: c = 2,40; Δ = 0,522; Γ = 0,066; Θ = −4,31;
    # Vega = 12,1; Rho = 8,91
    (CALL, 49, 50, 20 / 52, 0.05, 0.2, 0.0,
     2.40052732327171, 0.521604661066396, 0.0655440393478444, -4.30532982293257,
     12.1054798826288, 8.90696194960835),
    (PUT, 49, 50, 20 / 52, 0.05, 0.2, 0.0,
     2.44817544128185, -0.478395338933604, 0.0655440393478444, -1.85294741703207,
     12.1054798826288, -9.95751809578016),
    # Haug, The Complete Guide to Option Pricing Formulas (generalisiertes BSM): p = 4,0870
    (PUT, 75, 70, 0.5, 0.1, 0.35, 0.05,
     4.08695382863535, -0.299707917561591, 0.0184663838262898, -4.82964622893192,
     18.1778465790041, -13.2825238228773),
    # Hull, Indexoption mit Dividendenrendite: c = 51,83
    (CALL, 930, 900, 2 / 12, 0.08, 0.2, 0.03,
     51.8329567964908, 0.703418008601192, 0.00450740386169434, -106.531372855825,
     129.948453332648, 100.390965200436),
]  # fmt: skip


@pytest.mark.parametrize("case", TEXTBOOK)
def test_price_matches_textbook(case: tuple[float, ...]) -> None:
    kind, s, k, t, r, vol, q, price, *_ = case
    assert bs_price(OptionType(kind), s, k, t, r, vol, q) == pytest.approx(price, abs=TOL)


@pytest.mark.parametrize("case", TEXTBOOK)
def test_greeks_match_textbook(case: tuple[float, ...]) -> None:
    kind, s, k, t, r, vol, q, price, delta, gamma, theta, vega, rho = case
    greeks = bs_greeks(OptionType(kind), s, k, t, r, vol, q)
    assert isinstance(greeks, Greeks)
    assert greeks.price == pytest.approx(price, abs=TOL)
    assert greeks.delta == pytest.approx(delta, abs=TOL)
    assert greeks.gamma == pytest.approx(gamma, abs=TOL)
    assert greeks.theta == pytest.approx(theta, abs=TOL)
    assert greeks.vega == pytest.approx(vega, abs=TOL)
    assert greeks.rho == pytest.approx(rho, abs=TOL)


@pytest.mark.parametrize("case", TEXTBOOK)
def test_implied_volatility_recovers_textbook_sigma(case: tuple[float, ...]) -> None:
    kind, s, k, t, r, vol, q, price, *_ = case
    iv = implied_volatility(price, OptionType(kind), s, k, t, r, q)
    assert iv == pytest.approx(vol, abs=TOL)


_params = {
    "spot": st.floats(10, 500),
    "moneyness": st.floats(0.5, 1.5),
    "time": st.floats(0.02, 3.0),
    "rate": st.floats(-0.01, 0.1),
    "vol": st.floats(0.05, 1.5),
    "q": st.floats(0.0, 0.08),
}


@settings(max_examples=200, deadline=None)
@given(**_params)
def test_put_call_parity(
    spot: float, moneyness: float, time: float, rate: float, vol: float, q: float
) -> None:
    strike = spot * moneyness
    call = bs_price(CALL, spot, strike, time, rate, vol, q)
    put = bs_price(PUT, spot, strike, time, rate, vol, q)
    forward_diff = spot * math.exp(-q * time) - strike * math.exp(-rate * time)
    assert call - put == pytest.approx(forward_diff, abs=TOL)

    gc = bs_greeks(CALL, spot, strike, time, rate, vol, q)
    gp = bs_greeks(PUT, spot, strike, time, rate, vol, q)
    assert gc.delta - gp.delta == pytest.approx(math.exp(-q * time), abs=1e-12)
    assert gc.gamma == pytest.approx(gp.gamma, rel=1e-12)
    assert gc.vega == pytest.approx(gp.vega, rel=1e-12)
    assert gc.rho - gp.rho == pytest.approx(strike * time * math.exp(-rate * time), rel=1e-9)
    theta_diff = q * spot * math.exp(-q * time) - rate * strike * math.exp(-rate * time)
    assert gc.theta - gp.theta == pytest.approx(theta_diff, abs=1e-9)


@settings(max_examples=200, deadline=None)
@given(**_params, is_call=st.booleans())
def test_implied_volatility_round_trip(
    spot: float, moneyness: float, time: float, rate: float, vol: float, q: float, is_call: bool
) -> None:
    kind = CALL if is_call else PUT
    strike = spot * moneyness
    greeks = bs_greeks(kind, spot, strike, time, rate, vol, q)
    # Nur auswerten, wo der Preis überhaupt von der Volatilität abhängt.
    if greeks.vega < 1e-3:
        return
    iv = implied_volatility(greeks.price, kind, spot, strike, time, rate, q)
    assert iv == pytest.approx(vol, abs=1e-6)


@pytest.mark.parametrize("kind", [CALL, PUT])
def test_greeks_match_finite_differences(kind: OptionType) -> None:
    s, k, t, r, vol, q = 100.0, 105.0, 0.75, 0.03, 0.25, 0.02
    g = bs_greeks(kind, s, k, t, r, vol, q)
    h = 1e-4

    def price(**kw: float) -> float:
        args = {"spot": s, "time": t, "rate": r, "volatility": vol} | kw
        return bs_price(kind, args["spot"], k, args["time"], args["rate"], args["volatility"], q)

    assert g.delta == pytest.approx((price(spot=s + h) - price(spot=s - h)) / (2 * h), abs=1e-7)
    gamma_fd = (price(spot=s + h) - 2 * price() + price(spot=s - h)) / h**2
    assert g.gamma == pytest.approx(gamma_fd, abs=1e-5)
    assert g.theta == pytest.approx(-(price(time=t + h) - price(time=t - h)) / (2 * h), abs=1e-6)
    assert g.vega == pytest.approx(
        (price(volatility=vol + h) - price(volatility=vol - h)) / (2 * h), abs=1e-6
    )
    assert g.rho == pytest.approx((price(rate=r + h) - price(rate=r - h)) / (2 * h), abs=1e-6)


def test_option_type_values() -> None:
    assert OptionType("CALL") is CALL
    assert str(PUT) == "PUT"


@pytest.mark.parametrize(
    ("kind", "spot", "strike", "expected"),
    [(CALL, 110, 100, 10.0), (CALL, 90, 100, 0.0), (PUT, 90, 100, 10.0), (PUT, 110, 100, 0.0)],
)
def test_price_at_expiry_is_intrinsic(
    kind: OptionType, spot: float, strike: float, expected: float
) -> None:
    assert bs_price(kind, spot, strike, 0.0, 0.05, 0.3) == pytest.approx(expected)


def test_price_with_zero_volatility_is_discounted_forward_value() -> None:
    expected = 100 * math.exp(-0.02) - 90 * math.exp(-0.05)
    assert bs_price(CALL, 100, 90, 1.0, 0.05, 0.0, 0.02) == pytest.approx(expected)
    assert bs_price(PUT, 100, 90, 1.0, 0.05, 0.0, 0.02) == 0.0


@pytest.mark.parametrize(
    ("spot", "strike", "time", "vol", "message"),
    [
        (0.0, 100, 1.0, 0.2, "Kurs"),
        (100, -1.0, 1.0, 0.2, "Basispreis"),
        (100, 100, -0.1, 0.2, "Restlaufzeit"),
        (100, 100, 1.0, -0.2, "Volatilität"),
        (math.nan, 100, 1.0, 0.2, "Kurs"),
    ],
)
def test_invalid_inputs_raise(
    spot: float, strike: float, time: float, vol: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bs_price(CALL, spot, strike, time, 0.05, vol)


@pytest.mark.parametrize(("time", "vol"), [(0.0, 0.2), (1.0, 0.0)])
def test_greeks_need_positive_time_and_volatility(time: float, vol: float) -> None:
    with pytest.raises(ValueError, match="Griechen"):
        bs_greeks(CALL, 100, 100, time, 0.05, vol)


@pytest.mark.parametrize(
    ("price", "kind"),
    [(0.0, CALL), (-1.0, PUT), (100.0, CALL), (200.0, PUT), (5.0, CALL)],
)
def test_implied_volatility_rejects_prices_outside_bounds(price: float, kind: OptionType) -> None:
    # Call: Untergrenze ~ 9,75 (tief im Geld), Obergrenze 100; Put: Obergrenze ~ 85,6
    with pytest.raises(ValueError, match="Arbitragegrenzen"):
        implied_volatility(price, kind, 100, 90, 1.0, 0.05)


def test_implied_volatility_outside_search_interval() -> None:
    price = bs_price(CALL, 100, 100, 1.0, 0.05, 0.8)
    with pytest.raises(ValueError, match="Suchintervall"):
        implied_volatility(price, CALL, 100, 100, 1.0, 0.05, upper=0.5)
    price_low = bs_price(CALL, 100, 100, 1.0, 0.05, 0.1)
    with pytest.raises(ValueError, match="Suchintervall"):
        implied_volatility(price_low, CALL, 100, 100, 1.0, 0.05, lower=0.2)


def test_implied_volatility_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="Restlaufzeit"):
        implied_volatility(5.0, CALL, 100, 100, 0.0, 0.05)
    with pytest.raises(ValueError, match="Suchintervall"):
        implied_volatility(5.0, CALL, 100, 100, 1.0, 0.05, lower=1.0, upper=0.5)


def test_brentq_finds_roots_and_handles_edge_cases() -> None:
    assert _brentq(lambda x: x * x - 2, 0, 2, 1e-14, 1e-15, 100) == pytest.approx(math.sqrt(2))
    assert _brentq(lambda x: x, 0, 1, 1e-12, 1e-15, 100) == 0
    assert _brentq(lambda x: x - 1, 0, 1, 1e-12, 1e-15, 100) == 1
    assert _brentq(math.cos, 0, 3, 1e-14, 1e-15, 100) == pytest.approx(math.pi / 2)
    with pytest.raises(ValueError, match="Vorzeichenwechsel"):
        _brentq(lambda x: x * x + 1, -1, 1, 1e-12, 1e-15, 100)
    with pytest.raises(ValueError, match="Konvergenz"):
        _brentq(lambda x: x**3 - 0.3, 0, 1, 1e-15, 1e-15, 2)


def test_historical_volatility() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, 5000)
    closes = 100 * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    expected = float(np.std(returns, ddof=1) * math.sqrt(252))
    assert historical_volatility(closes) == pytest.approx(expected, rel=1e-12)
    assert historical_volatility(closes, periods_per_year=52) == pytest.approx(
        expected * math.sqrt(52 / 252), rel=1e-12
    )
    assert historical_volatility([100.0, 100.0, 100.0]) == 0.0


@pytest.mark.parametrize(
    ("closes", "ppy", "message"),
    [
        ([100.0, 101.0], 252, "mindestens drei"),
        ([100.0, 0.0, 101.0], 252, "größer als 0"),
        ([100.0, 101.0, 102.0], 0, "Perioden"),
    ],
)
def test_historical_volatility_invalid(closes: list[float], ppy: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        historical_volatility(closes, periods_per_year=ppy)
