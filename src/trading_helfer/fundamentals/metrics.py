"""Fundamentale Kennzahlen „point in time“ (docs/SPEC.md, „Kennzahlen (Fundamentaldaten)“).

Grundlage sind die Fundamentaldaten im Langformat aus dem Kursspeicher
(:meth:`~trading_helfer.data.storage.MarketStore.load_fundamentals`). Eine Zahl zählt
für den Stichtag ``as_of`` nur, wenn ihr ``reported_date`` ≤ ``as_of`` ist; so sieht ein
Backtest nie Werte, die am Stichtag noch nicht veröffentlicht waren.

Regeln der Berechnung:

- **Bezugsperiode** ist das jüngste Periodenende (Quartal oder Geschäftsjahr), zu dem am
  Stichtag Zahlen vorliegen.
- **Stromgrößen** (Umsatz, Gewinn, Cashflow …) gelten als TTM: Summe der letzten vier
  Quartale bis zur Bezugsperiode. Fehlt eines der vier Quartale, wird ersatzweise der
  Jahreswert verwendet, wenn das Geschäftsjahr an der Bezugsperiode endet.
- **Bestandsgrößen** (Bilanz, Aktienanzahl) stammen aus der Bezugsperiode.
- **Vergleichswerte** (Wachstum, Veränderungen, F-Score) stammen aus der Periode etwa
  ein bzw. drei/fünf Jahre vor der Bezugsperiode (± 31 Tage).
- **Kurs**: unbereinigter Schlusskurs des letzten Handelstags ≤ ``as_of``, höchstens
  :data:`MAX_PRICE_AGE` alt. Marktkapitalisierung = Kurs × ``shares_outstanding``.
- Quoten und Renditen sind Anteile (``0.25`` = 25 %).
- Division durch null, ein nicht positiver Nenner (etwa ein Verlust beim KGV) oder
  fehlende Eingangswerte ergeben ``None``. Fehlende Werte werden nie als ``0`` behandelt;
  einzige Ausnahme ist ein fehlender Lagerbestand bei der Quick Ratio.
- Mittelabflüsse (``capex``, ``dividends_paid``) gehen unabhängig vom Vorzeichen der
  Quelle als Betrag ein.
- ROIC nutzt den pauschalen US-Steuersatz :data:`ROIC_TAX_RATE`, weil die Steuerlast
  nicht gespeichert wird.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from statistics import median
from typing import Any, Final, cast

import pandas as pd

from trading_helfer.data.storage import DateLike, MarketStore

__all__ = [
    "MAX_PRICE_AGE",
    "METRIC_LABELS",
    "METRIC_NAMES",
    "ROIC_TAX_RATE",
    "MetricValues",
    "compute_metrics",
    "metrics_history",
    "sector_median",
]

MetricValues = dict[str, float | None]
"""Kennzahlen eines Werts: Name → Wert (``None`` = nicht berechenbar)."""

METRIC_LABELS: Final[Mapping[str, str]] = {
    "price": "Kurs",
    "shares_outstanding": "Aktienanzahl",
    "market_cap": "Marktkapitalisierung",
    "enterprise_value": "Enterprise Value",
    "pe_ratio": "KGV (trailing)",
    "ps_ratio": "KUV",
    "pb_ratio": "KBV",
    "pcf_ratio": "KCV",
    "ev_ebitda": "EV/EBITDA",
    "peg_ratio": "PEG-Ratio",
    "dividend_yield": "Dividendenrendite",
    "payout_ratio": "Ausschüttungsquote",
    "dividend_growth_5y": "Dividendenwachstum (5 Jahre, CAGR)",
    "fcf_yield": "Free-Cashflow-Rendite",
    "revenue_ttm": "Umsatz (TTM)",
    "net_income_ttm": "Nettogewinn (TTM)",
    "ebitda_ttm": "EBITDA (TTM)",
    "eps_ttm": "EPS (TTM, verwässert)",
    "free_cashflow": "Free Cashflow (TTM)",
    "revenue_growth_yoy": "Umsatzwachstum (YoY)",
    "revenue_growth_3y": "Umsatzwachstum (3 Jahre, CAGR)",
    "earnings_growth_yoy": "Gewinnwachstum (YoY)",
    "earnings_growth_3y": "Gewinnwachstum (3 Jahre, CAGR)",
    "eps_growth_yoy": "EPS-Wachstum (YoY)",
    "gross_margin": "Bruttomarge",
    "operating_margin": "Operative Marge",
    "net_margin": "Nettomarge",
    "roe": "Eigenkapitalrendite (ROE)",
    "roa": "Gesamtkapitalrendite (ROA)",
    "roic": "Rendite auf investiertes Kapital (ROIC)",
    "asset_turnover": "Kapitalumschlag",
    "debt_to_equity": "Verschuldungsgrad (Debt/Equity)",
    "net_debt_to_ebitda": "Netto-Verschuldung/EBITDA",
    "interest_coverage": "Zinsdeckungsgrad",
    "current_ratio": "Current Ratio",
    "quick_ratio": "Quick Ratio",
    "share_change_yoy": "Veränderung der Aktienanzahl (YoY)",
    "piotroski_f_score": "Piotroski-F-Score",
    "altman_z_score": "Altman-Z-Score",
}
"""Deutsche Anzeigenamen aller Kennzahlen, in Anzeigereihenfolge."""

METRIC_NAMES: Final[tuple[str, ...]] = tuple(METRIC_LABELS)
"""Namen aller Kennzahlen, die :func:`compute_metrics` liefert."""

MAX_PRICE_AGE: Final = dt.timedelta(days=7)
"""Ältester Schlusskurs (vor ``as_of``), der noch als aktueller Kurs gilt."""

ROIC_TAX_RATE: Final = 0.21
"""Pauschaler Steuersatz für NOPAT = operatives Ergebnis × (1 − Steuersatz)."""

_PERIOD_TOLERANCE: Final = pd.Timedelta(days=7)
_YEAR_TOLERANCE: Final = pd.Timedelta(days=31)
_TTM_WINDOW: Final = pd.Timedelta(days=320)


# ---------------------------------------------------------------------- öffentliche API


def compute_metrics(symbol: str, as_of: DateLike, *, store: MarketStore) -> MetricValues:
    """Berechnet alle Kennzahlen eines Werts zum Stichtag.

    Args:
        symbol: Tickersymbol.
        as_of: Stichtag; nur Zahlen mit ``reported_date ≤ as_of`` und Kurse bis
            einschließlich ``as_of`` gehen ein.
        store: Offener Kursspeicher.

    Returns:
        Dictionary mit genau den Schlüsseln :data:`METRIC_NAMES`; nicht berechenbare
        Kennzahlen sind ``None``.
    """
    day = _to_day(as_of)
    data = _Fundamentals.from_frame(store.load_fundamentals(symbol, as_of=day))
    price = _price_at(store, symbol, day)
    return _metrics(data, data.latest_period(), price)


def metrics_history(symbol: str, *, store: MarketStore) -> pd.DataFrame:
    """Kennzahlen je Quartal als Zeitreihe.

    Für jedes Periodenende (Quartale; ohne Quartalszahlen die Geschäftsjahre) werden die
    Kennzahlen so berechnet, wie sie am Tag der Veröffentlichung dieser Periode bekannt
    waren: Stichtag ist das späteste ``reported_date`` der Periode, der Kurs der
    Schlusskurs an diesem Tag.

    Args:
        symbol: Tickersymbol.
        store: Offener Kursspeicher.

    Returns:
        DataFrame mit den Spalten ``period_end``, ``as_of`` (beide ``datetime64[ns]``)
        und :data:`METRIC_NAMES` (``float64``, ``NaN`` = nicht berechenbar), aufsteigend
        nach ``period_end``; leer, wenn keine Fundamentaldaten vorliegen.
    """
    frame = store.load_fundamentals(symbol)
    bars = store.load_bars(symbol, adjusted=False)
    closes = pd.Series(
        bars["close"].to_numpy(dtype="float64"), index=pd.DatetimeIndex(bars["date"])
    )
    quarterly = frame[frame["period_type"] == "Q"]
    periods = quarterly if not quarterly.empty else frame
    rows: list[dict[str, object]] = []
    for period_end, group in periods.groupby("period_end", sort=True):
        as_of = pd.Timestamp(group["reported_date"].max())
        if pd.isna(as_of):
            continue
        visible = frame[frame["reported_date"] <= as_of]
        data = _Fundamentals.from_frame(visible)
        values = _metrics(data, pd.Timestamp(cast(Any, period_end)), _close_at(closes, as_of))
        rows.append({"period_end": period_end, "as_of": as_of, **values})
    columns = ["period_end", "as_of", *METRIC_NAMES]
    result = pd.DataFrame(rows, columns=columns)
    for column in ("period_end", "as_of"):
        result[column] = pd.to_datetime(result[column]).astype("datetime64[ns]")
    result[list(METRIC_NAMES)] = result[list(METRIC_NAMES)].astype("float64")
    return result


def sector_median(
    metric: str,
    as_of: DateLike,
    *,
    store: MarketStore,
    symbols: Iterable[str] | None = None,
) -> dict[str, float | None]:
    """Median einer Kennzahl je Sektor zum Stichtag.

    Grundlage sind alle Werte mit Sektor in den Stammdaten (``instruments``); nicht
    berechenbare Werte (``None``) zählen nicht mit.

    Args:
        metric: Name aus :data:`METRIC_NAMES`.
        as_of: Stichtag (siehe :func:`compute_metrics`).
        store: Offener Kursspeicher.
        symbols: Optional nur diese Symbole berücksichtigen (z. B. ein Indexuniversum).

    Returns:
        Sektor → Median, alphabetisch nach Sektor; ``None``, wenn im Sektor kein Wert
        die Kennzahl liefert.

    Raises:
        ValueError: Die Kennzahl ist unbekannt.
    """
    if metric not in METRIC_LABELS:
        raise ValueError(f"Unbekannte Kennzahl {metric!r}.")
    rows = store.connection.execute(
        "SELECT symbol, sector FROM instruments "
        "WHERE sector IS NOT NULL AND trim(sector) <> '' ORDER BY symbol"
    ).fetchall()
    wanted = None if symbols is None else {s.strip().upper() for s in symbols}
    values: dict[str, list[float]] = {}
    for symbol, sector in rows:
        if wanted is not None and symbol not in wanted:
            continue
        bucket = values.setdefault(str(sector), [])
        value = compute_metrics(str(symbol), as_of, store=store)[metric]
        if value is not None:
            bucket.append(value)
    return {sector: (median(v) if v else None) for sector, v in sorted(values.items())}


# ---------------------------------------------------------------------- Datenbasis


@dataclass(frozen=True)
class _Fundamentals:
    """Fundamentaldaten breit: je Periodenart eine Tabelle Periodenende × Metrik."""

    quarterly: pd.DataFrame
    annual: pd.DataFrame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> _Fundamentals:
        return cls(_wide(frame, "Q"), _wide(frame, "FY"))

    def period_ends(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.quarterly.index.union(self.annual.index)).sort_values()

    def latest_period(self) -> pd.Timestamp | None:
        ends = self.period_ends()
        return None if ends.empty else pd.Timestamp(ends[-1])

    def period_before(self, ref: pd.Timestamp, years: int) -> pd.Timestamp | None:
        """Periodenende etwa ``years`` Jahre vor ``ref`` (± 31 Tage), sonst ``None``."""
        target = ref - pd.DateOffset(years=years)
        candidates = [
            (abs(end - target), end)
            for end in map(pd.Timestamp, self.period_ends())
            if abs(end - target) <= _YEAR_TOLERANCE
        ]
        return min(candidates)[1] if candidates else None

    def flow(self, metric: str, ref: pd.Timestamp) -> float | None:
        """TTM-Wert einer Stromgröße bis ``ref``."""
        if metric in self.quarterly.columns:
            col = self.quarterly[metric]
            window = col[(col.index > ref - _TTM_WINDOW) & (col.index <= ref)].dropna()
            if len(window) == 4 and abs(window.index[-1] - ref) <= _PERIOD_TOLERANCE:
                return float(window.sum())
        return _value_at(self.annual, metric, ref)

    def stock(self, metric: str, ref: pd.Timestamp) -> float | None:
        """Bestandsgröße am Periodenende ``ref``."""
        value = _value_at(self.quarterly, metric, ref)
        return value if value is not None else _value_at(self.annual, metric, ref)


def _wide(frame: pd.DataFrame, period_type: str) -> pd.DataFrame:
    part = frame[frame["period_type"] == period_type]
    if part.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    wide = part.pivot_table(index="period_end", columns="metric", values="value", aggfunc="last")
    wide.index = pd.DatetimeIndex(wide.index)
    return wide.sort_index()


def _value_at(wide: pd.DataFrame, metric: str, ref: pd.Timestamp) -> float | None:
    if metric not in wide.columns or wide.empty:
        return None
    near = [
        float(value)
        for end, value in wide[metric].dropna().items()
        if abs(pd.Timestamp(cast(Any, end)) - ref) <= _PERIOD_TOLERANCE
    ]
    return near[-1] if near else None


# ---------------------------------------------------------------------- Kurs


def _price_at(store: MarketStore, symbol: str, day: pd.Timestamp) -> float | None:
    bars = store.load_bars(
        symbol, start=(day - MAX_PRICE_AGE).date(), end=day.date(), adjusted=False
    )
    if bars.empty:
        return None
    return float(bars["close"].iloc[-1])


def _close_at(closes: pd.Series, day: pd.Timestamp) -> float | None:
    window = closes[(closes.index <= day) & (closes.index >= day - MAX_PRICE_AGE)]
    return None if window.empty else float(window.iloc[-1])


# ---------------------------------------------------------------------- Berechnung


@dataclass(frozen=True)
class _Snapshot:
    """Eingangswerte zu einem Periodenende."""

    revenue: float | None
    gross_profit: float | None
    operating_income: float | None
    net_income: float | None
    ebitda: float | None
    eps: float | None
    interest_expense: float | None
    operating_cashflow: float | None
    capex: float | None
    dividends: float | None
    total_assets: float | None
    total_liabilities: float | None
    total_equity: float | None
    total_debt: float | None
    long_term_debt: float | None
    cash: float | None
    current_assets: float | None
    current_liabilities: float | None
    inventory: float | None
    retained_earnings: float | None
    shares: float | None

    @classmethod
    def at(cls, data: _Fundamentals, ref: pd.Timestamp) -> _Snapshot:
        flow = data.flow
        stock = data.stock
        revenue = flow("revenue", ref)
        gross = flow("gross_profit", ref)
        if gross is None:
            gross = _sub(revenue, flow("cost_of_revenue", ref))
        return cls(
            revenue=revenue,
            gross_profit=gross,
            operating_income=flow("operating_income", ref),
            net_income=flow("net_income", ref),
            ebitda=flow("ebitda", ref),
            eps=flow("eps_diluted", ref),
            interest_expense=_abs(flow("interest_expense", ref)),
            operating_cashflow=flow("operating_cashflow", ref),
            capex=_abs(flow("capex", ref)),
            dividends=_abs(flow("dividends_paid", ref)),
            total_assets=stock("total_assets", ref),
            total_liabilities=stock("total_liabilities", ref),
            total_equity=stock("total_equity", ref),
            total_debt=stock("total_debt", ref),
            long_term_debt=stock("long_term_debt", ref),
            cash=stock("cash", ref),
            current_assets=stock("current_assets", ref),
            current_liabilities=stock("current_liabilities", ref),
            inventory=stock("inventory", ref),
            retained_earnings=stock("retained_earnings", ref),
            shares=stock("shares_outstanding", ref),
        )

    @property
    def free_cashflow(self) -> float | None:
        return _sub(self.operating_cashflow, self.capex)

    @property
    def roa(self) -> float | None:
        return _ratio(self.net_income, self.total_assets)

    @property
    def gross_margin(self) -> float | None:
        return _ratio(self.gross_profit, self.revenue)

    @property
    def current_ratio(self) -> float | None:
        return _ratio(self.current_assets, self.current_liabilities)

    @property
    def asset_turnover(self) -> float | None:
        return _ratio(self.revenue, self.total_assets)

    @property
    def leverage(self) -> float | None:
        return _ratio(self.long_term_debt, self.total_assets)


def _metrics(data: _Fundamentals, ref: pd.Timestamp | None, price: float | None) -> MetricValues:
    result: MetricValues = dict.fromkeys(METRIC_NAMES)
    result["price"] = price
    if ref is None:
        return result
    now = _Snapshot.at(data, ref)
    ref_1y = data.period_before(ref, 1)
    prev = _Snapshot.at(data, ref_1y) if ref_1y is not None else None

    def back(years: int, metric: str) -> float | None:
        earlier = data.period_before(ref, years)
        return None if earlier is None else data.flow(metric, earlier)

    market_cap = _mul(price, now.shares)
    if market_cap is not None and market_cap <= 0:
        market_cap = None
    enterprise_value = _add(_sub(market_cap, now.cash), now.total_debt)
    net_debt = _sub(now.total_debt, now.cash)
    pe = _ratio(price, now.eps)
    if pe is None and now.eps is None:
        pe = _ratio(market_cap, now.net_income)
    eps_growth = _growth(now.eps, prev.eps if prev else None)
    invested = _sub(_add(now.total_equity, now.total_debt), now.cash)
    nopat = _mul(now.operating_income, 1 - ROIC_TAX_RATE)
    inventory = now.inventory if now.inventory is not None else 0.0

    result.update(
        shares_outstanding=now.shares,
        market_cap=market_cap,
        enterprise_value=enterprise_value,
        pe_ratio=pe,
        ps_ratio=_ratio(market_cap, now.revenue),
        pb_ratio=_ratio(market_cap, now.total_equity),
        pcf_ratio=_ratio(market_cap, now.operating_cashflow),
        ev_ebitda=_ratio(enterprise_value, now.ebitda),
        peg_ratio=_ratio(pe, _mul(eps_growth, 100.0)),
        dividend_yield=_ratio(now.dividends, market_cap),
        payout_ratio=_ratio(now.dividends, now.net_income),
        dividend_growth_5y=_cagr(now.dividends, back(5, "dividends_paid"), 5, absolute=True),
        fcf_yield=_ratio(now.free_cashflow, market_cap),
        revenue_ttm=now.revenue,
        net_income_ttm=now.net_income,
        ebitda_ttm=now.ebitda,
        eps_ttm=now.eps,
        free_cashflow=now.free_cashflow,
        revenue_growth_yoy=_growth(now.revenue, prev.revenue if prev else None),
        revenue_growth_3y=_cagr(now.revenue, back(3, "revenue"), 3),
        earnings_growth_yoy=_growth(now.net_income, prev.net_income if prev else None),
        earnings_growth_3y=_cagr(now.net_income, back(3, "net_income"), 3),
        eps_growth_yoy=eps_growth,
        gross_margin=now.gross_margin,
        operating_margin=_ratio(now.operating_income, now.revenue),
        net_margin=_ratio(now.net_income, now.revenue),
        roe=_ratio(now.net_income, now.total_equity),
        roa=now.roa,
        roic=_ratio(nopat, invested),
        asset_turnover=now.asset_turnover,
        debt_to_equity=_ratio(now.total_debt, now.total_equity),
        net_debt_to_ebitda=_ratio(net_debt, now.ebitda),
        interest_coverage=_ratio(now.operating_income, now.interest_expense),
        current_ratio=now.current_ratio,
        quick_ratio=_ratio(_sub(now.current_assets, inventory), now.current_liabilities),
        share_change_yoy=_growth(now.shares, prev.shares if prev else None),
        piotroski_f_score=_piotroski(now, prev),
        altman_z_score=_altman(now, market_cap),
    )
    return result


def _piotroski(now: _Snapshot, prev: _Snapshot | None) -> float | None:
    """Piotroski-F-Score (0–9); ``None``, wenn ein Kriterium nicht prüfbar ist."""
    if prev is None:
        return None
    checks: list[Callable[[], bool | None]] = [
        lambda: _gt(now.roa, 0.0),
        lambda: _gt(now.operating_cashflow, 0.0),
        lambda: _gt(now.roa, prev.roa),
        lambda: _gt(now.operating_cashflow, now.net_income),
        lambda: _gt(prev.leverage, now.leverage),
        lambda: _gt(now.current_ratio, prev.current_ratio),
        lambda: None if now.shares is None or prev.shares is None else now.shares <= prev.shares,
        lambda: _gt(now.gross_margin, prev.gross_margin),
        lambda: _gt(now.asset_turnover, prev.asset_turnover),
    ]
    score = 0
    for check in checks:
        passed = check()
        if passed is None:
            return None
        score += int(passed)
    return float(score)


def _altman(now: _Snapshot, market_cap: float | None) -> float | None:
    """Altman-Z-Score (Originalformel für börsennotierte Industrieunternehmen)."""
    assets = now.total_assets
    working_capital = _sub(now.current_assets, now.current_liabilities)
    parts = [
        _ratio(working_capital, assets),
        _ratio(now.retained_earnings, assets),
        _ratio(now.operating_income, assets),
        _ratio(market_cap, now.total_liabilities),
        _ratio(now.revenue, assets),
    ]
    if any(p is None for p in parts):
        return None
    weights = (1.2, 1.4, 3.3, 0.6, 1.0)
    return float(sum(w * p for w, p in zip(weights, parts, strict=True) if p is not None))


# ---------------------------------------------------------------------- Rechenhilfen


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """``numerator / denominator``; ``None`` bei fehlendem Wert oder Nenner ≤ 0."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _growth(current: float | None, previous: float | None) -> float | None:
    """Relative Veränderung; ``None``, wenn der Vorwert fehlt oder nicht positiv ist."""
    if current is None or previous is None or previous <= 0:
        return None
    return current / previous - 1.0


def _cagr(
    current: float | None, previous: float | None, years: int, *, absolute: bool = False
) -> float | None:
    """Jährliche Wachstumsrate; ``None``, wenn ein Wert fehlt oder nicht positiv ist."""
    if absolute:
        current, previous = _abs(current), _abs(previous)
    if current is None or previous is None or current <= 0 or previous <= 0:
        return None
    return float((current / previous) ** (1.0 / years) - 1.0)


def _add(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a + b


def _sub(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _mul(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a * b


def _abs(value: float | None) -> float | None:
    return None if value is None else abs(value)


def _gt(a: float | None, b: float | None) -> bool | None:
    return None if a is None or b is None else a > b


def _to_day(value: DateLike) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()
