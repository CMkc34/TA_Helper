"""Kennzahlen „point in time“ an einem handgerechneten Beispielunternehmen."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest

from trading_helfer.data.storage import MarketStore
from trading_helfer.fundamentals.metrics import (
    METRIC_LABELS,
    METRIC_NAMES,
    ROIC_TAX_RATE,
    compute_metrics,
    metrics_history,
    sector_median,
)

QUARTERS_2022 = ["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"]
QUARTERS_2023 = ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"]

# Stromgrößen je Quartal (in beiden Jahren jeweils vier gleiche Quartale).
FLOWS_2022 = {
    "revenue": 100.0,
    "cost_of_revenue": 60.0,  # Bruttogewinn wird abgeleitet: 40 je Quartal
    "operating_income": 20.0,
    "net_income": 15.0,
    "ebitda": 25.0,
    "eps_diluted": 0.15,
    "interest_expense": -2.0,
    "operating_cashflow": 20.0,
    "capex": -5.0,
    "dividends_paid": -3.0,
}
FLOWS_2023 = {
    "revenue": 125.0,
    "gross_profit": 55.0,
    "operating_income": 30.0,
    "net_income": 20.0,
    "ebitda": 35.0,
    "eps_diluted": 0.21,
    "interest_expense": -3.0,
    "operating_cashflow": 30.0,
    "capex": -6.0,
    "dividends_paid": -4.0,
}
# Bilanz jeweils zum Jahresende.
BALANCE_2022 = {
    "total_assets": 1000.0,
    "total_liabilities": 600.0,
    "total_equity": 400.0,
    "total_debt": 300.0,
    "long_term_debt": 250.0,
    "cash": 100.0,
    "current_assets": 300.0,
    "current_liabilities": 200.0,
    "inventory": 50.0,
    "retained_earnings": 200.0,
    "shares_outstanding": 100.0,
}
BALANCE_2023 = {
    "total_assets": 1100.0,
    "total_liabilities": 610.0,
    "total_equity": 490.0,
    "total_debt": 280.0,
    "long_term_debt": 220.0,
    "cash": 150.0,
    "current_assets": 360.0,
    "current_liabilities": 200.0,
    "inventory": 60.0,
    "retained_earnings": 260.0,
    "shares_outstanding": 102.0,
}
REPORT_LAG = pd.Timedelta(days=30)  # Q4 2023 wird am 2024-01-30 veröffentlicht


def rows(
    period_end: str,
    values: dict[str, float],
    period_type: str = "Q",
    lag: pd.Timedelta = REPORT_LAG,
) -> list[dict[str, object]]:
    end = pd.Timestamp(period_end)
    return [
        {
            "period_end": end,
            "period_type": period_type,
            "metric": metric,
            "value": value,
            "reported_date": end + lag,
        }
        for metric, value in values.items()
    ]


def example_fundamentals() -> pd.DataFrame:
    data: list[dict[str, object]] = []
    for end in QUARTERS_2022:
        values = dict(FLOWS_2022)
        if end.endswith("12-31"):
            values |= BALANCE_2022
        data += rows(end, values)
    for end in QUARTERS_2023:
        values = dict(FLOWS_2023)
        if end.endswith("12-31"):
            values |= BALANCE_2023
        data += rows(end, values)
    return pd.DataFrame(data)


def bars(closes: dict[str, float]) -> pd.DataFrame:
    dates = pd.to_datetime(list(closes))
    values = list(closes.values())
    return pd.DataFrame(
        {
            "date": dates,
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": [1000] * len(values),
            # Dividendenbereinigung darf die Marktkapitalisierung nicht verändern.
            "adj_close": [v * 0.9 for v in values],
        }
    )


@pytest.fixture
def store() -> Iterator[MarketStore]:
    with MarketStore(":memory:") as market_store:
        market_store.upsert_fundamentals("EXM", example_fundamentals())
        market_store.upsert_bars(
            "EXM",
            bars(
                {
                    "2023-10-30": 15.0,
                    "2024-01-29": 18.0,
                    "2024-02-01": 20.0,
                    "2024-03-01": 99.0,  # nach dem Stichtag
                }
            ),
        )
        yield market_store


# --- Beispielunternehmen --------------------------------------------------------------


def test_result_has_all_metric_names(store: MarketStore) -> None:
    result = compute_metrics("EXM", "2024-02-05", store=store)
    assert tuple(result) == METRIC_NAMES
    assert set(METRIC_NAMES) == set(METRIC_LABELS)
    assert all(isinstance(label, str) and label for label in METRIC_LABELS.values())


def test_example_company_hand_calculated(store: MarketStore) -> None:
    m = compute_metrics("exm", "2024-02-05", store=store)

    # Kurs 20 (unbereinigt), 102 Aktien
    market_cap = 20.0 * 102.0
    enterprise_value = market_cap - 150.0 + 280.0
    expected = {
        "price": 20.0,
        "shares_outstanding": 102.0,
        "market_cap": 2040.0,
        "enterprise_value": 2170.0,
        # TTM 2023: Umsatz 500, Bruttogewinn 220, EBIT 120, Gewinn 80, EBITDA 140,
        # EPS 0,84, Zinsen 12, OCF 120, Capex 24, Dividenden 16
        "revenue_ttm": 500.0,
        "net_income_ttm": 80.0,
        "ebitda_ttm": 140.0,
        "eps_ttm": 0.84,
        "free_cashflow": 96.0,
        "pe_ratio": 20.0 / 0.84,
        "ps_ratio": market_cap / 500.0,
        "pb_ratio": market_cap / 490.0,
        "pcf_ratio": market_cap / 120.0,
        "ev_ebitda": enterprise_value / 140.0,
        "dividend_yield": 16.0 / market_cap,
        "payout_ratio": 0.2,
        "fcf_yield": 96.0 / market_cap,
        # Vorjahr (TTM 2022): Umsatz 400, Gewinn 60, EPS 0,60
        "revenue_growth_yoy": 0.25,
        "earnings_growth_yoy": 80.0 / 60.0 - 1.0,
        "eps_growth_yoy": 0.4,
        "peg_ratio": (20.0 / 0.84) / 40.0,
        "gross_margin": 0.44,
        "operating_margin": 0.24,
        "net_margin": 0.16,
        "roe": 80.0 / 490.0,
        "roa": 80.0 / 1100.0,
        "roic": 120.0 * (1 - ROIC_TAX_RATE) / (490.0 + 280.0 - 150.0),
        "asset_turnover": 500.0 / 1100.0,
        "debt_to_equity": 280.0 / 490.0,
        "net_debt_to_ebitda": 130.0 / 140.0,
        "interest_coverage": 10.0,
        "current_ratio": 1.8,
        "quick_ratio": 1.5,
        "share_change_yoy": 0.02,
        # alle Kriterien erfüllt bis auf „keine neuen Aktien“ (100 → 102)
        "piotroski_f_score": 8.0,
        "altman_z_score": (
            1.2 * 160.0 / 1100.0
            + 1.4 * 260.0 / 1100.0
            + 3.3 * 120.0 / 1100.0
            + 0.6 * market_cap / 610.0
            + 1.0 * 500.0 / 1100.0
        ),
    }
    for name, value in expected.items():
        assert m[name] == pytest.approx(value), name
    # Für 3 bzw. 5 Jahre fehlt die Historie.
    assert m["revenue_growth_3y"] is None
    assert m["earnings_growth_3y"] is None
    assert m["dividend_growth_5y"] is None


def test_piotroski_counts_every_criterion() -> None:
    with MarketStore(":memory:") as store:
        frame = example_fundamentals()
        is_shares_2023 = (frame["metric"] == "shares_outstanding") & (
            frame["period_end"] == pd.Timestamp("2023-12-31")
        )
        frame.loc[is_shares_2023, "value"] = 98.0
        store.upsert_fundamentals("EXM", frame)
        m = compute_metrics("EXM", "2024-02-05", store=store)
    assert m["piotroski_f_score"] == 9.0
    assert m["share_change_yoy"] == pytest.approx(-0.02)
    assert m["price"] is None  # keine Kurse gespeichert
    assert m["market_cap"] is None
    assert m["pe_ratio"] is None
    assert m["altman_z_score"] is None


# --- point in time / Lookahead ---------------------------------------------------------


def test_no_lookahead_before_reported_date(store: MarketStore) -> None:
    # Q4 2023 erscheint am 2024-01-30; am Vortag gilt Q3 2023 als jüngste Periode.
    m = compute_metrics("EXM", "2024-01-29", store=store)
    assert m["price"] == 18.0
    assert m["revenue_ttm"] == pytest.approx(100.0 + 3 * 125.0)
    assert m["eps_ttm"] == pytest.approx(0.15 + 3 * 0.21)
    # TTM zu Q3 2022 bräuchte Q4 2021 → kein Vorjahresvergleich
    assert m["revenue_growth_yoy"] is None
    # Bilanzwerte gibt es nur zum Jahresende → zu Q3 nicht berechenbar
    assert m["shares_outstanding"] is None
    assert m["market_cap"] is None
    assert m["pe_ratio"] == pytest.approx(18.0 / 0.78)

    later = compute_metrics("EXM", "2024-01-30", store=store)
    assert later["revenue_ttm"] == pytest.approx(500.0)
    assert later["shares_outstanding"] == 102.0


def test_later_prices_are_ignored(store: MarketStore) -> None:
    assert compute_metrics("EXM", "2024-02-05", store=store)["price"] == 20.0
    assert compute_metrics("EXM", "2024-03-01", store=store)["price"] == 99.0


def test_stale_price_is_none(store: MarketStore) -> None:
    m = compute_metrics("EXM", "2024-02-20", store=store)
    assert m["price"] is None
    assert m["market_cap"] is None


def test_before_first_report_everything_is_none(store: MarketStore) -> None:
    m = compute_metrics("EXM", "2022-04-01", store=store)
    assert all(value is None for value in m.values())


def test_timezone_aware_as_of(store: MarketStore) -> None:
    as_of = pd.Timestamp("2024-02-05 10:00", tz="UTC")
    assert compute_metrics("EXM", as_of, store=store)["market_cap"] == pytest.approx(2040.0)


# --- Verlustfall und Division durch null ------------------------------------------------


def test_loss_company_gives_none() -> None:
    frame = example_fundamentals()
    with MarketStore(":memory:") as store:
        loss = frame.copy()
        is_2022 = loss["period_end"] < pd.Timestamp("2023-01-01")
        for metric, value in {"net_income": -5.0, "eps_diluted": -0.05}.items():
            loss.loc[loss["metric"] == metric, "value"] = value
        # 2022 mit Verlust, 2023 ebenfalls; Eigenkapital 0, Zinsen 0, kein EBITDA
        loss.loc[(loss["metric"] == "total_equity") & ~is_2022, "value"] = 0.0
        loss.loc[loss["metric"] == "interest_expense", "value"] = 0.0
        loss.loc[loss["metric"] == "ebitda", "value"] = -1.0
        store.upsert_fundamentals("LOSS", loss)
        store.upsert_bars("LOSS", bars({"2024-02-01": 20.0}))
        m = compute_metrics("LOSS", "2024-02-05", store=store)

    assert m["net_income_ttm"] == pytest.approx(-20.0)
    assert m["eps_ttm"] == pytest.approx(-0.2)
    for name in (
        "pe_ratio",
        "peg_ratio",
        "payout_ratio",
        "earnings_growth_yoy",
        "eps_growth_yoy",
        "pb_ratio",
        "roe",
        "debt_to_equity",
        "ev_ebitda",
        "net_debt_to_ebitda",
        "interest_coverage",
    ):
        assert m[name] is None, name
    assert m["net_margin"] == pytest.approx(-20.0 / 500.0)
    assert m["roa"] == pytest.approx(-20.0 / 1100.0)
    assert m["piotroski_f_score"] is not None
    assert m["piotroski_f_score"] <= 7.0


def test_pe_falls_back_to_net_income_without_eps() -> None:
    frame = example_fundamentals()
    with MarketStore(":memory:") as other:
        other.upsert_fundamentals("NOEPS", frame[frame["metric"] != "eps_diluted"])
        other.upsert_bars("NOEPS", bars({"2024-02-01": 20.0}))
        m = compute_metrics("NOEPS", "2024-02-05", store=other)
    assert m["eps_ttm"] is None
    assert m["pe_ratio"] == pytest.approx(2040.0 / 80.0)
    assert m["peg_ratio"] is None


def test_missing_inventory_counts_as_zero_for_quick_ratio() -> None:
    frame = example_fundamentals()
    with MarketStore(":memory:") as store:
        store.upsert_fundamentals("SOFT", frame[frame["metric"] != "inventory"])
        m = compute_metrics("SOFT", "2024-02-05", store=store)
    assert m["quick_ratio"] == pytest.approx(1.8)


# --- TTM aus Jahreszahlen, mehrjähriges Wachstum ----------------------------------------


def annual_store(store: MarketStore) -> None:
    data: list[dict[str, object]] = []
    revenue = {2018: 100.0, 2019: 110.0, 2020: 121.0, 2021: 133.1, 2022: 146.41, 2023: 161.051}
    net_income = {2018: -1.0, 2019: 8.0, 2020: 10.0, 2021: 12.0, 2022: 14.0, 2023: 16.0}
    dividends = {2018: -2.0, 2019: -2.5, 2020: -3.0, 2021: -3.5, 2022: -4.0, 2023: -2.0 * 2**5}
    for year in revenue:
        data += rows(
            f"{year}-12-31",
            {
                "revenue": revenue[year],
                "net_income": net_income[year],
                "dividends_paid": dividends[year],
            },
            period_type="FY",
            lag=pd.Timedelta(days=60),
        )
    store.upsert_fundamentals("ANN", pd.DataFrame(data))


def test_annual_only_data_uses_fiscal_year_as_ttm() -> None:
    with MarketStore(":memory:") as store:
        annual_store(store)
        m = compute_metrics("ANN", "2024-03-01", store=store)
        early = compute_metrics("ANN", "2022-03-01", store=store)
    assert m["revenue_ttm"] == pytest.approx(161.051)
    assert m["revenue_growth_yoy"] == pytest.approx(0.1)
    assert m["revenue_growth_3y"] == pytest.approx(0.1)
    assert m["earnings_growth_3y"] == pytest.approx((16.0 / 10.0) ** (1 / 3) - 1)
    assert m["dividend_growth_5y"] == pytest.approx(1.0)
    # Basis 2018 mit Verlust → kein 3-Jahres-Wachstum
    assert early["revenue_growth_3y"] == pytest.approx(0.1)
    assert early["earnings_growth_3y"] is None


def test_incomplete_quarters_fall_back_to_annual_value() -> None:
    data = rows("2023-12-31", {"revenue": 400.0}, period_type="FY")
    for end in QUARTERS_2023[1:]:
        data += rows(end, {"revenue": 125.0})
    with MarketStore(":memory:") as store:
        store.upsert_fundamentals("MIX", pd.DataFrame(data))
        m = compute_metrics("MIX", "2024-02-05", store=store)
    assert m["revenue_ttm"] == pytest.approx(400.0)


# --- Verlauf ---------------------------------------------------------------------------


def test_metrics_history_per_quarter(store: MarketStore) -> None:
    history = metrics_history("EXM", store=store)
    assert list(history.columns) == ["period_end", "as_of", *METRIC_NAMES]
    assert list(history["period_end"].dt.strftime("%Y-%m-%d")) == QUARTERS_2022 + QUARTERS_2023
    assert history["as_of"].iloc[-1] == pd.Timestamp("2024-01-30")
    # Die ersten drei Quartale haben noch kein TTM.
    assert history["revenue_ttm"].iloc[:3].isna().all()
    assert history["revenue_ttm"].iloc[3] == pytest.approx(400.0)
    assert history["revenue_ttm"].iloc[-1] == pytest.approx(500.0)
    # Q3 2023 (veröffentlicht 2023-10-30): Kurs 15
    assert history["price"].iloc[6] == 15.0
    assert history["pe_ratio"].iloc[6] == pytest.approx(15.0 / 0.78)
    # Q4 2023 (veröffentlicht 2024-01-30): Kurs vom 2024-01-29, Lookahead ausgeschlossen
    assert history["price"].iloc[-1] == 18.0
    assert history["market_cap"].iloc[-1] == pytest.approx(18.0 * 102.0)
    assert history["piotroski_f_score"].iloc[-1] == 8.0


def test_metrics_history_annual_only_and_empty() -> None:
    with MarketStore(":memory:") as store:
        annual_store(store)
        history = metrics_history("ANN", store=store)
        empty = metrics_history("NONE", store=store)
    assert len(history) == 6
    assert history["revenue_growth_3y"].iloc[-1] == pytest.approx(0.1)
    assert empty.empty
    assert list(empty.columns) == ["period_end", "as_of", *METRIC_NAMES]
    assert empty["period_end"].dtype == "datetime64[ns]"


# --- Sektor-Median ---------------------------------------------------------------------


def test_sector_median(store: MarketStore) -> None:
    frame = example_fundamentals()
    for symbol, factor in {"TWO": 2.0, "THREE": 3.0, "NOPRICE": 1.0}.items():
        store.upsert_fundamentals(symbol, frame)
        if symbol != "NOPRICE":
            store.upsert_bars(symbol, bars({"2024-02-01": 20.0 * factor}))
    store.upsert_instruments(
        pd.DataFrame(
            {
                "symbol": ["EXM", "TWO", "THREE", "NOPRICE", "NOSECTOR"],
                "sector": ["Technology", "Technology", "Technology", "Energy", None],
            }
        )
    )
    medians = sector_median("pe_ratio", "2024-02-05", store=store)
    assert list(medians) == ["Energy", "Technology"]
    assert medians["Technology"] == pytest.approx(40.0 / 0.84)
    assert medians["Energy"] is None

    subset = sector_median("pe_ratio", "2024-02-05", store=store, symbols=[" exm ", "three"])
    assert subset == {"Technology": pytest.approx(40.0 / 0.84)}
    assert sector_median("revenue_ttm", "2024-02-05", store=store)["Energy"] == 500.0


def test_sector_median_rejects_unknown_metric(store: MarketStore) -> None:
    with pytest.raises(ValueError, match="Kennzahl"):
        sector_median("kgv", "2024-02-05", store=store)
