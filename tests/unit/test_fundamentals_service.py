"""Fassade des Kennzahlen-Tabs: Formatierung, Sektor-Vergleich, Stichtag und Verlauf."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from trading_helfer.data.storage import MarketStore
from trading_helfer.fundamentals.metrics import METRIC_NAMES, compute_metrics
from trading_helfer.services.fundamentals_service import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    Comparison,
    FundamentalsService,
    compare,
    empty_history,
    format_metric,
)

QUARTERS = pd.date_range("2021-03-31", periods=12, freq="QE")
REPORT_LAG = pd.Timedelta(days=30)


def synthetic_fundamentals(scale: float, growth: float) -> pd.DataFrame:
    """Synthetisches Unternehmen: Quartalszahlen mit stetigem Wachstum."""
    rows: list[dict[str, object]] = []
    for i, end in enumerate(QUARTERS):
        factor = scale * (1.0 + growth) ** (i / 4)
        values = {
            "revenue": 100.0 * factor,
            "gross_profit": 40.0 * factor,
            "operating_income": 20.0 * factor,
            "net_income": 15.0 * factor,
            "ebitda": 25.0 * factor,
            "eps_diluted": 0.15 * factor,
            "operating_cashflow": 20.0 * factor,
            "capex": -5.0 * factor,
            "total_assets": 1000.0 * scale,
            "total_liabilities": 600.0 * scale,
            "total_equity": 400.0 * scale,
            "total_debt": 300.0 * scale,
            "cash": 100.0 * scale,
            "current_assets": 300.0 * scale,
            "current_liabilities": 200.0 * scale,
            "shares_outstanding": 100.0 * scale,
        }
        rows += [
            {
                "period_end": end,
                "period_type": "Q",
                "metric": metric,
                "value": value,
                "reported_date": end + REPORT_LAG,
            }
            for metric, value in values.items()
        ]
    return pd.DataFrame(rows)


def synthetic_bars(price: float) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-01", "2024-03-29")
    closes = [price] * len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * len(dates),
        }
    )


COMPANIES = {
    # Symbol: (Sektor, Größe, Wachstum p. a., Kurs)
    "AAA": ("Technologie", 1.0, 0.30, 20.0),
    "BBB": ("Technologie", 2.0, 0.10, 15.0),
    "CCC": ("Technologie", 1.5, 0.00, 30.0),
    "DDD": ("Versorger", 1.0, 0.05, 10.0),
    "NOS": (None, 1.0, 0.05, 12.0),
}


def fill(store: MarketStore) -> None:
    store.upsert_instruments(
        pd.DataFrame(
            {
                "symbol": list(COMPANIES),
                "sector": [c[0] for c in COMPANIES.values()],
            }
        )
    )
    for symbol, (_sector, scale, growth, price) in COMPANIES.items():
        store.upsert_fundamentals(symbol, synthetic_fundamentals(scale, growth))
        store.upsert_bars(symbol, synthetic_bars(price))


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "markt.duckdb"
    with MarketStore(path) as store:
        fill(store)
    return path


@pytest.fixture
def service(db_path: Path) -> FundamentalsService:
    return FundamentalsService(db_path)


AS_OF = dt.date(2024, 3, 1)


# ---------------------------------------------------------------------- Formatierung


@pytest.mark.parametrize(
    ("metric", "value", "expected"),
    [
        ("pe_ratio", 18.4, "18,40"),
        ("pe_ratio", 1234.5, "1.234,50"),
        ("gross_margin", 0.253, "25,3 %"),
        ("share_change_yoy", -0.012, "-1,2 %"),
        ("price", 182.5, "182,50 $"),
        ("market_cap", 2.51e12, "2,51 Bio. $"),
        ("revenue_ttm", 3.4e9, "3,40 Mrd. $"),
        ("free_cashflow", -12.5e6, "-12,50 Mio. $"),
        ("free_cashflow", 950.0, "950 $"),
        ("shares_outstanding", 15.2e9, "15,20 Mrd."),
        ("piotroski_f_score", 7.0, "7"),
        ("altman_z_score", 3.456, "3,46"),
        ("pe_ratio", None, "–"),
        ("pe_ratio", math.nan, "–"),
        ("gross_margin", math.inf, "–"),
    ],
)
def test_format_metric(metric: str, value: float | None, expected: str) -> None:
    assert format_metric(metric, value) == expected


def test_compare_directions() -> None:
    assert compare("roe", 0.2, 0.1) is Comparison.BETTER
    assert compare("roe", 0.05, 0.1) is Comparison.WORSE
    assert compare("pe_ratio", 10.0, 20.0) is Comparison.BETTER
    assert compare("pe_ratio", 30.0, 20.0) is Comparison.WORSE
    assert compare("pe_ratio", 20.0, 20.0) is Comparison.EQUAL
    assert compare("market_cap", 1e9, 2e9) is None  # ohne Richtung
    assert compare("roe", None, 0.1) is None
    assert compare("roe", 0.1, None) is None
    assert compare("roe", math.nan, 0.1) is None


def test_directions_are_known_metrics_and_disjoint() -> None:
    assert set(METRIC_NAMES) >= HIGHER_IS_BETTER | LOWER_IS_BETTER
    assert not HIGHER_IS_BETTER & LOWER_IS_BETTER


def test_empty_history_columns() -> None:
    frame = empty_history()
    assert frame.empty
    assert list(frame.columns) == ["period_end", "as_of", *METRIC_NAMES]
    assert frame["period_end"].dtype == "datetime64[ns]"


# ---------------------------------------------------------------------- Service


def test_load_fills_all_metrics_with_sector_median(
    service: FundamentalsService, db_path: Path
) -> None:
    data = service.load("aaa", AS_OF)
    assert data.symbol == "AAA"
    assert data.as_of == AS_OF
    assert data.sector == "Technologie"
    assert data.peer_count == 3
    assert data.has_values
    assert [r.metric for r in data.rows] == list(METRIC_NAMES)

    with MarketStore(db_path, read_only=True) as store:
        own = compute_metrics("AAA", AS_OF, store=store)
        peers = [compute_metrics(s, AS_OF, store=store) for s in ("AAA", "BBB", "CCC")]
    growth = data.row("revenue_growth_yoy")
    assert growth.value == pytest.approx(own["revenue_growth_yoy"])
    expected_median = sorted(p["revenue_growth_yoy"] or 0.0 for p in peers)[1]
    assert growth.sector_median == pytest.approx(expected_median)
    assert growth.comparison is Comparison.BETTER  # 30 % Wachstum ist das höchste
    assert data.row("pe_ratio").comparison is not None
    assert data.row("market_cap").comparison is None


def test_slowest_grower_is_worse(service: FundamentalsService) -> None:
    data = service.load("CCC", AS_OF)
    assert data.row("revenue_growth_yoy").comparison is Comparison.WORSE


def test_missing_values_stay_none(service: FundamentalsService) -> None:
    data = service.load("AAA", AS_OF)
    # Keine Dividenden und keine Zinsen in den Daten → nicht berechenbar.
    assert data.row("dividend_yield").value is None
    assert data.row("interest_coverage").value is None
    assert data.row("interest_coverage").comparison is None


def test_symbol_without_sector_has_no_comparison(service: FundamentalsService) -> None:
    data = service.load("NOS", AS_OF)
    assert data.sector is None
    assert data.peer_count == 0
    assert all(r.sector_median is None and r.comparison is None for r in data.rows)
    assert data.has_values


def test_unknown_symbol_is_empty(service: FundamentalsService) -> None:
    data = service.load("XYZ", AS_OF)
    assert data.sector is None
    assert not data.has_values
    assert data.history.empty


def test_as_of_limits_values_and_history(service: FundamentalsService) -> None:
    early = service.load("AAA", dt.date(2022, 6, 1))
    late = service.load("AAA", AS_OF)
    assert early.history["as_of"].max() <= pd.Timestamp("2022-06-01")
    assert len(early.history) < len(late.history)
    assert len(late.history) == len(QUARTERS)
    assert early.row("revenue_ttm").value != late.row("revenue_ttm").value
    before_any = service.load("AAA", dt.date(2020, 1, 1))
    assert not before_any.has_values
    assert before_any.history.empty


def test_caches_are_reused_and_invalidated(service: FundamentalsService) -> None:
    first = service.load("AAA", AS_OF)
    second = service.load("AAA", AS_OF)
    assert first.history is not second.history  # gefiltert, aber gleiche Werte
    pd.testing.assert_frame_equal(first.history, second.history)
    assert len(service._history) == 1
    assert len(service._peers) == 1
    service.invalidate()
    assert not service._history
    assert not service._peers


def test_cache_sizes_are_bounded(service: FundamentalsService) -> None:
    for day in range(1, 12):
        service.load("AAA", dt.date(2024, 1, day))
    assert len(service._peers) <= 8
    for symbol in COMPANIES:
        service.load(symbol, AS_OF)
    service.load("AAA", AS_OF)
    assert next(reversed(service._history)) == "AAA"


def test_missing_database_gives_empty_data(tmp_path: Path) -> None:
    for service in (FundamentalsService(), FundamentalsService(tmp_path / "fehlt.duckdb")):
        data = service.load("AAA", AS_OF)
        assert not data.has_values
        assert all(r.value is None for r in data.rows)
        assert data.history.empty


def test_custom_opener() -> None:
    @contextmanager
    def opener() -> Iterator[MarketStore | None]:
        with MarketStore(":memory:") as store:
            fill(store)
            yield store

    data = FundamentalsService(opener=opener).load("BBB", AS_OF)
    assert data.sector == "Technologie"
    assert data.has_values


def test_row_unknown_metric_raises(service: FundamentalsService) -> None:
    with pytest.raises(KeyError):
        service.load("AAA", AS_OF).row("gibt_es_nicht")
