"""YFinanceProvider.fetch_fundamentals mit gespeicherten Abschlüssen (Fixtures)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError, YFTickerMissingError

from trading_helfer.data.fundamentals import METRICS, normalize_fundamentals
from trading_helfer.data.providers import (
    FUNDAMENTALS_COLUMNS,
    FundamentalsProvider,
    ProviderError,
)
from trading_helfer.data.providers import yfinance_provider as module
from trading_helfer.data.providers.rate_limit import TokenBucket
from trading_helfer.data.providers.yfinance_provider import YFinanceProvider

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "yfinance"
STATEMENTS = {
    "get_income_stmt": "income",
    "get_balance_sheet": "balance",
    "get_cash_flow": "cashflow",
}


def load_statement(statement: str, freq: str) -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / f"SYNTH_{statement}_{freq}.csv", index_col=0)
    df.columns = pd.to_datetime(df.columns)
    return df


class StatementTicker:
    """Attrappe für die Abschluss-Methoden von ``yfinance.Ticker``."""

    def __init__(self, answers: dict[tuple[str, str], Any] | None = None) -> None:
        self.answers = answers
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _answer(self, method: str, kwargs: dict[str, Any]) -> Any:
        self.calls.append((method, kwargs))
        freq = kwargs["freq"]
        if self.answers is None:
            return load_statement(STATEMENTS[method], freq)
        value = self.answers.get((method, freq), pd.DataFrame())
        if isinstance(value, BaseException):
            raise value
        return value

    def get_income_stmt(self, *args: Any, **kwargs: Any) -> Any:
        return self._answer("get_income_stmt", kwargs)

    def get_balance_sheet(self, *args: Any, **kwargs: Any) -> Any:
        return self._answer("get_balance_sheet", kwargs)

    def get_cash_flow(self, *args: Any, **kwargs: Any) -> Any:
        return self._answer("get_cash_flow", kwargs)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_provider(ticker: StatementTicker, clock: Clock | None = None) -> YFinanceProvider:
    clock = clock or Clock()
    return YFinanceProvider(
        ticker_factory=lambda _symbol: ticker,  # type: ignore[arg-type,return-value]
        limiter=TokenBucket(clock=clock, sleep=clock.sleep),
        sleep=clock.sleep,
    )


def value(result: pd.DataFrame, period_end: str, period_type: str, metric: str) -> float:
    rows = result[
        (result["period_end"] == pd.Timestamp(period_end))
        & (result["period_type"] == period_type)
        & (result["metric"] == metric)
    ]
    assert len(rows) == 1, (period_end, period_type, metric)
    return float(rows["value"].iloc[0])


def has(result: pd.DataFrame, period_type: str, metric: str) -> bool:
    return bool(((result["period_type"] == period_type) & (result["metric"] == metric)).any())


@pytest.fixture
def result() -> pd.DataFrame:
    return make_provider(StatementTicker()).fetch_fundamentals("SYNTH")


def test_mapping_covers_all_metrics() -> None:
    mapped = [m for metrics in module._STATEMENT_METRICS.values() for m in metrics]
    assert sorted(mapped) == sorted(METRICS)


def test_provider_satisfies_protocol() -> None:
    assert isinstance(make_provider(StatementTicker()), FundamentalsProvider)


def test_requests_all_statements_quarterly_and_yearly() -> None:
    ticker = StatementTicker()
    make_provider(ticker).fetch_fundamentals("brk.b")
    assert sorted((m, kw["freq"]) for m, kw in ticker.calls) == sorted(
        (m, f) for m in STATEMENTS for f in ("quarterly", "yearly")
    )
    assert all(kw["pretty"] is False for _, kw in ticker.calls)


def test_fixture_normalized_to_fixed_names(result: pd.DataFrame) -> None:
    assert list(result.columns) == list(FUNDAMENTALS_COLUMNS)
    assert set(result["metric"]) <= set(METRICS)
    assert set(result["period_type"]) == {"Q", "FY"}
    # Ergebnis ist bereits normiert.
    pd.testing.assert_frame_equal(result, normalize_fundamentals(result))
    # Jahreswerte: alle 22 Metriken für 2023 vorhanden.
    fy_2023 = result[
        (result["period_type"] == "FY") & (result["period_end"] == pd.Timestamp("2023-12-31"))
    ]
    assert sorted(fy_2023["metric"]) == sorted(METRICS)
    assert value(result, "2023-12-31", "FY", "revenue") == 1_000_000_000
    assert value(result, "2023-12-31", "FY", "eps_diluted") == 1.5
    assert value(result, "2023-12-31", "FY", "total_liabilities") == 600_000_000
    assert value(result, "2023-12-31", "FY", "total_equity") == 900_000_000
    assert value(result, "2023-12-31", "FY", "shares_outstanding") == 100_000_000
    # Mittelabflüsse wie gemeldet negativ.
    assert value(result, "2023-12-31", "FY", "capex") == -60_000_000
    assert value(result, "2022-12-31", "FY", "dividends_paid") == -25_000_000


def test_fallback_names(result: pd.DataFrame) -> None:
    assert value(result, "2024-03-31", "Q", "revenue") == 270_000_000  # OperatingRevenue
    assert value(result, "2024-03-31", "Q", "total_equity") == 930_000_000
    assert value(result, "2024-03-31", "Q", "shares_outstanding") == 99_000_000
    # Kasse: Ersatzname nur dort, wo der Hauptname in der Periode fehlt.
    assert value(result, "2024-03-31", "Q", "cash") == 170_000_000
    assert value(result, "2023-12-31", "Q", "cash") == 150_000_000


def test_missing_metrics_are_absent_not_zero(result: pd.DataFrame) -> None:
    assert not has(result, "Q", "ebitda")
    assert not has(result, "Q", "inventory")
    assert not has(result, "Q", "dividends_paid")
    # Einzelner fehlender Wert und leere Spalte 2021.
    interest = result[(result["metric"] == "interest_expense")]
    assert interest["period_end"].tolist() == [pd.Timestamp("2023-12-31")]
    assert pd.Timestamp("2021-12-31") not in set(result["period_end"])
    assert result["value"].notna().all()


def test_conservative_reported_dates(result: pd.DataFrame) -> None:
    lag = (result["reported_date"] - result["period_end"]).dt.days
    assert set(lag[result["period_type"] == "Q"]) == {45}
    assert set(lag[result["period_type"] == "FY"]) == {75}


def test_empty_statements_give_empty_result() -> None:
    result = make_provider(StatementTicker({})).fetch_fundamentals("SYNTH")
    assert result.empty
    assert list(result.columns) == list(FUNDAMENTALS_COLUMNS)


def test_unknown_symbol_gives_empty_result() -> None:
    ticker = StatementTicker({("get_income_stmt", "quarterly"): YFTickerMissingError("X", "weg")})
    result = make_provider(ticker).fetch_fundamentals("XXXX")
    assert result.empty
    assert len(ticker.calls) == 1


def test_only_unmapped_rows_give_empty_result() -> None:
    raw = pd.DataFrame({pd.Timestamp("2023-12-31"): [1.0]}, index=["ResearchAndDevelopment"])
    ticker = StatementTicker({("get_income_stmt", "yearly"): raw})
    assert make_provider(ticker).fetch_fundamentals("SYNTH").empty


def test_duplicate_row_uses_first() -> None:
    raw = pd.DataFrame(
        {pd.Timestamp("2023-12-31"): [5.0, 7.0]}, index=["TotalRevenue", "TotalRevenue"]
    )
    ticker = StatementTicker({("get_income_stmt", "yearly"): raw})
    result = make_provider(ticker).fetch_fundamentals("SYNTH")
    assert result["value"].tolist() == [5.0]


def test_retries_then_provider_error() -> None:
    clock = Clock()
    ticker = StatementTicker({("get_income_stmt", "quarterly"): YFRateLimitError()})
    with pytest.raises(ProviderError, match="4 Versuchen"):
        make_provider(ticker, clock).fetch_fundamentals("SYNTH")
    assert clock.sleeps.count(2.0) >= 1
    assert len(ticker.calls) == 4


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ("kaputt", "Unerwartete Antwort"),
        (pd.DataFrame({"keinDatum": [1.0]}, index=["TotalRevenue"]), "Unerwartete Perioden"),
    ],
)
def test_unexpected_answers(answer: Any, message: str) -> None:
    ticker = StatementTicker({("get_income_stmt", "quarterly"): answer})
    with pytest.raises(ProviderError, match=message):
        make_provider(ticker).fetch_fundamentals("SYNTH")
