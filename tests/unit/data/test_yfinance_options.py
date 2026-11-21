"""YFinanceProvider.fetch_option_chain mit gespeicherten Antworten statt Netzwerk."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError, YFTickerMissingError

from trading_helfer.data.providers import (
    OPTION_CHAIN_COLUMNS,
    OptionChainProvider,
    ProviderError,
    empty_option_chain,
)
from trading_helfer.data.providers.rate_limit import RetryPolicy, TokenBucket
from trading_helfer.data.providers.yfinance_provider import YFinanceProvider

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "yfinance"
META = json.loads((FIXTURES / "SYNTH_options.json").read_text(encoding="utf-8"))


def chain_for(expiry: str, underlying: Any = None) -> SimpleNamespace:
    calls = pd.read_csv(FIXTURES / f"SYNTH_options_{expiry}_calls.csv")
    puts = pd.read_csv(FIXTURES / f"SYNTH_options_{expiry}_puts.csv")
    return SimpleNamespace(
        calls=calls, puts=puts, underlying=META["underlying"] if underlying is None else underlying
    )


class OptionTicker:
    """Attrappe für ``yfinance.Ticker`` mit Optionsketten aus Fixtures."""

    def __init__(self, options: Any = None, chains: dict[str, Any] | None = None) -> None:
        self._options = tuple(META["expirations"]) if options is None else options
        self._chains = chains
        self.chain_calls: list[str] = []

    @property
    def options(self) -> Any:
        if isinstance(self._options, BaseException):
            raise self._options
        return self._options

    def option_chain(self, date: str | None = None, *args: Any, **kwargs: Any) -> Any:
        assert date is not None
        self.chain_calls.append(date)
        if self._chains is not None:
            answer = self._chains[date]
            if isinstance(answer, BaseException):
                raise answer
            return answer
        return chain_for(date)


def make_provider(ticker: OptionTicker, seen: list[str] | None = None) -> YFinanceProvider:
    def factory(symbol: str) -> Any:
        if seen is not None:
            seen.append(symbol)
        return ticker

    return YFinanceProvider(
        ticker_factory=factory,
        limiter=TokenBucket(rate=1000.0, capacity=1000.0, sleep=lambda _s: None),
        retry_policy=RetryPolicy(delays=(0.0, 0.0), retry_on=(OSError, YFRateLimitError)),
        sleep=lambda _s: None,
    )


def test_fetch_option_chain_normalizes_all_expiries() -> None:
    seen: list[str] = []
    ticker = OptionTicker()
    provider = make_provider(ticker, seen)
    assert isinstance(provider, OptionChainProvider)

    chain = provider.fetch_option_chain("synth")

    assert list(chain.columns) == list(OPTION_CHAIN_COLUMNS)
    assert len(chain) == 20  # 2 Verfallstage x (5 Calls + 5 Puts)
    assert ticker.chain_calls == ["2026-12-18", "2027-01-15"]  # sortiert
    assert set(seen) == {"SYNTH"}
    assert chain["expiry"].dtype == "datetime64[ns]"
    assert chain["volume"].dtype == "Int64"
    assert chain["open_interest"].dtype == "Int64"
    assert list(chain["expiry"].drop_duplicates()) == [
        pd.Timestamp("2026-12-18"),
        pd.Timestamp("2027-01-15"),
    ]
    first = chain.iloc[0]
    assert first["type"] == "CALL"
    assert first["strike"] == 90.0
    assert first["last"] == 12.5
    assert first["bid"] == 12.4
    assert first["ask"] == 12.6
    assert first["iv"] == pytest.approx(0.32)
    assert first["underlying_price"] == 100.0
    no_volume = chain[(chain["type"] == "CALL") & (chain["strike"] == 110.0)].iloc[0]
    assert pd.isna(no_volume["volume"])
    assert no_volume["open_interest"] == 500
    puts = chain[chain["type"] == "PUT"]
    assert len(puts) == 10
    sorted_chain = chain.sort_values(["expiry", "type", "strike"], kind="mergesort")
    assert chain.equals(sorted_chain.reset_index(drop=True))


def test_fetch_option_chain_max_expiries_limits_requests() -> None:
    ticker = OptionTicker()
    chain = make_provider(ticker).fetch_option_chain("SYNTH", max_expiries=1)
    assert ticker.chain_calls == ["2026-12-18"]
    assert set(chain["expiry"]) == {pd.Timestamp("2026-12-18")}


def test_fetch_option_chain_rejects_invalid_max_expiries() -> None:
    with pytest.raises(ValueError, match="max_expiries"):
        make_provider(OptionTicker()).fetch_option_chain("SYNTH", max_expiries=0)


@pytest.mark.parametrize("options", [(), YFTickerMissingError("XXX", "no options")])
def test_no_options_or_unknown_symbol_gives_empty_chain(options: Any) -> None:
    chain = make_provider(OptionTicker(options=options)).fetch_option_chain("XXX")
    assert chain.empty
    assert list(chain.columns) == list(OPTION_CHAIN_COLUMNS)


def test_unexpected_expiry_list_is_provider_error() -> None:
    with pytest.raises(ProviderError, match="Verfallstage"):
        make_provider(OptionTicker(options="2026-12-18")).fetch_option_chain("SYNTH")


def test_network_error_after_retries_is_provider_error() -> None:
    ticker = OptionTicker(
        chains={"2026-12-18": OSError("weg"), "2027-01-15": chain_for("2027-01-15")}
    )
    with pytest.raises(ProviderError, match="3 Versuchen"):
        make_provider(ticker).fetch_option_chain("SYNTH")
    assert ticker.chain_calls == ["2026-12-18"] * 3


def test_missing_expiry_and_empty_sides_are_skipped() -> None:
    empty = SimpleNamespace(calls=pd.DataFrame(), puts=pd.DataFrame(), underlying=None)
    ticker = OptionTicker(
        chains={
            "2026-12-18": YFTickerMissingError("SYNTH", "expiry gone"),
            "2027-01-15": empty,
        }
    )
    chain = make_provider(ticker).fetch_option_chain("SYNTH")
    assert chain.empty
    assert chain.dtypes.equals(empty_option_chain().dtypes)


def test_underlying_fallbacks_and_missing_columns() -> None:
    calls = pd.DataFrame({"strike": [100.0, 0.0, None], "lastPrice": [1.0, 2.0, 3.0]})
    chain = SimpleNamespace(
        calls=calls,
        puts=pd.DataFrame(),
        underlying={"regularMarketPrice": None, "previousClose": 99.5},
    )
    ticker = OptionTicker(options=("2026-12-18",), chains={"2026-12-18": chain})
    result = make_provider(ticker).fetch_option_chain("SYNTH")
    assert len(result) == 1  # Basispreis 0 und fehlender Basispreis fallen weg
    row = result.iloc[0]
    assert row["last"] == 1.0
    assert pd.isna(row["bid"])
    assert pd.isna(row["volume"])
    assert row["underlying_price"] == 99.5


@pytest.mark.parametrize("underlying", ["kaputt", {"regularMarketPrice": True}])
def test_unusable_underlying_is_nan(underlying: Any) -> None:
    chain = SimpleNamespace(
        calls=pd.DataFrame({"strike": [100.0]}), puts=pd.DataFrame(), underlying=underlying
    )
    ticker = OptionTicker(options=("2026-12-18",), chains={"2026-12-18": chain})
    result = make_provider(ticker).fetch_option_chain("SYNTH")
    assert pd.isna(result.iloc[0]["underlying_price"])


def test_bad_chain_answers_are_provider_errors() -> None:
    no_frame = SimpleNamespace(calls=None, puts=pd.DataFrame(), underlying=None)
    ticker = OptionTicker(options=("2026-12-18",), chains={"2026-12-18": no_frame})
    with pytest.raises(ProviderError, match="calls"):
        make_provider(ticker).fetch_option_chain("SYNTH")

    no_strike = SimpleNamespace(calls=pd.DataFrame({"bid": [1.0]}), puts=None, underlying=None)
    ticker = OptionTicker(options=("2026-12-18",), chains={"2026-12-18": no_strike})
    with pytest.raises(ProviderError, match="strike"):
        make_provider(ticker).fetch_option_chain("SYNTH")

    ticker = OptionTicker(options=("kein-datum",), chains={"kein-datum": chain_for("2026-12-18")})
    with pytest.raises(ProviderError, match="Verfallstag"):
        make_provider(ticker).fetch_option_chain("SYNTH")
