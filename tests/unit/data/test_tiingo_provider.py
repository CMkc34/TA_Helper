"""TiingoProvider mit gespeicherten HTTP-Antworten (``responses``) statt Netzwerk."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests
import responses
from pydantic import SecretStr
from responses import matchers

from trading_helfer.core.bars import validate_bars
from trading_helfer.core.config import Settings
from trading_helfer.data.providers import ACTION_COLUMNS, PriceProvider, ProviderError
from trading_helfer.data.providers import tiingo_provider as module
from trading_helfer.data.providers.rate_limit import TokenBucket
from trading_helfer.data.providers.tiingo_provider import (
    BASE_URL,
    MISSING_KEY_MESSAGE,
    TiingoApiKeyMissingError,
    TiingoProvider,
    TiingoQuotaExceededError,
    to_tiingo_symbol,
)
from trading_helfer.data.providers.usage import MemoryRequestCounter, QuotaWindow

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "tiingo"
PRICES_URL = f"{BASE_URL}/tiingo/daily/synth/prices"
YF_COLUMNS = ["date", "open", "high", "low", "close", "volume", "adj_close"]
NOW = datetime(2024, 1, 15, 12, 0, 0)


def load_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeClock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def no_sleep(_: float) -> None:
    pass


def make_provider(
    api_key: str | SecretStr | None = "secret-key",
    *,
    counter: MemoryRequestCounter | None = None,
    clock: FakeClock | None = None,
    sleep: Any = no_sleep,
    **kwargs: Any,
) -> TiingoProvider:
    return TiingoProvider(
        api_key,
        counter=counter if counter is not None else MemoryRequestCounter(),
        limiter=TokenBucket(rate=1000.0, sleep=no_sleep),
        sleep=sleep,
        clock=clock if clock is not None else FakeClock(),
        **kwargs,
    )


@responses.activate
def test_fetch_daily_returns_same_columns_as_yfinance() -> None:
    responses.get(
        PRICES_URL,
        json=load_json("SYNTH_prices.json"),
        match=[
            matchers.query_param_matcher(
                {
                    "startDate": "2024-01-02",
                    "endDate": "2024-01-10",
                    "format": "json",
                    "resampleFreq": "daily",
                }
            ),
            matchers.header_matcher({"Authorization": "Token secret-key"}),
        ],
    )
    bars = make_provider().fetch_daily("synth", date(2024, 1, 2), date(2024, 1, 10))

    assert list(bars.columns) == YF_COLUMNS
    assert str(bars["date"].dtype) == "datetime64[ns]"
    assert str(bars["volume"].dtype) == "int64"
    assert len(bars) == 7
    assert bars["date"].iloc[0] == pd.Timestamp("2024-01-02")
    assert bars["close"].iloc[4] == pytest.approx(104.5)
    assert bars["adj_close"].iloc[0] == pytest.approx(100.0)
    pd.testing.assert_frame_equal(bars, validate_bars(bars))


@responses.activate
def test_fetch_daily_clips_to_requested_range() -> None:
    responses.get(PRICES_URL, json=load_json("SYNTH_prices.json"))
    bars = make_provider().fetch_daily("SYNTH", date(2024, 1, 3), date(2024, 1, 5))
    assert list(bars["date"].dt.day) == [3, 4, 5]


def test_fetch_daily_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="Start"):
        make_provider().fetch_daily("SYNTH", date(2024, 2, 1), date(2024, 1, 1))


@responses.activate
def test_fetch_daily_empty_answer_gives_empty_bars() -> None:
    responses.get(PRICES_URL, json=[])
    bars = make_provider().fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))
    assert bars.empty
    assert list(bars.columns) == YF_COLUMNS


@responses.activate
def test_unknown_symbol_gives_empty_result() -> None:
    responses.get(
        f"{BASE_URL}/tiingo/daily/nope/prices", json=load_json("not_found.json"), status=404
    )
    provider = make_provider()
    assert provider.fetch_daily("NOPE", date(2024, 1, 2), date(2024, 1, 3)).empty
    actions = provider.fetch_actions("NOPE")
    assert actions.empty
    assert list(actions.columns) == list(ACTION_COLUMNS)


@responses.activate
def test_fetch_actions_reads_splits_and_dividends() -> None:
    responses.get(
        PRICES_URL,
        json=load_json("SYNTH_prices.json"),
        match=[matchers.query_param_matcher({"startDate": "1900-01-01"}, strict_match=False)],
    )
    actions = make_provider().fetch_actions("SYNTH")

    assert list(actions.columns) == list(ACTION_COLUMNS)
    assert actions.to_dict("records") == [
        {"date": pd.Timestamp("2024-01-04"), "type": "DIVIDEND", "value": 0.25},
        {"date": pd.Timestamp("2024-01-08"), "type": "SPLIT", "value": 4.0},
    ]
    assert responses.calls[0].request.params["endDate"] == "2024-01-15"


@responses.activate
def test_fetch_actions_without_events_is_empty() -> None:
    rows = [r for r in load_json("SYNTH_prices.json") if r["date"].startswith("2024-01-02")]
    responses.get(PRICES_URL, json=rows)
    assert make_provider().fetch_actions("SYNTH").empty


@responses.activate
def test_fetch_actions_without_action_fields_is_empty() -> None:
    responses.get(PRICES_URL, json=[{"date": "2024-01-02T00:00:00.000Z", "close": 1.0}])
    assert make_provider().fetch_actions("SYNTH").empty


@responses.activate
def test_fetch_actions_without_date_field_is_error() -> None:
    responses.get(PRICES_URL, json=[{"close": 1.0}])
    with pytest.raises(ProviderError, match="date"):
        make_provider().fetch_actions("SYNTH")


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_key_gives_clear_message(key: str | None) -> None:
    provider = make_provider(key)
    assert not provider.has_api_key
    with pytest.raises(TiingoApiKeyMissingError, match=MISSING_KEY_MESSAGE) as info:
        provider.fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))
    assert isinstance(info.value, ProviderError)
    assert info.value.reason == "Tiingo-API-Key fehlt"
    with pytest.raises(TiingoApiKeyMissingError):
        provider.fetch_actions("SYNTH")


@responses.activate
def test_key_from_settings() -> None:
    responses.get(
        PRICES_URL,
        json=[],
        match=[matchers.header_matcher({"Authorization": "Token from-settings"})],
    )
    settings = Settings(tiingo_api_key=SecretStr("from-settings"))
    provider = TiingoProvider.from_settings(
        settings, MemoryRequestCounter(), sleep=no_sleep, clock=FakeClock()
    )
    assert provider.has_api_key
    assert provider.fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3)).empty


def test_settings_without_key_gives_missing_key() -> None:
    provider = TiingoProvider.from_settings(Settings(tiingo_api_key=None))
    with pytest.raises(TiingoApiKeyMissingError):
        provider.fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))


def test_satisfies_price_provider_protocol() -> None:
    provider = make_provider()
    assert isinstance(provider, PriceProvider)
    assert provider.name == "tiingo"


@pytest.mark.parametrize(
    ("symbol", "expected"), [("brk.b", "BRK-B"), (" aapl ", "AAPL"), ("MSFT", "MSFT")]
)
def test_to_tiingo_symbol(symbol: str, expected: str) -> None:
    assert to_tiingo_symbol(symbol) == expected


def test_to_tiingo_symbol_rejects_empty() -> None:
    with pytest.raises(ValueError, match="leer"):
        to_tiingo_symbol("  ")


@responses.activate
def test_symbol_with_dot_uses_hyphen_in_url() -> None:
    responses.get(f"{BASE_URL}/tiingo/daily/brk-b/prices", json=[])
    make_provider().fetch_daily("BRK.B", date(2024, 1, 2), date(2024, 1, 3))
    assert len(responses.calls) == 1


# -- Kontingent -----------------------------------------------------------------------


@responses.activate
def test_each_request_is_counted() -> None:
    responses.get(PRICES_URL, json=[])
    counter = MemoryRequestCounter()
    provider = make_provider(counter=counter)
    for _ in range(3):
        provider.fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))
    assert len(counter.timestamps_since("tiingo", NOW - timedelta(days=1))) == 3
    assert counter.timestamps_since("yfinance", NOW - timedelta(days=1)) == []


@responses.activate
def test_hourly_limit_pauses_provider_with_message() -> None:
    responses.get(PRICES_URL, json=[])
    counter = MemoryRequestCounter()
    clock = FakeClock()
    for minute in range(50):
        counter.record("tiingo", NOW - timedelta(minutes=59) + timedelta(seconds=minute))
    provider = make_provider(counter=counter, clock=clock)

    with pytest.raises(TiingoQuotaExceededError) as info:
        provider.fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))
    assert "Tiingo-Limit erreicht (50 Anfragen pro Stunde)" in str(info.value)
    assert "pausiert bis 15.01.2024 12:01 UTC" in str(info.value)
    assert info.value.resume_at == NOW + timedelta(minutes=1)
    assert len(responses.calls) == 0, "bei erschöpftem Kontingent keine Anfrage senden"

    clock.now = NOW + timedelta(minutes=1)
    provider.fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))
    assert len(responses.calls) == 1


@responses.activate
def test_daily_limit_pauses_provider() -> None:
    responses.get(PRICES_URL, json=[])
    counter = MemoryRequestCounter()
    start = NOW - timedelta(hours=23)
    for i in range(1000):
        counter.record("tiingo", start + timedelta(seconds=i))
    provider = make_provider(counter=counter)

    with pytest.raises(TiingoQuotaExceededError, match="1000 Anfragen pro Tag") as info:
        provider.fetch_actions("SYNTH")
    assert info.value.resume_at == start + timedelta(days=1)
    assert len(responses.calls) == 0


@responses.activate
def test_quota_counts_every_request_until_limit() -> None:
    responses.get(PRICES_URL, json=[])
    provider = make_provider(quotas=(QuotaWindow(3, timedelta(hours=1), "pro Stunde"),))
    for _ in range(3):
        provider.fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))
    with pytest.raises(TiingoQuotaExceededError):
        provider.fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))
    assert len(responses.calls) == 3


@responses.activate
def test_http_429_reports_quota_without_retry() -> None:
    responses.get(PRICES_URL, json={"detail": "Rate limit exceeded"}, status=429)
    with pytest.raises(TiingoQuotaExceededError, match="Rate limit exceeded") as info:
        make_provider().fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))
    assert info.value.resume_at is None
    assert len(responses.calls) == 1


# -- Fehler und Wiederholungen ------------------------------------------------------


@responses.activate
def test_server_error_is_retried_then_succeeds() -> None:
    responses.get(PRICES_URL, status=503, body="down")
    responses.get(PRICES_URL, json=load_json("SYNTH_prices.json"))
    delays: list[float] = []
    provider = make_provider(sleep=delays.append)
    bars = provider.fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 10))
    assert len(bars) == 7
    assert delays == [2.0]


@responses.activate
def test_network_error_after_all_retries_is_provider_error() -> None:
    responses.get(PRICES_URL, body=requests.ConnectionError("keine Verbindung"))
    counter = MemoryRequestCounter()
    provider = make_provider(counter=counter)
    with pytest.raises(ProviderError, match="nach 4 Versuchen") as info:
        provider.fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))
    assert not isinstance(info.value, TiingoQuotaExceededError)
    assert len(responses.calls) == 4
    assert len(counter.timestamps_since("tiingo", NOW - timedelta(hours=1))) == 4


@responses.activate
def test_other_request_exception_is_provider_error() -> None:
    responses.get(PRICES_URL, body=requests.TooManyRedirects("Schleife"))
    with pytest.raises(ProviderError, match="Unerwarteter Fehler"):
        make_provider().fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))


@pytest.mark.parametrize("status", [401, 403])
@responses.activate
def test_invalid_key(status: int) -> None:
    responses.get(PRICES_URL, json={"detail": "Invalid token."}, status=status)
    with pytest.raises(ProviderError, match="API-Key ungültig"):
        make_provider().fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))


@responses.activate
def test_unexpected_status() -> None:
    responses.get(PRICES_URL, body="", status=418)
    with pytest.raises(ProviderError, match="HTTP 418"):
        make_provider().fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))


@responses.activate
def test_invalid_json() -> None:
    responses.get(PRICES_URL, body="kein json", content_type="application/json")
    with pytest.raises(ProviderError, match="kein gültiges JSON"):
        make_provider().fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))


@responses.activate
def test_unexpected_payload_type() -> None:
    responses.get(PRICES_URL, json={"foo": 1})
    with pytest.raises(ProviderError, match="Unerwartete Antwort vom Typ dict"):
        make_provider().fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))


@responses.activate
def test_missing_price_fields() -> None:
    responses.get(PRICES_URL, json=[{"date": "2024-01-02T00:00:00.000Z", "close": 1.0}])
    with pytest.raises(ProviderError, match=r"Feld\(er\) fehlen.*open"):
        make_provider().fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))


@responses.activate
def test_invalid_price_values() -> None:
    rows = load_json("SYNTH_prices.json")[:1]
    rows[0]["volume"] = None
    responses.get(PRICES_URL, json=rows)
    with pytest.raises(ProviderError, match="Ungültige Kursdaten"):
        make_provider().fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 3))


def test_default_clock_is_naive_utc() -> None:
    now = module._utc_now()
    assert now.tzinfo is None
    assert abs(now - datetime.now(UTC).replace(tzinfo=None)) < timedelta(minutes=1)
