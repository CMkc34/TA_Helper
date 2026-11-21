"""YFinanceProvider mit gespeicherten Antworten (Fixtures) statt Netzwerk."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yfinance as yf
from yfinance.exceptions import (
    YFPricesMissingError,
    YFRateLimitError,
    YFTzMissingError,
)

from trading_helfer.core.bars import validate_bars
from trading_helfer.data.providers import (
    ACTION_COLUMNS,
    InstrumentInfo,
    InstrumentProvider,
    PriceProvider,
    ProviderError,
)
from trading_helfer.data.providers import yfinance_provider as module
from trading_helfer.data.providers.rate_limit import RetryPolicy, TokenBucket
from trading_helfer.data.providers.yfinance_provider import YFinanceProvider, to_yahoo_symbol

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "yfinance"


def load_frame(name: str) -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / name, index_col="Date")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df.index.name = "Date"
    return df


class FakeTicker:
    """Attrappe für ``yfinance.Ticker`` mit Antworten aus Fixtures."""

    def __init__(
        self,
        symbol: str,
        *,
        history: pd.DataFrame | BaseException | None = None,
        actions: pd.DataFrame | BaseException | None = None,
        info: Any = None,
    ) -> None:
        self.symbol = symbol
        self._history = history
        self._actions = actions
        self._info = info
        self.history_calls: list[dict[str, Any]] = []

    @staticmethod
    def _answer(value: Any) -> Any:
        if isinstance(value, BaseException):
            raise value
        return value

    def history(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        self.history_calls.append(kwargs)
        result: pd.DataFrame = self._answer(self._history)
        return result

    @property
    def actions(self) -> pd.DataFrame:
        result: pd.DataFrame = self._answer(self._actions)
        return result

    @property
    def info(self) -> Mapping[str, Any]:
        result: Mapping[str, Any] = self._answer(self._info)
        return result


class Recorder:
    """Ticker-Fabrik, die erzeugte Ticker und Symbole festhält."""

    def __init__(self, make: Callable[[str], FakeTicker]) -> None:
        self.make = make
        self.symbols: list[str] = []
        self.tickers: list[FakeTicker] = []

    def __call__(self, symbol: str) -> FakeTicker:
        self.symbols.append(symbol)
        ticker = self.make(symbol)
        self.tickers.append(ticker)
        return ticker


class NoWait:
    def __init__(self) -> None:
        self.sleeps: list[float] = []
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_provider(recorder: Recorder, clock: NoWait | None = None) -> YFinanceProvider:
    clock = clock or NoWait()
    return YFinanceProvider(
        ticker_factory=recorder,
        limiter=TokenBucket(clock=clock, sleep=clock.sleep),
        sleep=clock.sleep,
    )


@pytest.fixture
def synth() -> Recorder:
    return Recorder(
        lambda s: FakeTicker(
            s,
            history=load_frame("SYNTH_history.csv"),
            actions=load_frame("SYNTH_actions.csv"),
            info=json.loads((FIXTURES / "SYNTH_info.json").read_text(encoding="utf-8")),
        )
    )


# --- Symbole ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [("BRK.B", "BRK-B"), ("brk.b", "BRK-B"), (" AAPL ", "AAPL"), ("^GSPC", "^GSPC")],
)
def test_to_yahoo_symbol(symbol: str, expected: str) -> None:
    assert to_yahoo_symbol(symbol) == expected


def test_to_yahoo_symbol_rejects_empty() -> None:
    with pytest.raises(ValueError, match="leer"):
        to_yahoo_symbol("  ")


def test_provider_requests_mapped_symbol(synth: Recorder) -> None:
    make_provider(synth).fetch_daily("BRK.B", date(2024, 1, 1), date(2024, 1, 31))
    assert synth.symbols == ["BRK-B"]


def test_provider_satisfies_protocols(synth: Recorder) -> None:
    provider = make_provider(synth)
    assert isinstance(provider, PriceProvider)
    assert isinstance(provider, InstrumentProvider)
    assert provider.name == "yfinance"


# --- Tageskurse ------------------------------------------------------------------


def test_fetch_daily_normalized_columns(synth: Recorder) -> None:
    bars = make_provider(synth).fetch_daily("SYNTH", date(2024, 1, 1), date(2024, 1, 31))
    assert list(bars.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
    ]
    assert str(bars["date"].dtype) == "datetime64[ns]"
    assert bars["date"].dt.tz is None
    assert bars["date"].tolist() == [
        pd.Timestamp(d)
        for d in [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
    ]
    # Rohe Kurse (vor dem Split 4:1) plus bereinigter Schlusskurs.
    assert bars.loc[0, "close"] == 402.0
    assert bars.loc[0, "adj_close"] == 100.0
    assert bars["volume"].dtype == "int64"
    pd.testing.assert_frame_equal(bars, validate_bars(bars))


def test_fetch_daily_passes_raw_request(synth: Recorder) -> None:
    make_provider(synth).fetch_daily("SYNTH", date(2024, 1, 2), date(2024, 1, 9))
    (call,) = synth.tickers[0].history_calls
    assert call == {
        "start": "2024-01-02",
        "end": "2024-01-10",
        "interval": "1d",
        "auto_adjust": False,
        "actions": False,
        "repair": False,
    }


def test_fetch_daily_end_is_inclusive_and_range_is_clipped(synth: Recorder) -> None:
    bars = make_provider(synth).fetch_daily("SYNTH", date(2024, 1, 3), date(2024, 1, 8))
    assert bars["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-03",
        "2024-01-04",
        "2024-01-08",
    ]


def test_fetch_daily_rejects_inverted_range(synth: Recorder) -> None:
    with pytest.raises(ValueError, match="Start"):
        make_provider(synth).fetch_daily("SYNTH", date(2024, 2, 1), date(2024, 1, 1))


def test_fetch_daily_without_adj_close_uses_close() -> None:
    history = load_frame("SYNTH_history.csv").drop(columns=["Adj Close"])
    provider = make_provider(Recorder(lambda s: FakeTicker(s, history=history)))
    bars = provider.fetch_daily("SYNTH", date(2024, 1, 1), date(2024, 1, 31))
    assert (bars["adj_close"] == bars["close"]).all()


def test_fetch_daily_missing_column_is_provider_error() -> None:
    history = load_frame("SYNTH_history.csv").drop(columns=["Volume"])
    provider = make_provider(Recorder(lambda s: FakeTicker(s, history=history)))
    with pytest.raises(ProviderError, match="Volume"):
        provider.fetch_daily("SYNTH", date(2024, 1, 1), date(2024, 1, 31))


def test_fetch_daily_invalid_values_is_provider_error() -> None:
    history = load_frame("SYNTH_history.csv")
    history["Volume"] = history["Volume"].astype("object")
    history.iloc[0, history.columns.get_loc("Volume")] = "viel"
    provider = make_provider(Recorder(lambda s: FakeTicker(s, history=history)))
    with pytest.raises(ProviderError, match="Ungültige Kursdaten"):
        provider.fetch_daily("SYNTH", date(2024, 1, 1), date(2024, 1, 31))


def test_fetch_daily_unexpected_type_is_provider_error() -> None:
    provider = make_provider(Recorder(lambda s: FakeTicker(s, history="kaputt")))  # type: ignore[arg-type]
    with pytest.raises(ProviderError, match="Unerwartete Antwort"):
        provider.fetch_daily("SYNTH", date(2024, 1, 1), date(2024, 1, 31))


# --- Unbekannte Symbole ----------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        pd.DataFrame(),
        YFTzMissingError("NOPE"),
        YFPricesMissingError("NOPE", debug_info=""),
    ],
    ids=["leer", "tz-fehlt", "keine-kurse"],
)
def test_unknown_symbol_gives_empty_bars(answer: Any) -> None:
    provider = make_provider(Recorder(lambda s: FakeTicker(s, history=answer)))
    bars = provider.fetch_daily("NOPE", date(2024, 1, 1), date(2024, 1, 31))
    assert bars.empty
    assert list(bars.columns) == ["date", "open", "high", "low", "close", "volume", "adj_close"]


def test_unknown_symbol_gives_empty_actions_and_no_info() -> None:
    provider = make_provider(
        Recorder(lambda s: FakeTicker(s, actions=YFTzMissingError(s), info=YFTzMissingError(s)))
    )
    actions = provider.fetch_actions("NOPE")
    assert actions.empty
    assert list(actions.columns) == list(ACTION_COLUMNS)
    assert provider.fetch_instrument("NOPE") is None


# --- Netzwerkfehler und Wiederholungen -------------------------------------------


def failing_then(failures: int, error: BaseException, success: Any) -> Recorder:
    counter = {"n": 0}

    def make(symbol: str) -> FakeTicker:
        counter["n"] += 1
        answer = error if counter["n"] <= failures else success
        return FakeTicker(symbol, history=answer)

    return Recorder(make)


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("Verbindung abgelehnt"),
        TimeoutError("Zeitüberschreitung"),
        YFRateLimitError(),
    ],
)
def test_network_error_after_all_attempts_is_provider_error(error: BaseException) -> None:
    clock = NoWait()
    recorder = failing_then(99, error, None)
    provider = make_provider(recorder, clock)
    with pytest.raises(ProviderError) as info:
        provider.fetch_daily("SYNTH", date(2024, 1, 1), date(2024, 1, 31))
    assert len(recorder.symbols) == 4  # erster Versuch plus 3 Wiederholungen
    assert [s for s in clock.sleeps if s >= 1] == [2.0, 4.0, 8.0]
    assert info.value.provider == "yfinance"
    assert info.value.symbol == "SYNTH"
    assert "4 Versuchen" in info.value.reason
    assert info.value.__cause__ is error


def test_transient_error_recovers() -> None:
    clock = NoWait()
    recorder = failing_then(2, OSError("kurz weg"), load_frame("SYNTH_history.csv"))
    bars = make_provider(recorder, clock).fetch_daily("SYNTH", date(2024, 1, 1), date(2024, 1, 31))
    assert len(bars) == 6
    assert [s for s in clock.sleeps if s >= 1] == [2.0, 4.0]


def test_unexpected_error_is_provider_error_without_retry() -> None:
    recorder = failing_then(99, KeyError("chart"), None)
    with pytest.raises(ProviderError, match="Unerwarteter Fehler"):
        make_provider(recorder).fetch_daily("SYNTH", date(2024, 1, 1), date(2024, 1, 31))
    assert len(recorder.symbols) == 1


def test_requests_respect_rate_limit(synth: Recorder) -> None:
    clock = NoWait()
    provider = make_provider(synth, clock)
    for _ in range(3):
        provider.fetch_daily("SYNTH", date(2024, 1, 1), date(2024, 1, 31))
    assert clock.sleeps == pytest.approx([0.5, 0.5])


# --- Corporate Actions -----------------------------------------------------------


def test_fetch_actions(synth: Recorder) -> None:
    actions = make_provider(synth).fetch_actions("SYNTH")
    assert list(actions.columns) == ["date", "type", "value"]
    assert actions.to_dict("records") == [
        {"date": pd.Timestamp("2024-01-04"), "type": "DIVIDEND", "value": 0.25},
        {"date": pd.Timestamp("2024-01-08"), "type": "SPLIT", "value": 4.0},
    ]
    assert str(actions["date"].dtype) == "datetime64[ns]"


def test_fetch_actions_same_day_sorted_by_type() -> None:
    raw = pd.DataFrame(
        {"Stock Splits": [2.0], "Dividends": [0.1], "Capital Gains": [0.0]},
        index=pd.DatetimeIndex(["2020-05-01"]),
    )
    provider = make_provider(Recorder(lambda s: FakeTicker(s, actions=raw)))
    actions = provider.fetch_actions("X")
    assert actions["type"].tolist() == ["DIVIDEND", "SPLIT"]


@pytest.mark.parametrize(
    "raw",
    [
        pd.DataFrame(),
        pd.DataFrame({"Capital Gains": [1.0]}, index=pd.DatetimeIndex(["2020-01-02"])),
    ],
    ids=["leer", "ohne-bekannte-spalten"],
)
def test_fetch_actions_empty(raw: pd.DataFrame) -> None:
    provider = make_provider(Recorder(lambda s: FakeTicker(s, actions=raw)))
    actions = provider.fetch_actions("X")
    assert actions.empty
    assert list(actions.columns) == list(ACTION_COLUMNS)


def test_fetch_actions_unexpected_type() -> None:
    provider = make_provider(Recorder(lambda s: FakeTicker(s, actions=[1, 2])))  # type: ignore[arg-type]
    with pytest.raises(ProviderError, match="Unerwartete Antwort"):
        provider.fetch_actions("X")


# --- Stammdaten ------------------------------------------------------------------


def test_fetch_instrument(synth: Recorder) -> None:
    info = make_provider(synth).fetch_instrument("synth")
    assert info == InstrumentInfo(
        symbol="SYNTH",
        name="Synthetic Test Corp.",
        exchange="NMS",
        sector="Technology",
        industry="Software - Infrastructure",
        currency="USD",
        source="yfinance",
    )


def test_fetch_instrument_falls_back_to_short_name() -> None:
    raw = {"shortName": "Kurz", "longName": "  ", "sector": float("nan"), "currency": None}
    provider = make_provider(Recorder(lambda s: FakeTicker(s, info=raw)))
    info = provider.fetch_instrument("BRK.B")
    assert info is not None
    assert info.symbol == "BRK.B"
    assert info.name == "Kurz"
    assert info.sector is None
    assert info.currency is None


@pytest.mark.parametrize(
    "raw", [{}, None, {"trailingPegRatio": None}], ids=["leer", "none", "ohne-name"]
)
def test_fetch_instrument_unknown(raw: Any) -> None:
    provider = make_provider(Recorder(lambda s: FakeTicker(s, info=raw)))
    assert provider.fetch_instrument("NOPE") is None


def test_fetch_instrument_unexpected_type() -> None:
    provider = make_provider(Recorder(lambda s: FakeTicker(s, info=["x"])))
    with pytest.raises(ProviderError, match="Unerwartete Stammdaten"):
        provider.fetch_instrument("X")


# --- Standard-Fabrik mit echtem yfinance (gepatcht) ------------------------------


def test_default_factory_uses_yfinance_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str] = []

    def fake_ticker(symbol: str) -> FakeTicker:
        created.append(symbol)
        return FakeTicker(symbol, history=load_frame("SYNTH_history.csv"))

    monkeypatch.setattr(yf, "Ticker", fake_ticker)
    previous = yf.config.debug.hide_exceptions
    yf.config.debug.hide_exceptions = True
    try:
        clock = NoWait()
        provider = YFinanceProvider(
            sleep=clock.sleep, limiter=TokenBucket(clock=clock, sleep=clock.sleep)
        )
        bars = provider.fetch_daily("BRK.B", date(2024, 1, 1), date(2024, 1, 31))
        # Fehler sollen nicht als leeres Ergebnis verschluckt werden.
        hidden = yf.config.debug.hide_exceptions
    finally:
        yf.config.debug.hide_exceptions = previous
    assert created == ["BRK-B"]
    assert len(bars) == 6
    assert hidden is False


def test_default_policy_retries_network_errors() -> None:
    assert module.RETRYABLE_ERRORS[0] is OSError
    assert RetryPolicy().delays == (2.0, 4.0, 8.0)


def test_provider_error_message() -> None:
    err = ProviderError("yfinance", None, "kaputt")
    assert str(err) == "yfinance: kaputt"
    assert err.symbol is None
