"""Tests für das inkrementelle Datenupdate (Issue D-06) mit einem Fake-Provider."""

from __future__ import annotations

import datetime as dt
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_helfer.core import config
from trading_helfer.core.config import Settings
from trading_helfer.data import update as update_module
from trading_helfer.data.providers.base import InstrumentInfo, ProviderError, empty_actions
from trading_helfer.data.providers.tiingo_provider import TiingoProvider
from trading_helfer.data.providers.yfinance_provider import YFinanceProvider
from trading_helfer.data.storage import MarketStore
from trading_helfer.data.update import (
    BENCHMARKS,
    UpdateError,
    UpdateResult,
    build_price_providers,
    collect_symbols,
    update_universe,
)

START = dt.date(2024, 1, 1)
DAY1 = dt.date(2024, 3, 28)
DAY2 = dt.date(2024, 4, 12)


class FakeProvider:
    """Liefert synthetische Kurse an Werktagen: Trend 100, 101, … ab ``START``.

    ``split_on`` legt einen Split 4:1 an (ab diesem Tag Rohkurse durch 4, davor
    ``adj_close`` durch 4); er ist nur sichtbar, wenn ``end`` nach dem Splittag liegt.
    """

    def __init__(
        self,
        name: str = "fake",
        *,
        failing: set[str] | None = None,
        empty: set[str] | None = None,
        split_on: dict[str, dt.date] | None = None,
        broken: bool = False,
        with_instrument: bool = False,
    ) -> None:
        self.name = name
        self.failing = failing or set()
        self.empty = empty or set()
        self.split_on = split_on or {}
        self.broken = broken
        self.calls: Counter[str] = Counter()
        self.ranges: list[tuple[str, dt.date, dt.date]] = []
        if with_instrument:
            self.fetch_instrument = self._fetch_instrument

    def _check(self, symbol: str) -> None:
        self.calls[symbol] += 1
        if self.broken or symbol in self.failing:
            raise ProviderError(self.name, symbol, "Quelle nicht erreichbar")

    def fetch_daily(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        self._check(symbol)
        self.ranges.append((symbol, start, end))
        if symbol in self.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        days = pd.bdate_range(START, end)
        base = pd.Series(range(len(days)), dtype="float64") + 100.0
        close = base.copy()
        adj = base.copy()
        split = self.split_on.get(symbol)
        if split is not None and split <= end:
            after = days >= pd.Timestamp(split)
            close[after] = close[after] / 4
            adj[after] = adj[after] / 4
            adj[~after] = adj[~after] / 4
        frame = pd.DataFrame(
            {
                "date": days,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": 1000,
                "adj_close": adj,
            }
        )
        mask = (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))
        return frame.loc[mask].reset_index(drop=True)

    def fetch_actions(self, symbol: str) -> pd.DataFrame:
        self._check(symbol)
        split = self.split_on.get(symbol)
        if split is None:
            return empty_actions()
        return pd.DataFrame({"date": [pd.Timestamp(split)], "type": ["SPLIT"], "value": [4.0]})

    def _fetch_instrument(self, symbol: str) -> InstrumentInfo | None:
        self._check(symbol)
        if symbol == "QQQ":
            raise ProviderError(self.name, symbol, "keine Stammdaten")
        if symbol == "SPY":
            return None
        return InstrumentInfo(symbol=symbol, name=f"{symbol} Inc.", currency="USD")


@pytest.fixture
def store() -> Iterator[MarketStore]:
    with MarketStore(":memory:") as db:
        yield db


def run(
    store: MarketStore, providers: list[Any], symbols: list[str], today: dt.date, **kwargs: Any
) -> UpdateResult:
    return update_universe(
        [], store=store, providers=providers, symbols=symbols, start=START, today=today, **kwargs
    )


def log_rows(store: MarketStore) -> list[tuple[Any, ...]]:
    return list(
        store.connection.execute(
            "SELECT id, finished_at IS NOT NULL, provider, symbols_ok, symbols_failed, message "
            "FROM update_log ORDER BY id"
        ).fetchall()
    )


# ---------------------------------------------------------------------- Erstlauf


def test_first_run_loads_full_history_and_benchmarks(store: MarketStore) -> None:
    provider = FakeProvider(with_instrument=True)
    progress: list[tuple[int, int, str]] = []
    result = update_universe(
        [],
        lambda done, total, symbol: progress.append((done, total, symbol)),
        store=store,
        providers=[provider],
        symbols=["aapl", " msft "],
        start=START,
        today=DAY1,
    )
    assert result.ok == ["SPY", "QQQ", "AAPL", "MSFT"]
    assert result.failed == {}
    assert progress == [(1, 4, "SPY"), (2, 4, "QQQ"), (3, 4, "AAPL"), (4, 4, "MSFT")]
    expected_days = len(pd.bdate_range(START, DAY1))
    assert result.rows_written == 4 * expected_days
    assert store.symbols() == ["AAPL", "MSFT", "QQQ", "SPY"]
    bars = store.load_bars("AAPL")
    assert bars["date"].iloc[0].date() == START
    assert bars["date"].iloc[-1].date() == DAY1
    assert provider.ranges[0] == ("SPY", START, DAY1)
    info = store.get_instrument("AAPL")
    assert info is not None
    assert info["name"] == "AAPL Inc."
    assert info["first_date"] == START
    assert info["last_date"] == DAY1
    assert info["source"] == "fake"
    spy = store.get_instrument("SPY")
    assert spy is not None
    assert spy["name"] is None
    assert spy["last_date"] == DAY1
    rows = log_rows(store)
    assert len(rows) == 1
    _, finished, providers, ok, failed, message = rows[0]
    assert finished
    assert (providers, ok, failed) == ("fake", 4, 0)
    assert message == "4 von 4 Symbolen aktualisiert."
    assert result.log_id == rows[0][0]


# ---------------------------------------------------------------------- Folgelauf


def test_second_run_same_day_makes_at_most_one_request_per_symbol(store: MarketStore) -> None:
    provider = FakeProvider(with_instrument=True)
    run(store, [provider], ["AAPL", "MSFT"], DAY1)
    provider.calls.clear()
    provider.ranges.clear()
    result = run(store, [provider], ["AAPL", "MSFT"], DAY1)
    assert result.failed == {}
    assert result.reloaded == []
    assert set(provider.calls) == {"SPY", "QQQ", "AAPL", "MSFT"}
    assert max(provider.calls.values()) == 1
    assert all(start == DAY1 - dt.timedelta(days=5) for _, start, _ in provider.ranges)
    assert len(store.load_bars("AAPL")) == len(pd.bdate_range(START, DAY1))


def test_follow_up_run_appends_new_days_from_last_date_minus_five(store: MarketStore) -> None:
    provider = FakeProvider()
    run(store, [provider], ["AAPL"], DAY1)
    provider.ranges.clear()
    result = run(store, [provider], ["AAPL"], DAY2)
    assert ("AAPL", DAY1 - dt.timedelta(days=5), DAY2) in provider.ranges
    bars = store.load_bars("AAPL")
    assert bars["date"].iloc[-1].date() == DAY2
    assert len(bars) == len(pd.bdate_range(START, DAY2))
    assert bars["date"].is_unique
    info = store.get_instrument("AAPL")
    assert info is not None
    assert info["last_date"] == DAY2
    assert result.rows_written == 3 * len(pd.bdate_range(DAY1 - dt.timedelta(days=5), DAY2))
    assert len(log_rows(store)) == 2


# ---------------------------------------------------------------------- Split


def test_split_after_last_date_reloads_whole_history(store: MarketStore) -> None:
    split_day = dt.date(2024, 4, 5)
    provider = FakeProvider(split_on={"AAPL": split_day})
    run(store, [provider], ["AAPL"], DAY1)
    old_first = store.load_bars("AAPL", adjusted=False)["adj_factor"].iloc[0]
    assert old_first == pytest.approx(1.0)
    provider.ranges.clear()
    result = run(store, [provider], ["AAPL"], DAY2)
    assert result.reloaded == ["AAPL"]
    assert ("AAPL", START, DAY2) in provider.ranges
    raw = store.load_bars("AAPL", adjusted=False)
    assert len(raw) == len(pd.bdate_range(START, DAY2))
    assert raw["adj_factor"].iloc[0] == pytest.approx(0.25)
    adjusted = store.load_bars("AAPL")
    assert adjusted["close"].iloc[0] == pytest.approx(25.0)
    actions = store.load_actions("AAPL")
    assert len(actions) == 1
    assert actions["value"].iloc[0] == pytest.approx(4.0)
    assert "1 neu geladen (Split/Dividende)" in result.summary()


# ---------------------------------------------------------------------- Fallback und Fehler


def test_fallback_to_next_provider_on_error(store: MarketStore) -> None:
    first = FakeProvider("eins", failing={"AAPL"})
    second = FakeProvider("zwei")
    result = run(store, [first, second], ["AAPL", "MSFT"], DAY1)
    assert result.failed == {}
    assert second.calls["AAPL"] == 2  # Kurse und Corporate Actions
    assert second.calls["MSFT"] == 0
    info = store.get_instrument("AAPL")
    assert info is not None
    assert info["source"] == "zwei"
    assert set(store.load_bars("AAPL", adjusted=False).columns) >= {"date", "close"}
    source = store.connection.execute(
        "SELECT DISTINCT source FROM bars_daily WHERE symbol = 'AAPL'"
    ).fetchall()
    assert source == [("zwei",)]
    assert log_rows(store)[0][2] == "eins,zwei"


def test_failing_symbol_does_not_block_others(store: MarketStore) -> None:
    first = FakeProvider("eins", failing={"BAD"})
    second = FakeProvider("zwei", failing={"BAD"})
    result = run(store, [first, second], ["AAPL", "BAD", "MSFT"], DAY1)
    assert result.ok == ["SPY", "QQQ", "AAPL", "MSFT"]
    assert list(result.failed) == ["BAD"]
    assert "eins: Quelle nicht erreichbar" in result.failed["BAD"]
    assert "zwei: Quelle nicht erreichbar" in result.failed["BAD"]
    assert "BAD" not in store.symbols()
    _, _, _, ok, failed, message = log_rows(store)[0]
    assert (ok, failed) == (4, 1)
    assert "1 fehlgeschlagen (BAD)" in message


def test_invalid_bars_fall_back_to_next_provider(store: MarketStore) -> None:
    class Broken(FakeProvider):
        def fetch_daily(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
            frame = super().fetch_daily(symbol, start, end)
            frame["high"] = float("nan")  # fehlende Werte
            return frame

    result = run(store, [Broken("kaputt"), FakeProvider("gut")], ["AAPL"], DAY1)
    assert result.failed == {}
    info = store.get_instrument("AAPL")
    assert info is not None
    assert info["source"] == "gut"


def test_incremental_fallback_and_error(store: MarketStore) -> None:
    run(store, [FakeProvider()], ["AAPL"], DAY1)
    broken = FakeProvider("weg", broken=True)
    result = run(store, [broken], ["AAPL"], DAY2)
    assert set(result.failed) == {"SPY", "QQQ", "AAPL"}
    assert store.last_date("AAPL") == DAY1


def test_symbol_without_data_counts_as_ok_without_bars(store: MarketStore) -> None:
    provider = FakeProvider(empty={"GONE"})
    second = FakeProvider("zwei", empty={"GONE"})
    result = run(store, [provider, second], ["GONE"], DAY1)
    assert result.no_data == ["GONE"]
    assert "GONE" in result.ok
    assert second.calls["GONE"] == 1
    assert "GONE" not in store.symbols()
    assert "1 ohne Kurse" in result.summary()


def test_empty_answer_falls_back_to_next_provider(store: MarketStore) -> None:
    second = FakeProvider("zwei")
    result = run(store, [FakeProvider(empty={"GONE"}), second], ["GONE"], DAY1)
    assert result.no_data == []
    assert store.last_date("GONE") == DAY1
    assert store.get_instrument("GONE")["source"] == "zwei"  # type: ignore[index]


def test_failing_fallback_after_empty_answer_is_no_error(store: MarketStore) -> None:
    result = run(
        store, [FakeProvider(empty={"GONE"}), FakeProvider("zwei", broken=True)], ["GONE"], DAY1
    )
    assert "GONE" in result.no_data
    assert "GONE" not in result.failed
    assert "Quelle nicht erreichbar" in result.fallback_errors["GONE"]
    assert "ohne Ersatzquelle" in result.summary()


def test_incremental_asks_stored_source_first_and_stops_on_empty(store: MarketStore) -> None:
    run(store, [FakeProvider(empty={"GONE"}), FakeProvider("zwei")], ["GONE"], DAY1)
    first = FakeProvider(empty={"GONE"})
    second = FakeProvider("zwei", empty={"GONE"})
    run(store, [first, second], ["GONE"], DAY2)
    assert second.calls["GONE"] == 1
    assert first.calls["GONE"] == 0


class ForeignProvider(FakeProvider):
    """Wie :class:`FakeProvider`, aber mit Kursblöcken × 10 (Fremdkurse)."""

    def __init__(self, name: str = "fremd", foreign: set[str] | None = None) -> None:
        super().__init__(name)
        self.foreign = foreign if foreign is not None else {"GR"}

    def fetch_daily(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        frame = super().fetch_daily(symbol, start, end)
        if symbol in self.foreign and len(frame) > 40:
            for first in (10, 20, 30):
                rows = frame.index[first : first + 3]
                for column in ("open", "high", "low", "close", "adj_close"):
                    frame.loc[rows, column] = frame.loc[rows, column] * 10
        return frame


def test_foreign_prices_fall_back_to_next_provider(store: MarketStore) -> None:
    result = run(store, [ForeignProvider(), FakeProvider("zwei")], ["GR"], DAY2)
    assert store.get_instrument("GR")["source"] == "zwei"  # type: ignore[index]
    assert result.excluded == {}
    assert store.excluded_symbols() == {}


def test_foreign_prices_kept_and_excluded_without_better_source(store: MarketStore) -> None:
    result = run(store, [ForeignProvider(), ForeignProvider("zwei")], ["GR"], DAY2)
    assert store.get_instrument("GR")["source"] == "fremd"  # type: ignore[index]
    assert "GR" in result.excluded
    assert "Tagessprünge" in store.excluded_symbols()["GR"]
    assert "1 wegen Fremdkursen ausgeschlossen" in result.summary()


def test_recheck_reloads_foreign_prices_from_better_source(store: MarketStore) -> None:
    run(store, [ForeignProvider()], ["GR", "AAPL"], DAY2)
    assert "GR" in store.excluded_symbols()
    second = FakeProvider("zwei")
    result = run(store, [ForeignProvider(), second], ["GR", "AAPL"], DAY2, recheck=True)
    assert "GR" in result.reloaded
    assert "AAPL" not in result.reloaded
    assert store.get_instrument("GR")["source"] == "zwei"  # type: ignore[index]
    assert store.excluded_symbols() == {}


def test_incremental_empty_answer_keeps_data(store: MarketStore) -> None:
    run(store, [FakeProvider()], ["AAPL"], DAY1)
    result = run(store, [FakeProvider(empty={"AAPL"})], ["AAPL"], DAY2)
    assert "AAPL" in result.ok
    assert store.last_date("AAPL") == DAY1


def test_no_providers_raises(store: MarketStore) -> None:
    with pytest.raises(UpdateError, match="Keine Kursquelle"):
        run(store, [], ["AAPL"], DAY1)


# ---------------------------------------------------------------------- Abbruch


def test_keyboard_interrupt_during_write_rolls_back_symbol(
    store: MarketStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(split_on={"MSFT": dt.date(2024, 2, 1)})
    original = store.upsert_actions

    def interrupt(symbol: str, df: pd.DataFrame) -> int:
        if symbol == "MSFT":
            raise KeyboardInterrupt
        return original(symbol, df)

    monkeypatch.setattr(store, "upsert_actions", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(store, [provider], ["AAPL", "MSFT", "ZZZ"], DAY1)
    assert store.symbols() == ["AAPL", "QQQ", "SPY"]
    assert store.get_instrument("MSFT") is None
    count = store.connection.execute(
        "SELECT count(*) FROM corporate_actions WHERE symbol = 'MSFT'"
    ).fetchall()
    assert count == [(0,)]
    _, finished, _, ok, failed, message = log_rows(store)[0]
    assert finished
    assert (ok, failed) == (3, 0)
    assert message.startswith("Abgebrochen: 3 von 5 Symbolen")
    # Die Datenbank bleibt benutzbar; der nächste Lauf holt MSFT nach.
    monkeypatch.setattr(store, "upsert_actions", original)
    result = run(store, [provider], ["AAPL", "MSFT", "ZZZ"], DAY1)
    assert result.failed == {}
    assert "MSFT" in store.symbols()


def test_interrupt_during_reload_keeps_old_history(
    store: MarketStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(split_on={"AAPL": dt.date(2024, 4, 5)})
    run(store, [provider], ["AAPL"], DAY1)
    before = store.load_bars("AAPL", adjusted=False)

    def interrupt(symbol: str, df: pd.DataFrame) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "upsert_actions", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(store, [provider], ["AAPL"], DAY2)
    after = store.load_bars("AAPL", adjusted=False)
    pd.testing.assert_frame_equal(before, after)


def test_should_cancel_stops_cleanly(store: MarketStore) -> None:
    seen: list[str] = []
    result = update_universe(
        [],
        lambda done, total, symbol: seen.append(symbol),
        store=store,
        providers=[FakeProvider()],
        symbols=["AAPL", "MSFT"],
        start=START,
        today=DAY1,
        should_cancel=lambda: len(seen) >= 2,
    )
    assert result.cancelled
    assert result.ok == ["SPY", "QQQ"]
    assert log_rows(store)[0][5].startswith("Abgebrochen:")


# ---------------------------------------------------------------------- Indizes


def test_collect_symbols_includes_former_members_and_benchmarks() -> None:
    symbols = collect_symbols(["NDX100"], ["zzzz"], start=dt.date(2020, 1, 1), end=DAY1)
    assert symbols[:2] == list(BENCHMARKS)
    assert "ZZZZ" in symbols
    assert "AAPL" in symbols
    assert len(symbols) == len(set(symbols))
    assert symbols[2:] == sorted(symbols[2:])


def test_collect_symbols_unknown_index() -> None:
    with pytest.raises(UpdateError):
        collect_symbols(["DAX"])


def test_update_with_index_writes_membership(
    store: MarketStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(update_module, "all_members", lambda code, start, end: {"AAPL"})
    result = update_universe(
        ["ndx100"], store=store, providers=[FakeProvider()], start=START, today=DAY1
    )
    assert result.ok == ["SPY", "QQQ", "AAPL"]
    count = store.connection.execute(
        "SELECT count(*), count(DISTINCT symbol), count(*) FILTER (WHERE end_date IS NULL) "
        "FROM index_membership WHERE index_code = 'NDX100'"
    ).fetchall()[0]
    assert count[0] >= 100
    assert count[2] >= 100
    # Ein zweiter Lauf ersetzt die Mitgliedschaften, statt sie zu verdoppeln.
    update_universe(["NDX100"], store=store, providers=[FakeProvider()], start=START, today=DAY1)
    again = store.connection.execute(
        "SELECT count(*) FROM index_membership WHERE index_code = 'NDX100'"
    ).fetchall()[0][0]
    assert again == count[0]


def test_write_membership_unknown_index(store: MarketStore) -> None:
    with pytest.raises(UpdateError, match="Unbekannter Index"):
        update_module._write_membership(store, "DAX", START)


def test_membership_without_changes_uses_start(
    store: MarketStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(
        [("SP500", "AAA", None, None)],
        columns=["index_code", "symbol", "start_date", "end_date"],
    )
    monkeypatch.setattr(update_module, "membership_intervals", lambda code: frame)
    update_module._write_membership(store, "SP500", START)
    rows = store.connection.execute("SELECT symbol, start_date FROM index_membership").fetchall()
    assert rows == [("AAA", START)]


# ---------------------------------------------------------------------- Provider aus Einstellungen


def test_build_price_providers_in_settings_order(store: MarketStore, tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, providers=["tiingo", "yfinance"])
    providers = build_price_providers(settings, store)
    assert isinstance(providers[0], TiingoProvider)
    assert isinstance(providers[1], YFinanceProvider)


def test_build_price_providers_unknown(store: MarketStore, tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, providers=["quandl"])
    with pytest.raises(UpdateError, match="Unbekannte Kursquelle 'quandl'"):
        build_price_providers(settings, store)


def test_summary_lists_at_most_twenty_failures() -> None:
    result = UpdateResult(log_id=1, total=25, failed={f"S{i:02d}": "x" for i in range(25)})
    text = result.summary()
    assert "25 fehlgeschlagen" in text
    assert "und 5 weitere" in text


# ---------------------------------------------------------------------- CLI


@pytest.fixture
def settings_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Leitet Einstellungen und Datenbank nach ``tmp_path`` um."""
    for name in list(os.environ):
        if name.upper().startswith("TH_"):
            monkeypatch.delenv(name)
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "user_data_dir", lambda app, appauthor: str(data_dir))
    monkeypatch.setattr(config, "user_config_dir", lambda app, appauthor: str(tmp_path / "config"))
    return data_dir


@pytest.fixture
def fake_cli(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Ersetzt Provider und Update-Funktion, damit die CLI offline läuft."""
    calls: dict[str, Any] = {}
    provider = FakeProvider(failing={"BAD"})
    real_update = update_module.update_universe

    def fake_update(index_codes: list[str], progress_cb: Any = None, **kwargs: Any) -> Any:
        calls["index_codes"] = list(index_codes)
        calls["symbols"] = list(kwargs["symbols"])
        kwargs["today"] = DAY1
        kwargs["start"] = START
        if index_codes:
            monkeypatch.setattr(update_module, "all_members", lambda code, start, end: set())
        return real_update(index_codes, progress_cb, **kwargs)

    monkeypatch.setattr(update_module, "build_price_providers", lambda settings, store: [provider])
    monkeypatch.setattr(update_module, "update_universe", fake_update)
    return calls


def test_cli_update_symbols(
    settings_dir: Path, fake_cli: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    from trading_helfer.app import main

    assert main(["update", "--symbols", "AAPL", "MSFT"]) == 0
    assert fake_cli["index_codes"] == []
    assert fake_cli["symbols"] == ["AAPL", "MSFT"]
    out = capsys.readouterr().out
    assert "[1/4] SPY" in out
    assert "[4/4] MSFT" in out
    assert "4 von 4 Symbolen aktualisiert." in out
    with MarketStore(settings_dir / "market.duckdb", read_only=True) as db:
        assert db.symbols() == ["AAPL", "MSFT", "QQQ", "SPY"]


def test_cli_update_default_indexes_and_failure(
    settings_dir: Path,
    fake_cli: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_helfer.app import main

    db = tmp_path / "eigene.duckdb"
    assert main(["update", "--db", str(db)]) == 0
    assert fake_cli["index_codes"] == ["SP500", "NDX100"]
    assert main(["update", "--db", str(db), "--index", "ndx100", "--symbols", "BAD"]) == 1
    assert fake_cli["index_codes"] == ["NDX100"]
    captured = capsys.readouterr()
    assert "Fehler BAD:" in captured.err


def test_cli_update_rejects_unknown_index(settings_dir: Path) -> None:
    from trading_helfer.app import main

    with pytest.raises(SystemExit):
        main(["update", "--index", "DAX"])


def test_cli_update_config_and_update_errors(
    settings_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_helfer.app import main

    monkeypatch.setenv("TH_PROVIDERS", "quandl")
    assert main(["update", "--symbols", "AAPL"]) == 2
    assert "Unbekannte Kursquelle" in capsys.readouterr().err
    monkeypatch.setenv("TH_DEFAULT_CAPITAL", "-1")
    assert main(["update"]) == 2
    assert "default_capital" in capsys.readouterr().err


def test_cli_update_keyboard_interrupt(
    settings_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from trading_helfer.app import main

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(update_module, "build_price_providers", lambda settings, store: [])
    monkeypatch.setattr(update_module, "update_universe", interrupt)
    assert main(["update", "--symbols", "AAPL"]) == 130
    assert "Abgebrochen" in capsys.readouterr().err


def test_cli_update_recheck_reports_fallback_and_exclusions(
    settings_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from trading_helfer.app import main

    providers = [ForeignProvider(), FakeProvider("zwei", failing={"GR", "GONE"})]
    real_update = update_module.update_universe
    seen: dict[str, Any] = {}

    def fake_update(index_codes: list[str], progress_cb: Any = None, **kwargs: Any) -> Any:
        seen["recheck"] = kwargs["recheck"]
        kwargs["today"] = DAY2
        kwargs["start"] = START
        return real_update(index_codes, progress_cb, **kwargs)

    monkeypatch.setattr(update_module, "build_price_providers", lambda settings, store: providers)
    monkeypatch.setattr(update_module, "update_universe", fake_update)
    assert main(["update", "--symbols", "GR", "--recheck"]) == 0
    assert seen["recheck"] is True
    out = capsys.readouterr()
    assert "Wegen Fremdkursen ausgeschlossen: GR" in out.out
    assert "ohne Ersatzquelle" in out.err


class QuotaError(ProviderError):
    def __init__(self, symbol: str, resume_at: dt.datetime | None) -> None:
        super().__init__("tiingo", symbol, "Tiingo-Limit erreicht")
        self.resume_at = resume_at


class LimitedProvider(FakeProvider):
    """Meldet die ersten ``blocked`` Anfragen ein erschöpftes Kontingent."""

    def __init__(self, blocked: int, resume_at: dt.datetime | None) -> None:
        super().__init__("tiingo", with_instrument=True)
        self.blocked = blocked
        self.resume_at = resume_at

    def fetch_daily(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        if self.blocked > 0:
            self.blocked -= 1
            raise QuotaError(symbol, self.resume_at)
        return super().fetch_daily(symbol, start, end)


def test_wait_for_quota_sleeps_and_retries(store: MarketStore) -> None:
    soon = dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(minutes=10)
    pauses: list[float] = []
    provider = LimitedProvider(1, soon)
    result = run(
        store,
        [FakeProvider(empty={"GONE"}), provider],
        ["GONE"],
        DAY1,
        wait_for_quota=True,
        sleep=pauses.append,
    )
    assert len(pauses) == 1
    assert 600 <= pauses[0] <= 660
    assert result.fallback_errors == {}
    assert store.get_instrument("GONE")["source"] == "tiingo"  # type: ignore[index]
    assert store.get_instrument("GONE")["name"] == "GONE Inc."  # type: ignore[index]


def test_wait_for_quota_unknown_time_and_give_up(store: MarketStore) -> None:
    pauses: list[float] = []
    provider = LimitedProvider(10, None)
    result = run(
        store,
        [FakeProvider(empty={"GONE"}), provider],
        ["GONE"],
        DAY1,
        wait_for_quota=True,
        sleep=pauses.append,
    )
    assert pauses == [3600.0] * update_module.QUOTA_WAIT_MAX_TRIES
    assert "Tiingo-Limit" in result.fallback_errors["GONE"]


def test_without_wait_quota_is_not_awaited(store: MarketStore) -> None:
    pauses: list[float] = []
    result = run(
        store,
        [FakeProvider(empty={"GONE"}), LimitedProvider(1, None)],
        ["GONE"],
        DAY1,
        sleep=pauses.append,
    )
    assert pauses == []
    assert "GONE" in result.fallback_errors


def test_wait_does_not_retry_other_errors(store: MarketStore) -> None:
    pauses: list[float] = []
    result = run(
        store,
        [FakeProvider(failing={"BAD"})],
        ["BAD"],
        DAY1,
        wait_for_quota=True,
        sleep=pauses.append,
    )
    assert pauses == []
    assert "BAD" in result.failed


def test_cli_update_wait_flag(
    settings_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from trading_helfer.app import main

    seen: dict[str, Any] = {}

    def fake_update(index_codes: list[str], progress_cb: Any = None, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return UpdateResult(log_id=1, total=0)

    monkeypatch.setattr(update_module, "build_price_providers", lambda settings, store: [])
    monkeypatch.setattr(update_module, "update_universe", fake_update)
    assert main(["update", "--symbols", "AAPL", "--wait"]) == 0
    assert seen["wait_for_quota"] is True
    assert seen["recheck"] is False
