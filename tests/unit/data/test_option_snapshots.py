"""Tägliche Options-Schnappschüsse: Speichern, Laden, Lauf über die Watchlist, CLI."""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_helfer.core import config
from trading_helfer.data import option_snapshots as module
from trading_helfer.data.option_snapshots import (
    SNAPSHOT_COLUMNS,
    OptionSnapshotError,
    load_option_snapshot,
    save_option_snapshot,
    snapshot_date_for,
    snapshot_dates,
    snapshot_options,
    validate_option_chain,
)
from trading_helfer.data.providers import OPTION_CHAIN_COLUMNS, ProviderError, empty_option_chain
from trading_helfer.data.storage import MarketStore

FRIDAY = dt.date(2026, 11, 20)
MONDAY = dt.date(2026, 11, 23)


def make_chain(
    underlying: float = 100.0, expiries: tuple[str, ...] = ("2026-12-18", "2027-01-15")
) -> pd.DataFrame:
    """Synthetische Kette: je Verfall drei Basispreise, Calls und Puts."""
    rows = []
    for expiry in expiries:
        for strike in (95.0, 100.0, 105.0):
            for kind in ("CALL", "PUT"):
                intrinsic = (
                    max(underlying - strike, 0) if kind == "CALL" else max(strike - underlying, 0)
                )
                last = intrinsic + 2.0
                rows.append(
                    {
                        "expiry": pd.Timestamp(expiry),
                        "strike": strike,
                        "type": kind,
                        "bid": last - 0.1,
                        "ask": last + 0.1,
                        "last": last,
                        "volume": 10,
                        "open_interest": 100,
                        "iv": 0.3,
                        "underlying_price": underlying,
                    }
                )
    return pd.DataFrame(rows)


class FakeOptionProvider:
    """Optionsquelle ohne Netzwerk."""

    name = "fake"

    def __init__(
        self,
        chains: dict[str, pd.DataFrame] | None = None,
        failing: set[str] | None = None,
    ) -> None:
        self.chains = chains or {}
        self.failing = failing or set()
        self.calls: list[tuple[str, int | None]] = []

    def fetch_option_chain(self, symbol: str, max_expiries: int | None = None) -> pd.DataFrame:
        self.calls.append((symbol, max_expiries))
        if symbol in self.failing:
            raise ProviderError(self.name, symbol, "Server antwortet nicht.")
        return self.chains.get(symbol, empty_option_chain())


@pytest.fixture
def store() -> Iterator[MarketStore]:
    with MarketStore(":memory:") as db:
        yield db


# ---------------------------------------------------------------------- Schema und Prüfung


def test_migration_creates_table_with_issue_columns(store: MarketStore) -> None:
    rows = store.connection.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'option_snapshots' ORDER BY ordinal_position"
    ).fetchall()
    columns = [r[0] for r in rows]
    assert columns == list(SNAPSHOT_COLUMNS)
    for required in (
        "symbol",
        "snapshot_date",
        "expiry",
        "strike",
        "type",
        "bid",
        "ask",
        "last",
        "volume",
        "open_interest",
        "iv",
    ):
        assert required in columns


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (dt.date(2026, 11, 20), dt.date(2026, 11, 20)),  # Freitag
        (dt.date(2026, 11, 21), dt.date(2026, 11, 20)),  # Samstag
        (dt.date(2026, 11, 22), dt.date(2026, 11, 20)),  # Sonntag
        (dt.date(2026, 11, 23), dt.date(2026, 11, 23)),  # Montag
    ],
)
def test_snapshot_date_on_weekend_is_previous_friday(day: dt.date, expected: dt.date) -> None:
    assert snapshot_date_for(day) == expected


def test_validate_normalizes_types_and_order() -> None:
    chain = make_chain().iloc[::-1].copy()
    chain["type"] = chain["type"].str.lower()
    chain["expiry"] = pd.to_datetime(chain["expiry"]).dt.tz_localize("America/New_York")
    chain["volume"] = [None, *([10.4] * (len(chain) - 1))]
    result = validate_option_chain(chain)
    assert list(result.columns) == list(OPTION_CHAIN_COLUMNS)
    assert set(result["type"]) == {"CALL", "PUT"}
    assert result["expiry"].dtype == "datetime64[ns]"
    assert result["volume"].dtype == "Int64"
    assert result["volume"].isna().sum() == 1
    assert result.iloc[0][["expiry", "type", "strike"]].tolist() == [
        pd.Timestamp("2026-12-18"),
        "CALL",
        95.0,
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.drop(columns=["iv"]), "Spalte"),
        (lambda c: c.assign(type="FUTURE"), "unbekannte Art"),
        (lambda c: c.assign(strike=0.0), "Basispreis"),
        (lambda c: c.assign(strike=None), "Basispreis"),
        (lambda c: c.assign(bid=-1.0), "negative"),
        (lambda c: c.assign(open_interest=-5), "negative"),
        (lambda c: c.assign(ask=float("inf")), "unendliche"),
        (lambda c: c.assign(expiry=None), "Verfallstag fehlt"),
        (lambda c: c.assign(expiry="kein Datum"), "ungültiger Verfallstag"),
        (lambda c: pd.concat([c, c.iloc[:1]]), "doppelt"),
    ],
)
def test_validate_rejects_invalid_chains(change: Any, message: str) -> None:
    with pytest.raises(OptionSnapshotError, match=message):
        validate_option_chain(change(make_chain()))


# ---------------------------------------------------------------------- Speichern und Laden


def test_save_and_load_round_trip(store: MarketStore) -> None:
    chain = make_chain()
    chain.loc[0, "volume"] = None
    assert save_option_snapshot(store, " synth ", FRIDAY, chain, source="fake") == 12
    loaded = load_option_snapshot(store, "SYNTH", FRIDAY)
    assert list(loaded.columns) == list(SNAPSHOT_COLUMNS)
    assert len(loaded) == 12
    assert set(loaded["symbol"]) == {"SYNTH"}
    assert set(loaded["snapshot_date"]) == {pd.Timestamp(FRIDAY)}
    assert set(loaded["source"]) == {"fake"}
    assert loaded["volume"].isna().sum() == 1
    expected = validate_option_chain(chain)
    pd.testing.assert_frame_equal(
        loaded.loc[:, list(OPTION_CHAIN_COLUMNS)], expected, check_dtype=True
    )


def test_second_run_same_day_replaces_snapshot(store: MarketStore) -> None:
    save_option_snapshot(store, "SYNTH", FRIDAY, make_chain(100.0), source="fake")
    smaller = make_chain(101.0, expiries=("2026-12-18",))
    assert save_option_snapshot(store, "SYNTH", FRIDAY, smaller, source="fake") == 6
    loaded = load_option_snapshot(store, "SYNTH", FRIDAY)
    assert len(loaded) == 6
    assert set(loaded["underlying_price"]) == {101.0}


def test_snapshots_build_a_history(store: MarketStore) -> None:
    save_option_snapshot(store, "SYNTH", FRIDAY, make_chain(100.0))
    save_option_snapshot(store, "SYNTH", MONDAY, make_chain(102.0))
    save_option_snapshot(store, "OTHER", MONDAY, make_chain(50.0))
    assert snapshot_dates(store, "synth") == [FRIDAY, MONDAY]
    assert snapshot_dates(store) == [FRIDAY, MONDAY]
    assert snapshot_dates(store, "OTHER") == [MONDAY]
    latest = load_option_snapshot(store, "SYNTH")
    assert set(latest["underlying_price"]) == {102.0}
    friday = load_option_snapshot(store, "SYNTH", FRIDAY, expiry=dt.date(2027, 1, 15))
    assert len(friday) == 6
    assert set(friday["expiry"]) == {pd.Timestamp("2027-01-15")}
    assert set(friday["underlying_price"]) == {100.0}


def test_load_without_data_is_empty_with_columns(store: MarketStore) -> None:
    for result in (
        load_option_snapshot(store, "NONE"),
        load_option_snapshot(store, "NONE", FRIDAY),
    ):
        assert result.empty
        assert list(result.columns) == list(SNAPSHOT_COLUMNS)
        assert result["expiry"].dtype == "datetime64[ns]"


def test_save_empty_chain_only_deletes(store: MarketStore) -> None:
    save_option_snapshot(store, "SYNTH", FRIDAY, make_chain())
    assert save_option_snapshot(store, "SYNTH", FRIDAY, empty_option_chain()) == 0
    assert load_option_snapshot(store, "SYNTH", FRIDAY).empty


def test_invalid_chain_keeps_previous_snapshot(store: MarketStore) -> None:
    save_option_snapshot(store, "SYNTH", FRIDAY, make_chain())
    with pytest.raises(OptionSnapshotError):
        save_option_snapshot(store, "SYNTH", FRIDAY, make_chain().assign(type="X"))
    assert len(load_option_snapshot(store, "SYNTH", FRIDAY)) == 12


def test_empty_symbol_is_rejected(store: MarketStore) -> None:
    with pytest.raises(OptionSnapshotError, match="Symbol"):
        save_option_snapshot(store, "  ", FRIDAY, make_chain())


# ---------------------------------------------------------------------- Lauf


def test_snapshot_run_over_watchlist(store: MarketStore) -> None:
    provider = FakeOptionProvider(
        chains={"AAPL": make_chain(190.0), "SPY": make_chain(500.0)}, failing={"BAD"}
    )
    progress: list[tuple[int, int, str]] = []
    result = snapshot_options(
        ["aapl", "SPY", "BAD", "NOOPT", "AAPL"],
        lambda *args: progress.append(args),
        store=store,
        provider=provider,
        today=dt.date(2026, 11, 21),  # Samstag → Freitag
        max_expiries=2,
    )
    assert result.snapshot_date == FRIDAY
    assert result.total == 4
    assert result.ok == ["AAPL", "SPY", "NOOPT"]
    assert result.no_options == ["NOOPT"]
    assert set(result.failed) == {"BAD"}
    assert result.rows_written == 24
    assert progress[-1] == (4, 4, "NOOPT")
    assert [c[1] for c in provider.calls] == [2, 2, 2, 2]
    assert snapshot_dates(store, "AAPL") == [FRIDAY]
    log = store.connection.execute(
        "SELECT provider, symbols_ok, symbols_failed, message, finished_at FROM update_log "
        "WHERE id = ?",
        [result.log_id],
    ).fetchone()
    assert log[0] == "fake:optionen"
    assert (log[1], log[2]) == (3, 1)
    assert log[3] == result.summary()
    assert "1 fehlgeschlagen (BAD)" in log[3]
    assert "1 ohne Optionen" in log[3]
    assert log[4] is not None


def test_snapshot_run_twice_same_day_has_no_duplicates(store: MarketStore) -> None:
    provider = FakeOptionProvider(chains={"AAPL": make_chain()})
    for _ in range(2):
        snapshot_options(["AAPL"], store=store, provider=provider, today=MONDAY)
    count = store.connection.execute("SELECT count(*) FROM option_snapshots").fetchone()[0]
    assert count == 12


def test_invalid_chain_counts_as_failure(store: MarketStore) -> None:
    provider = FakeOptionProvider(chains={"AAPL": make_chain().assign(strike=-1.0)})
    result = snapshot_options(["AAPL"], store=store, provider=provider, today=MONDAY)
    assert "Basispreis" in result.failed["AAPL"]


def test_snapshot_run_needs_symbols(store: MarketStore) -> None:
    with pytest.raises(OptionSnapshotError, match="Keine Symbole"):
        snapshot_options([" "], store=store, provider=FakeOptionProvider())


def test_should_cancel_stops_cleanly(store: MarketStore) -> None:
    provider = FakeOptionProvider(chains={"A": make_chain(), "B": make_chain()})
    asked: list[int] = []

    def cancel() -> bool:
        asked.append(1)
        return len(asked) > 1

    result = snapshot_options(
        ["A", "B"], store=store, provider=provider, today=MONDAY, should_cancel=cancel
    )
    assert result.cancelled
    assert result.ok == ["A"]
    assert result.summary().startswith("Abgebrochen:")


def test_keyboard_interrupt_is_logged(store: MarketStore) -> None:
    class Interrupting(FakeOptionProvider):
        def fetch_option_chain(self, symbol: str, max_expiries: int | None = None) -> Any:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        snapshot_options(["A"], store=store, provider=Interrupting(), today=MONDAY)
    message = store.connection.execute("SELECT message FROM update_log").fetchone()[0]
    assert message.startswith("Abgebrochen:")


def test_build_option_provider_is_yfinance() -> None:
    assert module.build_option_provider().name == "yfinance"


# ---------------------------------------------------------------------- CLI


@pytest.fixture
def settings_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Leitet Einstellungen, GUI-Zustand und Datenbank nach ``tmp_path`` um."""
    for name in list(os.environ):
        if name.upper().startswith("TH_"):
            monkeypatch.delenv(name)
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    monkeypatch.setattr(config, "user_data_dir", lambda app, appauthor: str(data_dir))
    monkeypatch.setattr(config, "user_config_dir", lambda app, appauthor: str(config_dir))
    return tmp_path


@pytest.fixture
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> FakeOptionProvider:
    provider = FakeOptionProvider(
        chains={"AAPL": make_chain(190.0), "MSFT": make_chain(400.0)}, failing={"BAD"}
    )
    monkeypatch.setattr(module, "build_option_provider", lambda: provider)
    return provider


def write_watchlists(settings_dir: Path, watchlists: dict[str, list[str]]) -> None:
    path = settings_dir / "config" / "gui_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"watchlists": watchlists}), encoding="utf-8")


def stored_symbols(db: Path) -> list[str]:
    with MarketStore(db, read_only=True) as store:
        rows = store.connection.execute(
            "SELECT DISTINCT symbol FROM option_snapshots ORDER BY 1"
        ).fetchall()
    return [r[0] for r in rows]


def test_cli_uses_default_watchlist(
    settings_dir: Path, fake_provider: FakeOptionProvider, capsys: pytest.CaptureFixture[str]
) -> None:
    from trading_helfer.app import main

    write_watchlists(settings_dir, {"Eigene": ["AAPL", "MSFT"]})
    assert main(["options-snapshot", "--max-expiries", "3"]) == 0
    out = capsys.readouterr().out
    assert "[2/2] MSFT" in out
    assert "2 von 2 Symbolen gespeichert, 24 Kontrakte." in out
    assert fake_provider.calls == [("AAPL", 3), ("MSFT", 3)]
    assert stored_symbols(settings_dir / "data" / "market.duckdb") == ["AAPL", "MSFT"]


def test_cli_symbols_named_watchlist_and_failure(
    settings_dir: Path,
    fake_provider: FakeOptionProvider,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_helfer.app import main

    write_watchlists(settings_dir, {"Eigene": ["MSFT"], "Optionen": ["BAD"]})
    db = tmp_path / "eigene.duckdb"
    assert main(["options-snapshot", "--symbols", "AAPL", "--db", str(db)]) == 0
    assert [c[0] for c in fake_provider.calls] == ["AAPL"]
    code = main(
        ["options-snapshot", "--symbols", "AAPL", "--watchlist", "Optionen", "--db", str(db)]
    )
    assert code == 1
    assert "Fehler BAD:" in capsys.readouterr().err
    assert stored_symbols(db) == ["AAPL"]


def test_cli_errors(
    settings_dir: Path,
    fake_provider: FakeOptionProvider,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_helfer.app import main

    assert main(["options-snapshot"]) == 2
    assert "Watchlist „Eigene“ ist leer" in capsys.readouterr().err
    assert main(["options-snapshot", "--watchlist", "Gibt es nicht"]) == 2
    assert "gibt es nicht" in capsys.readouterr().err
    assert main(["options-snapshot", "--symbols", "AAPL", "--max-expiries", "0"]) == 2
    assert "--max-expiries" in capsys.readouterr().err
    assert main(["options-snapshot", "--symbols", " "]) == 2
    assert "Keine Symbole" in capsys.readouterr().err
    monkeypatch.setenv("TH_DEFAULT_CAPITAL", "-1")
    assert main(["options-snapshot", "--symbols", "AAPL"]) == 2
    assert "default_capital" in capsys.readouterr().err


def test_cli_keyboard_interrupt(
    settings_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from trading_helfer.app import main

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "build_option_provider", FakeOptionProvider)
    monkeypatch.setattr(module, "snapshot_options", interrupt)
    assert main(["options-snapshot", "--symbols", "AAPL"]) == 130
    assert "Abgebrochen" in capsys.readouterr().err
