"""Scanner (S-04): Signale, nur abgeschlossene Kerzen, Konsistenz mit dem Backtest, Filter."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.broker_sim import Bar, SimBroker
from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import run_backtest
from trading_helfer.core.config import Settings
from trading_helfer.core.types import SignalType, Timeframe
from trading_helfer.data.resample import resample
from trading_helfer.data.storage import MarketStore
from trading_helfer.data.universe import members_on
from trading_helfer.scanner.scan import (
    RESULT_COLUMNS,
    ScanError,
    ScanFilters,
    export_csv,
    load_signals,
    scan,
    signal_of,
)
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    OrderSide,
    Param,
    Params,
    Sizing,
    Strategy,
)

START = "2015-01-02"


class SmaCross(Strategy):
    """Long bei Kreuzung der schnellen über die langsame SMA, Ausstieg bei Rückkreuzung."""

    id = "test_sma_cross"
    name = "SMA-Kreuzung (Test)"
    version = "1.0.0"
    description = "Test"
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1, Timeframe.D3, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        "fast": Param(3, 2, 50, 1, "schnell"),
        "slow": Param(8, 3, 200, 1, "langsam"),
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        close = bars["close"]
        fast = close.rolling(p.fast).mean()
        slow = close.rolling(p.slow).mean()
        return pd.DataFrame({"fast": fast, "slow": slow, "strength": fast / slow - 1.0})

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and ctx.crossed_above("fast", "slow"):
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        if ctx.position is not None and ctx.crossed_below("fast", "slow"):
            return ctx.exit()
        return []


class AlwaysShort(Strategy):
    """Gibt auf jeder Kerze Ausstieg (falls Position) und danach eine Short-Order ab."""

    id = "test_always_short"
    name = "Immer short"
    version = "1.0.0"
    description = "Test"
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1]

    def on_bar(self, ctx: BarContext) -> list[Order]:
        return [*ctx.exit(), ctx.sell_short(qty=1, tag="probe")]


class Alternating(Strategy):
    """Ohne Position Short-Einstieg, mit Position nur Ausstieg."""

    id = "test_alternating"
    name = "Abwechselnd"
    version = "1.0.0"
    description = "Test"
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1]

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None:
            return [ctx.sell_short(qty=1)]
        return ctx.exit()


class Broken(Strategy):
    id = "test_broken"
    name = "Kaputt"
    version = "1.0.0"
    description = "Test"
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1]

    def on_bar(self, ctx: BarContext) -> list[Order]:
        raise RuntimeError("absichtlich")


def wave_bars(
    start: str = START, days: int = 600, period: float = 60.0, phase: float = 0.0
) -> pd.DataFrame:
    """Synthetische Tageskurse: Sinuswelle auf leichtem Aufwärtstrend."""
    dates = pd.bdate_range(start, periods=days)
    t = np.arange(days, dtype=float)
    close = 100.0 + 0.05 * t + 10.0 * np.sin(2 * np.pi * (t + phase) / period)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.3,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000,
        }
    )


@pytest.fixture
def store() -> Iterator[MarketStore]:
    with MarketStore(":memory:") as s:
        yield s


@pytest.fixture
def filled(store: MarketStore) -> MarketStore:
    for i, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        store.upsert_bars(symbol, wave_bars(phase=7.0 * i))
    store.upsert_instruments(
        pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC", "DDD"],
                "sector": ["Technology", "Energy", "Technology", None],
            }
        )
    )
    return store


NO_COSTS = CostModel(fee_per_order=0.0, fee_pct=0.0, slippage_pct=0.0)


def reference_entries(
    strategy: Strategy, bars_daily: pd.DataFrame, timeframe: Timeframe, symbol: str
) -> list[tuple[dt.date, dt.date]]:
    """Unabhängiger Einzelwert-Backtest: (Signalkerze, Ausführungskerze) je Einstieg."""
    bars = resample(bars_daily, timeframe)
    bars = bars.loc[~bars["is_partial"]].drop(columns=["is_partial"]).reset_index(drop=True)
    p = strategy.make_params()
    series = strategy.prepare(bars, p)
    broker = SimBroker(1000.0, NO_COSTS)
    dates = [d.date() for d in bars["date"]]
    signal_day: dict[int, dt.date] = {}
    for t in range(len(bars)):
        if t > 0:
            row = bars.iloc[t]
            broker.process_bar(symbol, Bar(dates[t], row.open, row.high, row.low, row.close))
        ctx = BarContext.at(
            series,
            t,
            symbol=symbol,
            timeframe=timeframe,
            position=broker.position(symbol),
            cash=broker.cash,
            equity=broker.equity(),
        )
        for order in strategy.on_bar(ctx):
            pending = broker.submit(order, date=dates[t], close=float(bars["close"].iloc[t]))
            assert pending is not None
            signal_day[pending.id] = dates[t]
    return [(signal_day[f.id], f.date) for f in broker.fills if f.side.is_entry]


# -- Grundfunktion ------------------------------------------------------------


def test_scan_lists_symbols_with_signal_on_last_closed_bar(filled: MarketStore) -> None:
    result = scan(SmaCross, None, ["AAA", "BBB", "CCC", "DDD"], "W1", store=filled)
    assert list(result.columns[: len(RESULT_COLUMNS)]) == list(RESULT_COLUMNS)
    assert {"fast", "slow"} <= set(result.columns)
    assert "strength" not in result.columns[len(RESULT_COLUMNS) :]
    assert result["signal"].isin(["BUY", "SELL", "EXIT"]).all()
    # Alle Signale liegen auf der letzten abgeschlossenen Wochenkerze.
    last_daily = filled.load_bars("AAA")["date"].iloc[-1]
    weekly = resample(filled.load_bars("AAA"), Timeframe.W1)
    last_closed = weekly.loc[~weekly["is_partial"], "date"].iloc[-1]
    assert last_closed <= last_daily
    assert (result["date"] == last_closed).all()
    assert isinstance(result.attrs["scan_id"], str)


def test_strength_comes_from_indicator_column(filled: MarketStore) -> None:
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    results = [
        scan(SmaCross, None, symbols, Timeframe.D1, as_of=day, store=filled, save=False)
        for day in pd.bdate_range("2015-06-01", periods=30).date
    ]
    result = pd.concat([r for r in results if not r.empty], ignore_index=True)
    assert len(result) >= 2
    expected = result["fast"] / result["slow"] - 1.0
    np.testing.assert_allclose(result["strength"], expected)


def test_result_is_deterministic(filled: MarketStore) -> None:
    a = scan(AlwaysShort, None, ["AAA", "BBB", "CCC", "DDD"], "1W", store=filled, save=False)
    b = scan(AlwaysShort, None, ["DDD", "CCC", "BBB", "AAA"], "W1", store=filled, save=False)
    assert len(a) == 4
    pd.testing.assert_frame_equal(a, b)


def test_partial_bar_is_ignored(store: MarketStore) -> None:
    # Letzter Tag ist ein Mittwoch: die laufende Woche ist unvollständig.
    bars = wave_bars(days=300)
    bars = bars.loc[bars["date"] <= "2016-02-17"]
    store.upsert_bars("AAA", bars)
    result = scan(AlwaysShort, None, ["AAA"], "W1", store=store)
    assert len(result) == 1
    assert result["date"].iloc[0] == pd.Timestamp("2016-02-12")  # Freitag davor


def test_as_of_cuts_data_and_skips_unclosed_week(filled: MarketStore) -> None:
    # Mittwoch: Woche bis 2015-06-12 ist am 2015-06-10 noch offen.
    result = scan(AlwaysShort, None, ["AAA"], "W1", as_of="2015-06-10", store=filled)
    assert result["date"].iloc[0] == pd.Timestamp("2015-06-05")
    # Freitag: die Woche ist abgeschlossen.
    result = scan(AlwaysShort, None, ["AAA"], "W1", as_of=dt.date(2015, 6, 12), store=filled)
    assert result["date"].iloc[0] == pd.Timestamp("2015-06-12")


def test_signal_types_and_order_choice(filled: MarketStore) -> None:
    result = scan(AlwaysShort, None, ["AAA"], "W1", as_of="2015-06-12", store=filled)
    # Mit offener Position: Short-Einstieg wird vor dem Ausstieg gemeldet.
    row = result.iloc[0]
    assert row["signal"] == "SELL"
    assert row["note"] == "SELL_SHORT MARKET probe"
    assert pd.isna(row["strength"])
    # Abwechselnd: Einstieg und Ausstieg wechseln sich wöchentlich ab.
    signals = {
        scan(Alternating, None, ["AAA"], "W1", as_of=day, store=filled)["signal"].iloc[0]
        for day in ("2015-06-12", "2015-06-19")
    }
    assert signals == {"SELL", "EXIT"}


def test_signal_of_maps_all_sides() -> None:
    assert signal_of(Order("A", OrderSide.BUY, qty=1)) is SignalType.BUY
    assert signal_of(Order("A", OrderSide.SELL_SHORT, qty=1)) is SignalType.SELL
    assert signal_of(Order("A", OrderSide.SELL)) is SignalType.EXIT
    assert signal_of(Order("A", OrderSide.BUY_TO_COVER)) is SignalType.EXIT


# -- Konsistenz mit dem Backtest ----------------------------------------------


@pytest.mark.parametrize("timeframe", [Timeframe.D3, Timeframe.W1])
def test_scan_matches_backtest_entries(store: MarketStore, timeframe: Timeframe) -> None:
    daily = wave_bars(days=500, period=45.0)
    store.upsert_bars("AAA", daily)
    strategy = SmaCross()
    entries = reference_entries(strategy, daily, timeframe, "AAA")
    assert len(entries) >= 3
    signal_days = {signal for signal, _ in entries}

    bars = resample(daily, timeframe)
    closed = [d.date() for d in bars.loc[~bars["is_partial"], "date"]]
    settings = Settings(fee_per_order=0.0, slippage_pct=0.0)
    found = set()
    for day in closed:
        result = scan(strategy, None, ["AAA"], timeframe, as_of=day, store=store, settings=settings)
        if not result.empty and result["signal"].iloc[0] == "BUY":
            assert result["date"].iloc[0].date() == day
            found.add(day)
    assert found == signal_days
    # Ausführung im Backtest zum Open der Kerze nach dem Signal.
    for signal, fill in entries:
        assert closed.index(fill) == closed.index(signal) + 1


def test_scan_matches_engine_backtest(store: MarketStore) -> None:
    """Scan an jedem Stichtag = Einstiege eines Backtests der Engine mit gleichem Beginn."""
    daily = wave_bars(start="2012-01-03", days=1500, period=50.0)
    store.upsert_bars("AAA", daily)
    settings = Settings(fee_per_order=1.0, slippage_pct=0.1)
    begin, end = dt.date(2015, 1, 2), dt.date(2017, 12, 29)
    backtest = run_backtest(
        SmaCross(),
        None,
        "AAA",
        Timeframe.W1,
        begin,
        end,
        settings,
        bars=store.load_bars("AAA").drop(columns=["adj_factor"]),
    )
    entries = backtest.orders.loc[backtest.orders["side"] == "BUY", "submitted"]
    expected = {d.date() for d in entries}
    assert len(expected) >= 5

    weekly = resample(daily, Timeframe.W1)
    days = [d.date() for d in weekly["date"] if begin <= d.date() <= end]
    found = set()
    for day in days:
        result = scan(
            SmaCross,
            None,
            ["AAA"],
            "W1",
            as_of=day,
            start=begin,
            store=store,
            settings=settings,
            save=False,
        )
        if not result.empty and result["signal"].iloc[0] == "BUY":
            found.add(result["date"].iloc[0].date())
    assert found == expected


# -- Universum ----------------------------------------------------------------


def test_index_universe_uses_members_on_as_of(store: MarketStore) -> None:
    day = dt.date(2015, 6, 12)
    members = sorted(members_on("SP500", day))[:3]
    for symbol in members:
        store.upsert_bars(symbol, wave_bars())
    result = scan(AlwaysShort, None, "sp500", "W1", as_of=day, store=store, save=False)
    assert sorted(result["symbol"]) == members
    skipped = result.attrs["skipped"]
    assert len(skipped) == len(members_on("SP500", day)) - 3
    assert set(skipped.values()) == {"keine Kurse"}


def test_index_universe_without_as_of_uses_last_stored_day(store: MarketStore) -> None:
    symbol = sorted(members_on("NDX100", dt.date(2016, 4, 1)))[0]
    store.upsert_bars(symbol, wave_bars(days=300))
    result = scan(AlwaysShort, None, "NDX100", "W1", store=store, save=False)
    assert list(result["symbol"]) == [symbol]


def test_entries_only_while_member(store: MarketStore, monkeypatch: pytest.MonkeyPatch) -> None:
    import trading_helfer.scanner.scan as mod

    store.upsert_bars("AAA", wave_bars())
    members = {dt.date(2015, 6, 12): {"AAA"}, dt.date(2015, 6, 19): set()}

    def fake_members(code: str, day: dt.date) -> set[str]:
        return members.get(day, {"AAA"})

    monkeypatch.setattr(mod, "members_on", fake_members)
    inside = scan(AlwaysShort, None, "SP500", "W1", as_of="2015-06-12", store=store)
    assert list(inside["signal"]) == ["SELL"]
    # Am Signaltag kein Mitglied: Einstieg fällt weg, der Ausstieg bleibt.
    outside = scan(AlwaysShort, None, "SP500", "W1", as_of="2015-06-21", store=store)
    assert list(outside["signal"]) == ["EXIT"]


def test_strategy_error_skips_symbol(filled: MarketStore) -> None:
    result = scan(Broken(), None, ["AAA", "ZZZ"], "W1", store=filled)
    assert result.empty
    message = result.attrs["skipped"]["AAA"]
    assert message.startswith("Fehler der Strategie:")
    assert "absichtlich" in message
    assert result.attrs["skipped"]["ZZZ"] == "keine Kurse"


def test_symbol_without_closed_bar_is_skipped(store: MarketStore) -> None:
    store.upsert_bars("AAA", wave_bars(start="2015-01-05", days=2))  # Montag, Dienstag
    result = scan(AlwaysShort, None, ["AAA"], "W1", store=store)
    assert result.attrs["skipped"] == {"AAA": "keine abgeschlossene Kerze"}


def test_progress_is_reported(filled: MarketStore) -> None:
    calls: list[tuple[int, int]] = []
    scan(
        AlwaysShort,
        None,
        ["AAA", "BBB"],
        "W1",
        store=filled,
        progress=lambda d, n: calls.append((d, n)),
    )
    assert calls == [(1, 2), (2, 2)]


# -- Filter -------------------------------------------------------------------


def shares(store: MarketStore, symbol: str, count: float, net_income: float = 10.0) -> None:
    rows = [
        (q, "Q", metric, value, pd.Timestamp(q) + pd.Timedelta(days=20))
        for q in ["2014-03-31", "2014-06-30", "2014-09-30", "2014-12-31"]
        for metric, value in (("shares_outstanding", count), ("net_income", net_income))
    ]
    frame = pd.DataFrame(
        rows, columns=["period_end", "period_type", "metric", "value", "reported_date"]
    )
    store.upsert_fundamentals(symbol, frame)


def test_sector_filter(filled: MarketStore) -> None:
    filters = ScanFilters(sectors=frozenset({"technology"}))
    result = scan(
        AlwaysShort,
        None,
        ["AAA", "BBB", "CCC", "DDD"],
        "W1",
        filters=filters,
        as_of="2015-06-12",
        store=filled,
    )
    assert list(result["symbol"]) == ["AAA", "CCC"]
    assert list(result["sector"]) == ["Technology", "Technology"]


def test_market_cap_filter(filled: MarketStore) -> None:
    shares(filled, "AAA", 1_000_000.0)
    shares(filled, "BBB", 10_000_000.0)
    # CCC ohne Fundamentaldaten fällt heraus.
    filters = ScanFilters(min_market_cap=500_000_000.0)
    result = scan(
        AlwaysShort,
        None,
        ["AAA", "BBB", "CCC"],
        "W1",
        filters=filters,
        as_of="2015-06-12",
        store=filled,
    )
    assert list(result["symbol"]) == ["BBB"]
    assert result["market_cap"].iloc[0] > 500_000_000.0
    filters = ScanFilters(max_market_cap=500_000_000.0)
    result = scan(
        AlwaysShort,
        None,
        ["AAA", "BBB", "CCC"],
        "W1",
        filters=filters,
        as_of="2015-06-12",
        store=filled,
    )
    assert list(result["symbol"]) == ["AAA"]


def test_metric_filter(filled: MarketStore) -> None:
    shares(filled, "AAA", 1_000_000.0, net_income=1_000_000.0)  # KGV ≈ 27
    shares(filled, "BBB", 1_000_000.0, net_income=100_000.0)  # KGV ≈ 270
    filters = ScanFilters(metrics={"pe_ratio": (None, 50.0), "roe": (None, None)})
    result = scan(
        AlwaysShort, None, ["AAA", "BBB"], "W1", filters=filters, as_of="2015-06-12", store=filled
    )
    assert list(result["symbol"]) == ["AAA"]


def test_signal_filter(filled: MarketStore) -> None:
    filters = ScanFilters(signals=frozenset({SignalType.EXIT}))
    result = scan(
        AlwaysShort, None, ["AAA"], "W1", filters=filters, as_of="2015-06-12", store=filled
    )
    assert result.empty
    assert list(result.columns) == list(RESULT_COLUMNS)


def test_filter_validation() -> None:
    with pytest.raises(ScanError, match="Unbekannte Kennzahl"):
        ScanFilters(metrics={"kgv": (1.0, 2.0)})
    with pytest.raises(ScanError, match="Untergrenze"):
        ScanFilters(min_market_cap=10.0, max_market_cap=1.0)
    assert not ScanFilters(metrics={"roe": (None, None)}).needs_metrics
    assert ScanFilters(max_market_cap=1.0).needs_metrics
    assert ScanFilters().match_sector(None)
    assert not ScanFilters(sectors=frozenset({"Energy"})).match_sector(None)


# -- Speichern und Export -----------------------------------------------------


def test_results_are_saved_in_signals(filled: MarketStore) -> None:
    result = scan(AlwaysShort, None, ["AAA", "BBB", "CCC", "DDD"], "W1", store=filled)
    assert len(result) == 4
    stored = load_signals(filled, result.attrs["scan_id"])
    assert list(stored["symbol"]) == sorted(result["symbol"])
    assert set(stored["strategy_id"]) == {"test_always_short"}
    assert len(load_signals(filled)) == len(result)
    unsaved = scan(SmaCross, None, ["AAA"], "1D", store=filled, save=False)
    assert load_signals(filled, unsaved.attrs["scan_id"]).empty


def test_export_csv(filled: MarketStore, tmp_path: Path) -> None:
    result = scan(AlwaysShort, None, ["AAA", "BBB"], "W1", as_of="2015-06-12", store=filled)
    path = export_csv(result, tmp_path / "out" / "scan.csv")
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("symbol,date,signal,strength,close,sector,market_cap,note")
    assert text[1].startswith("AAA,2015-06-12,SELL,")


# -- Eingaben -----------------------------------------------------------------


def test_invalid_inputs(filled: MarketStore) -> None:
    with pytest.raises(ScanError, match="Zeitrahmen 1M"):
        scan(SmaCross, None, ["AAA"], "1M", store=filled)
    with pytest.raises(ScanError, match="Unbekannter Zeitrahmen"):
        scan(SmaCross, None, ["AAA"], "2W", store=filled)
    with pytest.raises(ScanError, match="Unbekannter Index"):
        scan(SmaCross, None, "DAX", "W1", store=filled)
    with pytest.raises(ScanError, match="Ungültiges Datum"):
        scan(SmaCross, None, ["AAA"], "W1", as_of="gestern", store=filled)
    with pytest.raises(ScanError):
        scan(SmaCross, {"fast": 1000}, ["AAA"], "W1", store=filled)


def test_params_and_start(filled: MarketStore) -> None:
    p = SmaCross.make_params({"fast": 2, "slow": 5})
    a = scan(SmaCross, p, ["AAA", "BBB"], "1D", store=filled, save=False)
    b = scan(SmaCross, {"fast": 2, "slow": 5}, ["AAA", "BBB"], "1D", store=filled, save=False)
    pd.testing.assert_frame_equal(a, b)
    late = scan(
        AlwaysShort,
        None,
        ["AAA"],
        "W1",
        start=dt.datetime(2015, 6, 1),
        as_of="2015-06-12",
        store=filled,
    )
    assert late["signal"].iloc[0] == "SELL"


def test_history_window(store: MarketStore, monkeypatch: pytest.MonkeyPatch) -> None:
    import trading_helfer.scanner.scan as mod

    store.upsert_bars("AAA", wave_bars(start="2000-01-03", days=4000))
    seen: list[tuple[dt.date, dt.date]] = []
    original = mod.run_backtest

    def spy(strategy, params, symbol, timeframe, start, end, settings, **kw):  # type: ignore[no-untyped-def]
        seen.append((start, kw["bars"]["date"].iloc[0].date()))
        return original(strategy, params, symbol, timeframe, start, end, settings, **kw)

    monkeypatch.setattr(mod, "run_backtest", spy)
    scan(AlwaysShort, None, ["AAA"], "W1", store=store, save=False)
    scan(AlwaysShort, None, ["AAA"], "W1", as_of="2012-02-29", store=store, save=False)
    scan(AlwaysShort, None, ["AAA"], "W1", history_years=None, store=store, save=False)
    scan(AlwaysShort, None, ["AAA"], "W1", start="2010-01-01", store=store, save=False)
    scan(AlwaysShort, None, ["AAA"], "W1", start="1995-01-01", store=store, save=False)
    last = store.last_date("AAA")
    assert last is not None
    ten = last.replace(year=last.year - mod.DEFAULT_HISTORY_YEARS)
    assert seen[0][0] == ten
    assert seen[0][1] >= ten.replace(year=ten.year - mod.WARMUP_YEARS)
    assert seen[1][0] == dt.date(2007, 2, 28)
    assert seen[2] == (dt.date(2000, 1, 3), dt.date(2000, 1, 3))
    assert seen[3][0] == dt.date(2010, 1, 1)
    assert seen[3][1] >= dt.date(2008, 1, 1)
    assert seen[4] == (dt.date(2000, 1, 3), dt.date(2000, 1, 3))
    late = scan(AlwaysShort, None, ["AAA"], "W1", start="2016-01-01", store=store)
    assert late.attrs["skipped"] == {"AAA": "keine Kurse im Zeitraum"}
