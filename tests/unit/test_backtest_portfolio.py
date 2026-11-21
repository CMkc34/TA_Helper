"""Tests des Portfolio-Backtests (B-04)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest import portfolio as portfolio_module
from trading_helfer.backtest.broker_sim import SimBroker
from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import (
    EQUITY_COLUMNS,
    ORDER_COLUMNS,
    TRADE_COLUMNS,
    BacktestError,
    LookaheadError,
    StrategyError,
)
from trading_helfer.backtest.portfolio import (
    DELISTING_REASON,
    PortfolioResult,
    run_portfolio_backtest,
)
from trading_helfer.core.config import Settings
from trading_helfer.core.types import Timeframe
from trading_helfer.data.storage import MarketStore
from trading_helfer.data.universe import SURVIVORSHIP_WARNING, IndexChange, IndexHistory
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    OrderError,
    OrderSide,
    Param,
    Params,
    Sizing,
    Strategy,
)

NO_COSTS = CostModel(fee_per_order=0.0, slippage_pct=0.0)
START, END = "2020-01-01", "2030-12-31"


def _bars(close: np.ndarray | list[float], start: str = "2020-01-02") -> pd.DataFrame:
    """Tageskurse an Werktagen; Open = Vortages-Close, High/Low ±1 %."""
    close_arr = np.asarray(close, dtype=float)
    opens = np.concatenate([[close_arr[0]], close_arr[:-1]])
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=len(close_arr)),
            "open": opens,
            "high": np.maximum(opens, close_arr) * 1.01,
            "low": np.minimum(opens, close_arr) * 0.99,
            "close": close_arr,
            "volume": 1000,
        }
    )


def _trend(n: int = 60, start: float = 100.0, step: float = 0.5) -> np.ndarray:
    return start + step * np.arange(n)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


class HoldRanked(Strategy):
    """Kauft jeden Wert einmal (Mitglieder), Stärke aus :attr:`STRENGTH`; verkauft nach N Kerzen."""

    id = "test_hold_ranked"
    name = "Halten nach Rang"
    version = "1.0.0"
    description = "Kauft und hält; Signalstärke je Symbol fest vorgegeben."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1, Timeframe.W1]
    params: ClassVar[dict[str, Param]] = {
        "hold": Param(0, 0, 1000, 1, "Haltedauer in Kerzen (0 = bis zum Ende)"),
        "pct": Param(30.0, 1.0, 100.0, 1.0, "Anteil am Kapital in %"),
    }
    STRENGTH: ClassVar[dict[str, float]] = {"AAA": 1.0, "BBB": 3.0, "CCC": 2.0}

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None:
            if not ctx.is_member:
                return []
            return [
                ctx.buy(
                    sizing=Sizing.percent_equity(self.pct),
                    strength=self.STRENGTH.get(ctx.symbol),
                    tag="Einstieg",
                )
            ]
        held = int(np.sum(ctx.bars.dates >= np.datetime64(ctx.position.entry_date)))
        if self.hold and held > self.hold:
            return ctx.exit(tag="Zeit")
        return []

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        self.hold, self.pct = int(p.hold), float(p.pct)  # Params lässt sich nicht pickeln
        return pd.DataFrame(index=bars.index)


class IndicatorStrength(HoldRanked):
    """Wie :class:`HoldRanked`, aber Stärke aus der Indikatorspalte ``strength``."""

    id = "test_indicator_strength"
    STRENGTH: ClassVar[dict[str, float]] = {}
    LEVEL: ClassVar[dict[float, float]] = {}

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        self.hold, self.pct = int(p.hold), float(p.pct)  # Params lässt sich nicht pickeln
        first = float(bars["close"].iloc[0])
        return pd.DataFrame({"strength": np.full(len(bars), self.LEVEL.get(first, 0.0))})


def _three(n: int = 60) -> dict[str, pd.DataFrame]:
    return {
        "AAA": _bars(_trend(n, 100.0)),
        "BBB": _bars(_trend(n, 50.0)),
        "CCC": _bars(_trend(n, 20.0)),
    }


def _run(
    settings: Settings,
    bars: dict[str, pd.DataFrame],
    universe: object = ("AAA", "BBB", "CCC"),
    strategy: Strategy | type[Strategy] = HoldRanked,
    params: dict[str, object] | None = None,
    **kwargs: object,
) -> PortfolioResult:
    kwargs.setdefault("costs", NO_COSTS)
    kwargs.setdefault("workers", 1)
    return run_portfolio_backtest(
        strategy,
        params,
        universe,  # type: ignore[arg-type]
        kwargs.pop("timeframe", Timeframe.D1),  # type: ignore[arg-type]
        kwargs.pop("start", START),  # type: ignore[arg-type]
        kwargs.pop("end", END),  # type: ignore[arg-type]
        settings,
        bars=bars,
        **kwargs,  # type: ignore[arg-type]
    )


def _entries(result: PortfolioResult) -> pd.DataFrame:
    orders = result.orders
    return orders[orders["side"] == "BUY"]


# ---------------------------------------------------------------------------
# Rangfolge
# ---------------------------------------------------------------------------


def test_ranking_by_order_strength(settings: Settings) -> None:
    result = _run(settings, _three(), max_positions=2)
    held = sorted(p.symbol for p in result.open_positions)
    assert held == ["BBB", "CCC"]
    entries = _entries(result)
    first_day = entries["submitted"].min()
    first = entries[entries["submitted"] == first_day].set_index("symbol")
    assert first.loc["BBB", "status"] == "FILLED"
    assert first.loc["CCC", "status"] == "FILLED"
    assert first.loc["AAA", "status"] == "REJECTED"
    assert first.loc["AAA", "order_id"] == 0
    assert "Kein freier Platz" in first.loc["AAA", "reason"]
    # BBB (stärkstes Signal) wird zuerst ausgeführt.
    assert first.loc["BBB", "order_id"] < first.loc["CCC", "order_id"]


def test_ranking_single_slot_takes_strongest(settings: Settings) -> None:
    result = _run(settings, _three(), max_positions=1)
    assert [p.symbol for p in result.open_positions] == ["BBB"]


def test_ranking_by_indicator_column_and_tie_break(settings: Settings) -> None:
    class Strat(IndicatorStrength):
        LEVEL: ClassVar[dict[float, float]] = {100.0: 5.0, 50.0: 5.0, 20.0: 9.0}

    result = _run(settings, _three(), strategy=Strat, max_positions=2)
    # CCC hat die höchste Stärke; AAA vor BBB bei Gleichstand (alphabetisch).
    assert sorted(p.symbol for p in result.open_positions) == ["AAA", "CCC"]


def test_freed_slot_is_reused(settings: Settings) -> None:
    result = _run(settings, _three(), params={"hold": 5}, max_positions=1)
    trades = result.trades
    assert len(trades) >= 3
    # Nie mehr als eine Position gleichzeitig.
    for _, row in trades.iterrows():
        overlap = trades[
            (trades["entry_date"] < row["exit_date"]) & (trades["exit_date"] > row["entry_date"])
        ]
        assert len(overlap) == 1
    # Der frei werdende Platz geht an das nächststärkste Signal (CCC), nie an AAA.
    assert set(trades["symbol"]) == {"BBB", "CCC"}
    exits = set(trades["exit_date"])
    assert any(entry in exits for entry in trades["entry_date"])


# ---------------------------------------------------------------------------
# Mitgliedschaft
# ---------------------------------------------------------------------------


def _history(changes: list[IndexChange], current: set[str]) -> IndexHistory:
    return IndexHistory("TEST", frozenset(current), tuple(changes))


def test_only_members_may_open(settings: Settings) -> None:
    bars = _three()
    joined = date(2020, 2, 3)
    history = _history(
        [IndexChange(joined, frozenset({"CCC"}), frozenset())], {"AAA", "BBB", "CCC"}
    )
    result = _run(settings, bars, universe=history, max_positions=3)
    ccc = result.orders
    ccc = ccc[(ccc["symbol"] == "CCC") & (ccc["side"] == "BUY")]
    assert len(ccc) == 1
    assert ccc["submitted"].iloc[0].date() >= joined
    assert ccc["status"].iloc[0] == "FILLED"
    assert result.universe == "TEST"
    assert result.symbol == "TEST"


def test_non_member_entry_is_rejected(settings: Settings) -> None:
    class AlwaysBuy(HoldRanked):
        def on_bar(self, ctx: BarContext) -> list[Order]:
            if ctx.position is None:
                return [ctx.buy(sizing=Sizing.percent_equity(20))]
            return []

    removed = date(2020, 1, 3)  # AAA ab dem 2. Handelstag kein Mitglied mehr
    history = _history([IndexChange(removed, frozenset(), frozenset({"AAA"}))], {"BBB"})
    bars = {"AAA": _bars(_trend()), "BBB": _bars(_trend())}
    result = _run(settings, bars, universe=history, strategy=AlwaysBuy)
    aaa = result.orders[result.orders["symbol"] == "AAA"]
    # Am 2.1. noch Mitglied: Kauf; danach keine Aufrufe mehr ohne Position/Order.
    assert aaa["status"].tolist() == ["FILLED"]
    assert {p.symbol for p in result.open_positions} == {"AAA", "BBB"}


def test_removed_member_may_close_position(settings: Settings) -> None:
    seen: list[tuple[date, bool]] = []

    class Recorder(HoldRanked):
        def on_bar(self, ctx: BarContext) -> list[Order]:
            if ctx.symbol == "AAA":
                seen.append((ctx.date, ctx.is_member))
            return super().on_bar(ctx)

    removed = date(2020, 1, 10)
    history = _history([IndexChange(removed, frozenset(), frozenset({"AAA"}))], {"BBB"})
    bars = {"AAA": _bars(_trend()), "BBB": _bars(_trend())}
    result = _run(settings, bars, universe=history, strategy=Recorder, params={"hold": 10})
    aaa = result.trades[result.trades["symbol"] == "AAA"]
    assert len(aaa) == 1
    assert aaa["exit_reason"].iloc[0] == "Zeit"
    assert aaa["exit_date"].iloc[0].date() > removed
    assert any(not member for _, member in seen)
    # Nach dem Schließen wird AAA nicht mehr aufgerufen.
    assert max(d for d, _ in seen) <= aaa["exit_date"].iloc[0].date()


def test_index_code_universe_and_survivorship_warning(settings: Settings) -> None:
    bars = {"AAPL": _bars(_trend(30), start="1995-01-03")}
    result = _run(
        settings, bars, universe="SP500", start="1995-01-01", end="1995-03-31", max_positions=1
    )
    assert SURVIVORSHIP_WARNING in result.log
    assert any("ohne Kurse" in line for line in result.log)
    assert result.symbols == ("AAPL",)


# ---------------------------------------------------------------------------
# Delisting
# ---------------------------------------------------------------------------


def test_delisting_closes_at_last_close(settings: Settings) -> None:
    bars = _three()
    bars["BBB"] = bars["BBB"].iloc[:25].reset_index(drop=True)
    costs = CostModel(fee_per_order=1.0, slippage_pct=0.5)
    result = _run(settings, bars, max_positions=3, costs=costs)
    delisted = result.trades[result.trades["exit_reason"] == DELISTING_REASON]
    assert delisted["symbol"].tolist() == ["BBB"]
    last = bars["BBB"].iloc[-1]
    assert delisted["exit_date"].iloc[0] == last["date"]
    assert delisted["exit_price"].iloc[0] == pytest.approx(last["close"])  # ohne Slippage
    assert result.delisted == {"BBB": last["date"].date()}
    assert "BBB" not in {p.symbol for p in result.open_positions}
    order = result.orders[result.orders["tag"] == DELISTING_REASON]
    assert order["status"].tolist() == ["FILLED"]
    assert any("Delisting" in line for line in result.log)


def test_weekly_delisting_mid_week(settings: Settings) -> None:
    bars = {"AAA": _bars(_trend(200)), "BBB": _bars(_trend(200, 50.0))}
    bars["BBB"] = bars["BBB"].iloc[:105].reset_index(drop=True)  # endet an einem Mittwoch
    assert bars["BBB"]["date"].iloc[-1].day_name() == "Wednesday"
    result = _run(settings, bars, universe=["AAA", "BBB"], timeframe=Timeframe.W1)
    delisted = result.trades[result.trades["exit_reason"] == DELISTING_REASON]
    assert delisted["symbol"].tolist() == ["BBB"]
    assert delisted["exit_price"].iloc[0] == pytest.approx(bars["BBB"]["close"].iloc[-1])


def test_delisted_without_position_cancels_orders(settings: Settings) -> None:
    class StopEntry(HoldRanked):
        def on_bar(self, ctx: BarContext) -> list[Order]:
            if ctx.position is None and not self.__dict__.get("sent"):
                self.sent = True
                return [ctx.buy(qty=1, type="STOP", stop_price=10_000.0)]  # type: ignore[arg-type]
            return []

    bars = _three()
    bars["AAA"] = bars["AAA"].iloc[:10].reset_index(drop=True)
    result = _run(settings, bars, strategy=StopEntry)
    aaa = result.orders[result.orders["symbol"] == "AAA"]
    assert aaa["status"].tolist() == ["CANCELLED"]
    assert result.delisted["AAA"] == bars["AAA"]["date"].iloc[-1].date()


# ---------------------------------------------------------------------------
# Bilanz
# ---------------------------------------------------------------------------


def test_balance_equation_with_costs(settings: Settings) -> None:
    rng = np.random.default_rng(3)
    bars = {
        sym: _bars(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 250))))
        for sym in ("AAA", "BBB", "CCC", "DDD")
    }
    bars["DDD"] = bars["DDD"].iloc[:120].reset_index(drop=True)
    costs = CostModel(fee_per_order=1.0, fee_pct=0.1, slippage_pct=0.05)
    result = _run(settings, bars, params={"hold": 7}, max_positions=2, costs=costs)
    assert len(result.trades) > 10
    assert (result.trades["fees"] > 0).all()
    assert result.open_pnl == pytest.approx(0.0, abs=1e-6) or result.open_positions
    realized = float(result.trades["pnl"].sum())
    gross = float(
        ((result.trades["exit_price"] - result.trades["entry_price"]) * result.trades["qty"]).sum()
    )
    fees = float(result.trades["fees"].sum())
    assert realized == pytest.approx(gross - fees)
    if not result.open_positions:
        assert realized == pytest.approx(result.final_equity - result.initial_capital)
    assert realized + result.open_pnl == pytest.approx(result.final_equity - result.initial_capital)


def test_balance_equation_all_closed(settings: Settings) -> None:
    bars = _three(40)
    for sym in bars:
        bars[sym] = bars[sym].iloc[: 30 + len(sym)].reset_index(drop=True)
    bars["ZZZ"] = _bars(_trend(40, 10.0))  # verlängert die Zeitachse: alle anderen delistet
    # ZZZ ist nur am ersten Tag Mitglied (dort schwächstes Signal) und wird nie gekauft.
    history = _history(
        [IndexChange(date(2020, 1, 3), frozenset(), frozenset({"ZZZ"}))], {"AAA", "BBB", "CCC"}
    )
    costs = CostModel(fee_per_order=1.0, slippage_pct=0.05)
    result = _run(settings, bars, universe=history, max_positions=3, costs=costs)
    assert not result.open_positions
    total = float(result.trades["pnl"].sum())
    gross = float(
        ((result.trades["exit_price"] - result.trades["entry_price"]) * result.trades["qty"]).sum()
    )
    fees = float(result.trades["fees"].sum())
    assert gross - fees == pytest.approx(result.final_equity - result.initial_capital)
    assert total == pytest.approx(result.final_equity - result.initial_capital)
    assert result.open_pnl == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Ergebnis, Determinismus, Parallelisierung
# ---------------------------------------------------------------------------


def test_result_shape_and_determinism(settings: Settings) -> None:
    a = _run(settings, _three(), params={"hold": 4}, max_positions=2)
    b = _run(settings, _three(), params={"hold": 4}, max_positions=2, workers=2)
    assert a.fingerprint() == b.fingerprint()
    assert tuple(a.trades.columns) == TRADE_COLUMNS
    assert tuple(a.equity.columns) == EQUITY_COLUMNS
    assert tuple(a.orders.columns) == ORDER_COLUMNS
    assert a.symbols == ("AAA", "BBB", "CCC")
    assert a.settings["max_positions"] == 2
    assert a.max_positions == 2
    assert a.universe == "Liste"
    assert (a.equity["drawdown"] <= 0).all()
    assert a.warmup_end == date(2020, 1, 2)
    c = _run(settings, _three(), params={"hold": 4}, max_positions=1)
    assert c.fingerprint() != a.fingerprint()


def _walks(n: int = 600) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(11)
    return {
        sym: _bars(100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n))))
        for sym in ("AAA", "BBB", "CCC", "DDD")
    }


def _sma_run(settings: Settings, workers: int) -> PortfolioResult:
    from trading_helfer.strategies.builtin.sma_cross import SmaCross

    return _run(
        settings,
        _walks(),
        universe=["AAA", "BBB", "CCC", "DDD"],
        strategy=SmaCross,
        params={"fast": 5, "slow": 10},
        timeframe=Timeframe.W1,
        max_positions=2,
        workers=workers,
    )


def test_process_pool_matches_serial(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_helfer.strategies.builtin.sma_cross import SmaCross

    monkeypatch.setattr(portfolio_module, "_PROCESS_MIN_SYMBOLS", 2)
    assert portfolio_module._process_safe(SmaCross())
    serial = _sma_run(settings, workers=1)
    parallel = _sma_run(settings, workers=2)
    assert len(serial.trades) > 0
    assert serial.fingerprint() == parallel.fingerprint()


def test_process_safe_rejects_unimportable_strategies() -> None:
    class Local(HoldRanked):  # lokale Klasse lässt sich nicht pickeln
        pass

    main_cls = type("MainStrat", (HoldRanked,), {"__module__": "__main__"})
    missing_cls = type("Missing", (HoldRanked,), {"__module__": "gibt_es_nicht.modul"})
    assert not portfolio_module._process_safe(Local())
    assert not portfolio_module._process_safe(main_cls())
    assert not portfolio_module._process_safe(missing_cls())


def test_unimportable_strategy_uses_threads(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(portfolio_module, "_PROCESS_MIN_SYMBOLS", 2)
    threaded = _run(settings, _three(), params={"hold": 4}, max_positions=2, workers=2)
    serial = _run(settings, _three(), params={"hold": 4}, max_positions=2, workers=1)
    assert threaded.fingerprint() == serial.fingerprint()


def test_broken_process_pool_falls_back(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from concurrent.futures.process import BrokenProcessPool

    class Broken:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise BrokenProcessPool("kaputt")

    monkeypatch.setattr(portfolio_module, "_PROCESS_MIN_SYMBOLS", 2)
    monkeypatch.setattr(portfolio_module, "ProcessPoolExecutor", Broken)
    result = _sma_run(settings, workers=2)
    assert "Prozess-Pool" in caplog.text
    assert result.fingerprint() == _sma_run(settings, workers=1).fingerprint()


def test_strategy_state_is_per_symbol(settings: Settings) -> None:
    class OncePerInstance(HoldRanked):
        def on_bar(self, ctx: BarContext) -> list[Order]:
            if self.__dict__.get("done"):
                return []
            self.done = True
            return [ctx.buy(qty=1)]

    result = _run(settings, _three(), strategy=OncePerInstance(), max_positions=3)
    assert len(result.open_positions) == 3


def test_load_from_database(settings: Settings) -> None:
    bars = _three()
    with MarketStore(settings.database_path) as store:
        for sym, frame in bars.items():
            store.upsert_bars(sym, frame, source="test")
    from_db = run_portfolio_backtest(
        HoldRanked,
        None,
        ["AAA", "BBB", "CCC", "XXX"],
        Timeframe.D1,
        START,
        END,
        settings,
        max_positions=2,
        costs=NO_COSTS,
        workers=1,
    )
    direct = _run(settings, bars, universe=["AAA", "BBB", "CCC", "XXX"], max_positions=2)
    assert from_db.final_equity == pytest.approx(direct.final_equity)
    assert any("XXX" in line for line in from_db.log)


def test_missing_database(settings: Settings) -> None:
    with pytest.raises(BacktestError, match="Datenbank"):
        run_portfolio_backtest(HoldRanked, None, ["AAA"], Timeframe.D1, START, END, settings)


# ---------------------------------------------------------------------------
# Fehlerfälle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_positions": 0}, "max_positions"),
        ({"start": "2021-01-01", "end": "2020-01-01"}, "nach dem Ende"),
        ({"start": "kein Datum"}, "kein gültiges Datum"),
        ({"timeframe": Timeframe.MN1}, "unterstützt den Zeitrahmen"),
        ({"universe": []}, "keine Symbole"),
        ({"universe": ["QQQ"]}, "Keine Kurse"),
        ({"start": "2029-01-01"}, "Keine abgeschlossenen"),
    ],
)
def test_invalid_inputs(settings: Settings, kwargs: dict[str, object], match: str) -> None:
    universe = kwargs.pop("universe", ("AAA", "BBB", "CCC"))
    with pytest.raises(BacktestError, match=match):
        _run(settings, _three(), universe=universe, **kwargs)


def test_empty_bars_are_skipped(settings: Settings) -> None:
    bars = _three()
    bars["DDD"] = bars["AAA"].iloc[:0]
    bars["EEE"] = _bars(_trend(5), start="2019-01-02")  # endet vor dem Start
    result = _run(settings, bars, universe=["AAA", "BBB", "CCC", "DDD", "EEE"], start="2020-01-01")
    assert any("übersprungen" in line for line in result.log)
    assert "DDD" not in result.symbols


def test_indicators_never_complete(settings: Settings) -> None:
    class NeverReady(HoldRanked):
        def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
            self.hold, self.pct = int(p.hold), float(p.pct)  # Params lässt sich nicht pickeln
            return pd.DataFrame({"x": np.full(len(bars), np.nan)})

    result = _run(settings, _three(), strategy=NeverReady)
    assert result.trades.empty
    assert result.warmup_end is None
    assert any("nie vollständig" in line for line in result.log)


class _Bad(HoldRanked):
    MODE: ClassVar[str] = ""

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        self.hold, self.pct = int(p.hold), float(p.pct)  # Params lässt sich nicht pickeln
        if self.MODE == "ind_raise":
            raise RuntimeError("kaputt")
        if self.MODE == "ind_type":
            return [1, 2]  # type: ignore[return-value]
        if self.MODE == "ind_len":
            return pd.DataFrame({"x": [1.0]})
        return pd.DataFrame(index=bars.index)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        mode = self.MODE
        if mode == "raise":
            raise RuntimeError("Fehler")
        if mode == "peek":
            return [ctx.buy(qty=1)] if ctx.bars.close[len(ctx.bars)] > 0 else []
        if mode == "none":
            return None  # type: ignore[return-value]
        if mode == "type":
            return "x"  # type: ignore[return-value]
        if mode == "item":
            return [1]  # type: ignore[list-item]
        if mode == "other":
            return [Order("ZZZ", OrderSide.BUY, qty=1)]
        return []


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("ind_raise", StrategyError),
        ("ind_type", StrategyError),
        ("ind_len", StrategyError),
        ("raise", StrategyError),
        ("peek", LookaheadError),
        ("type", StrategyError),
        ("item", StrategyError),
        ("other", StrategyError),
    ],
)
def test_strategy_errors(settings: Settings, mode: str, error: type[Exception]) -> None:
    strat = type("BadStrat", (_Bad,), {"MODE": mode})
    with pytest.raises(error):
        _run(settings, _three(), strategy=strat)


def test_on_bar_none_is_no_orders(settings: Settings) -> None:
    strat = type("NoneStrat", (_Bad,), {"MODE": "none"})
    result = _run(settings, _three(), strategy=strat)
    assert result.orders.empty


def test_broker_rejections_are_logged(settings: Settings) -> None:
    result = _run(settings, _three(), params={"pct": 100.0}, max_positions=3)
    # Erst BBB mit 100 %, danach reicht das Kapital für CCC/AAA nicht mehr.
    assert (result.orders["status"] == "REJECTED").any()
    assert any("Kapital reicht nicht" in line for line in result.log)


def test_exit_only_orders_for_symbol_without_position(settings: Settings) -> None:
    class ExitOnly(HoldRanked):
        def on_bar(self, ctx: BarContext) -> list[Order]:
            return [Order(ctx.symbol, OrderSide.SELL, type="STOP", stop_price=1.0)]  # type: ignore[arg-type]

    result = _run(settings, _three(), strategy=ExitOnly, end="2020-01-06")
    assert (result.orders["status"] == "OPEN").all()
    assert any("noch offen" in line for line in result.log)


# ---------------------------------------------------------------------------
# Order.strength und Broker
# ---------------------------------------------------------------------------


def test_order_strength_validation() -> None:
    assert Order("AAA", OrderSide.BUY, qty=1, strength=2.5).strength == 2.5
    with pytest.raises(OrderError, match="strength"):
        Order("AAA", OrderSide.BUY, qty=1, strength=float("nan"))


def test_force_close_without_position() -> None:
    broker = SimBroker(1000.0, NO_COSTS)
    assert broker.force_close("AAA", date(2020, 1, 2), 10.0, "x") is None
