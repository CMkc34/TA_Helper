"""Tests der Backtest-Engine für einen Wert (B-03)."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from trading_helfer.backtest.costs import CostModel
from trading_helfer.backtest.engine import (
    EQUITY_COLUMNS,
    ORDER_COLUMNS,
    TRADE_COLUMNS,
    BacktestError,
    LookaheadError,
    StrategyError,
    run_backtest,
)
from trading_helfer.core.config import Settings
from trading_helfer.core.types import OrderType, Timeframe
from trading_helfer.data.storage import MarketStore
from trading_helfer.strategies.base import (
    BarContext,
    Order,
    OrderSide,
    Param,
    Params,
    Sizing,
    Strategy,
)

ALL_TIMEFRAMES = [Timeframe.D1, Timeframe.D3, Timeframe.W1, Timeframe.MN1]
NO_COSTS = CostModel(fee_per_order=0.0, slippage_pct=0.0)


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


def _sawtooth(cycles: int = 10, half: int = 10) -> np.ndarray:
    up = np.linspace(100.0, 120.0, half, endpoint=False)
    down = np.linspace(120.0, 100.0, half, endpoint=False)
    return np.tile(np.concatenate([up, down]), cycles)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


class BuyAndHold(Strategy):
    id = "test_buy_and_hold"
    name = "Kaufen und Halten"
    version = "1.0.0"
    description = "Kauft auf der ersten Kerze mit dem ganzen Kapital."
    timeframes: ClassVar[list[Timeframe]] = ALL_TIMEFRAMES

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and not self._bought:
            self._bought = True
            return [ctx.buy(sizing=Sizing.percent_equity(100))]
        return []

    def __init__(self) -> None:
        self._bought = False


class SmaCross(Strategy):
    id = "test_sma_cross"
    name = "SMA-Kreuzung"
    version = "1.0.0"
    description = "SMA-Kreuzung für Tests."
    timeframes: ClassVar[list[Timeframe]] = ALL_TIMEFRAMES
    params: ClassVar[dict[str, Param]] = {
        "fast": Param(3, 1, 50, 1, "schnell"),
        "slow": Param(8, 2, 200, 1, "langsam"),
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        close = bars["close"]
        return pd.DataFrame(
            {"fast": close.rolling(p.fast).mean(), "slow": close.rolling(p.slow).mean()}
        )

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.position is None and ctx.crossed_above("fast", "slow"):
            return [ctx.buy(sizing=Sizing.percent_equity(100), tag="Kreuzung oben")]
        if ctx.position is not None and ctx.crossed_below("fast", "slow"):
            return ctx.exit(tag="Kreuzung unten")
        return []


class PeekAhead(Strategy):
    id = "test_peek_ahead"
    name = "Blick in die Zukunft"
    version = "1.0.0"
    description = "Greift auf t+1 zu."
    timeframes: ClassVar[list[Timeframe]] = ALL_TIMEFRAMES

    def on_bar(self, ctx: BarContext) -> list[Order]:
        t = len(ctx.bars) - 1
        tomorrow = ctx.bars.close[t + 1]
        return [ctx.buy(qty=1)] if tomorrow > ctx.bars.close[t] else []


def _expected_sma_trades(close: np.ndarray, fast: int, slow: int) -> int:
    """Unabhängige, vektorisierte Zählung abgeschlossener Kreuzungs-Trades."""
    s = pd.Series(close)
    f, sl = s.rolling(fast).mean().to_numpy(), s.rolling(slow).mean().to_numpy()
    above = (f[:-1] <= sl[:-1]) & (f[1:] > sl[1:])
    below = (f[:-1] >= sl[:-1]) & (f[1:] < sl[1:])
    # Signal auf Kerze t+1 (Index i+1); Ausführung auf der Folgekerze, muss existieren.
    holding, trades = False, 0
    for i in range(len(above)):
        t = i + 1
        if t + 1 >= len(close):
            break
        if not holding and above[i]:
            holding = True
        elif holding and below[i]:
            holding = False
            trades += 1
    return trades


# ---------------------------------------------------------------------------
# Akzeptanzkriterien
# ---------------------------------------------------------------------------


def test_buy_and_hold_without_costs(settings: Settings) -> None:
    bars = _bars(np.linspace(50.0, 80.0, 300))
    result = run_backtest(
        BuyAndHold(),
        None,
        "SYN",
        Timeframe.D1,
        "2020-01-01",
        "2030-12-31",
        settings,
        bars=bars,
        costs=NO_COSTS,
    )
    open_start, close_end = bars["open"].iloc[1], bars["close"].iloc[-1]
    expected = 1000.0 * close_end / open_start
    assert result.final_equity == pytest.approx(expected, rel=1e-4)
    assert len(result.open_positions) == 1
    assert result.trades.empty


def test_buy_and_hold_with_costs(settings: Settings) -> None:
    bars = _bars(np.linspace(50.0, 80.0, 300))
    result = run_backtest(
        BuyAndHold,
        {},
        "SYN",
        Timeframe.D1,
        date(2020, 1, 1),
        date(2030, 12, 31),
        settings,
        bars=bars,
    )
    assert settings.fee_per_order == 1.0
    open_start, close_end = bars["open"].iloc[1], bars["close"].iloc[-1]
    slip = settings.slippage_pct / 100.0
    # 1 USD Gebühr plus 1 USD Gebührenreserve für den späteren Ausstieg
    qty = math.floor((1000.0 - 2.0) / (open_start * (1 + slip)) * 1e4) / 1e4
    expected = 1000.0 - 1.0 - qty * open_start * (1 + slip) + qty * close_end
    assert result.final_equity == pytest.approx(expected, rel=1e-4)
    assert result.final_equity == pytest.approx(1000.0 * close_end / open_start, rel=2e-3)
    assert result.final_equity < 1000.0 * close_end / open_start


def test_sma_cross_on_sawtooth_has_known_trade_count(settings: Settings) -> None:
    close = _sawtooth(cycles=10, half=10)
    result = run_backtest(
        SmaCross(),
        {"fast": 3, "slow": 8},
        "SAW",
        Timeframe.D1,
        "2000-01-01",
        "2030-01-01",
        settings,
        bars=_bars(close),
        costs=NO_COSTS,
    )
    assert len(result.trades) == _expected_sma_trades(close, 3, 8)
    assert len(result.trades) == 9
    assert set(result.trades["exit_reason"]) == {"Kreuzung unten"}
    assert (result.trades["entry_date"] < result.trades["exit_date"]).all()
    # Warm-up: erstes Signal frühestens auf der Kerze, auf der die langsame SMA steht.
    assert result.warmup_end == _bars(close)["date"].iloc[7].date()


def test_lookahead_trap_raises(settings: Settings) -> None:
    with pytest.raises(LookaheadError, match="nach der aktuellen Kerze"):
        run_backtest(
            PeekAhead(),
            None,
            "SYN",
            Timeframe.D1,
            "2020-01-01",
            "2030-01-01",
            settings,
            bars=_bars(np.linspace(10, 20, 50)),
        )


def test_determinism_hash(settings: Settings) -> None:
    bars = _bars(_sawtooth(cycles=6, half=7))

    def run() -> str:
        return run_backtest(
            SmaCross(),
            {"fast": 2, "slow": 5},
            "SAW",
            Timeframe.W1,
            "2020-01-01",
            "2030-01-01",
            settings,
            bars=bars,
        ).fingerprint()

    first = run()
    assert first == run()
    other = run_backtest(
        SmaCross(),
        {"fast": 2, "slow": 6},
        "SAW",
        Timeframe.W1,
        "2020-01-01",
        "2030-01-01",
        settings,
        bars=bars,
    ).fingerprint()
    assert other != first


# ---------------------------------------------------------------------------
# Ablauf und Ergebnis
# ---------------------------------------------------------------------------


class RecordingStrategy(Strategy):
    id = "test_recording"
    name = "Protokoll"
    version = "0.1.0"
    description = "Merkt sich die Kontexte."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.W1, Timeframe.D1]
    params: ClassVar[dict[str, Param]] = {"period": Param(3, 1, 10, 1, "SMA")}

    def __init__(self) -> None:
        self.seen: list[tuple[date, int, float]] = []

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        return pd.DataFrame({"sma": bars["close"].rolling(p.period).mean()})

    def on_bar(self, ctx: BarContext) -> list[Order]:
        self.seen.append((ctx.date, len(ctx.bars), float(ctx.ind["sma"])))
        return []


def test_strategy_sees_only_closed_bars_from_start(settings: Settings) -> None:
    daily = _bars(np.linspace(100, 130, 60), start="2024-01-01")
    strat = RecordingStrategy()
    result = run_backtest(
        strat,
        {"period": 3},
        "SYN",
        Timeframe.W1,
        "2024-01-20",
        "2024-03-20",
        settings,
        bars=daily,
    )
    dates = [d for d, _, _ in strat.seen]
    assert dates[0] >= date(2024, 1, 20)
    assert dates[-1] <= date(2024, 3, 15)  # Woche bis 2024-03-20 ist unvollständig
    assert any("Unvollständige letzte Kerze" in line for line in result.log)
    # Indikatoren nutzen die Historie vor start: kein NaN schon auf der ersten Kerze.
    assert not any(math.isnan(v) for _, _, v in strat.seen)
    assert strat.seen[0][1] > 1
    assert list(result.equity.columns) == list(EQUITY_COLUMNS)
    assert result.equity["date"].iloc[0] == pd.Timestamp(dates[0])
    assert (result.equity["equity"] == 1000.0).all()
    assert (result.equity["drawdown"] == 0.0).all()
    assert result.total_return_pct == 0.0


def test_warmup_inside_period_is_logged(settings: Settings) -> None:
    strat = RecordingStrategy()
    result = run_backtest(
        strat,
        {"period": 5},
        "SYN",
        Timeframe.D1,
        "2020-01-01",
        "2030-01-01",
        settings,
        bars=_bars(np.linspace(1, 2, 20)),
    )
    assert len(strat.seen) == 16
    assert any("Warm-up" in line for line in result.log)
    assert len(result.equity) == 20


def test_indicators_never_complete(settings: Settings) -> None:
    strat = RecordingStrategy()
    result = run_backtest(
        strat,
        {"period": 10},
        "SYN",
        Timeframe.D1,
        "2020-01-01",
        "2030-01-01",
        settings,
        bars=_bars(np.linspace(1, 2, 5)),
    )
    assert strat.seen == []
    assert result.warmup_end is None
    assert any("nie vollständig" in line for line in result.log)


class StopEntry(Strategy):
    id = "test_stop_entry"
    name = "Stop-Einstieg mit Risiko"
    version = "1.0.0"
    description = "Kauft mit Schutz-Stop und Risiko-Sizing, Ziel per Limit."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1]

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if len(ctx.bars) == 2:
            close = ctx.bars.close[-1]
            return [
                Order(ctx.symbol, OrderSide.SELL, OrderType.STOP, stop_price=close * 0.9),
                ctx.buy(sizing=Sizing.risk(1)),
                Order(ctx.symbol, OrderSide.SELL, OrderType.LIMIT, limit_price=close * 1.05),
                ctx.buy(sizing=Sizing.fixed_amount(10_000_000)),
            ]
        return []


def test_orders_frame_and_rejections(settings: Settings) -> None:
    result = run_backtest(
        StopEntry(),
        None,
        "SYN",
        Timeframe.D1,
        "2020-01-01",
        "2030-01-01",
        settings,
        bars=_bars(np.linspace(100, 120, 30)),
    )
    orders = result.orders
    assert list(orders.columns) == list(ORDER_COLUMNS)
    assert list(orders["status"]) == ["CANCELLED", "FILLED", "FILLED", "REJECTED"]
    assert "Kapital reicht nicht" in orders["reason"].iloc[3]
    assert any("verworfen" in line for line in result.log)
    assert list(result.trades.columns) == list(TRADE_COLUMNS)
    assert len(result.trades) == 1
    assert result.trades["exit_reason"].iloc[0] == "LIMIT"
    assert orders["sizing"].iloc[1] == "RISK:1.0"


class TrailingAtr(Strategy):
    id = "test_trailing_atr"
    name = "Trailing ATR"
    version = "1.0.0"
    description = "Trailing Stop in ATR-Vielfachen und Volatilitäts-Ziel."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1]

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if len(ctx.bars) == 25:
            return [ctx.buy(sizing=Sizing.volatility_target(10))]
        if len(ctx.bars) == 26 and ctx.position is not None:
            return ctx.exit(type=OrderType.TRAILING_STOP, trail_atr=2.0)
        return []


def test_trailing_atr_and_volatility_target(settings: Settings) -> None:
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.02, 80)))
    close[60:] = close[59] * np.linspace(1.0, 0.6, 20)
    result = run_backtest(
        TrailingAtr(),
        None,
        "SYN",
        Timeframe.D1,
        "2020-01-01",
        "2030-01-01",
        settings,
        bars=_bars(close),
    )
    assert list(result.orders["status"]) == ["FILLED", "FILLED"]
    assert result.trades["exit_reason"].iloc[0] == "TRAILING_STOP"
    assert result.open_positions == []
    assert (result.equity["drawdown"] <= 0).all()
    assert result.equity["exposure"].max() > 0


def test_open_order_at_end_is_logged(settings: Settings) -> None:
    class LastBarBuy(BuyAndHold):
        def on_bar(self, ctx: BarContext) -> list[Order]:
            return [ctx.buy(qty=1)] if len(ctx.bars) == 10 else []

    result = run_backtest(
        LastBarBuy(),
        None,
        "SYN",
        Timeframe.D1,
        "2020-01-01",
        "2030-01-01",
        settings,
        bars=_bars(np.linspace(1, 2, 10)),
    )
    assert list(result.orders["status"]) == ["OPEN"]
    assert any("am Ende noch offen" in line for line in result.log)


# ---------------------------------------------------------------------------
# Fehlerfälle
# ---------------------------------------------------------------------------


class Broken(Strategy):
    id = "test_broken"
    name = "Kaputt"
    version = "1.0.0"
    description = "Liefert Fehler."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1]
    params: ClassVar[dict[str, Param]] = {
        "mode": Param("raise", choices=("raise", "type", "item", "symbol", "none", "ind"))
    }

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        if p.mode == "ind":
            raise RuntimeError("kaputt")
        return pd.DataFrame(index=bars.index)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        mode = self.mode
        if mode == "raise":
            raise ZeroDivisionError("durch null")
        if mode == "type":
            return "BUY"  # type: ignore[return-value]
        if mode == "item":
            return ["BUY"]  # type: ignore[list-item]
        if mode == "symbol":
            return [Order("ANDERS", OrderSide.BUY, qty=1)]
        return None  # type: ignore[return-value]

    mode = "raise"


@pytest.mark.parametrize(
    ("mode", "error", "message"),
    [
        ("raise", StrategyError, "fehlgeschlagen"),
        ("type", StrategyError, "Liste von Orders"),
        ("item", StrategyError, "statt Order"),
        ("symbol", StrategyError, "gehandelt wird nur"),
        ("ind", StrategyError, "Indikatoren"),
    ],
)
def test_strategy_errors(
    settings: Settings, mode: str, error: type[Exception], message: str
) -> None:
    strat = Broken()
    strat.mode = mode
    with pytest.raises(error, match=message):
        run_backtest(
            strat,
            {"mode": mode},
            "SYN",
            Timeframe.D1,
            "2020-01-01",
            "2030-01-01",
            settings,
            bars=_bars(np.linspace(1, 2, 10)),
        )


def test_none_from_on_bar_is_tolerated(settings: Settings) -> None:
    strat = Broken()
    strat.mode = "none"
    result = run_backtest(
        strat,
        None,
        "SYN",
        Timeframe.D1,
        "2020-01-01",
        "2030-01-01",
        settings,
        bars=_bars(np.linspace(1, 2, 10)),
    )
    assert result.orders.empty


class BadIndicators(Strategy):
    id = "test_bad_ind"
    name = "Falsche Indikatoren"
    version = "1.0.0"
    description = "Liefert falsche Indikatoren."
    timeframes: ClassVar[list[Timeframe]] = [Timeframe.D1]
    params: ClassVar[dict[str, Param]] = {"short": Param(False)}

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        if p.short:
            return pd.DataFrame({"x": [1.0]})
        return [1, 2]  # type: ignore[return-value]

    def on_bar(self, ctx: BarContext) -> list[Order]:
        return []


@pytest.mark.parametrize(("short", "message"), [(True, "Zeilen"), (False, "DataFrame")])
def test_bad_indicator_output(settings: Settings, short: bool, message: str) -> None:
    with pytest.raises(StrategyError, match=message):
        run_backtest(
            BadIndicators(),
            {"short": short},
            "SYN",
            Timeframe.D1,
            "2020-01-01",
            "2030-01-01",
            settings,
            bars=_bars(np.linspace(1, 2, 10)),
        )


@pytest.mark.parametrize(
    ("timeframe", "start", "end", "message"),
    [
        (Timeframe.MN1, "2020-01-01", "2030-01-01", "unterstützt den Zeitrahmen"),
        (Timeframe.D1, "2021-01-01", "2020-01-01", "liegt nach dem Ende"),
        (Timeframe.D1, "kein Datum", "2020-01-01", "kein gültiges Datum"),
        (Timeframe.D1, "2025-01-01", "2030-01-01", "Keine Kerzen"),
        (Timeframe.D1, "2010-01-01", "2010-12-31", "Keine abgeschlossenen"),
    ],
)
def test_invalid_inputs(
    settings: Settings, timeframe: Timeframe, start: str, end: str, message: str
) -> None:
    with pytest.raises(BacktestError, match=message):
        run_backtest(
            RecordingStrategy(),
            None,
            "SYN",
            timeframe,
            start,
            end,
            settings,
            bars=_bars(np.linspace(1, 2, 30)),
        )


def test_nat_date_rejected(settings: Settings) -> None:
    with pytest.raises(BacktestError, match="kein gültiges Datum"):
        run_backtest(
            RecordingStrategy(),
            None,
            "SYN",
            Timeframe.D1,
            pd.NaT,
            "2020-01-01",
            settings,
            bars=_bars(np.linspace(1, 2, 30)),
        )


# ---------------------------------------------------------------------------
# Laden aus der Datenbank
# ---------------------------------------------------------------------------


def test_loads_bars_from_database(settings: Settings) -> None:
    bars = _bars(np.linspace(50.0, 80.0, 120))
    with MarketStore(settings.database_path) as store:
        store.upsert_bars("SYN", bars, source="test")
    from_db = run_backtest(
        BuyAndHold(), None, "SYN", Timeframe.D1, "2020-01-01", "2020-04-30", settings
    )
    direct = run_backtest(
        BuyAndHold(),
        None,
        "SYN",
        Timeframe.D1,
        "2020-01-01",
        "2020-04-30",
        settings,
        bars=bars,
    )
    assert from_db.final_equity == pytest.approx(direct.final_equity)
    assert from_db.equity["date"].iloc[-1] == pd.Timestamp("2020-04-30")


def test_missing_database(settings: Settings) -> None:
    with pytest.raises(BacktestError, match="existiert nicht"):
        run_backtest(BuyAndHold(), None, "SYN", Timeframe.D1, "2020-01-01", "2020-04-30", settings)
