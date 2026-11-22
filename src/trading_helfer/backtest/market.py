"""Marktfilter: Kurse und Indikatoren eines Markt-Symbols (z. B. SPY) für Strategien.

Eine Strategie mit ``market_symbol`` bekommt zu jeder Kerze die Spalte ``market_close``
und die Spalten aus :meth:`~trading_helfer.strategies.base.Strategy.market_indicators`
(mit Präfix ``market_``) in ihren Indikatoren zu sehen. Die Marktkurse werden in den
Zeitrahmen der Strategie umgerechnet; je Kerze des Werts gilt die letzte abgeschlossene
Marktkerze mit Datum bis einschließlich des Kerzendatums. So sieht die Strategie nie
Marktdaten nach ``t``.

Beispiel::

    market = load_market_daily(strategy, settings, end, market=None, bars=bars, log=log)
    table = market_table(strategy, params, market, Timeframe.W1)
    ind = attach_market(ind, frame["date"], table)
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date as Date  # noqa: N812
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from trading_helfer.core.bars import DATE_COLUMN
from trading_helfer.core.types import Timeframe
from trading_helfer.data.resample import PARTIAL_COLUMN, resample
from trading_helfer.strategies.base import Params, Strategy

if TYPE_CHECKING:
    from trading_helfer.core.config import Settings

__all__ = [
    "MARKET_PREFIX",
    "MarketError",
    "attach_market",
    "load_market_daily",
    "market_table",
]

MARKET_PREFIX: Final = "market_"
"""Präfix der Marktspalten in den Indikatoren einer Strategie."""


class MarketError(ValueError):
    """``market_indicators`` ist fehlgeschlagen oder liefert Ungültiges; Text auf Deutsch."""


def load_market_daily(
    strategy: Strategy,
    settings: Settings,
    end: Date,
    *,
    market: pd.DataFrame | None,
    bars: Mapping[str, pd.DataFrame] | None,
    log: list[str],
) -> pd.DataFrame | None:
    """Bereinigte Tageskurse des Markt-Symbols der Strategie.

    Reihenfolge: ``market``, dann ``bars[strategy.market_symbol]``, dann die Datenbank.
    Fehlen die Kurse, steht ein Hinweis in ``log`` und das Ergebnis ist ``None``; die
    Strategie läuft dann ohne Marktspalten.

    Args:
        strategy: Strategie; ohne ``market_symbol`` ist das Ergebnis ``None``.
        settings: Einstellungen (Pfad der Datenbank).
        end: Letzter Tag der Daten (einschließlich).
        market: Vorgegebene Tageskurse des Markts.
        bars: Tageskurse je Symbol, die dem Backtest übergeben wurden.
        log: Protokoll des Laufs.

    Returns:
        Tageskurse bis ``end`` oder ``None``.
    """
    symbol = strategy.market_symbol
    if not symbol:
        return None
    frame = market
    if frame is None and bars is not None:
        frame = bars.get(symbol)
    if frame is None:
        frame = _load_from_store(settings, symbol, end)
    if frame is None or frame.empty:
        log.append(f"Keine Kurse für den Markt {symbol}: Marktfilter ist ausgeschaltet.")
        return None
    dates = pd.to_datetime(frame[DATE_COLUMN])
    frame = frame.loc[(dates <= pd.Timestamp(end)).to_numpy()]
    if frame.empty:
        log.append(f"Keine Kurse für den Markt {symbol} bis {end}: Marktfilter ist ausgeschaltet.")
        return None
    return frame.reset_index(drop=True)


def _load_from_store(settings: Settings, symbol: str, end: Date) -> pd.DataFrame | None:
    db_path = settings.db_path  # ohne Datenordner (z. B. im Validator) nicht gesetzt
    if db_path is None or not Path(db_path).exists():
        return None
    path = Path(db_path)
    from trading_helfer.data.storage import MarketStore

    with MarketStore(path, read_only=True) as store:
        frame = store.load_bars(symbol, end=end, adjusted=True)
    return frame if len(frame) else None


def market_table(
    strategy: Strategy,
    params: Params,
    daily: pd.DataFrame | None,
    timeframe: Timeframe,
) -> pd.DataFrame | None:
    """Marktkerzen im Zeitrahmen mit ``market_close`` und den Marktindikatoren.

    Unvollständige letzte Kerzen werden verworfen.

    Args:
        strategy: Strategie mit :meth:`~Strategy.market_indicators`.
        params: Geprüfte Parameter.
        daily: Tageskurse des Markts (``None`` = kein Markt).
        timeframe: Zeitrahmen der Strategie.

    Returns:
        DataFrame mit Spalte ``date`` und Spalten mit Präfix ``market_`` oder ``None``.

    Raises:
        MarketError: ``market_indicators`` löst einen Fehler aus oder liefert kein
            DataFrame mit passender Länge.
    """
    if daily is None:
        return None
    frame = resample(daily, timeframe)
    frame = frame.loc[~frame[PARTIAL_COLUMN].to_numpy(dtype=bool)]
    frame = frame.drop(columns=[PARTIAL_COLUMN]).reset_index(drop=True)
    if frame.empty:
        return None
    try:
        ind = strategy.market_indicators(frame.copy(), params)
    except Exception as exc:
        raise MarketError(
            f"Strategie {strategy.id!r}: Marktindikatoren konnten nicht berechnet werden: {exc}"
        ) from exc
    if not isinstance(ind, pd.DataFrame):
        raise MarketError(
            f"Strategie {strategy.id!r}: market_indicators() muss ein DataFrame liefern, "
            f"erhalten: {type(ind).__name__}."
        )
    if len(ind) != len(frame):
        raise MarketError(
            f"Strategie {strategy.id!r}: market_indicators() liefert {len(ind)} Zeilen, "
            f"erwartet {len(frame)}."
        )
    table = pd.DataFrame(
        {
            DATE_COLUMN: _ns(frame[DATE_COLUMN]),
            f"{MARKET_PREFIX}close": frame["close"].to_numpy(dtype=float),
        }
    )
    for name in ind.columns:
        column = f"{MARKET_PREFIX}{name}"
        if column in table.columns:
            continue
        table[column] = ind[name].to_numpy()
    return table


def attach_market(
    ind: pd.DataFrame, dates: pd.Series | pd.DatetimeIndex, table: pd.DataFrame | None
) -> pd.DataFrame:
    """Hängt die Marktspalten je Kerze an die Indikatoren eines Werts an.

    Je Kerze gilt die letzte Marktkerze mit Datum bis einschließlich des Kerzendatums;
    davor sind die Werte ``NaN``. Bestehende Spalten gleichen Namens werden ersetzt.

    Args:
        ind: Indikatoren des Werts (gleiche Zeilenzahl wie ``dates``).
        dates: Kerzendaten des Werts, aufsteigend.
        table: Ergebnis von :func:`market_table` (``None`` = nichts anhängen).

    Returns:
        Indikatoren mit Marktspalten (neuer, fortlaufender Index).
    """
    ind = ind.reset_index(drop=True)
    if table is None:
        return ind
    left = pd.DataFrame({DATE_COLUMN: _ns(pd.Series(dates))})
    right = table.assign(**{DATE_COLUMN: _ns(table[DATE_COLUMN])})
    merged = pd.merge_asof(left, right, on=DATE_COLUMN, direction="backward")
    result = ind.copy()
    for name in table.columns:
        if name != DATE_COLUMN:
            result[name] = merged[name].to_numpy()
    return result


def _ns(values: pd.Series) -> pd.Series:
    """Datumswerte einheitlich als ``datetime64[ns]`` (für ``merge_asof``)."""
    return pd.Series(pd.to_datetime(values).to_numpy().astype("datetime64[ns]"))
