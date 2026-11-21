"""Relative Stärke: RS-Linie gegen einen Benchmark und RS-Rang im Universum.

Beide Funktionen arbeiten auf Kursreihen nach :mod:`trading_helfer.core.bars`
(Spalte ``date`` oder ein ``DatetimeIndex``) und nutzen nur Kerzen bis zum jeweiligen
Datum. Sie stehen nicht in der Indikator-Registry, weil sie mehr als eine Kursreihe
brauchen.

- :func:`rs_line`: ``Close / Close_Benchmark``, normiert auf 100 am Startdatum.
- :func:`rs_rank`: Perzentilrang (0 bis 100) der Rendite über ``lookback`` Kerzen im
  Universum. Mit ``index`` zählen nur Werte, die am Stichtag Indexmitglied waren
  (point in time, siehe :mod:`trading_helfer.data.universe`).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from trading_helfer.data.universe import IndexHistory, UniverseError, members_on
from trading_helfer.indicators.registry import IndicatorError
from trading_helfer.indicators.trend import _check_period

__all__ = ["rs_line", "rs_rank", "rs_returns"]

DateLike = dt.date | str | pd.Timestamp


def _price_series(df: pd.DataFrame, source: str, what: str) -> pd.Series[float]:
    """Spalte ``source`` als Serie mit Datumsindex (sortiert, ohne Lücken in den Kursen)."""
    if source not in df.columns:
        raise IndicatorError(f"Spalte {source!r} fehlt in den Kursdaten von {what}.")
    if "date" in df.columns:
        index = _dates(df)
    elif isinstance(df.index, pd.DatetimeIndex):
        index = df.index
    else:
        raise IndicatorError(
            f"Kursdaten von {what} brauchen eine Spalte 'date' oder einen Datumsindex."
        )
    series = pd.Series(df[source].to_numpy(dtype=np.float64), index=index.normalize())
    if series.index.has_duplicates:
        raise IndicatorError(f"Kursdaten von {what} enthalten doppelte Datumswerte.")
    return series.sort_index()


def _dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    try:
        return pd.DatetimeIndex(pd.to_datetime(df["date"], format="ISO8601"))
    except (TypeError, ValueError):
        raise IndicatorError("Spalte 'date' enthält ungültige Datumswerte.") from None


def _to_timestamp(value: DateLike) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise IndicatorError(f"Ungültiges Datum {value!r}.") from None
    if pd.isna(stamp):
        raise IndicatorError(f"Ungültiges Datum {value!r}.")
    return stamp.tz_localize(None).normalize() if stamp.tzinfo else stamp.normalize()


def rs_line(
    symbol: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    start: DateLike | None = None,
    source: str = "close",
) -> pd.Series[float]:
    """Relative-Stärke-Linie ``Close / Close_Benchmark``, normiert auf 100 am Startdatum.

    Das Startdatum ist ``start`` oder, ohne Angabe, der erste Tag, an dem beide Reihen
    einen Kurs haben. Fehlt am Startdatum ein Kurs, gilt der erste gemeinsame Tag danach.

    Args:
        symbol: Kursreihe des Wertes.
        benchmark: Kursreihe des Benchmarks (z. B. SPY oder QQQ).
        start: Tag, an dem die Linie 100 ist; davor ist sie NaN.
        source: Kursspalte, Standard ``close``.

    Returns:
        Serie ``rs_line`` mit derselben Länge und Reihenfolge wie ``symbol``; NaN vor dem
        Startdatum und an Tagen ohne Benchmark-Kurs.

    Raises:
        IndicatorError: Spalte oder Datum fehlt, oder es gibt keinen gemeinsamen Tag.
    """
    own = _price_series(symbol, source, "Wert")
    bench = _price_series(benchmark, source, "Benchmark")
    ratio = own / bench.reindex(own.index)
    ratio = ratio.where(np.isfinite(ratio))
    if start is not None:
        ratio = ratio[ratio.index >= _to_timestamp(start)]
    valid = ratio.dropna()
    if valid.empty:
        raise IndicatorError("Wert und Benchmark haben ab dem Startdatum keinen gemeinsamen Tag.")
    line = 100.0 * ratio / valid.iloc[0]
    line = line[line.index >= valid.index[0]]
    # Zurück in Reihenfolge und Index der Eingabe.
    dates = _dates(symbol) if "date" in symbol.columns else symbol.index
    values = line.reindex(pd.DatetimeIndex(dates).normalize()).to_numpy()
    return pd.Series(values, index=symbol.index, name="rs_line", dtype="float64")


def rs_returns(
    universe: Mapping[str, pd.DataFrame],
    date: DateLike,
    lookback: int,
    *,
    source: str = "close",
) -> pd.Series[float]:
    """Rendite je Wert über ``lookback`` Kerzen bis einschließlich ``date``.

    Je Wert gilt die letzte Kerze am oder vor ``date`` und die Kerze ``lookback``
    Positionen davor. Werte mit zu kurzer Historie oder ungültigen Kursen fehlen.

    Args:
        universe: Kursreihen je Symbol.
        date: Stichtag.
        lookback: Anzahl Kerzen.
        source: Kursspalte, Standard ``close``.

    Returns:
        Rendite als Bruch (0,1 = +10 %), Index = Symbol in Großbuchstaben.
    """
    _check_period(lookback, "lookback")
    day = _to_timestamp(date)
    returns: dict[str, float] = {}
    for raw_symbol, df in universe.items():
        prices = _price_series(df, source, raw_symbol)
        window = prices[prices.index <= day].to_numpy()
        if len(window) <= lookback:
            continue
        last, first = window[-1], window[-1 - lookback]
        if first > 0 and np.isfinite(first) and np.isfinite(last):
            returns[raw_symbol.strip().upper()] = last / first - 1.0
    return pd.Series(returns, dtype="float64", name="return")


def rs_rank(
    universe: Mapping[str, pd.DataFrame],
    date: DateLike,
    lookback: int,
    *,
    index: str | IndexHistory | Iterable[str] | None = None,
    source: str = "close",
) -> pd.Series[float]:
    """RS-Rang: Perzentilrang der Rendite über ``lookback`` Kerzen im Universum.

    Der schwächste Wert erhält 0, der stärkste 100, dazwischen linear nach Rang
    (``(Rang − 1) / (n − 1) · 100``, gleiche Renditen teilen sich den mittleren Rang).
    Ein einzelner Wert erhält 100.

    Args:
        universe: Kursreihen je Symbol (bereinigte Kurse).
        date: Stichtag; nur Kerzen bis einschließlich dieses Tages zählen.
        lookback: Anzahl Kerzen für die Rendite.
        index: Nur Mitglieder am Stichtag zählen: Index-Code (``"SP500"``,
            ``"NDX100"``), eine :class:`IndexHistory` oder eine feste Symbolmenge.
            ``None`` nimmt alle Werte aus ``universe``.
        source: Kursspalte, Standard ``close``.

    Returns:
        Serie ``rs_rank`` (0 bis 100), Index = Symbol, absteigend nach Rang sortiert
        (bei Gleichstand alphabetisch). Werte ohne ausreichende Historie fehlen.

    Raises:
        IndicatorError: Ungültiger ``lookback``, fehlende Spalte oder unbekannter Index.
    """
    returns = rs_returns(universe, date, lookback, source=source)
    if index is not None:
        allowed = _members(index, date)
        returns = returns[returns.index.isin(allowed)]
    if returns.empty:
        return pd.Series(dtype="float64", name="rs_rank")
    n = len(returns)
    if n == 1:
        ranks = pd.Series(100.0, index=returns.index)
    else:
        ranks = (returns.rank(method="average") - 1.0) / (n - 1) * 100.0
    frame = pd.DataFrame({"rank": ranks, "symbol": ranks.index})
    frame = frame.sort_values(["rank", "symbol"], ascending=[False, True])
    return pd.Series(frame["rank"].to_numpy(), index=frame["symbol"].to_numpy(), name="rs_rank")


def _members(index: str | IndexHistory | Iterable[str], date: DateLike) -> set[str]:
    """Symbole, die am Stichtag Mitglied sind, in Großbuchstaben."""
    day = _to_timestamp(date)
    if isinstance(index, IndexHistory):
        members = index.members_on(day)
    elif isinstance(index, str):
        try:
            members = members_on(index, day)
        except UniverseError as exc:
            raise IndicatorError(str(exc)) from None
    else:
        members = set(index)
    return {symbol.strip().upper() for symbol in members}
