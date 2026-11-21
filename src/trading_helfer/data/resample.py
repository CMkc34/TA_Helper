"""Zeitrahmen 3D, 1W und 1M aus Tageskursen berechnen.

Regeln (siehe ``docs/SPEC.md``, Abschnitt „Zeitrahmen-Regeln“):

- **1W**: Woche Montag bis Freitag; Kerzendatum ist der letzte Handelstag der Woche
  laut Börsenkalender XNYS (in Feiertagswochen z. B. der Donnerstag vor Karfreitag).
- **1M**: Kalendermonat; Kerzendatum ist der letzte Handelstag des Monats.
- **3D**: je drei aufeinanderfolgende Handelstage des XNYS-Kalenders, gezählt ab dem
  festen Anker 2000-01-03; Handelstage vor dem Anker werden rückwärts in Dreiergruppen
  gezählt. Die Gruppengrenzen hängen daher nur vom Anker ab, nicht von den Daten.
  Beginnt ein Symbol mitten in einer Gruppe, wird diese erste, unvollständige Gruppe
  verworfen. Kerzendatum ist der letzte Handelstag der Gruppe.
- Aggregation: Open = erstes Open, High = Maximum, Low = Minimum, Close = letztes
  Close, Volume = Summe, ``adj_close`` (falls vorhanden) = letzter Wert.
- Die letzte Kerze erhält ``is_partial = True``, wenn ihr Zeitraum laut Kalender noch
  Handelstage nach dem letzten vorhandenen Tag hat. Ihr Datum ist dann der letzte
  vorhandene Tag. Alle anderen Kerzen haben ``is_partial = False``.

Fehlende Tage mitten in den Daten (Lücken) werden nicht aufgefüllt; die Kerze wird aus
den vorhandenen Tagen gebildet. Tage, die kein XNYS-Handelstag sind, zählen für 3D zum
nächsten Handelstag.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from datetime import date
from functools import lru_cache
from typing import Any, Final

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from trading_helfer.core.bars import (
    ADJ_CLOSE_COLUMN,
    BAR_COLUMNS,
    DATE_COLUMN,
    VOLUME_COLUMN,
    empty_bars,
    validate_bars,
)
from trading_helfer.core.types import Timeframe

__all__ = [
    "ANCHOR_3D",
    "CALENDAR_NAME",
    "PARTIAL_COLUMN",
    "ResampleCache",
    "resample",
    "resample_cached",
]

CALENDAR_NAME: Final = "XNYS"
"""Börsenkalender (``exchange_calendars``) für Handelstage."""

ANCHOR_3D: Final = pd.Timestamp("2000-01-03")
"""Erster Handelstag der 3D-Gruppe, ab der alle 3D-Gruppen gezählt werden."""

PARTIAL_COLUMN: Final = "is_partial"
"""Spalte, die die unvollständige letzte Kerze markiert (``bool``)."""

_GROUP_SIZE_3D: Final = 3
_CALENDAR_START: Final = pd.Timestamp("1980-01-01")
_CALENDAR_MARGIN: Final = pd.Timedelta(days=400)
_CALENDAR_YEAR_STEP: Final = 5


def resample(df_daily: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Berechnet Kerzen eines Zeitrahmens aus Tageskursen.

    Args:
        df_daily: Tageskurse im Schema aus :mod:`trading_helfer.core.bars`; wird mit
            :func:`~trading_helfer.core.bars.validate_bars` geprüft und nie verändert.
        timeframe: Ziel-Zeitrahmen. Für ``Timeframe.D1`` werden die geprüften
            Tageskurse unverändert zurückgegeben (alle ``is_partial = False``).

    Returns:
        Neues DataFrame mit den Spalten ``date, open, high, low, close, volume``,
        ``adj_close`` (nur wenn in der Eingabe vorhanden) und ``is_partial``; nach Datum
        sortiert, ``RangeIndex``. Zusätzliche Eingabespalten werden nicht übernommen.

    Raises:
        BarsValidationError: ``df_daily`` verletzt das Kursreihen-Schema.
        ValueError: ``timeframe`` ist kein bekannter Zeitrahmen.
    """
    timeframe = Timeframe(timeframe)
    daily = validate_bars(df_daily)
    columns = [*BAR_COLUMNS]
    if ADJ_CLOSE_COLUMN in daily.columns:
        columns.append(ADJ_CLOSE_COLUMN)
    daily = daily.loc[:, columns]

    if daily.empty:
        result = empty_bars(with_adj_close=ADJ_CLOSE_COLUMN in columns)
        result[PARTIAL_COLUMN] = pd.Series(dtype="bool")
        return result
    if timeframe is Timeframe.D1:
        daily[PARTIAL_COLUMN] = False
        return daily

    dates = pd.DatetimeIndex(daily[DATE_COLUMN])
    sessions = _sessions(dates[0], dates[-1])
    if timeframe is Timeframe.W1:
        day_keys = _week_keys(dates)
        session_keys = _week_keys(sessions)
    elif timeframe is Timeframe.MN1:
        day_keys = _month_keys(dates)
        session_keys = _month_keys(sessions)
    else:
        day_keys, session_keys = _keys_3d(dates, sessions)

    # Letzter Handelstag und Anzahl Handelstage je Zeitraum laut Kalender.
    period_end = pd.Series(sessions, index=session_keys).groupby(level=0).max()
    bars = _aggregate(daily, day_keys)
    ends = period_end.reindex(bars.index)
    last_date = dates[-1]
    partial = (ends > last_date).to_numpy()
    bars[DATE_COLUMN] = np.where(partial, bars["_last_day"], ends.to_numpy())
    bars[PARTIAL_COLUMN] = partial

    if timeframe is Timeframe.D3:
        # Erste Gruppe verwerfen, wenn das Symbol mitten in ihr beginnt.
        first_key = bars.index[0]
        first_group_start = sessions[session_keys == first_key].min()
        if dates[0] > first_group_start:
            bars = bars.iloc[1:]

    result = bars.loc[:, [*columns, PARTIAL_COLUMN]].reset_index(drop=True)
    result[DATE_COLUMN] = result[DATE_COLUMN].astype("datetime64[ns]")
    result[PARTIAL_COLUMN] = result[PARTIAL_COLUMN].astype("bool")
    return result


class ResampleCache:
    """LRU-Zwischenspeicher für :func:`resample` je ``(symbol, timeframe, last_date)``.

    Zum Schlüssel gehören zusätzlich erstes Datum und Zeilenzahl der Tageskurse, damit
    nachgeladene Historie nicht übersehen wird. Ein- und Ausgaben werden kopiert, damit
    Aufrufer den Speicher nicht verändern können.
    """

    def __init__(self, maxsize: int = 256) -> None:
        """Legt einen leeren Speicher für höchstens ``maxsize`` Einträge an.

        Raises:
            ValueError: ``maxsize`` ist kleiner als 1.
        """
        if maxsize < 1:
            raise ValueError("maxsize muss mindestens 1 sein.")
        self.maxsize = maxsize
        self._entries: OrderedDict[tuple[Any, ...], pd.DataFrame] = OrderedDict()

    def __len__(self) -> int:
        """Anzahl gespeicherter Einträge."""
        return len(self._entries)

    def clear(self) -> None:
        """Entfernt alle Einträge."""
        self._entries.clear()

    def get(self, symbol: str, df_daily: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
        """Liefert die Kerzen aus dem Speicher oder berechnet sie mit :func:`resample`.

        Args:
            symbol: Symbol der Tageskurse (Teil des Schlüssels).
            df_daily: Tageskurse des Symbols.
            timeframe: Ziel-Zeitrahmen.

        Returns:
            Neue Kopie der Kerzen.
        """
        timeframe = Timeframe(timeframe)
        key = (symbol, timeframe, *_data_key(df_daily))
        cached = self._entries.get(key)
        if cached is None:
            cached = resample(df_daily, timeframe)
            self._entries[key] = cached
            if len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(key)
        return cached.copy()


_DEFAULT_CACHE = ResampleCache()


def resample_cached(symbol: str, df_daily: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Wie :func:`resample`, aber über einen prozessweiten :class:`ResampleCache`.

    Args:
        symbol: Symbol der Tageskurse.
        df_daily: Tageskurse des Symbols.
        timeframe: Ziel-Zeitrahmen.

    Returns:
        Neue Kopie der Kerzen.
    """
    return _DEFAULT_CACHE.get(symbol, df_daily, timeframe)


def _data_key(df_daily: pd.DataFrame) -> tuple[Any, ...]:
    """Schlüsselteil aus letztem Datum, erstem Datum und Zeilenzahl der Tageskurse."""
    if DATE_COLUMN not in df_daily.columns or df_daily.empty:
        return (None, None, len(df_daily))
    dates = pd.to_datetime(df_daily[DATE_COLUMN])
    return (dates.max(), dates.min(), len(df_daily))


def _sessions(first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    """Handelstage, die die Daten, den 3D-Anker und alle betroffenen Zeiträume abdecken."""
    start = min(first, _CALENDAR_START)
    end = max(last, ANCHOR_3D) + _CALENDAR_MARGIN
    # Endjahr auf ein Vielfaches von _CALENDAR_YEAR_STEP runden, damit der Kalender
    # selten neu aufgebaut wird.
    end_year = -(-end.year // _CALENDAR_YEAR_STEP) * _CALENDAR_YEAR_STEP
    return _calendar_sessions(start.year, end_year)


@lru_cache(maxsize=8)
def _calendar_sessions(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """Alle XNYS-Handelstage vom 1.1. ``start_year`` bis 31.12. ``end_year``."""
    with warnings.catch_warnings():
        # exchange_calendars löst unter neueren pandas-Versionen Deprecation-Warnungen aus.
        warnings.simplefilter("ignore")
        calendar = xcals.get_calendar(
            CALENDAR_NAME,
            start=date(start_year, 1, 1).isoformat(),
            end=date(end_year, 12, 31).isoformat(),
        )
    sessions = pd.DatetimeIndex(calendar.sessions).tz_localize(None)
    return pd.DatetimeIndex(sessions.as_unit("ns"))


def _week_keys(dates: pd.DatetimeIndex) -> np.ndarray[Any, Any]:
    """Montag der Kalenderwoche je Datum."""
    return (dates - pd.to_timedelta(dates.day_of_week, unit="D")).to_numpy()


def _month_keys(dates: pd.DatetimeIndex) -> np.ndarray[Any, Any]:
    """Monatsnummer (Jahr * 12 + Monat) je Datum."""
    return (dates.year * 12 + dates.month - 1).to_numpy()


def _keys_3d(
    dates: pd.DatetimeIndex, sessions: pd.DatetimeIndex
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """3D-Gruppennummer je Datum und je Handelstag, relativ zum Anker."""
    anchor_pos = int(sessions.searchsorted(ANCHOR_3D))
    session_keys = (np.arange(len(sessions)) - anchor_pos) // _GROUP_SIZE_3D
    # Tage ohne Handel laut Kalender zählen zum nächsten Handelstag.
    day_pos = sessions.searchsorted(dates)
    day_keys = (day_pos - anchor_pos) // _GROUP_SIZE_3D
    return np.asarray(day_keys), session_keys


def _aggregate(daily: pd.DataFrame, keys: np.ndarray[Any, Any]) -> pd.DataFrame:
    """Fasst Tageskurse je Schlüssel zusammen; ``_last_day`` = letzter vorhandener Tag."""
    fast = _aggregate_sorted(daily, keys)
    if fast is not None:
        return fast
    grouped = daily.groupby(keys, sort=True)
    spec: dict[str, tuple[str, str]] = {
        "open": ("open", "first"),
        "high": ("high", "max"),
        "low": ("low", "min"),
        "close": ("close", "last"),
        VOLUME_COLUMN: (VOLUME_COLUMN, "sum"),
        "_last_day": (DATE_COLUMN, "max"),
    }
    if ADJ_CLOSE_COLUMN in daily.columns:
        spec[ADJ_CLOSE_COLUMN] = (ADJ_CLOSE_COLUMN, "last")
    bars = grouped.agg(**spec)
    bars[VOLUME_COLUMN] = bars[VOLUME_COLUMN].astype("int64")
    return bars


def _aggregate_sorted(daily: pd.DataFrame, keys: np.ndarray[Any, Any]) -> pd.DataFrame | None:
    """Schneller Weg für aufsteigende Schlüssel ohne fehlende Werte (Normalfall).

    Liefert dasselbe wie die ``groupby``-Aggregation oder ``None``, wenn die Voraussetzungen
    nicht erfüllt sind.
    """
    n = len(keys)
    if n == 0 or len(daily) != n:
        return None
    if n > 1 and not bool(np.all(keys[1:] >= keys[:-1])):
        return None
    float_columns = ["open", "high", "low", "close"]
    if ADJ_CLOSE_COLUMN in daily.columns:
        float_columns.append(ADJ_CLOSE_COLUMN)
    values = {c: daily[c].to_numpy(dtype="float64") for c in float_columns}
    if any(np.isnan(v).any() for v in values.values()):
        return None
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:] - 1, n - 1]
    dates = daily[DATE_COLUMN].to_numpy(dtype="datetime64[ns]")
    volume = daily[VOLUME_COLUMN].to_numpy(dtype="int64")
    data: dict[str, Any] = {
        "open": values["open"][starts],
        "high": np.maximum.reduceat(values["high"], starts),
        "low": np.minimum.reduceat(values["low"], starts),
        "close": values["close"][ends],
        VOLUME_COLUMN: np.add.reduceat(volume, starts),
        "_last_day": dates[ends],
    }
    if ADJ_CLOSE_COLUMN in values:
        data[ADJ_CLOSE_COLUMN] = values[ADJ_CLOSE_COLUMN][ends]
    return pd.DataFrame(data, index=pd.Index(keys[starts]))
