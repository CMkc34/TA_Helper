"""Kursreihen (Bars): Spaltenschema und Validierung.

Eine Kursreihe ist ein ``pd.DataFrame`` mit einer Zeile je Kerze. Pflichtspalten sind
``date, open, high, low, close, volume``; ``adj_close`` ist optional. Alle Module, die
Kurse erzeugen oder lesen (Provider, Speicher, Resampling, Indikatoren, Backtest),
verwenden dieses Schema und prüfen fremde Eingaben mit :func:`validate_bars`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

__all__ = [
    "ADJ_CLOSE_COLUMN",
    "BAR_COLUMNS",
    "BAR_DTYPES",
    "DATE_COLUMN",
    "OPTIONAL_BAR_COLUMNS",
    "PRICE_COLUMNS",
    "VOLUME_COLUMN",
    "BarsValidationError",
    "empty_bars",
    "validate_bars",
]

DATE_COLUMN: Final = "date"
"""Spalte mit dem Handelstag (``datetime64[ns]``, ohne Zeitzone, auf Mitternacht normalisiert)."""

PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")
"""Pflicht-Kursspalten (``float64``)."""

VOLUME_COLUMN: Final = "volume"
"""Spalte mit dem Handelsvolumen (``int64``)."""

ADJ_CLOSE_COLUMN: Final = "adj_close"
"""Optionale Spalte mit dem um Splits und Dividenden bereinigten Schlusskurs (``float64``)."""

BAR_COLUMNS: Final[tuple[str, ...]] = (DATE_COLUMN, *PRICE_COLUMNS, VOLUME_COLUMN)
"""Pflichtspalten einer Kursreihe in kanonischer Reihenfolge."""

OPTIONAL_BAR_COLUMNS: Final[tuple[str, ...]] = (ADJ_CLOSE_COLUMN,)
"""Optionale Spalten einer Kursreihe in kanonischer Reihenfolge."""

BAR_DTYPES: Final[Mapping[str, str]] = {
    DATE_COLUMN: "datetime64[ns]",
    **dict.fromkeys(PRICE_COLUMNS, "float64"),
    VOLUME_COLUMN: "int64",
    ADJ_CLOSE_COLUMN: "float64",
}
"""Ziel-Datentyp je Pflicht- und optionaler Spalte."""

_MAX_LISTED = 5


class BarsValidationError(ValueError):
    """Eine Kursreihe verletzt das Schema; ``reason`` nennt den Grund auf Deutsch."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def empty_bars(*, with_adj_close: bool = True) -> pd.DataFrame:
    """Leere Kursreihe mit allen Spalten und den richtigen Datentypen.

    Args:
        with_adj_close: ``True`` fügt die optionale Spalte ``adj_close`` hinzu.

    Returns:
        DataFrame ohne Zeilen mit ``RangeIndex``; besteht :func:`validate_bars`.
    """
    columns = BAR_COLUMNS + (OPTIONAL_BAR_COLUMNS if with_adj_close else ())
    return pd.DataFrame(
        {name: pd.Series(dtype=BAR_DTYPES[name]) for name in columns},
        index=pd.RangeIndex(0),
    )


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Prüft eine Kursreihe und gibt eine bereinigte Kopie zurück.

    Erwartet werden die Spalten aus :data:`BAR_COLUMNS` und optional ``adj_close``.
    Das Eingabe-DataFrame wird nie verändert.

    Bereinigung (kein Fehler):

    - ``date`` wird nach ``datetime64[ns]`` umgewandelt (auch aus Text oder
      ``datetime.date``). Zeitzonen werden entfernt, die lokale Uhrzeit bleibt erhalten
      (``2024-01-02 00:00-05:00`` wird zu ``2024-01-02``); danach wird auf den Tag
      normalisiert (Uhrzeit 00:00).
    - ``open``, ``high``, ``low``, ``close`` und ``adj_close`` werden ``float64``,
      ``volume`` wird ``int64`` (aus Ganzzahlen, ganzzahligen Gleitkommazahlen oder
      Text wie ``"100"``).
    - Zeilen werden stabil nach ``date`` sortiert, der Index wird ein ``RangeIndex``.
    - Zeilen, die in allen Spalten exakt übereinstimmen, bleiben nur einmal erhalten.
    - Spaltenreihenfolge: Pflichtspalten, ``adj_close`` (falls vorhanden), danach
      alle zusätzlichen Spalten unverändert in ihrer ursprünglichen Reihenfolge.

    Fehler (:class:`BarsValidationError` mit Grund):

    - Eingabe ist kein DataFrame, Spaltennamen sind doppelt oder Pflichtspalten fehlen.
    - Werte lassen sich nicht in den Ziel-Datentyp umwandeln.
    - Fehlende Werte (NaN/NaT/None) oder unendliche Werte in ``date``, den Kursspalten,
      ``volume`` oder einem vorhandenen ``adj_close``.
    - ``volume`` enthält Nachkommastellen.
    - Ein Datum kommt mehrfach mit abweichenden Werten vor (auch, wenn erst die
      Normalisierung auf den Tag die Daten gleich macht).

    Inhaltliche Plausibilität (High < Low, Kurse ≤ 0, Lücken gegen den Börsenkalender)
    prüft diese Funktion bewusst nicht; dafür ist die Datenqualitätsprüfung zuständig.

    Args:
        df: Zu prüfende Kursreihe.

    Returns:
        Neue, bereinigte Kursreihe.

    Raises:
        BarsValidationError: Die Kursreihe ist ungültig; ``reason`` nennt den Grund.
    """
    if not isinstance(df, pd.DataFrame):
        raise BarsValidationError(f"Erwartet wird ein DataFrame, erhalten: {type(df).__name__}.")
    if df.columns.has_duplicates:
        duplicated = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
        raise BarsValidationError(f"Doppelte Spaltennamen: {', '.join(duplicated)}.")
    missing = [name for name in BAR_COLUMNS if name not in df.columns]
    if missing:
        raise BarsValidationError(f"Pflichtspalte(n) fehlen: {', '.join(missing)}.")

    result = df.reset_index(drop=True)  # neue Kopie, Index unabhängig von der Eingabe
    dates = _to_dates(result[DATE_COLUMN])
    result[DATE_COLUMN] = dates

    float_columns = [*PRICE_COLUMNS, *(c for c in OPTIONAL_BAR_COLUMNS if c in result.columns)]
    for name in float_columns:
        result[name] = _to_float(result[name], name, dates)
    result[VOLUME_COLUMN] = _to_volume(result[VOLUME_COLUMN], dates)

    result = result.sort_values(DATE_COLUMN, kind="mergesort")
    if result[DATE_COLUMN].duplicated().any():
        result = _drop_identical_rows(result)
        conflicting = result.loc[result[DATE_COLUMN].duplicated(), DATE_COLUMN]
        if not conflicting.empty:
            raise BarsValidationError(
                "Doppelte Daten mit abweichenden Werten: "
                f"{_format_dates(conflicting.drop_duplicates())}."
            )

    known = [*BAR_COLUMNS, *(c for c in OPTIONAL_BAR_COLUMNS if c in result.columns)]
    extra = [c for c in result.columns if c not in known]
    return result.loc[:, [*known, *extra]].reset_index(drop=True)


def _to_dates(values: pd.Series[Any]) -> pd.Series[Any]:
    """Wandelt ``values`` in tageweise normalisierte ``datetime64[ns]`` ohne Zeitzone."""
    try:
        dates = pd.to_datetime(values)
        if isinstance(dates.dtype, pd.DatetimeTZDtype):
            dates = dates.dt.tz_localize(None)
        if dates.isna().any():
            raise BarsValidationError(
                f"Spalte {DATE_COLUMN!r} enthält fehlende Werte in {_count(dates.isna())} Zeile(n)."
            )
        return dates.dt.normalize().astype("datetime64[ns]")
    except BarsValidationError:
        raise
    except (ValueError, TypeError, OverflowError) as exc:
        raise BarsValidationError(
            f"Spalte {DATE_COLUMN!r} lässt sich nicht in Datumswerte umwandeln: {exc}"
        ) from exc


def _to_numeric(values: pd.Series[Any], name: str) -> pd.Series[Any]:
    try:
        return pd.to_numeric(values, errors="raise")
    except (ValueError, TypeError) as exc:
        raise BarsValidationError(
            f"Spalte {name!r} lässt sich nicht in Zahlen umwandeln: {exc}"
        ) from exc


def _to_float(values: pd.Series[Any], name: str, dates: pd.Series[Any]) -> pd.Series[Any]:
    """Wandelt eine Kursspalte in endliche ``float64``-Werte."""
    floats = _to_numeric(values, name).astype("float64")
    _reject_invalid(floats, name, dates)
    return floats


def _to_volume(values: pd.Series[Any], dates: pd.Series[Any]) -> pd.Series[Any]:
    """Wandelt die Volumenspalte in ``int64``; Nachkommastellen sind ein Fehler."""
    numeric = _to_numeric(values, VOLUME_COLUMN)
    if numeric.isna().any():
        _reject_invalid(numeric.astype("float64"), VOLUME_COLUMN, dates)
    if is_integer_dtype(numeric.dtype) or is_bool_dtype(numeric.dtype):
        return numeric.astype("int64")
    floats = numeric.astype("float64")
    _reject_invalid(floats, VOLUME_COLUMN, dates)
    fractional = (floats % 1) != 0
    if fractional.any():
        raise BarsValidationError(
            f"Spalte {VOLUME_COLUMN!r} muss ganzzahlig sein; Nachkommastellen am "
            f"{_format_dates(dates[fractional])}."
        )
    return floats.astype("int64")


def _reject_invalid(floats: pd.Series[float], name: str, dates: pd.Series[Any]) -> None:
    """Wirft einen Fehler bei fehlenden oder unendlichen Werten."""
    missing = floats.isna()
    if missing.any():
        raise BarsValidationError(
            f"Spalte {name!r} enthält fehlende Werte am {_format_dates(dates[missing])}."
        )
    infinite = pd.Series(np.isinf(floats.to_numpy()), index=floats.index)
    if infinite.any():
        raise BarsValidationError(
            f"Spalte {name!r} enthält unendliche Werte am {_format_dates(dates[infinite])}."
        )


def _drop_identical_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Entfernt exakt identische Zeilen; bei nicht hashbaren Zusatzspalten unverändert."""
    try:
        return df.drop_duplicates(keep="first")
    except TypeError:
        return df


def _count(mask: pd.Series[bool]) -> int:
    return int(mask.sum())


def _format_dates(dates: pd.Series[Any]) -> str:
    """Formatiert höchstens ``_MAX_LISTED`` Daten als ``YYYY-MM-DD``-Liste."""
    shown = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates.iloc[:_MAX_LISTED]]
    rest = len(dates) - len(shown)
    text = ", ".join(shown)
    return f"{text} (und {rest} weitere)" if rest > 0 else text
