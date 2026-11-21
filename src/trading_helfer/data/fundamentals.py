"""Fundamentaldaten: feste Metriknamen, konservatives Veröffentlichungsdatum, Speichern.

Jeder Provider liefert Fundamentaldaten im Langformat mit den Spalten
``period_end, period_type, metric, value, reported_date`` (siehe ``docs/SPEC.md``,
Abschnitt „Datenmodell“). Dieses Modul legt die erlaubten Metriknamen fest
(:data:`METRICS`), bringt die Tabelle in eine einheitliche Form
(:func:`normalize_fundamentals`) und speichert sie (:func:`update_fundamentals`).

Point in time: Liefert die Quelle kein Veröffentlichungsdatum, gilt ein Quartalswert
erst 45 Tage und ein Jahreswert erst 75 Tage nach Periodenende
(:func:`conservative_reported_date`). So sieht ein Backtest keine Zahl, bevor sie
öffentlich war.

Fehlende Metriken fehlen in der Tabelle; sie werden nie als ``0`` gespeichert.

Vorzeichen: Werte werden so übernommen, wie die Quelle sie meldet. Bei Yahoo sind
Mittelabflüsse (``capex``, ``dividends_paid``) negativ.

Fundamentaldaten ändern sich nur quartalsweise; im Datenupdate reicht ein Abruf je
Woche (:data:`UPDATE_INTERVAL`).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from typing import TYPE_CHECKING, Final, Literal

import pandas as pd

from trading_helfer.data.providers.base import FUNDAMENTALS_COLUMNS, FundamentalsProvider

if TYPE_CHECKING:
    from trading_helfer.data.storage import MarketStore

__all__ = [
    "ANNUAL_REPORT_LAG",
    "METRICS",
    "PERIOD_TYPES",
    "QUARTER_REPORT_LAG",
    "UPDATE_INTERVAL",
    "PeriodType",
    "conservative_reported_date",
    "empty_fundamentals",
    "normalize_fundamentals",
    "update_fundamentals",
]

PeriodType = Literal["Q", "FY"]
"""Art der Periode: Quartal (``Q``) oder Geschäftsjahr (``FY``)."""

PERIOD_TYPES: Final[tuple[PeriodType, ...]] = ("Q", "FY")
"""Erlaubte Periodenarten."""

METRICS: Final[tuple[str, ...]] = (
    # Gewinn- und Verlustrechnung
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "ebitda",
    "eps_diluted",
    "interest_expense",
    # Bilanz
    "total_assets",
    "total_liabilities",
    "total_equity",
    "total_debt",
    "long_term_debt",
    "cash",
    "current_assets",
    "current_liabilities",
    "inventory",
    "retained_earnings",
    "shares_outstanding",
    # Kapitalflussrechnung
    "operating_cashflow",
    "capex",
    "dividends_paid",
)
"""Feste Metriknamen, die in der Tabelle ``fundamentals`` vorkommen dürfen."""

QUARTER_REPORT_LAG: Final = timedelta(days=45)
"""Abstand Periodenende → Veröffentlichung für Quartalszahlen ohne Datum der Quelle."""

ANNUAL_REPORT_LAG: Final = timedelta(days=75)
"""Abstand Periodenende → Veröffentlichung für Jahreszahlen ohne Datum der Quelle."""

UPDATE_INTERVAL: Final = timedelta(days=7)
"""Fundamentaldaten müssen höchstens einmal pro Woche neu geladen werden."""

_LAGS: Final[dict[str, timedelta]] = {"Q": QUARTER_REPORT_LAG, "FY": ANNUAL_REPORT_LAG}


def conservative_reported_date(period_end: date | pd.Timestamp, period_type: str) -> pd.Timestamp:
    """Spätestes plausibles Veröffentlichungsdatum einer Periode.

    Args:
        period_end: Letzter Tag der Periode.
        period_type: ``"Q"`` (Quartal, + 45 Tage) oder ``"FY"`` (Jahr, + 75 Tage).

    Returns:
        Datum ohne Uhrzeit und Zeitzone.

    Raises:
        ValueError: ``period_type`` ist unbekannt.
    """
    lag = _LAGS.get(str(period_type).strip().upper())
    if lag is None:
        raise ValueError(f"Unbekannte Periodenart {period_type!r} (erlaubt: Q, FY).")
    return _to_day(pd.Timestamp(period_end)) + lag


def empty_fundamentals() -> pd.DataFrame:
    """Leere Fundamentaldaten-Tabelle mit den richtigen Datentypen."""
    return pd.DataFrame(
        {
            "period_end": pd.Series(dtype="datetime64[ns]"),
            "period_type": pd.Series(dtype="object"),
            "metric": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
            "reported_date": pd.Series(dtype="datetime64[ns]"),
        },
        index=pd.RangeIndex(0),
    )


def normalize_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """Bringt Fundamentaldaten im Langformat in die einheitliche Form.

    - Datumsspalten ohne Uhrzeit und Zeitzone (``datetime64[ns]``),
    - Zeilen ohne Wert (``NaN``, ``None``, nicht numerisch) entfallen,
    - fehlendes ``reported_date`` wird konservativ ergänzt
      (:func:`conservative_reported_date`); ein ``reported_date`` vor dem
      Periodenende ist unmöglich und wird ebenso ersetzt,
    - sortiert nach ``period_end, period_type, metric``.

    Args:
        df: Spalten ``period_end, period_type, metric, value`` und optional
            ``reported_date``.

    Returns:
        Spalten ``period_end, period_type, metric, value, reported_date``.

    Raises:
        ValueError: Spalten fehlen, eine Metrik oder Periodenart ist unbekannt, oder
            ein Eintrag kommt doppelt vor.
    """
    required = [c for c in FUNDAMENTALS_COLUMNS if c != "reported_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fundamentaldaten ohne Spalte(n): {', '.join(missing)}.")
    if df.empty:
        return empty_fundamentals()

    src = df.reset_index(drop=True)
    period_type = src["period_type"].astype("string").str.strip().str.upper()
    unknown_types = sorted(set(period_type.dropna()) - set(PERIOD_TYPES))
    if unknown_types or period_type.isna().any():
        raise ValueError(f"Unbekannte Periodenart: {unknown_types or ['<leer>']}.")
    metric = src["metric"].astype("string").str.strip()
    unknown_metrics = sorted(set(metric.dropna()) - set(METRICS))
    if unknown_metrics or metric.isna().any():
        raise ValueError(f"Unbekannte Metrik(en): {unknown_metrics or ['<leer>']}.")

    period_end = _to_day_column(src["period_end"])
    if period_end.isna().any():
        raise ValueError("Fundamentaldaten mit fehlendem Periodenende.")
    if "reported_date" in src.columns:
        reported = _to_day_column(src["reported_date"])
    else:
        reported = pd.Series(pd.NaT, index=src.index, dtype="datetime64[ns]")
    fallback = period_end + period_type.map(_LAGS).astype("timedelta64[ns]")
    reported = reported.where(reported.notna() & (reported >= period_end), fallback)

    result = pd.DataFrame(
        {
            "period_end": period_end,
            "period_type": period_type.astype("object"),
            "metric": metric.astype("object"),
            "value": pd.to_numeric(src["value"], errors="coerce").astype("float64"),
            "reported_date": reported.astype("datetime64[ns]"),
        }
    )
    result = result[result["value"].notna()]
    key = ["period_end", "period_type", "metric"]
    duplicated = result.duplicated(subset=key, keep=False)
    if duplicated.any():
        first = result.loc[duplicated, key].iloc[0]
        raise ValueError(
            "Doppelter Eintrag in den Fundamentaldaten: "
            f"{first['period_end'].date()} {first['period_type']} {first['metric']}."
        )
    result = result.sort_values(key, kind="mergesort")
    return result.loc[:, list(FUNDAMENTALS_COLUMNS)].reset_index(drop=True)


def update_fundamentals(
    store: MarketStore,
    provider: FundamentalsProvider,
    symbols: Iterable[str],
) -> dict[str, int]:
    """Lädt Fundamentaldaten und schreibt sie in den Speicher.

    Vorhandene Werte derselben Periode und Metrik werden überschrieben, andere bleiben
    erhalten. Fehler der Quelle (:class:`~trading_helfer.data.providers.ProviderError`)
    werden nicht abgefangen; das Datenupdate entscheidet, wie es damit umgeht.

    Args:
        store: Offener Speicher.
        provider: Quelle der Fundamentaldaten.
        symbols: Symbole in App-Schreibweise.

    Returns:
        Anzahl geschriebener Zeilen je Symbol.
    """
    source = getattr(provider, "name", None)
    written: dict[str, int] = {}
    for symbol in symbols:
        frame = normalize_fundamentals(provider.fetch_fundamentals(symbol))
        written[symbol] = store.upsert_fundamentals(symbol, frame, source=source)
    return written


def _to_day(value: pd.Timestamp) -> pd.Timestamp:
    if value.tzinfo is not None:
        value = value.tz_localize(None)
    return value.normalize()


def _to_day_column(values: pd.Series) -> pd.Series:
    converted = pd.to_datetime(values, errors="coerce")
    if isinstance(converted.dtype, pd.DatetimeTZDtype):
        converted = converted.dt.tz_localize(None)
    return converted.dt.normalize().astype("datetime64[ns]")
