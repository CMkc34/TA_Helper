"""Fassade für den Kennzahlen-Tab: Kennzahlen, Sektor-Median und Verlauf je Wert.

Das Modul importiert kein Qt. Grundlage sind :func:`~trading_helfer.fundamentals.metrics.
compute_metrics` und :func:`~trading_helfer.fundamentals.metrics.metrics_history` auf dem
DuckDB-Kursspeicher. Jede Anfrage öffnet die Datenbank nur lesend und schließt sie wieder,
damit der Service in Hintergrund-Threads laufen kann.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Final

import pandas as pd

from trading_helfer.data.storage import MarketStore
from trading_helfer.fundamentals.metrics import (
    METRIC_LABELS,
    METRIC_NAMES,
    MetricValues,
    compute_metrics,
    metrics_history,
)

__all__ = [
    "HIGHER_IS_BETTER",
    "LOWER_IS_BETTER",
    "METRIC_FORMATS",
    "Comparison",
    "FundamentalsData",
    "FundamentalsService",
    "MetricFormat",
    "MetricRow",
    "StoreOpener",
    "compare",
    "empty_history",
    "format_metric",
]

StoreOpener = Callable[[], AbstractContextManager[MarketStore | None]]
"""Öffnet den Kursspeicher (``None``, wenn es keinen gibt) und schließt ihn danach."""


class Comparison(StrEnum):
    """Einordnung eines Werts gegenüber dem Sektor-Median."""

    BETTER = "better"
    WORSE = "worse"
    EQUAL = "equal"


class MetricFormat(StrEnum):
    """Darstellungsart einer Kennzahl."""

    PRICE = "price"
    """Kurs in USD mit zwei Nachkommastellen."""
    AMOUNT = "amount"
    """Großer USD-Betrag (Mio., Mrd., Bio.)."""
    COUNT = "count"
    """Stückzahl (Mio., Mrd.)."""
    PERCENT = "percent"
    """Anteil, angezeigt in Prozent (``0.25`` → ``25,0 %``)."""
    RATIO = "ratio"
    """Verhältniszahl mit zwei Nachkommastellen."""
    SCORE = "score"
    """Punktwert ohne Nachkommastellen (Piotroski)."""


HIGHER_IS_BETTER: Final[frozenset[str]] = frozenset(
    {
        "dividend_yield",
        "dividend_growth_5y",
        "fcf_yield",
        "revenue_growth_yoy",
        "revenue_growth_3y",
        "earnings_growth_yoy",
        "earnings_growth_3y",
        "eps_growth_yoy",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "roe",
        "roa",
        "roic",
        "asset_turnover",
        "interest_coverage",
        "current_ratio",
        "quick_ratio",
        "piotroski_f_score",
        "altman_z_score",
    }
)
"""Kennzahlen, bei denen ein höherer Wert als der Sektor-Median besser ist."""

LOWER_IS_BETTER: Final[frozenset[str]] = frozenset(
    {
        "pe_ratio",
        "ps_ratio",
        "pb_ratio",
        "pcf_ratio",
        "ev_ebitda",
        "peg_ratio",
        "debt_to_equity",
        "net_debt_to_ebitda",
        "share_change_yoy",
    }
)
"""Kennzahlen, bei denen ein niedrigerer Wert besser ist (Bewertung, Verschuldung)."""

METRIC_FORMATS: Final[Mapping[str, MetricFormat]] = {
    "price": MetricFormat.PRICE,
    "eps_ttm": MetricFormat.PRICE,
    "shares_outstanding": MetricFormat.COUNT,
    "market_cap": MetricFormat.AMOUNT,
    "enterprise_value": MetricFormat.AMOUNT,
    "revenue_ttm": MetricFormat.AMOUNT,
    "net_income_ttm": MetricFormat.AMOUNT,
    "ebitda_ttm": MetricFormat.AMOUNT,
    "free_cashflow": MetricFormat.AMOUNT,
    "dividend_yield": MetricFormat.PERCENT,
    "payout_ratio": MetricFormat.PERCENT,
    "dividend_growth_5y": MetricFormat.PERCENT,
    "fcf_yield": MetricFormat.PERCENT,
    "revenue_growth_yoy": MetricFormat.PERCENT,
    "revenue_growth_3y": MetricFormat.PERCENT,
    "earnings_growth_yoy": MetricFormat.PERCENT,
    "earnings_growth_3y": MetricFormat.PERCENT,
    "eps_growth_yoy": MetricFormat.PERCENT,
    "gross_margin": MetricFormat.PERCENT,
    "operating_margin": MetricFormat.PERCENT,
    "net_margin": MetricFormat.PERCENT,
    "roe": MetricFormat.PERCENT,
    "roa": MetricFormat.PERCENT,
    "roic": MetricFormat.PERCENT,
    "share_change_yoy": MetricFormat.PERCENT,
    "piotroski_f_score": MetricFormat.SCORE,
}
"""Darstellung je Kennzahl; nicht aufgeführte Kennzahlen sind :attr:`MetricFormat.RATIO`."""

MISSING: Final = "–"
"""Anzeige für fehlende oder nicht berechenbare Werte."""

_EQUAL_TOLERANCE: Final = 1e-9
_HISTORY_CACHE_SIZE: Final = 16
_PEER_CACHE_SIZE: Final = 8


# ---------------------------------------------------------------------- Hilfsfunktionen


def _german(value: float, decimals: int) -> str:
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def format_metric(metric: str, value: float | None) -> str:
    """Formatiert den Wert einer Kennzahl deutsch; fehlende Werte als „–“.

    Args:
        metric: Name aus :data:`~trading_helfer.fundamentals.metrics.METRIC_NAMES`.
        value: Wert oder ``None``/``NaN``.

    Returns:
        Anzeigetext, z. B. ``"25,3 %"``, ``"2,51 Bio. $"`` oder ``"18,40"``.
    """
    if value is None or not math.isfinite(value):
        return MISSING
    kind = METRIC_FORMATS.get(metric, MetricFormat.RATIO)
    if kind is MetricFormat.PERCENT:
        return f"{_german(value * 100.0, 1)} %"
    if kind is MetricFormat.SCORE:
        return _german(value, 0)
    if kind is MetricFormat.PRICE:
        return f"{_german(value, 2)} $"
    if kind in (MetricFormat.AMOUNT, MetricFormat.COUNT):
        suffix = " $" if kind is MetricFormat.AMOUNT else ""
        for factor, unit in ((1e12, "Bio."), (1e9, "Mrd."), (1e6, "Mio.")):
            if abs(value) >= factor:
                return f"{_german(value / factor, 2)} {unit}{suffix}"
        return f"{_german(value, 0)}{suffix}"
    return _german(value, 2)


def compare(metric: str, value: float | None, sector_median: float | None) -> Comparison | None:
    """Ordnet ``value`` gegenüber dem Sektor-Median ein.

    Args:
        metric: Name der Kennzahl.
        value: Wert des Symbols.
        sector_median: Median des Sektors.

    Returns:
        :class:`Comparison` oder ``None``, wenn ein Wert fehlt oder die Kennzahl keine
        Richtung hat (z. B. Marktkapitalisierung).
    """
    if value is None or sector_median is None:
        return None
    if not (math.isfinite(value) and math.isfinite(sector_median)):
        return None
    if metric in HIGHER_IS_BETTER:
        sign = 1.0
    elif metric in LOWER_IS_BETTER:
        sign = -1.0
    else:
        return None
    diff = (value - sector_median) * sign
    if abs(diff) <= _EQUAL_TOLERANCE * max(1.0, abs(sector_median)):
        return Comparison.EQUAL
    return Comparison.BETTER if diff > 0 else Comparison.WORSE


def empty_history() -> pd.DataFrame:
    """Leerer Kennzahlenverlauf im Format von :func:`metrics_history`."""
    frame = pd.DataFrame(columns=["period_end", "as_of", *METRIC_NAMES])
    for column in ("period_end", "as_of"):
        frame[column] = frame[column].astype("datetime64[ns]")
    frame[list(METRIC_NAMES)] = frame[list(METRIC_NAMES)].astype("float64")
    return frame


# ---------------------------------------------------------------------- Ergebnis


@dataclass(frozen=True)
class MetricRow:
    """Eine Zeile der Kennzahlentabelle.

    Attributes:
        metric: Interner Name.
        label: Deutscher Anzeigename.
        value: Wert des Symbols (``None`` = nicht berechenbar).
        sector_median: Median im Sektor (``None`` = kein Sektor oder kein Wert).
        comparison: Einordnung gegenüber dem Median.
    """

    metric: str
    label: str
    value: float | None
    sector_median: float | None
    comparison: Comparison | None


@dataclass(frozen=True)
class FundamentalsData:
    """Kennzahlen eines Symbols zum Stichtag samt Verlauf.

    Attributes:
        symbol: Tickersymbol.
        as_of: Stichtag.
        sector: Sektor aus den Stammdaten (``None``, wenn unbekannt).
        peer_count: Anzahl Werte im Sektor (inklusive des Symbols).
        rows: Je Kennzahl eine Zeile, in Reihenfolge von ``METRIC_NAMES``.
        history: Verlauf je Periodenende wie :func:`metrics_history`, nur Perioden, die
            am Stichtag veröffentlicht waren.
    """

    symbol: str
    as_of: dt.date
    sector: str | None
    peer_count: int
    rows: tuple[MetricRow, ...]
    history: pd.DataFrame

    @property
    def has_values(self) -> bool:
        """``True``, wenn mindestens eine Kennzahl außer dem Kurs einen Wert hat."""
        return any(r.value is not None for r in self.rows if r.metric != "price")

    def row(self, metric: str) -> MetricRow:
        """Zeile der Kennzahl ``metric``.

        Raises:
            KeyError: Die Kennzahl ist unbekannt.
        """
        for row in self.rows:
            if row.metric == metric:
                return row
        raise KeyError(metric)


# ---------------------------------------------------------------------- Service


def _store_opener(db_path: Path | None) -> StoreOpener:
    @contextmanager
    def open_store() -> Iterator[MarketStore | None]:
        if db_path is None or not db_path.exists():
            yield None
            return
        with MarketStore(db_path, read_only=True) as store:
            yield store

    return open_store


class FundamentalsService:
    """Liefert Kennzahlen, Sektor-Mediane und Verlauf für den Kennzahlen-Tab.

    Args:
        db_path: Pfad der DuckDB-Datei (``Settings.database_path``). ``None`` oder eine
            fehlende Datei ergeben leere Kennzahlen.
        opener: Alternativ eine eigene Funktion, die den Speicher öffnet (für Tests).
    """

    def __init__(self, db_path: Path | None = None, *, opener: StoreOpener | None = None) -> None:
        self._open: StoreOpener = opener if opener is not None else _store_opener(db_path)
        self._history: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._peers: OrderedDict[tuple[str, dt.date], tuple[dict[str, float | None], int]] = (
            OrderedDict()
        )

    def invalidate(self) -> None:
        """Verwirft zwischengespeicherte Verläufe und Sektor-Mediane."""
        self._history.clear()
        self._peers.clear()

    def load(self, symbol: str, as_of: dt.date) -> FundamentalsData:
        """Kennzahlen von ``symbol`` zum Stichtag ``as_of``.

        Args:
            symbol: Tickersymbol.
            as_of: Stichtag; es zählen nur Zahlen, die an diesem Tag veröffentlicht waren.

        Returns:
            Tabelle aller Kennzahlen mit Sektor-Median und Einordnung sowie Verlauf.
        """
        symbol = symbol.strip().upper()
        with self._open() as store:
            if store is None:
                values: MetricValues = dict.fromkeys(METRIC_NAMES)
                return self._build(symbol, as_of, None, values, {}, 0, empty_history())
            values = compute_metrics(symbol, as_of, store=store)
            sector = _sector_of(store, symbol)
            medians: dict[str, float | None] = {}
            peer_count = 0
            if sector is not None:
                medians, peer_count = self._sector_medians(store, sector, as_of)
            history = self._symbol_history(store, symbol)
        visible = history[history["as_of"] <= pd.Timestamp(as_of)].reset_index(drop=True)
        return self._build(symbol, as_of, sector, values, medians, peer_count, visible)

    @staticmethod
    def _build(
        symbol: str,
        as_of: dt.date,
        sector: str | None,
        values: MetricValues,
        medians: Mapping[str, float | None],
        peer_count: int,
        history: pd.DataFrame,
    ) -> FundamentalsData:
        rows = tuple(
            MetricRow(
                metric=name,
                label=METRIC_LABELS[name],
                value=values.get(name),
                sector_median=medians.get(name),
                comparison=compare(name, values.get(name), medians.get(name)),
            )
            for name in METRIC_NAMES
        )
        return FundamentalsData(symbol, as_of, sector, peer_count, rows, history)

    def _symbol_history(self, store: MarketStore, symbol: str) -> pd.DataFrame:
        cached = self._history.get(symbol)
        if cached is None:
            cached = metrics_history(symbol, store=store)
            self._history[symbol] = cached
            if len(self._history) > _HISTORY_CACHE_SIZE:
                self._history.popitem(last=False)
        else:
            self._history.move_to_end(symbol)
        return cached

    def _sector_medians(
        self, store: MarketStore, sector: str, as_of: dt.date
    ) -> tuple[dict[str, float | None], int]:
        key = (sector, as_of)
        cached = self._peers.get(key)
        if cached is not None:
            self._peers.move_to_end(key)
            return cached
        peers = [
            str(row[0])
            for row in store.connection.execute(
                "SELECT symbol FROM instruments WHERE sector = ? ORDER BY symbol", [sector]
            ).fetchall()
        ]
        buckets: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
        for peer in peers:
            for name, value in compute_metrics(peer, as_of, store=store).items():
                if value is not None and math.isfinite(value):
                    buckets[name].append(value)
        result = ({n: (median(v) if v else None) for n, v in buckets.items()}, len(peers))
        self._peers[key] = result
        if len(self._peers) > _PEER_CACHE_SIZE:
            self._peers.popitem(last=False)
        return result


def _sector_of(store: MarketStore, symbol: str) -> str | None:
    instrument = store.get_instrument(symbol)
    if instrument is None:
        return None
    sector = instrument.get("sector")
    if sector is None or not str(sector).strip():
        return None
    return str(sector)
