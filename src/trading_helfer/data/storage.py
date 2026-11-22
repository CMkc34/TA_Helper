"""Persistenter Speicher aller Marktdaten in einer DuckDB-Datei.

:class:`MarketStore` öffnet (oder erzeugt) die Datenbank, bringt das Schema über die
Migrationen in ``data/migrations/`` auf den neuesten Stand und bietet Methoden zum
Schreiben und Lesen von Kursen, Stammdaten, Corporate Actions und Fundamentaldaten.

Gespeichert werden nur Tageskurse mit Rohwerten und dem Bereinigungsfaktor
``adj_factor = adj_close / close``. :meth:`MarketStore.load_bars` liefert wahlweise
bereinigte (OHLC × ``adj_factor``) oder rohe Kurse.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self

import duckdb
import pandas as pd

from trading_helfer.core.bars import (
    ADJ_CLOSE_COLUMN,
    BAR_COLUMNS,
    BAR_DTYPES,
    PRICE_COLUMNS,
    validate_bars,
)
from trading_helfer.data.schema import MIGRATIONS_DIR, run_migrations

__all__ = [
    "ACTION_DIVIDEND",
    "ACTION_SPLIT",
    "ACTION_TYPES",
    "ADJ_FACTOR_COLUMN",
    "EXCLUSION_ALLOWED",
    "EXCLUSION_AUTO",
    "EXCLUSION_MANUAL",
    "EXCLUSION_SOURCES",
    "INSTRUMENT_COLUMNS",
    "LOADED_BAR_COLUMNS",
    "PERIOD_TYPES",
    "DateLike",
    "MarketStore",
    "StorageError",
]

DateLike = dt.date | str | pd.Timestamp
"""Datumsangabe: ``datetime.date``, ``pd.Timestamp`` oder ISO-Text wie ``"2024-01-31"``."""

ADJ_FACTOR_COLUMN: Final = "adj_factor"
"""Spalte mit dem Bereinigungsfaktor ``adj_close / close``."""

LOADED_BAR_COLUMNS: Final[tuple[str, ...]] = (*BAR_COLUMNS, ADJ_CLOSE_COLUMN, ADJ_FACTOR_COLUMN)
"""Spalten des Ergebnisses von :meth:`MarketStore.load_bars` in dieser Reihenfolge."""

INSTRUMENT_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "name",
    "exchange",
    "sector",
    "industry",
    "currency",
    "is_active",
    "first_date",
    "last_date",
    "source",
)
"""Spalten der Tabelle ``instruments``."""

ACTION_SPLIT: Final = "split"
"""Corporate Action: Aktiensplit, ``value`` = Verhältnis (4:1 → 4.0)."""
ACTION_DIVIDEND: Final = "dividend"
"""Corporate Action: Dividende, ``value`` = Betrag je Aktie."""
ACTION_TYPES: Final = frozenset({ACTION_SPLIT, ACTION_DIVIDEND})
"""Erlaubte Werte für ``corporate_actions.type``."""

EXCLUSION_AUTO: Final = "auto"
"""Ausschluss durch die automatische Prüfung auf Fremdkurse (``data.quality``)."""
EXCLUSION_MANUAL: Final = "manuell"
"""Von Hand ausgeschlossen; die automatische Prüfung ändert den Eintrag nie."""
EXCLUSION_ALLOWED: Final = "freigegeben"
"""Von Hand erlaubt; die automatische Prüfung schließt den Wert nicht mehr aus."""
EXCLUSION_SOURCES: Final = frozenset({EXCLUSION_AUTO, EXCLUSION_MANUAL, EXCLUSION_ALLOWED})
"""Erlaubte Werte für ``symbol_exclusions.source``."""

PERIOD_TYPES: Final = frozenset({"Q", "FY"})
"""Erlaubte Werte für ``fundamentals.period_type``."""

_FUNDAMENTAL_COLUMNS: Final = ("period_end", "period_type", "metric", "value", "reported_date")
_DATE_INSTRUMENT_COLUMNS: Final = ("first_date", "last_date")


class StorageError(ValueError):
    """Ungültige Eingabe für den Speicher; die Meldung ist deutsch."""


class MarketStore:
    """Zugriff auf die DuckDB-Datenbank mit allen Tabellen des Datenmodells.

    Beim Öffnen wird die Datenbank bei Bedarf angelegt und alle fehlenden Migrationen
    werden ausgeführt. Bereits angewendete Migrationen laufen nie doppelt.

    Symbole werden ohne umgebende Leerzeichen und in Großbuchstaben gespeichert.

    Beispiel::

        with MarketStore(settings.database_path) as store:
            store.upsert_bars("AAPL", bars)
            adjusted = store.load_bars("AAPL", start="2020-01-01")
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        read_only: bool = False,
        migrations_dir: Path = MIGRATIONS_DIR,
    ) -> None:
        """Öffnet die Datenbank und wendet fehlende Migrationen an.

        Args:
            db_path: Pfad der DuckDB-Datei; fehlende Ordner werden angelegt.
                ``":memory:"`` erzeugt eine flüchtige Datenbank (für Tests).
            read_only: Nur lesend öffnen; dann werden keine Migrationen ausgeführt.
            migrations_dir: Ordner mit den Migrationsdateien.
        """
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Any = duckdb.connect(self._path, read_only=read_only)
        self.applied_migrations: list[int] = (
            [] if read_only else run_migrations(self._conn, migrations_dir)
        )
        """Versionen, die beim Öffnen dieses Objekts neu angewendet wurden."""

    # ------------------------------------------------------------------ Verwaltung

    @property
    def path(self) -> str:
        """Pfad der Datenbank (oder ``":memory:"``)."""
        return self._path

    @property
    def connection(self) -> Any:
        """Die zugrunde liegende DuckDB-Verbindung für Module mit eigenen Abfragen."""
        return self._conn

    def schema_version(self) -> int:
        """Höchste angewendete Migrationsversion (0, wenn keine angewendet ist)."""
        rows = self._conn.execute("SELECT coalesce(max(version), 0) FROM schema_version").fetchall()
        return int(rows[0][0])

    def close(self) -> None:
        """Schließt die Verbindung; weitere Zugriffe schlagen fehl."""
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """Kontextmanager für eine Transaktion; bei einer Ausnahme wird zurückgerollt."""
        self._conn.execute("BEGIN TRANSACTION")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    # ------------------------------------------------------------------ Kurse

    def upsert_bars(self, symbol: str, df: pd.DataFrame, *, source: str | None = None) -> int:
        """Schreibt Tageskurse eines Symbols; vorhandene Tage werden überschrieben.

        Die Eingabe wird mit :func:`~trading_helfer.core.bars.validate_bars` geprüft.
        Fehlt ``adj_close``, gilt ``adj_close = close`` (Faktor 1). Eine Spalte
        ``source`` in ``df`` hat Vorrang vor dem Argument ``source``.

        Args:
            symbol: Tickersymbol.
            df: Kursreihe mit ``date, open, high, low, close, volume`` und optional
                ``adj_close`` und ``source``.
            source: Name der Datenquelle für alle Zeilen ohne eigene ``source``.

        Returns:
            Anzahl geschriebener Zeilen.

        Raises:
            StorageError: Symbol leer oder ``close`` ist 0 (Faktor nicht berechenbar).
            trading_helfer.core.bars.BarsValidationError: Die Kursreihe ist ungültig.
        """
        sym = _normalize_symbol(symbol)
        bars = validate_bars(df)
        if bars.empty:
            return 0
        if ADJ_CLOSE_COLUMN not in bars.columns:
            bars[ADJ_CLOSE_COLUMN] = bars["close"]
        if (bars["close"] == 0).any():
            raise StorageError(
                f"{sym}: Schlusskurs 0 – Bereinigungsfaktor adj_close / close nicht berechenbar."
            )
        frame = pd.DataFrame(
            {
                "symbol": sym,
                "date": bars["date"].dt.date,
                "open": bars["open"],
                "high": bars["high"],
                "low": bars["low"],
                "close": bars["close"],
                "volume": bars["volume"],
                "adj_close": bars[ADJ_CLOSE_COLUMN],
                ADJ_FACTOR_COLUMN: bars[ADJ_CLOSE_COLUMN] / bars["close"],
                "source": bars["source"] if "source" in bars.columns else source,
            }
        )
        frame["source"] = frame["source"].astype("object").where(frame["source"].notna(), source)
        self._upsert("bars_daily", frame, key=("symbol", "date"))
        return len(frame)

    def load_bars(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Lädt Tageskurse eines Symbols, aufsteigend nach Datum.

        Args:
            symbol: Tickersymbol.
            start: Erster Tag (einschließlich); ``None`` = ab Beginn.
            end: Letzter Tag (einschließlich); ``None`` = bis zum Ende.
            adjusted: ``True`` liefert ``open, high, low, close`` × ``adj_factor``
                (bereinigt um Splits und Dividenden), ``False`` die Rohkurse.
                ``volume``, ``adj_close`` und ``adj_factor`` bleiben unverändert.

        Returns:
            DataFrame mit den Spalten :data:`LOADED_BAR_COLUMNS` und ``RangeIndex``;
            leer (mit allen Spalten), wenn keine Kurse vorhanden sind.
        """
        sym = _normalize_symbol(symbol)
        if adjusted:
            prices = ", ".join(f"{c} * adj_factor AS {c}" for c in PRICE_COLUMNS)
        else:
            prices = ", ".join(PRICE_COLUMNS)
        sql = (
            f"SELECT CAST(date AS TIMESTAMP_NS) AS date, {prices}, volume, adj_close, "
            "adj_factor FROM bars_daily WHERE symbol = ?"
        )
        params: list[Any] = [sym]
        if start is not None:
            sql += " AND date >= ?"
            params.append(_to_date(start))
        if end is not None:
            sql += " AND date <= ?"
            params.append(_to_date(end))
        sql += " ORDER BY date"
        result: pd.DataFrame = self._conn.execute(sql, params).df()
        return _with_bar_dtypes(result)

    def last_date(self, symbol: str) -> dt.date | None:
        """Letzter gespeicherter Handelstag eines Symbols (``None`` ohne Kurse)."""
        rows = self._conn.execute(
            "SELECT max(date) FROM bars_daily WHERE symbol = ?", [_normalize_symbol(symbol)]
        ).fetchall()
        value = rows[0][0]
        return value if isinstance(value, dt.date) else None

    def symbols(self) -> list[str]:
        """Alle Symbole mit gespeicherten Kursen, alphabetisch sortiert."""
        rows = self._conn.execute(
            "SELECT DISTINCT symbol FROM bars_daily ORDER BY symbol"
        ).fetchall()
        return [str(row[0]) for row in rows]

    # ------------------------------------------------------------------ Stammdaten

    def upsert_instruments(self, df: pd.DataFrame) -> int:
        """Schreibt Stammdaten; vorhandene Symbole werden aktualisiert.

        Nur die in ``df`` enthaltenen Spalten werden geschrieben, andere Felder eines
        vorhandenen Symbols bleiben erhalten.

        Args:
            df: Pflichtspalte ``symbol``, weitere Spalten aus :data:`INSTRUMENT_COLUMNS`.

        Returns:
            Anzahl geschriebener Zeilen.

        Raises:
            StorageError: ``symbol`` fehlt, ist leer oder doppelt, oder eine Spalte ist
                unbekannt.
        """
        if "symbol" not in df.columns:
            raise StorageError("Stammdaten: Pflichtspalte symbol fehlt.")
        unknown = [str(c) for c in df.columns if c not in INSTRUMENT_COLUMNS]
        if unknown:
            raise StorageError(f"Stammdaten: unbekannte Spalte(n): {', '.join(unknown)}.")
        if df.empty:
            return 0
        frame = df.reset_index(drop=True).copy()
        frame["symbol"] = [_normalize_symbol(s) for s in frame["symbol"]]
        _reject_duplicates(frame, ("symbol",), "Stammdaten")
        for column in _DATE_INSTRUMENT_COLUMNS:
            if column in frame.columns:
                frame[column] = _to_date_column(frame[column])
        if "is_active" in frame.columns:
            frame["is_active"] = frame["is_active"].astype("boolean")
        self._upsert("instruments", frame, key=("symbol",))
        return len(frame)

    def get_instrument(self, symbol: str) -> dict[str, Any] | None:
        """Stammdaten eines Symbols als Dictionary (``None``, wenn unbekannt).

        Fehlende Werte sind ``None``, Datumsfelder ``datetime.date``.
        """
        cursor = self._conn.execute(
            f"SELECT {', '.join(INSTRUMENT_COLUMNS)} FROM instruments WHERE symbol = ?",
            [_normalize_symbol(symbol)],
        )
        rows = cursor.fetchall()
        if not rows:
            return None
        return dict(zip(INSTRUMENT_COLUMNS, rows[0], strict=True))

    # ------------------------------------------------------------------ Corporate Actions

    def upsert_actions(self, symbol: str, df: pd.DataFrame) -> int:
        """Schreibt Splits und Dividenden eines Symbols; gleiche Einträge werden überschrieben.

        Args:
            symbol: Tickersymbol.
            df: Spalten ``date, type, value``; ``type`` ist ``"split"`` oder
                ``"dividend"`` (Groß-/Kleinschreibung egal).

        Returns:
            Anzahl geschriebener Zeilen.

        Raises:
            StorageError: Spalten fehlen, ``type`` ist unbekannt, Werte fehlen oder ein
                Eintrag kommt doppelt vor.
        """
        sym = _normalize_symbol(symbol)
        _require_columns(df, ("date", "type", "value"), "Corporate Actions")
        if df.empty:
            return 0
        frame = pd.DataFrame(
            {
                "symbol": sym,
                "date": _to_date_column(df["date"].reset_index(drop=True)),
                "type": df["type"].reset_index(drop=True).astype("string").str.strip().str.lower(),
                "value": pd.to_numeric(df["value"].reset_index(drop=True), errors="coerce"),
            }
        )
        if frame[["date", "type", "value"]].isna().any().any():
            raise StorageError(f"{sym}: Corporate Actions mit fehlenden Werten.")
        unknown = sorted(set(frame["type"]) - ACTION_TYPES)
        if unknown:
            raise StorageError(f"{sym}: unbekannte Art(en) von Corporate Action: {unknown}.")
        _reject_duplicates(frame, ("date", "type"), f"{sym}: Corporate Actions")
        self._upsert("corporate_actions", frame, key=("symbol", "date", "type"))
        return len(frame)

    def load_actions(self, symbol: str) -> pd.DataFrame:
        """Corporate Actions eines Symbols mit Spalten ``date, type, value``, nach Datum."""
        result: pd.DataFrame = self._conn.execute(
            "SELECT date, type, value FROM corporate_actions WHERE symbol = ? ORDER BY date, type",
            [_normalize_symbol(symbol)],
        ).df()
        result["date"] = pd.to_datetime(result["date"]).astype("datetime64[ns]")
        return result

    # ------------------------------------------------------------------ Ausschlüsse

    def load_exclusions(self) -> pd.DataFrame:
        """Alle Einträge der Tabelle ``symbol_exclusions``, alphabetisch.

        Returns:
            Spalten ``symbol, reason, source, created_at``; leer, wenn es keine gibt oder
            die Datenbank (nur lesend geöffnet) die Tabelle noch nicht hat.
        """
        columns = ["symbol", "reason", "source", "created_at"]
        try:
            result: pd.DataFrame = self._conn.execute(
                f"SELECT {', '.join(columns)} FROM symbol_exclusions ORDER BY symbol"
            ).df()
        except duckdb.CatalogException:
            return pd.DataFrame(columns=columns)
        return result

    def excluded_symbols(self) -> dict[str, str]:
        """Werte, die Backtests nicht verwenden, mit Grund (``auto`` und ``manuell``)."""
        frame = self.load_exclusions()
        active = frame[frame["source"] != EXCLUSION_ALLOWED]
        return dict(zip(active["symbol"], active["reason"], strict=True))

    def set_exclusion(self, symbol: str, reason: str, source: str = EXCLUSION_MANUAL) -> None:
        """Legt einen Ausschluss (oder eine Freigabe) für ``symbol`` an oder ersetzt ihn.

        Args:
            symbol: Tickersymbol.
            reason: Grund auf Deutsch, z. B. „Fremdkurse: 5 Sprünge über 80 %“.
            source: Einer der Werte aus :data:`EXCLUSION_SOURCES`.

        Raises:
            StorageError: Leeres Symbol, leerer Grund oder unbekannte Quelle.
        """
        sym = _normalize_symbol(symbol)
        if not reason.strip():
            raise StorageError("Ausschluss: Grund darf nicht leer sein.")
        if source not in EXCLUSION_SOURCES:
            allowed = ", ".join(sorted(EXCLUSION_SOURCES))
            raise StorageError(f"Ausschluss: unbekannte Quelle {source!r} (erlaubt: {allowed}).")
        self._conn.execute(
            "INSERT OR REPLACE INTO symbol_exclusions (symbol, reason, source, created_at) "
            "VALUES (?, ?, ?, ?)",
            [sym, reason.strip(), source, dt.datetime.now().replace(microsecond=0)],
        )

    def remove_exclusion(self, symbol: str) -> bool:
        """Entfernt den Eintrag für ``symbol``; ``True``, wenn es einen gab."""
        sym = _normalize_symbol(symbol)
        found = self._conn.execute(
            "SELECT count(*) FROM symbol_exclusions WHERE symbol = ?", [sym]
        ).fetchall()[0][0]
        self._conn.execute("DELETE FROM symbol_exclusions WHERE symbol = ?", [sym])
        return bool(found)

    # ------------------------------------------------------------------ Fundamentaldaten

    def upsert_fundamentals(
        self, symbol: str, df: pd.DataFrame, *, source: str | None = None
    ) -> int:
        """Schreibt Fundamentaldaten eines Symbols im Langformat.

        Vorhandene Einträge mit gleichem ``period_end, period_type, metric`` werden
        überschrieben.

        Args:
            symbol: Tickersymbol.
            df: Spalten ``period_end, period_type, metric, value, reported_date``
                (``period_type`` ist ``"Q"`` oder ``"FY"``) und optional ``source``.
            source: Datenquelle für Zeilen ohne eigene ``source``.

        Returns:
            Anzahl geschriebener Zeilen.

        Raises:
            StorageError: Spalten fehlen, ``period_type`` ist unbekannt, Schlüsselwerte
                fehlen oder ein Eintrag kommt doppelt vor.
        """
        sym = _normalize_symbol(symbol)
        _require_columns(df, _FUNDAMENTAL_COLUMNS, "Fundamentaldaten")
        if df.empty:
            return 0
        src = df.reset_index(drop=True)
        frame = pd.DataFrame(
            {
                "symbol": sym,
                "period_end": _to_date_column(src["period_end"]),
                "period_type": src["period_type"].astype("string").str.strip().str.upper(),
                "metric": src["metric"].astype("string").str.strip(),
                "value": pd.to_numeric(src["value"], errors="coerce").astype("float64"),
                "reported_date": _to_date_column(src["reported_date"]),
                "source": src["source"] if "source" in src.columns else source,
            }
        )
        frame["source"] = frame["source"].astype("object").where(frame["source"].notna(), source)
        if frame[["period_end", "period_type", "metric"]].isna().any().any():
            raise StorageError(f"{sym}: Fundamentaldaten mit fehlendem Schlüssel.")
        unknown = sorted(set(frame["period_type"]) - PERIOD_TYPES)
        if unknown:
            raise StorageError(f"{sym}: unbekannte(r) period_type: {unknown} (erlaubt: Q, FY).")
        key = ("period_end", "period_type", "metric")
        _reject_duplicates(frame, key, f"{sym}: Fundamentaldaten")
        self._upsert("fundamentals", frame, key=("symbol", *key))
        return len(frame)

    def load_fundamentals(self, symbol: str, as_of: DateLike | None = None) -> pd.DataFrame:
        """Fundamentaldaten eines Symbols im Langformat.

        Args:
            symbol: Tickersymbol.
            as_of: Stichtag. Nur Werte mit ``reported_date ≤ as_of`` werden geliefert;
                Werte ohne ``reported_date`` fehlen dann ebenfalls, weil ihr
                Veröffentlichungstag unbekannt ist (kein Blick in die Zukunft).
                ``None`` liefert alle Werte.

        Returns:
            DataFrame mit ``period_end, period_type, metric, value, reported_date,
            source``, sortiert nach ``period_end, period_type, metric``; Datumsspalten
            als ``datetime64[ns]``.
        """
        sql = (
            "SELECT period_end, period_type, metric, value, reported_date, source "
            "FROM fundamentals WHERE symbol = ?"
        )
        params: list[Any] = [_normalize_symbol(symbol)]
        if as_of is not None:
            sql += " AND reported_date <= ?"
            params.append(_to_date(as_of))
        sql += " ORDER BY period_end, period_type, metric"
        result: pd.DataFrame = self._conn.execute(sql, params).df()
        for column in ("period_end", "reported_date"):
            result[column] = pd.to_datetime(result[column]).astype("datetime64[ns]")
        result["value"] = result["value"].astype("float64")
        return result

    # ------------------------------------------------------------------ intern

    def _upsert(self, table: str, frame: pd.DataFrame, key: Sequence[str]) -> None:
        """Fügt ``frame`` in ``table`` ein; Zeilen mit gleichem Schlüssel werden ersetzt.

        Nur die Spalten von ``frame`` werden aktualisiert.
        """
        columns = [str(c) for c in frame.columns]
        quoted = ", ".join(f'"{c}"' for c in columns)
        updates = [c for c in columns if c not in key]
        if updates:
            conflict = "DO UPDATE SET " + ", ".join(f'"{c}" = excluded."{c}"' for c in updates)
        else:
            conflict = "DO NOTHING"
        key_list = ", ".join(f'"{c}"' for c in key)
        view = "_upsert_source"
        self._conn.register(view, frame)
        try:
            self._conn.execute(
                f"INSERT INTO {table} ({quoted}) SELECT {quoted} FROM {view} "
                f"ON CONFLICT ({key_list}) {conflict}"
            )
        finally:
            self._conn.unregister(view)


def _normalize_symbol(symbol: str) -> str:
    """Entfernt Leerzeichen und wandelt in Großbuchstaben; leere Symbole sind ein Fehler."""
    if not isinstance(symbol, str) or not symbol.strip():
        raise StorageError(f"Ungültiges Symbol: {symbol!r}.")
    return symbol.strip().upper()


def _to_date(value: DateLike) -> dt.date:
    """Wandelt eine Datumsangabe in ``datetime.date``."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise StorageError(f"Ungültiges Datum: {value!r}.") from exc
    if pd.isna(timestamp):
        raise StorageError(f"Ungültiges Datum: {value!r}.")
    result: dt.date = timestamp.date()
    return result


def _to_date_column(values: pd.Series[Any]) -> pd.Series[Any]:
    """Wandelt eine Spalte in ``datetime.date``-Objekte (fehlende Werte → ``None``)."""
    try:
        stamps = pd.to_datetime(values)
    except (TypeError, ValueError) as exc:
        raise StorageError(f"Ungültige Datumswerte in Spalte {values.name}.") from exc
    if getattr(stamps.dt, "tz", None) is not None:
        stamps = stamps.dt.tz_localize(None)
    return pd.Series(
        [None if pd.isna(s) else s.date() for s in stamps], index=values.index, dtype="object"
    )


def _require_columns(df: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise StorageError(f"{what}: Pflichtspalte(n) fehlen: {', '.join(missing)}.")


def _reject_duplicates(frame: pd.DataFrame, key: Sequence[str], what: str) -> None:
    duplicated = frame.duplicated(subset=list(key), keep=False)
    if duplicated.any():
        raise StorageError(f"{what}: doppelte Einträge für denselben Schlüssel ({', '.join(key)}).")


def _with_bar_dtypes(result: pd.DataFrame) -> pd.DataFrame:
    """Bringt das Abfrageergebnis von :meth:`MarketStore.load_bars` auf feste Datentypen.

    Nur Spalten mit abweichendem Datentyp werden umgewandelt (schnell bei vielen Abfragen).
    """
    dtypes: dict[str, Any] = {**BAR_DTYPES, ADJ_FACTOR_COLUMN: "float64"}
    out = result.loc[:, list(LOADED_BAR_COLUMNS)]
    wrong = {
        name: dtypes[name] for name in LOADED_BAR_COLUMNS if str(out[name].dtype) != dtypes[name]
    }
    if wrong:
        out = out.astype(wrong)
    if not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index(drop=True)
    return out
