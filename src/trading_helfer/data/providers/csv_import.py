"""Import von Tageskursen aus CSV-Dateien (Kaggle und eigene Dateien).

Unterstützt werden zwei Dateiformen:

- **Eine Datei je Symbol**: Die Datei hat keine Symbolspalte; das Symbol ist der
  Dateiname ohne Endung (``AAPL.csv`` → ``AAPL``).
- **Eine Datei mit Symbolspalte**: Eine Spalte ``Symbol``/``Ticker`` (o. Ä.) nennt das
  Symbol jeder Zeile, z. B. Kaggle-Datensätze mit allen S&P-500-Werten.

Die Spalten werden über ein Mapping (Quellspalte → Zielspalte) zugeordnet. Ohne
Mapping werden übliche Namen erkannt (``Date, Open, High, Low, Close, Adj Close,
Volume, Symbol/Ticker``), unabhängig von Groß-/Kleinschreibung, Leerzeichen und
Unterstrichen. Ein Mapping kann auch aus einer TOML-Datei kommen (siehe
:func:`load_mapping`).

Zeilen, die die Prüfung nicht bestehen, brechen den Import nicht ab, sondern landen
im Fehlerbericht (:class:`CsvImportReport`). Jede importierte Zeile erhält
``source = 'csv:<dateiname>'``.
"""

from __future__ import annotations

import csv
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from trading_helfer.core.bars import empty_bars
from trading_helfer.data.providers.base import ProviderError, empty_actions
from trading_helfer.data.storage import MarketStore

__all__ = [
    "KNOWN_COLUMN_NAMES",
    "OPTIONAL_TARGETS",
    "REQUIRED_TARGETS",
    "TARGET_COLUMNS",
    "CsvImportProvider",
    "CsvImportReport",
    "CsvMappingError",
    "RowError",
    "import_csv",
    "load_mapping",
    "read_csv_bars",
    "resolve_mapping",
]

REQUIRED_TARGETS: Final[tuple[str, ...]] = ("date", "open", "high", "low", "close", "volume")
"""Zielspalten, die jede Datei liefern muss."""

OPTIONAL_TARGETS: Final[tuple[str, ...]] = ("adj_close", "symbol")
"""Zielspalten, die fehlen dürfen (``symbol`` nur bei einer Datei je Symbol)."""

TARGET_COLUMNS: Final[frozenset[str]] = frozenset((*REQUIRED_TARGETS, *OPTIONAL_TARGETS))
"""Alle erlaubten Zielspalten eines Mappings."""

KNOWN_COLUMN_NAMES: Final[Mapping[str, tuple[str, ...]]] = {
    "date": ("date", "datetime", "timestamp", "datum", "time"),
    "open": ("open", "eroeffnung"),
    "high": ("high", "hoch"),
    "low": ("low", "tief"),
    "close": ("close", "schluss", "schlusskurs"),
    "adj_close": ("adjclose", "adjustedclose", "closeadjusted"),
    "volume": ("volume", "vol", "volumen"),
    "symbol": ("symbol", "ticker", "name"),
}
"""Erkannte Spaltennamen je Zielspalte (normalisiert, in absteigender Priorität)."""

_PRICE_TARGETS: Final = ("open", "high", "low", "close")
_SOURCE_PREFIX: Final = "csv:"
_PROVIDER_NAME: Final = "csv"


class CsvMappingError(ValueError):
    """Das Spalten-Mapping passt nicht zur Datei; die Meldung ist deutsch."""


@dataclass(frozen=True, slots=True)
class RowError:
    """Eine Zeile, die nicht importiert wurde."""

    file: str
    """Dateiname (ohne Ordner)."""
    line: int
    """Zeilennummer in der Datei (Kopfzeile = 1)."""
    symbol: str | None
    """Symbol der Zeile, falls bekannt."""
    reason: str
    """Grund auf Deutsch."""


@dataclass(slots=True)
class CsvImportReport:
    """Ergebnis eines Imports: geschriebene Zeilen und Fehlerbericht."""

    files: list[str] = field(default_factory=list)
    """Importierte Dateien (Namen ohne Ordner)."""
    rows_read: int = 0
    """Gelesene Datenzeilen (ohne Kopfzeile)."""
    rows_imported: int = 0
    """In die Datenbank geschriebene Zeilen."""
    symbols: set[str] = field(default_factory=set)
    """Symbole mit mindestens einer importierten Zeile."""
    errors: list[RowError] = field(default_factory=list)
    """Abgelehnte Zeilen mit Grund."""

    @property
    def rows_rejected(self) -> int:
        """Anzahl abgelehnter Zeilen."""
        return len(self.errors)

    def summary(self) -> str:
        """Kurze deutsche Zusammenfassung für Konsole oder Protokoll."""
        return (
            f"{len(self.files)} Datei(en), {self.rows_read} Zeilen gelesen, "
            f"{self.rows_imported} importiert ({len(self.symbols)} Symbole), "
            f"{self.rows_rejected} abgelehnt."
        )

    def write_errors(self, path: str | Path) -> Path:
        """Schreibt den Fehlerbericht als CSV (``file, line, symbol, reason``).

        Args:
            path: Zieldatei; fehlende Ordner werden angelegt.

        Returns:
            Pfad der geschriebenen Datei.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["file", "line", "symbol", "reason"])
            for error in self.errors:
                writer.writerow([error.file, error.line, error.symbol or "", error.reason])
        return target


# ---------------------------------------------------------------------- Mapping


def _normalize_name(name: str) -> str:
    """Kleinschreibung ohne Leerzeichen, Unter-, Binde- und Schrägstriche und Punkte."""
    return "".join(ch for ch in str(name).lower() if ch not in " _-./")


def load_mapping(path: str | Path) -> dict[str, str]:
    """Liest ein Spalten-Mapping aus einer TOML-Datei.

    Erwartet wird eine Tabelle ``[columns]`` mit ``Quellspalte = "zielspalte"``;
    fehlt sie, gilt die oberste Ebene der Datei als Mapping. Beispiel::

        [columns]
        "Datum" = "date"
        "Eröffnung" = "open"
        "Adj Close" = "adj_close"

    Args:
        path: Pfad der TOML-Datei.

    Returns:
        Mapping Quellspalte → Zielspalte.

    Raises:
        CsvMappingError: Datei fehlt, ist kein gültiges TOML oder enthält Werte, die
            keine Texte sind.
    """
    file = Path(path)
    try:
        with file.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise CsvMappingError(f"Mapping-Datei nicht lesbar: {file} ({exc.strerror}).") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CsvMappingError(f"Mapping-Datei {file.name} ist kein gültiges TOML: {exc}.") from exc
    table: Any = data.get("columns", data)
    if not isinstance(table, dict) or not all(isinstance(v, str) for v in table.values()):
        raise CsvMappingError(
            f'Mapping-Datei {file.name}: erwartet wird [columns] mit Quellspalte = "ziel".'
        )
    return {str(k): v for k, v in table.items()}


def resolve_mapping(
    columns: Iterable[str], mapping: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Bestimmt für die Spalten einer Datei das vollständige Mapping.

    Ein übergebenes Mapping hat Vorrang; nicht zugeordnete Zielspalten werden über
    :data:`KNOWN_COLUMN_NAMES` erkannt.

    Args:
        columns: Spaltennamen der Datei.
        mapping: Optionales Mapping Quellspalte → Zielspalte.

    Returns:
        Mapping Quellspalte → Zielspalte für alle erkannten Spalten.

    Raises:
        CsvMappingError: Das Mapping nennt eine unbekannte Zielspalte, eine Quellspalte,
            die es in der Datei nicht gibt, eine Zielspalte doppelt, oder eine
            Pflichtspalte bleibt ohne Quelle.
    """
    available = [str(c) for c in columns]
    result: dict[str, str] = {}
    for source, target in (mapping or {}).items():
        target_norm = target.strip().lower()
        if target_norm not in TARGET_COLUMNS:
            allowed = ", ".join(sorted(TARGET_COLUMNS))
            raise CsvMappingError(
                f"Unbekannte Zielspalte {target!r} für {source!r} (erlaubt: {allowed})."
            )
        if source not in available:
            raise CsvMappingError(
                f"Spalte {source!r} aus dem Mapping fehlt in der Datei "
                f"(vorhanden: {', '.join(available)})."
            )
        if target_norm in result.values():
            raise CsvMappingError(f"Zielspalte {target_norm!r} ist mehrfach zugeordnet.")
        result[source] = target_norm

    normalized = {_normalize_name(c): c for c in reversed(available)}
    for target, aliases in KNOWN_COLUMN_NAMES.items():
        if target in result.values():
            continue
        for alias in aliases:
            found = normalized.get(alias)
            if found is not None and found not in result:
                result[found] = target
                break

    missing = [t for t in REQUIRED_TARGETS if t not in result.values()]
    if missing:
        raise CsvMappingError(
            f"Pflichtspalte(n) ohne Quelle: {', '.join(missing)} "
            f"(Spalten der Datei: {', '.join(available)}). Bitte ein Mapping angeben."
        )
    return result


# ---------------------------------------------------------------------- Lesen und Prüfen


@dataclass(slots=True)
class _ParsedFile:
    """Geprüfter Inhalt einer Datei: gültige Kurse und abgelehnte Zeilen."""

    name: str
    rows_read: int
    bars: pd.DataFrame
    """Spalten ``symbol, date, open, high, low, close, volume[, adj_close]``."""
    errors: list[RowError]


def _to_float(values: pd.Series[Any]) -> pd.Series[Any]:
    """Zahlen als ``float64``; Text, der keine Zahl ist, wird ``NaN``."""
    if values.dtype.kind in "fiub":
        return values.astype("float64")
    return pd.to_numeric(values.astype("string").str.strip(), errors="coerce").astype("float64")


def _symbol_codes(values: pd.Series[Any]) -> tuple[Any, list[str]]:
    """Normalisierte Symbole als Codes (``-1`` = fehlt) und Liste der Symbolnamen.

    Leerzeichen werden entfernt, Buchstaben groß geschrieben; jeder eindeutige Wert
    wird nur einmal bearbeitet.
    """
    codes, uniques = pd.factorize(values, use_na_sentinel=True)
    names: list[str] = []
    index: dict[str, int] = {}
    remap = np.full(len(uniques) + 1, -1, dtype="int64")  # letzter Eintrag: fehlend
    for i, raw in enumerate(uniques):
        name = str(raw).strip().upper()
        if name:
            remap[i] = index.setdefault(name, len(names))
            if remap[i] == len(names):
                names.append(name)
    return remap[codes], names


def _parse_dates(values: pd.Series[Any], date_format: str | None) -> pd.Series[Any]:
    """Wandelt Text in Tagesdaten ohne Zeitzone; nicht lesbare Werte werden ``NaT``."""
    try:
        parsed = pd.to_datetime(values, errors="coerce", format=date_format)
    except (ValueError, TypeError):  # z. B. gemischte Zeitzonen-Offsets
        text = values.astype("string").str.strip().str.slice(0, 10)
        parsed = pd.to_datetime(text, errors="coerce", format="%Y-%m-%d")
    if not isinstance(parsed.dtype, np.dtype):  # zeitzonenbehaftet
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize().astype("datetime64[ns]")


def _parse_file(
    path: Path,
    mapping: Mapping[str, str] | None,
    date_format: str | None,
    default_symbol: str | None,
) -> _ParsedFile:
    """Liest eine CSV-Datei, wendet das Mapping an und prüft jede Zeile."""
    try:
        header = pd.read_csv(path, nrows=0, skipinitialspace=True)
        resolved = resolve_mapping(header.columns, mapping) if len(header.columns) else {}
        if not resolved:
            raise CsvMappingError(f"{path.name}: Datei ist leer.")
        text_columns = [c for c, t in resolved.items() if t in ("date", "symbol")]
        raw = pd.read_csv(
            path,
            usecols=list(resolved),
            dtype=dict.fromkeys(text_columns, str),
            keep_default_na=False,
            na_values=[""],
            skipinitialspace=True,
        )
    except pd.errors.EmptyDataError as exc:
        raise CsvMappingError(f"{path.name}: Datei ist leer.") from exc
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise CsvMappingError(f"{path.name}: Datei nicht lesbar ({exc}).") from exc

    frame = raw.loc[:, list(resolved)].rename(columns=resolved)
    if "symbol" not in frame.columns:
        if default_symbol is None:
            raise CsvMappingError(f"{path.name}: keine Symbolspalte und kein Symbol angegeben.")
        frame["symbol"] = default_symbol

    count = len(frame)
    reason_texts: list[str] = []
    reason_codes = np.full(count, -1, dtype="int16")  # -1 = gültig

    def reject(mask: Any, reason: str) -> None:
        hit = np.asarray(mask, dtype=bool) & (reason_codes < 0)
        if hit.any():
            reason_texts.append(reason)
            reason_codes[hit] = len(reason_texts) - 1

    symbol_codes, symbol_names = _symbol_codes(frame["symbol"])
    reject(symbol_codes < 0, "Symbol fehlt")
    dates = _parse_dates(frame["date"], date_format)
    reject(dates.isna(), "Datum fehlt oder ist nicht lesbar")

    out: dict[str, Any] = {"date": dates}
    float_targets = [*_PRICE_TARGETS, *(["adj_close"] if "adj_close" in frame.columns else [])]
    for target in float_targets:
        number = _to_float(frame[target])
        reject(~np.isfinite(number.to_numpy()), f"{target} fehlt oder ist keine Zahl")
        out[target] = number
    reject(out["close"].to_numpy() <= 0, "close ist nicht positiv")

    volume = _to_float(frame["volume"]).to_numpy()
    reject(~np.isfinite(volume), "volume fehlt oder ist keine Zahl")
    with np.errstate(invalid="ignore"):
        reject((volume % 1 != 0) | (volume < 0), "volume ist keine nichtnegative ganze Zahl")
    out["volume"] = volume

    # Doppelte Tage: identische Zeilen einmal übernehmen, abweichende ablehnen.
    valid = reason_codes < 0
    keep = valid.copy()
    keys = pd.DataFrame({"s": symbol_codes, "d": dates.to_numpy()})
    key_dupes = keys[valid].duplicated(keep=False)
    if key_dupes.any():
        dupe_rows = np.flatnonzero(valid)[key_dupes.to_numpy()]
        values = pd.DataFrame(out).iloc[dupe_rows].assign(s=symbol_codes[dupe_rows])
        exact = values.duplicated(keep="first").to_numpy()
        keep[dupe_rows[exact]] = False
        rest = values.loc[~exact, ["s", "date"]]
        conflict = rest.duplicated(keep=False).to_numpy()
        mask = np.zeros(count, dtype=bool)
        mask[dupe_rows[~exact][conflict]] = True
        reject(mask, "Datum doppelt mit abweichenden Werten")
        keep &= reason_codes < 0

    bad = np.flatnonzero(reason_codes >= 0)
    errors = [
        RowError(
            file=path.name,
            line=int(row) + 2,
            symbol=symbol_names[symbol_codes[row]] if symbol_codes[row] >= 0 else None,
            reason=reason_texts[reason_codes[row]],
        )
        for row in bad
    ]
    names = np.asarray([*symbol_names, ""], dtype=object)
    good = pd.DataFrame(
        {"symbol": names[symbol_codes[keep]], **{k: v[keep] for k, v in out.items()}}
    )
    good["volume"] = good["volume"].astype("int64")
    return _ParsedFile(name=path.name, rows_read=count, bars=good, errors=errors)


def _csv_files(path: Path) -> list[Path]:
    """Alle CSV-Dateien eines Ordners (sortiert) oder die Datei selbst."""
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
        if not files:
            raise CsvMappingError(f"Keine CSV-Dateien in {path}.")
        return files
    if not path.is_file():
        raise CsvMappingError(f"Datei nicht gefunden: {path}.")
    return [path]


def read_csv_bars(
    path: str | Path,
    mapping: Mapping[str, str] | None = None,
    *,
    date_format: str | None = None,
) -> tuple[pd.DataFrame, CsvImportReport]:
    """Liest und prüft eine CSV-Datei oder alle CSV-Dateien eines Ordners.

    Args:
        path: CSV-Datei oder Ordner mit CSV-Dateien.
        mapping: Optionales Mapping Quellspalte → Zielspalte.
        date_format: Optionales ``strftime``-Format der Datumsspalte; ohne Angabe
            wird das Format erkannt.

    Returns:
        Gültige Kurse mit Spalten ``symbol, date, open, high, low, close, volume``,
        ``adj_close`` (falls vorhanden) und ``source``, sowie ein Bericht mit den
        abgelehnten Zeilen (``rows_imported`` bleibt 0).

    Raises:
        CsvMappingError: Pfad fehlt, Datei unlesbar oder Mapping passt nicht.
    """
    report = CsvImportReport()
    frames: list[pd.DataFrame] = []
    for file in _csv_files(Path(path)):
        parsed = _parse_file(file, mapping, date_format, file.stem.strip().upper())
        parsed.bars["source"] = _SOURCE_PREFIX + parsed.name
        frames.append(parsed.bars)
        report.files.append(parsed.name)
        report.rows_read += parsed.rows_read
        report.errors.extend(parsed.errors)
    if not frames:  # pragma: no cover - _csv_files liefert immer mindestens eine Datei
        return pd.DataFrame(), report
    bars = pd.concat(frames, ignore_index=True)
    # Kommt ein Tag in mehreren Dateien vor, gilt wie beim Import die spätere Datei.
    bars = bars.drop_duplicates(subset=["symbol", "date"], keep="last").reset_index(drop=True)
    return bars, report


# ---------------------------------------------------------------------- Import

_INSERT_COLUMNS: Final = (
    "symbol, date, open, high, low, close, volume, adj_close, adj_factor, source"
)


def _write_bars(store: MarketStore, bars: pd.DataFrame, source: str) -> None:
    """Schreibt geprüfte Kurse vieler Symbole in einem Schritt (vorhandene Tage ersetzen).

    Entspricht :meth:`MarketStore.upsert_bars` je Symbol, ist bei Millionen Zeilen
    aber deutlich schneller. Die Zeilen sind bereits geprüft (Schlusskurs > 0, ein
    Eintrag je Symbol und Tag).
    """
    adj_close = bars["adj_close"] if "adj_close" in bars.columns else bars["close"]
    frame = pd.DataFrame(
        {
            "symbol": bars["symbol"].astype(object),
            "date": bars["date"],
            "open": bars["open"],
            "high": bars["high"],
            "low": bars["low"],
            "close": bars["close"],
            "volume": bars["volume"],
            "adj_close": adj_close,
            "adj_factor": adj_close / bars["close"],
            "source": source,
        }
    )
    conn = store.connection
    view = "_csv_import_source"
    conn.register(view, frame)
    try:
        with store.transaction():
            conn.execute(
                f"INSERT INTO bars_daily ({_INSERT_COLUMNS}) "
                "SELECT symbol, CAST(date AS DATE), open, high, low, close, volume, adj_close, "
                f"adj_factor, source FROM {view} "
                "ON CONFLICT (symbol, date) DO UPDATE SET open = excluded.open, "
                "high = excluded.high, low = excluded.low, close = excluded.close, "
                "volume = excluded.volume, adj_close = excluded.adj_close, "
                "adj_factor = excluded.adj_factor, source = excluded.source"
            )
    finally:
        conn.unregister(view)


def import_csv(
    store: MarketStore,
    path: str | Path,
    mapping: Mapping[str, str] | None = None,
    *,
    date_format: str | None = None,
) -> CsvImportReport:
    """Importiert Tageskurse aus CSV in die Datenbank.

    Jede Datei wird für sich gelesen und in einer Transaktion geschrieben. Zeilen,
    die die Prüfung nicht bestehen (fehlende oder ungültige Werte, Schlusskurs ≤ 0,
    Volumen mit Nachkommastellen, doppelte Tage mit abweichenden Werten), landen im
    Fehlerbericht. Vorhandene Tage eines Symbols werden überschrieben.

    Args:
        store: Geöffneter Speicher.
        path: CSV-Datei oder Ordner mit CSV-Dateien.
        mapping: Optionales Mapping Quellspalte → Zielspalte (siehe
            :func:`resolve_mapping`).
        date_format: Optionales ``strftime``-Format der Datumsspalte.

    Returns:
        Bericht mit gelesenen, importierten und abgelehnten Zeilen.

    Raises:
        CsvMappingError: Pfad fehlt, eine Datei ist unlesbar oder das Mapping passt
            nicht; in diesem Fall wird die betroffene Datei nicht importiert.
    """
    report = CsvImportReport()
    for file in _csv_files(Path(path)):
        parsed = _parse_file(file, mapping, date_format, file.stem.strip().upper())
        report.files.append(parsed.name)
        report.rows_read += parsed.rows_read
        report.errors.extend(parsed.errors)
        source = _SOURCE_PREFIX + parsed.name
        bars = parsed.bars
        if bars.empty:
            continue
        _write_bars(store, bars, source)
        report.rows_imported += len(bars)
        report.symbols.update(bars["symbol"].unique().tolist())
    return report


# ---------------------------------------------------------------------- Provider


class CsvImportProvider:
    """Preis-Provider, der Tageskurse aus CSV-Dateien liefert.

    Die Dateien werden beim ersten Zugriff einmal gelesen und geprüft; ungültige
    Zeilen stehen danach in :attr:`report`.

    Args:
        path: CSV-Datei oder Ordner mit CSV-Dateien.
        mapping: Optionales Mapping Quellspalte → Zielspalte.
        date_format: Optionales ``strftime``-Format der Datumsspalte.
    """

    name: str = _PROVIDER_NAME

    def __init__(
        self,
        path: str | Path,
        mapping: Mapping[str, str] | None = None,
        *,
        date_format: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._mapping = dict(mapping) if mapping else None
        self._date_format = date_format
        self._bars: pd.DataFrame | None = None
        self._report = CsvImportReport()

    @property
    def report(self) -> CsvImportReport:
        """Bericht des Lesevorgangs (leer vor dem ersten Zugriff)."""
        return self._report

    def _load(self) -> pd.DataFrame:
        if self._bars is None:
            try:
                self._bars, self._report = read_csv_bars(
                    self._path, self._mapping, date_format=self._date_format
                )
            except CsvMappingError as exc:
                raise ProviderError(self.name, None, str(exc)) from exc
        return self._bars

    def symbols(self) -> list[str]:
        """Alle Symbole mit gültigen Zeilen, alphabetisch sortiert."""
        bars = self._load()
        return sorted(bars["symbol"].unique().tolist()) if not bars.empty else []

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Tageskurse des Symbols von ``start`` bis einschließlich ``end``.

        Returns:
            Spalten ``date, open, high, low, close, volume, adj_close``; fehlt
            ``adj_close`` in der Datei, gilt ``adj_close = close``. Leer, wenn das
            Symbol nicht vorkommt.

        Raises:
            ProviderError: Datei fehlt, ist unlesbar oder das Mapping passt nicht.
        """
        bars = self._load()
        if bars.empty:
            return empty_bars()
        sym = symbol.strip().upper()
        mask = (
            (bars["symbol"] == sym)
            & (bars["date"] >= pd.Timestamp(start))
            & (bars["date"] <= pd.Timestamp(end))
        )
        selected = bars.loc[mask]
        if selected.empty:
            return empty_bars()
        adj_close = selected["adj_close"] if "adj_close" in selected.columns else selected["close"]
        result = pd.DataFrame(
            {
                "date": selected["date"],
                "open": selected["open"],
                "high": selected["high"],
                "low": selected["low"],
                "close": selected["close"],
                "volume": selected["volume"],
                "adj_close": adj_close,
            }
        )
        return result.sort_values("date", kind="mergesort").reset_index(drop=True)

    def fetch_actions(self, symbol: str) -> pd.DataFrame:
        """CSV-Dateien enthalten keine Corporate Actions; das Ergebnis ist immer leer."""
        return empty_actions()
