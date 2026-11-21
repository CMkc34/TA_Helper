"""Aktualisiert die Universum-CSV (Indexmitglieder) aus den englischen Wikipedia-Artikeln.

Das Skript läuft nur manuell; die App liest ausschließlich die CSV-Dateien in
``src/trading_helfer/data/universe/``.

Ablauf je Index:

1. Artikel laden (oder ``--html`` für eine gespeicherte Seite) und alle Tabellen lesen.
2. Aktuelle Mitgliederliste aus der Bestandstabelle (Spalte „Symbol“ bzw. „Ticker“).
3. Änderungen aus der Tabelle mit den Spalten „Added“ und „Removed“
   („Selected changes“). Übernommen werden nur Zeilen nach dem letzten Datum der CSV,
   die gepflegte Historie bleibt unverändert.
4. Abgleich: Weicht die neue Bestandsliste von „alte Liste + neue Änderungen“ ab,
   weil Wikipedia nicht jede Änderung aufführt, wird die Differenz als Zeile mit der
   Quelle „Abgleich Wikipedia-Bestandsliste“ zum Stichtag ``--as-of`` ergänzt.

Aufruf::

    uv run python scripts/build_universe.py              # beide Indizes
    uv run python scripts/build_universe.py --index SP500 --dry-run
    uv run python scripts/build_universe.py --index NDX100 --html seite.html
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from trading_helfer.data.universe import (
    CHANGE_COLUMNS,
    INDEX_CODES,
    UNIVERSE_DIR,
    load_history,
)

WIKIPEDIA_URLS: dict[str, str] = {
    "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "NDX100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}
SOURCE_WIKIPEDIA = "Wikipedia"
SOURCE_RECONCILE = "Abgleich Wikipedia-Bestandsliste"
USER_AGENT = "trading-helfer-universe-builder/1.0 (private use)"
_DATE_FORMATS = ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%B %d %Y")
_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{0,5}(\.[A-Z])?$")

Table = list[list[str]]
Fetcher = Callable[[str], str]


@dataclass
class ChangeRow:
    """Eine Zeile der Änderungs-CSV."""

    date: dt.date
    added: str
    removed: str
    source: str


@dataclass
class UpdateResult:
    """Ergebnis der Aktualisierung eines Index."""

    index_code: str
    current: set[str]
    new_rows: list[ChangeRow] = field(default_factory=list)
    reconciled: list[ChangeRow] = field(default_factory=list)


# --- HTML-Tabellen -------------------------------------------------------------------


class _TableParser(HTMLParser):
    """Liest alle ``<table>``-Elemente als Textraster (rowspan/colspan aufgelöst)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[Table] = []
        self._stack: list[list[tuple[list[str], list[int], list[int]]]] = []
        self._row: tuple[list[str], list[int], list[int]] | None = None
        self._cell: list[str] | None = None
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "table":
            self._stack.append([])
        elif not self._stack:
            return
        elif tag == "tr":
            self._row = ([], [], [])
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
            self._row[1].append(_span(values.get("rowspan")))
            self._row[2].append(_span(values.get("colspan")))
        elif tag in ("sup", "style", "script"):
            self._skip += 1
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        if tag in ("sup", "style", "script") and self._skip:
            self._skip -= 1
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row[0].append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self._stack[-1].append(self._row)
            self._row = None
        elif tag == "table":
            self.tables.append(_expand(self._stack.pop()))

    def handle_data(self, data: str) -> None:
        if self._cell is not None and not self._skip:
            self._cell.append(data)


def _span(value: str | None) -> int:
    try:
        return max(1, int(value or "1"))
    except ValueError:
        return 1


def _expand(rows: list[tuple[list[str], list[int], list[int]]]) -> Table:
    """Löst rowspan und colspan auf, sodass jede Zeile alle Spalten enthält."""
    grid: Table = []
    pending: dict[int, tuple[str, int]] = {}
    for texts, rowspans, colspans in rows:
        line: list[str] = []
        cells = list(zip(texts, rowspans, colspans, strict=True))
        col = 0
        while cells or col in pending:
            if col in pending:
                text, left = pending.pop(col)
                line.append(text)
                if left > 1:
                    pending[col] = (text, left - 1)
                col += 1
                continue
            text, rowspan, colspan = cells.pop(0)
            for _ in range(colspan):
                line.append(text)
                if rowspan > 1:
                    pending[col] = (text, rowspan - 1)
                col += 1
        grid.append(line)
    return grid


def parse_tables(html: str) -> list[Table]:
    """Liest alle Tabellen einer HTML-Seite als Liste von Textzeilen."""
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    return parser.tables


# --- Auswertung ----------------------------------------------------------------------


def normalize_ticker(text: str) -> str | None:
    """Wandelt Zelltext in ein Symbol in App-Schreibweise (``BRK.B``) oder ``None``."""
    cleaned = text.strip().upper().replace("-", ".")
    return cleaned if _TICKER_PATTERN.match(cleaned) else None


def parse_date(text: str) -> dt.date | None:
    """Liest Datumsangaben wie ``2025-09-22`` oder ``September 22, 2025``."""
    cleaned = " ".join(text.replace("\xa0", " ").split())
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _header_rows(table: Table) -> int:
    """Anzahl der Kopfzeilen: Zeilen bis zur ersten Zeile mit gültigem Datum."""
    for index, row in enumerate(table):
        if any(parse_date(cell) for cell in row):
            return index
    return len(table)


def extract_current(tables: Sequence[Table], minimum: int = 90) -> set[str]:
    """Sucht die Bestandstabelle und liefert ihre Symbole.

    Gewählt wird die erste Tabelle mit einer Kopfspalte „Symbol“ oder „Ticker“ und
    mindestens ``minimum`` gültigen Symbolen.
    """
    for table in tables:
        if not table:
            continue
        header = [cell.lower() for cell in table[0]]
        for col, name in enumerate(header):
            if name not in ("symbol", "ticker", "ticker symbol"):
                continue
            symbols = {
                ticker
                for row in table[1:]
                if col < len(row) and (ticker := normalize_ticker(row[col]))
            }
            if len(symbols) >= minimum:
                return symbols
    raise ValueError("Keine Bestandstabelle mit Spalte „Symbol“ oder „Ticker“ gefunden.")


def extract_changes(tables: Sequence[Table]) -> list[tuple[dt.date, set[str], set[str]]]:
    """Liest die Änderungstabelle (Spalten „Date“, „Added“, „Removed“).

    Returns:
        Liste von (Datum, aufgenommen, entfernt), nach Datum sortiert; leer, wenn die
        Seite keine solche Tabelle enthält.
    """
    for table in tables:
        heads = _header_rows(table)
        if heads == 0 or heads >= len(table):
            continue
        labels = [
            " ".join(row[col] for row in table[:heads] if col < len(row)).lower()
            for col in range(max(len(row) for row in table[:heads]))
        ]
        date_col = _find_column(labels, "date")
        added_col = _find_column(labels, "added", ("ticker", "symbol"))
        removed_col = _find_column(labels, "removed", ("ticker", "symbol"))
        if date_col is None or added_col is None or removed_col is None:
            continue
        by_date: dict[dt.date, tuple[set[str], set[str]]] = {}
        for row in table[heads:]:
            if max(date_col, added_col, removed_col) >= len(row):
                continue
            day = parse_date(row[date_col])
            if day is None:
                continue
            added, removed = by_date.setdefault(day, (set(), set()))
            if ticker := normalize_ticker(row[added_col]):
                added.add(ticker)
            if ticker := normalize_ticker(row[removed_col]):
                removed.add(ticker)
        return [(day, a, r) for day, (a, r) in sorted(by_date.items()) if a or r]
    return []


def _find_column(labels: list[str], word: str, prefer: tuple[str, ...] = ()) -> int | None:
    matches = [col for col, label in enumerate(labels) if word in label]
    for col in matches:
        if any(p in labels[col] for p in prefer):
            return col
    return matches[0] if matches else None


# --- CSV aktualisieren ----------------------------------------------------------------


def update_index(
    index_code: str,
    html: str,
    directory: Path = UNIVERSE_DIR,
    as_of: dt.date | None = None,
) -> UpdateResult:
    """Berechnet neue Änderungszeilen und die neue Bestandsliste (ohne zu schreiben)."""
    as_of = as_of or dt.date.today()
    history = load_history(index_code, directory)
    last = history.changes[-1].date if history.changes else None
    tables = parse_tables(html)
    current = extract_current(tables)
    result = UpdateResult(history.index_code, current)

    expected = set(history.current)
    for day, added, removed in extract_changes(tables):
        if (last is not None and day <= last) or day > as_of:
            continue
        added_now, removed_now = added - expected, removed & expected
        expected = (expected | added_now) - removed_now
        result.new_rows.extend(_rows(day, added_now, removed_now, SOURCE_WIKIPEDIA))

    missing, extra = current - expected, expected - current
    result.reconciled = _rows(as_of, missing, extra, SOURCE_RECONCILE)
    return result


def _rows(day: dt.date, added: set[str], removed: set[str], source: str) -> list[ChangeRow]:
    adds, rems = sorted(added), sorted(removed)
    return [
        ChangeRow(
            day,
            adds[i] if i < len(adds) else "",
            rems[i] if i < len(rems) else "",
            source,
        )
        for i in range(max(len(adds), len(rems)))
    ]


def write_update(result: UpdateResult, directory: Path = UNIVERSE_DIR) -> None:
    """Hängt neue Zeilen an die Änderungs-CSV an und schreibt die Bestandsliste neu."""
    prefix = INDEX_CODES[result.index_code]
    changes_path = directory / f"{prefix}_changes.csv"
    with changes_path.open(encoding="utf-8", newline="") as handle:
        existing = list(csv.reader(handle))
    rows = existing[1:] + [
        [row.date.isoformat(), row.added, row.removed, row.source]
        for row in [*result.new_rows, *result.reconciled]
    ]
    rows.sort(key=lambda row: row[0])
    with changes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CHANGE_COLUMNS)
        writer.writerows(rows)
    with (directory / f"{prefix}_current.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("symbol\n")
        handle.writelines(f"{symbol}\n" for symbol in sorted(result.current))
    load_history(result.index_code, directory)  # prüft die neue Fassung


def fetch_html(url: str) -> str:
    """Lädt eine Wikipedia-Seite (nur im manuellen Skriptlauf)."""
    import requests

    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.text


def main(argv: Sequence[str] | None = None, fetcher: Fetcher = fetch_html) -> int:
    """Kommandozeile: aktualisiert die CSV eines oder beider Indizes."""
    parser = argparse.ArgumentParser(description="Indexmitglieder aus Wikipedia aktualisieren.")
    parser.add_argument("--index", choices=[*INDEX_CODES, "all"], default="all")
    parser.add_argument("--html", type=Path, help="gespeicherte Seite statt Download")
    parser.add_argument("--dir", type=Path, default=UNIVERSE_DIR, help="Ordner der CSV")
    parser.add_argument("--as-of", type=dt.date.fromisoformat, help="Stichtag (Standard: heute)")
    parser.add_argument("--dry-run", action="store_true", help="nur anzeigen, nichts schreiben")
    args = parser.parse_args(argv)

    codes = list(INDEX_CODES) if args.index == "all" else [args.index]
    if args.html and len(codes) != 1:
        parser.error("--html braucht genau einen --index.")
    for code in codes:
        html = args.html.read_text(encoding="utf-8") if args.html else fetcher(WIKIPEDIA_URLS[code])
        result = update_index(code, html, args.dir, args.as_of)
        print(
            f"{code}: {len(result.current)} aktuelle Mitglieder, "
            f"{len(result.new_rows)} neue Änderungen, {len(result.reconciled)} Abgleichzeilen."
        )
        for row in [*result.new_rows, *result.reconciled]:
            print(f"  {row.date} +{row.added or '-'} -{row.removed or '-'} ({row.source})")
        if not args.dry_run:
            write_update(result, args.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
