"""Einstiegspunkt der App (Konsolenskript ``trading-helfer``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from trading_helfer import __version__


def build_parser() -> argparse.ArgumentParser:
    """Erzeugt den Argument-Parser mit allen Unterbefehlen."""
    parser = argparse.ArgumentParser(
        prog="trading-helfer",
        description="Trading Helfer: technische Analyse, Kennzahlen und Backtesting.",
    )
    parser.add_argument("--version", action="version", version=f"trading-helfer {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<befehl>")
    quality = commands.add_parser(
        "quality",
        help="Datenqualität der gespeicherten Kurse prüfen",
        description="Prüft gespeicherte Kurse auf Lücken, High < Low, Kurse ≤ 0, Sprünge "
        "und fehlende Daten von Indexmitgliedern.",
    )
    quality.add_argument(
        "symbols", nargs="*", metavar="SYMBOL", help="nur diese Symbole prüfen (Standard: alle)"
    )
    quality.add_argument(
        "--db", type=Path, default=None, help="DuckDB-Datei (Standard: aus den Einstellungen)"
    )
    quality.add_argument(
        "--csv", type=Path, default=None, help="Bericht zusätzlich als CSV-Datei speichern"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Startet ohne Unterbefehl die GUI und gibt den Exit-Code zurück."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        # Erst hier importieren, damit Unterbefehle ohne Qt auskommen.
        from trading_helfer.gui.application import run_gui

        return run_gui()
    if args.command == "quality":
        return run_quality(args.symbols, db_path=args.db, csv_path=args.csv)
    return 0  # pragma: no cover - weitere Unterbefehle folgen in späteren Issues


def run_quality(
    symbols: Sequence[str] = (), *, db_path: Path | None = None, csv_path: Path | None = None
) -> int:
    """Führt die Datenqualitätsprüfung aus und gibt den Bericht auf der Konsole aus.

    Args:
        symbols: Nur diese Symbole prüfen; leer = alle.
        db_path: DuckDB-Datei; ``None`` = Pfad aus den Einstellungen.
        csv_path: Wenn gesetzt, wird der Bericht zusätzlich als CSV gespeichert.

    Returns:
        Exit-Code: 0 ohne Fehler (Warnungen erlaubt), 1 bei mindestens einem Fehler.
    """
    from trading_helfer.core.config import load_settings
    from trading_helfer.data.quality import Severity, check_all
    from trading_helfer.data.storage import MarketStore

    settings = load_settings()
    path = db_path if db_path is not None else settings.database_path
    if not path.exists():
        print(f"Datenbank {path} nicht gefunden.", file=sys.stderr)
        return 2
    with MarketStore(path, read_only=True) as store:
        report = check_all(store, symbols=symbols or None, history_start=settings.start_date)
    if csv_path is not None:
        report.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    if report.empty:
        print("Keine Auffälligkeiten gefunden.")
        return 0
    errors = int((report["severity"] == Severity.ERROR.value).sum())
    warnings = len(report) - errors
    print(report.to_string(index=False, max_colwidth=120))
    print(f"\n{len(report)} Befunde: {errors} Fehler, {warnings} Warnungen.")
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
