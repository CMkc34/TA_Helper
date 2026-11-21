"""Einstiegspunkt der App (Konsolenskript ``trading-helfer``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from trading_helfer import __version__

INDEX_CHOICES = ("SP500", "NDX100")
"""Indizes, die ``trading-helfer update --index`` kennt."""


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
    update = commands.add_parser(
        "update",
        help="Kurse inkrementell aktualisieren",
        description="Lädt neue Tageskurse für Indexmitglieder, eigene Symbole und die "
        "Benchmarks SPY und QQQ. Ohne --index und --symbols werden alle Indizes geladen.",
    )
    update.add_argument(
        "--index",
        nargs="+",
        metavar="INDEX",
        type=str.upper,
        choices=sorted(INDEX_CHOICES),
        default=None,
        help="Indizes (SP500, NDX100)",
    )
    update.add_argument(
        "--symbols", nargs="+", metavar="SYMBOL", default=None, help="zusätzliche Symbole"
    )
    update.add_argument(
        "--db", type=Path, default=None, help="DuckDB-Datei (Standard: aus den Einstellungen)"

    import_csv = commands.add_parser(
        "import-csv",
        help="Tageskurse aus CSV-Dateien (z. B. Kaggle) importieren",
        description=(
            "Importiert Tageskurse aus einer CSV-Datei oder allen CSV-Dateien eines Ordners. "
            "Abgelehnte Zeilen stehen im Fehlerbericht <pfad>.fehler.csv."
        ),
    )
    import_csv.add_argument("path", type=Path, metavar="<pfad>", help="CSV-Datei oder Ordner")
    import_csv.add_argument(
        "--mapping",
        type=Path,
        metavar="datei.toml",
        help='TOML-Datei mit [columns] Quellspalte = "zielspalte"',
    )
    import_csv.add_argument(
        "--date-format", metavar="FORMAT", help="Datumsformat, z. B. %%d.%%m.%%Y"
    )
    import_csv.add_argument(
        "--db", type=Path, metavar="datei.duckdb", help="Datenbank (Standard: aus Einstellungen)"
    )
    return parser


def _run_import_csv(args: argparse.Namespace) -> int:
    """Führt ``import-csv`` aus und gibt den Exit-Code zurück."""
    from trading_helfer.data.providers.csv_import import (
        CsvMappingError,
        import_csv,
        load_mapping,
    )
    from trading_helfer.data.storage import MarketStore

    path: Path = args.path
    try:
        mapping = load_mapping(args.mapping) if args.mapping else None
        db_path = args.db
        if db_path is None:
            from trading_helfer.core.config import load_settings

            db_path = load_settings().database_path
        with MarketStore(db_path) as store:
            report = import_csv(store, path, mapping, date_format=args.date_format)
    except CsvMappingError as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 2
    print(report.summary())
    if report.errors:
        error_file = report.write_errors(path.with_name(path.name + ".fehler.csv"))
        print(f"Fehlerbericht: {error_file}")
    return 0


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
    if args.command == "update":
        return run_update(args.index, args.symbols, db_path=args.db)
    if args.command == "import-csv":
        return _run_import_csv(args)
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


def run_update(
    index_codes: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    *,
    db_path: Path | None = None,
) -> int:
    """Führt das Datenupdate aus und meldet den Fortschritt auf der Konsole.

    Args:
        index_codes: Indizes; ``None`` = alle, wenn auch ``symbols`` fehlt, sonst keine.
        symbols: Zusätzliche Symbole.
        db_path: DuckDB-Datei; ``None`` = Pfad aus den Einstellungen.

    Returns:
        Exit-Code: 0 ohne Fehler, 1 wenn Symbole fehlschlugen, 2 bei falschen
        Einstellungen, 130 nach Abbruch mit Strg+C.
    """
    from trading_helfer.core.config import ConfigError, load_settings
    from trading_helfer.data.storage import MarketStore
    from trading_helfer.data.update import UpdateError, build_price_providers, update_universe

    if index_codes is None:
        index_codes = [] if symbols else list(INDEX_CHOICES)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    path = db_path if db_path is not None else settings.database_path

    def progress(done: int, total: int, symbol: str) -> None:
        print(f"[{done}/{total}] {symbol}", flush=True)

    with MarketStore(path) as store:
        try:
            result = update_universe(
                index_codes,
                progress,
                store=store,
                providers=build_price_providers(settings, store),
                symbols=symbols or (),
                start=settings.start_date,
            )
        except UpdateError as exc:
            print(exc, file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("\nAbgebrochen. Bereits geladene Symbole sind gespeichert.", file=sys.stderr)
            return 130
    for symbol, reason in sorted(result.failed.items()):
        print(f"Fehler {symbol}: {reason}", file=sys.stderr)
    print(result.summary())
    return 1 if result.failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
