"""Einstiegspunkt der App (Konsolenskript ``trading-helfer``)."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from trading_helfer import __version__

INDEX_CHOICES = ("SP500", "NDX100")
"""Indizes, die ``trading-helfer update --index`` kennt."""

DEFAULT_OPTIONS_WATCHLIST = "Eigene"
"""Watchlist, deren Optionsketten ``options-snapshot`` ohne weitere Angabe speichert."""


SELFTEST_STDOUT = Path("-")
"""Wert von ``--selftest`` ohne Datei: Bericht nur auf der Konsole."""


def ensure_std_streams() -> None:
    """Ersetzt fehlende ``stdout``/``stderr`` durch ``os.devnull``.

    Das Fenster-Paket (PyInstaller ohne Konsole) startet mit ``sys.stdout = None``;
    ``print`` und Logging würden dann abstürzen.
    """
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115


def build_parser() -> argparse.ArgumentParser:
    """Erzeugt den Argument-Parser mit allen Unterbefehlen."""
    parser = argparse.ArgumentParser(
        prog="trading-helfer",
        description="Trading Helfer: technische Analyse, Kennzahlen und Backtesting.",
    )
    parser.add_argument("--version", action="version", version=f"trading-helfer {__version__}")
    parser.add_argument(
        "--selftest",
        nargs="?",
        const=SELFTEST_STDOUT,
        default=None,
        type=Path,
        metavar="BERICHT",
        help="Selbsttest: Paket prüfen, Hauptfenster kurz zeigen und beenden "
        "(Exit-Code 0 = bestanden); optional Bericht in die Datei BERICHT schreiben",
    )
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
    quality.add_argument(
        "--exclude",
        action="store_true",
        help="Werte mit Fremdkursen automatisch aus Backtests ausschließen "
        "(und unauffällig gewordene wieder zulassen)",
    )
    exclude = commands.add_parser(
        "exclude",
        help="Werte aus Backtests ausschließen oder wieder zulassen",
        description="Ausgeschlossene Werte (z. B. Fremdkurse unter neu vergebenem Ticker) "
        "verwenden Portfolio-Backtests und Studien nicht. Ohne Symbole wird die Liste "
        "ausgegeben.",
    )
    exclude.add_argument("symbols", nargs="*", metavar="SYMBOL", help="Symbole")
    exclude_mode = exclude.add_mutually_exclusive_group()
    exclude_mode.add_argument(
        "--allow",
        action="store_true",
        help="trotz Befund verwenden (die automatische Prüfung schließt sie nicht mehr aus)",
    )
    exclude_mode.add_argument(
        "--remove", action="store_true", help="Eintrag löschen (automatische Prüfung gilt wieder)"
    )
    exclude.add_argument(
        "--reason", default="von Hand ausgeschlossen", help="Grund für den Eintrag"
    )
    exclude.add_argument(
        "--db", type=Path, default=None, help="DuckDB-Datei (Standard: aus den Einstellungen)"
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
        "--wait",
        action="store_true",
        help="bei erreichtem Tiingo-Limit warten, bis wieder Anfragen frei sind "
        "(statt den Wert ohne Tiingo abzuschließen); für langes Nachladen über Nacht",
    )
    update.add_argument(
        "--recheck",
        action="store_true",
        help="Werte mit Fremdkursen komplett neu laden und dabei alle Kursquellen fragen",
    )
    update.add_argument(
        "--db", type=Path, default=None, help="DuckDB-Datei (Standard: aus den Einstellungen)"
    )
    options = commands.add_parser(
        "options-snapshot",
        help="Optionsketten der Watchlist als Tages-Schnappschuss speichern",
        description="Speichert die aktuellen Optionsketten (yfinance) als Schnappschuss des "
        "Tages in der Tabelle option_snapshots. Ohne --symbols und --watchlist gilt die "
        f"Watchlist „{DEFAULT_OPTIONS_WATCHLIST}“. Für eine eigene Historie täglich nach "
        "Börsenschluss ausführen (z. B. über die Windows-Aufgabenplanung).",
    )
    options.add_argument("--symbols", nargs="+", metavar="SYMBOL", default=None, help="Symbole")
    options.add_argument(
        "--watchlist", metavar="NAME", default=None, help="Symbole dieser Watchlist nehmen"
    )
    options.add_argument(
        "--max-expiries",
        type=int,
        metavar="N",
        default=None,
        help="nur die nächsten N Verfallstage je Symbol (Standard: alle)",
    )
    options.add_argument(
        "--db", type=Path, default=None, help="DuckDB-Datei (Standard: aus den Einstellungen)"
    )
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
    from trading_helfer.scanner.cli import add_scan_parser

    add_scan_parser(commands)
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
    # Im gebauten Paket starten Kindprozesse (Strategie-Prüfung, Portfolio-Vorberechnung)
    # dieselbe .exe; freeze_support() übernimmt sie, bevor die GUI startet.
    multiprocessing.freeze_support()
    ensure_std_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selftest is not None:
        from trading_helfer.gui.application import run_selftest

        report = None if args.selftest == SELFTEST_STDOUT else args.selftest
        return run_selftest(report_path=report)
    if args.command is None:
        # Erst hier importieren, damit Unterbefehle ohne Qt auskommen.
        from trading_helfer.gui.application import run_gui

        return run_gui()
    if args.command == "quality":
        return run_quality(args.symbols, db_path=args.db, csv_path=args.csv, exclude=args.exclude)
    if args.command == "exclude":
        return run_exclude(
            args.symbols,
            allow=args.allow,
            remove=args.remove,
            reason=args.reason,
            db_path=args.db,
        )
    if args.command == "update":
        return run_update(
            args.index, args.symbols, db_path=args.db, recheck=args.recheck, wait=args.wait
        )
    if args.command == "import-csv":
        return _run_import_csv(args)
    if args.command == "scan":
        from trading_helfer.scanner.cli import run_scan

        return run_scan(args)
    if args.command == "options-snapshot":
        return run_options_snapshot(
            args.symbols,
            watchlist=args.watchlist,
            max_expiries=args.max_expiries,
            db_path=args.db,
        )
    return 0  # pragma: no cover - weitere Unterbefehle folgen in späteren Issues


def run_quality(
    symbols: Sequence[str] = (),
    *,
    db_path: Path | None = None,
    csv_path: Path | None = None,
    exclude: bool = False,
) -> int:
    """Führt die Datenqualitätsprüfung aus und gibt den Bericht auf der Konsole aus.

    Args:
        symbols: Nur diese Symbole prüfen; leer = alle.
        db_path: DuckDB-Datei; ``None`` = Pfad aus den Einstellungen.
        csv_path: Wenn gesetzt, wird der Bericht zusätzlich als CSV gespeichert.
        exclude: Ausschlüsse wegen Fremdkursen neu bewerten und speichern.

    Returns:
        Exit-Code: 0 ohne Fehler (Warnungen erlaubt), 1 bei mindestens einem Fehler.
    """
    from trading_helfer.core.config import load_settings
    from trading_helfer.data.quality import Severity, check_all, refresh_exclusions
    from trading_helfer.data.storage import MarketStore

    settings = load_settings()
    path = db_path if db_path is not None else settings.database_path
    if not path.exists():
        print(f"Datenbank {path} nicht gefunden.", file=sys.stderr)
        return 2
    with MarketStore(path, read_only=not exclude) as store:
        report = check_all(store, symbols=symbols or None, history_start=settings.start_date)
        if exclude:
            flagged = refresh_exclusions(store, symbols or None, history_start=settings.start_date)
            names = ", ".join(sorted(flagged)) or "keine"
            print(f"Wegen Fremdkursen aus Backtests ausgeschlossen: {names}.")
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


def run_exclude(
    symbols: Sequence[str] = (),
    *,
    allow: bool = False,
    remove: bool = False,
    reason: str = "von Hand ausgeschlossen",
    db_path: Path | None = None,
) -> int:
    """Pflegt die Liste der aus Backtests ausgeschlossenen Werte.

    Args:
        symbols: Symbole; leer = nur die Liste ausgeben.
        allow: Symbole freigeben (Quelle ``freigegeben``).
        remove: Einträge der Symbole löschen.
        reason: Grund für neue Einträge.
        db_path: DuckDB-Datei; ``None`` = Pfad aus den Einstellungen.

    Returns:
        Exit-Code: 0 bei Erfolg, 2 ohne Datenbank.
    """
    from trading_helfer.core.config import load_settings
    from trading_helfer.data.storage import EXCLUSION_ALLOWED, EXCLUSION_MANUAL, MarketStore

    path = db_path if db_path is not None else load_settings().database_path
    if not path.exists():
        print(f"Datenbank {path} nicht gefunden.", file=sys.stderr)
        return 2
    with MarketStore(path) as store:
        for symbol in symbols:
            if remove:
                store.remove_exclusion(symbol)
            elif allow:
                store.set_exclusion(symbol, reason, EXCLUSION_ALLOWED)
            else:
                store.set_exclusion(symbol, reason, EXCLUSION_MANUAL)
        frame = store.load_exclusions()
    if frame.empty:
        print("Keine Einträge.")
        return 0
    for row in frame.itertuples(index=False):
        print(f"{row.symbol:<8} {row.source:<12} {row.reason}")
    return 0


def run_update(
    index_codes: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    *,
    db_path: Path | None = None,
    recheck: bool = False,
    wait: bool = False,
) -> int:
    """Führt das Datenupdate aus und meldet den Fortschritt auf der Konsole.

    Args:
        index_codes: Indizes; ``None`` = alle, wenn auch ``symbols`` fehlt, sonst keine.
        symbols: Zusätzliche Symbole.
        db_path: DuckDB-Datei; ``None`` = Pfad aus den Einstellungen.
        recheck: Werte mit Fremdkursen komplett neu laden.
        wait: Bei erschöpftem Kontingent einer Quelle warten statt überspringen.

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
                recheck=recheck,
                wait_for_quota=wait,
            )
        except UpdateError as exc:
            print(exc, file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("\nAbgebrochen. Bereits geladene Symbole sind gespeichert.", file=sys.stderr)
            return 130
    for symbol, reason in sorted(result.failed.items()):
        print(f"Fehler {symbol}: {reason}", file=sys.stderr)
    if result.fallback_errors:
        sample = next(iter(sorted(result.fallback_errors.items())))
        print(
            f"Hinweis: {len(result.fallback_errors)} Werte ohne Ersatzquelle, "
            f"z. B. {sample[0]}: {sample[1]}",
            file=sys.stderr,
        )
    if result.excluded:
        print("Wegen Fremdkursen ausgeschlossen: " + ", ".join(sorted(result.excluded)))
    print(result.summary())
    return 1 if result.failed else 0


def run_options_snapshot(
    symbols: Sequence[str] | None = None,
    *,
    watchlist: str | None = None,
    max_expiries: int | None = None,
    db_path: Path | None = None,
) -> int:
    """Speichert die Optionsketten als Tages-Schnappschuss und meldet den Fortschritt.

    Args:
        symbols: Symbole; zusammen mit ``watchlist`` werden beide genommen.
        watchlist: Name einer Watchlist; ohne ``symbols`` und ``watchlist`` gilt
            :data:`DEFAULT_OPTIONS_WATCHLIST`.
        max_expiries: Nur die nächsten ``n`` Verfallstage je Symbol.
        db_path: DuckDB-Datei; ``None`` = Pfad aus den Einstellungen.

    Returns:
        Exit-Code: 0 ohne Fehler, 1 wenn Symbole fehlschlugen, 2 bei falschen
        Einstellungen oder ohne Symbole, 130 nach Abbruch mit Strg+C.
    """
    from trading_helfer.core.config import ConfigError, load_settings
    from trading_helfer.data.option_snapshots import (
        OptionSnapshotError,
        build_option_provider,
        snapshot_options,
    )
    from trading_helfer.data.storage import MarketStore
    from trading_helfer.services.app_service import AppService, WatchlistError

    if max_expiries is not None and max_expiries < 1:
        print("--max-expiries muss mindestens 1 sein.", file=sys.stderr)
        return 2
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    selected = list(symbols or [])
    name = watchlist if watchlist is not None else DEFAULT_OPTIONS_WATCHLIST
    if watchlist is not None or not selected:
        try:
            selected += AppService(settings).symbols(name)
        except WatchlistError as exc:
            print(exc, file=sys.stderr)
            return 2
    if not selected:
        print(
            f"Keine Symbole: die Watchlist „{name}“ ist leer. "
            "Symbole mit --symbols angeben oder zur Watchlist hinzufügen.",
            file=sys.stderr,
        )
        return 2
    path = db_path if db_path is not None else settings.database_path

    def progress(done: int, total: int, symbol: str) -> None:
        print(f"[{done}/{total}] {symbol}", flush=True)

    with MarketStore(path) as store:
        try:
            result = snapshot_options(
                selected,
                progress,
                store=store,
                provider=build_option_provider(),
                max_expiries=max_expiries,
            )
        except OptionSnapshotError as exc:
            print(exc, file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("\nAbgebrochen. Bereits gespeicherte Symbole bleiben erhalten.", file=sys.stderr)
            return 130
    for symbol, reason in sorted(result.failed.items()):
        print(f"Fehler {symbol}: {reason}", file=sys.stderr)
    print(result.summary())
    return 1 if result.failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
