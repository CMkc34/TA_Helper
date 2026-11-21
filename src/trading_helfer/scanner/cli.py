"""Kommandozeile ``trading-helfer scan`` (Argumente und Ausführung)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from trading_helfer.core.types import SignalType

__all__ = ["add_scan_parser", "parse_filters", "parse_params", "run_scan"]


def add_scan_parser(commands: Any) -> argparse.ArgumentParser:
    """Registriert den Unterbefehl ``scan`` bei ``commands`` (``add_subparsers``-Objekt)."""
    parser: argparse.ArgumentParser = commands.add_parser(
        "scan",
        help="Universum mit einer Strategie auf aktuelle Signale prüfen",
        description=(
            "Wendet eine Strategie auf alle Werte eines Index oder einer Symbolliste an "
            "und listet die Werte mit Signal auf der letzten abgeschlossenen Kerze."
        ),
    )
    parser.add_argument("--strategy", required=True, metavar="ID", help="ID der Strategie")
    universe = parser.add_mutually_exclusive_group(required=True)
    universe.add_argument("--index", metavar="CODE", help="Index-Code, z. B. SP500 oder NDX100")
    universe.add_argument(
        "--symbols", nargs="+", metavar="SYMBOL", help="eigene Symbolliste statt eines Index"
    )
    parser.add_argument("--tf", default="W1", metavar="ZEITRAHMEN", help="D3, W1 … (Standard W1)")
    parser.add_argument("--as-of", metavar="JJJJ-MM-TT", help="Stichtag (Standard: letzte Daten)")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=WERT",
        help="Parameter der Strategie (mehrfach möglich)",
    )
    parser.add_argument("--sector", action="append", default=[], help="nur dieser Sektor")
    parser.add_argument("--min-cap", type=float, metavar="USD", help="Mindest-Marktkapitalisierung")
    parser.add_argument("--max-cap", type=float, metavar="USD", help="Höchste Marktkapitalisierung")
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        metavar="NAME:MIN:MAX",
        help="Kennzahlfilter, z. B. pe_ratio::25 (leere Grenze = offen)",
    )
    parser.add_argument(
        "--signal", action="append", default=[], choices=[s.value for s in SignalType]
    )
    parser.add_argument("--csv", type=Path, metavar="DATEI", help="Ergebnis als CSV speichern")
    parser.add_argument("--db", type=Path, metavar="DATEI", help="DuckDB-Datei")
    parser.add_argument("--no-save", action="store_true", help="nicht in der Datenbank speichern")
    return parser


def parse_params(items: list[str]) -> dict[str, Any]:
    """Wandelt ``NAME=WERT`` in ein Dictionary; Zahlen und ``true``/``false`` werden erkannt.

    Raises:
        ValueError: Eintrag ohne ``=``.
    """
    result: dict[str, Any] = {}
    for item in items:
        name, sep, raw = item.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"Parameter {item!r} hat nicht die Form NAME=WERT.")
        result[name.strip()] = _value(raw.strip())
    return result


def parse_filters(args: argparse.Namespace) -> Any:
    """Erzeugt :class:`~trading_helfer.scanner.scan.ScanFilters` aus den Argumenten.

    Raises:
        ValueError: Kennzahlfilter nicht in der Form ``NAME:MIN:MAX``.
    """
    from trading_helfer.scanner.scan import ScanFilters

    metrics: dict[str, tuple[float | None, float | None]] = {}
    for item in args.metric:
        parts = item.split(":")
        if len(parts) != 3 or not parts[0].strip():
            raise ValueError(f"Kennzahlfilter {item!r} hat nicht die Form NAME:MIN:MAX.")
        metrics[parts[0].strip()] = (_bound(parts[1]), _bound(parts[2]))
    return ScanFilters(
        sectors=frozenset(args.sector) if args.sector else None,
        min_market_cap=args.min_cap,
        max_market_cap=args.max_cap,
        metrics=metrics,
        signals=frozenset(SignalType(s) for s in args.signal) if args.signal else None,
    )


def run_scan(args: argparse.Namespace) -> int:
    """Führt ``scan`` aus, gibt das Ergebnis aus und liefert den Exit-Code.

    Returns:
        0 bei Erfolg (auch ohne Signale), 2 bei ungültiger Eingabe oder fehlender Datenbank.
    """
    from trading_helfer.core.config import load_settings
    from trading_helfer.data.storage import MarketStore
    from trading_helfer.scanner.scan import ScanError, export_csv, scan
    from trading_helfer.strategies.registry import StrategyNotFoundError, StrategyRegistry

    settings = load_settings()
    db_path: Path = args.db if args.db is not None else settings.database_path
    if not db_path.exists():
        print(f"Datenbank {db_path} nicht gefunden.", file=sys.stderr)
        return 2
    registry = StrategyRegistry(settings.plugins_path)
    registry.load()
    try:
        strategy = registry.create(args.strategy)
        params = parse_params(args.param)
        filters = parse_filters(args)
        universe = args.index if args.index else args.symbols
        with MarketStore(db_path, read_only=args.no_save) as store:
            result = scan(
                strategy,
                params,
                universe,
                args.tf,
                args.as_of,
                filters,
                store=store,
                settings=settings,
                save=not args.no_save,
            )
    except (StrategyNotFoundError, ScanError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"Fehler: {message}", file=sys.stderr)
        return 2
    _print(result, saved=not args.no_save)
    if args.csv is not None:
        print(f"CSV: {export_csv(result, args.csv)}")
    return 0


def _print(result: pd.DataFrame, *, saved: bool) -> None:
    if result.empty:
        print("Keine Signale.")
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(result.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(f"\n{len(result)} Signal(e).")
    skipped = result.attrs.get("skipped", {})
    if skipped:
        print(f"{len(skipped)} Wert(e) übersprungen (z. B. ohne Kurse).")
    if saved and not result.empty:
        print(f"Scan-ID: {result.attrs['scan_id']}")


def _bound(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Grenze {text!r} ist keine Zahl.") from exc


def _value(text: str) -> Any:
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text
