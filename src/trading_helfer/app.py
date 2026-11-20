"""Einstiegspunkt der App (Konsolenskript ``trading-helfer``)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from trading_helfer import __version__


def build_parser() -> argparse.ArgumentParser:
    """Erzeugt den Argument-Parser mit allen Unterbefehlen."""
    parser = argparse.ArgumentParser(
        prog="trading-helfer",
        description="Trading Helfer: technische Analyse, Kennzahlen und Backtesting.",
    )
    parser.add_argument("--version", action="version", version=f"trading-helfer {__version__}")
    parser.add_subparsers(dest="command", metavar="<befehl>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Startet die App bzw. führt einen Unterbefehl aus und gibt den Exit-Code zurück."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        print(f"trading-helfer {__version__}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
