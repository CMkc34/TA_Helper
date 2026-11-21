"""Kommandozeile für ``research/run_study.py``.

Beispiele::

    uv run python research/run_study.py --strategy trend_sma_cross --tf W1 --index SP500 \\
        --param fast=10 --param slow=40 --in-sample-only
    uv run python research/run_study.py --strategy trend_sma_cross --tf W1 --index SP500 \\
        --param fast=10 --param slow=40
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date as Date  # noqa: N812
from pathlib import Path
from typing import Any

__all__ = ["DEFAULT_OUT_DIR", "build_parser", "default_report_dir", "main", "parse_grid"]

DEFAULT_OUT_DIR = Path("research") / "reports"
"""Standardordner für Berichte (je Studie ein Unterordner)."""


def build_parser() -> argparse.ArgumentParser:
    """Argumente von ``run_study.py``."""
    parser = argparse.ArgumentParser(
        prog="run_study.py",
        description=(
            "Prüft eine Strategie nach dem festen Studienablauf (In-Sample 2000–2015, "
            "Walk-Forward 5/1, Monte-Carlo, doppelte Kosten, Marktphasen, Out-of-Sample "
            "ab 2016) und schreibt einen Bericht (Markdown plus Grafiken)."
        ),
    )
    parser.add_argument("--strategy", required=True, metavar="ID", help="ID der Strategie")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--symbol", metavar="SYMBOL", help="einzelner Wert")
    target.add_argument("--index", metavar="CODE", help="Index-Code, z. B. SP500 oder NDX100")
    target.add_argument("--symbols", nargs="+", metavar="SYMBOL", help="eigene Symbolliste")
    parser.add_argument("--tf", default="W1", metavar="ZEITRAHMEN", help="D3 oder W1 (Standard)")
    parser.add_argument(
        "--param", action="append", default=[], metavar="NAME=WERT", help="Parameter"
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="NAME=W1,W2,…",
        help="eigenes Walk-Forward-Raster (Standard: Parameter-Nachbarschaft)",
    )
    parser.add_argument(
        "--max-positions", type=int, default=10, metavar="N", help="Positionen im Portfolio"
    )
    parser.add_argument(
        "--in-sample-only",
        action="store_true",
        help="Kurse nach 2015 abschneiden; Out-of-Sample bleibt ungesehen",
    )
    parser.add_argument("--end", metavar="JJJJ-MM-TT", help="letzter Datentag")
    parser.add_argument("--simulations", type=int, default=1000, help="Monte-Carlo-Simulationen")
    parser.add_argument("--seed", type=int, default=42, help="Seed der Monte-Carlo-Simulation")
    parser.add_argument("--workers", type=int, metavar="N", help="Prozesse (Standard: CPU-Kerne)")
    parser.add_argument("--out", type=Path, metavar="ORDNER", help="Zielordner des Berichts")
    parser.add_argument("--db", type=Path, metavar="DATEI", help="DuckDB-Datei")
    return parser


def parse_grid(items: Sequence[str]) -> dict[str, list[Any]]:
    """Wandelt ``NAME=W1,W2`` in ein Raster; Zahlen und ``true``/``false`` werden erkannt.

    Raises:
        ValueError: Eintrag ohne ``=`` oder ohne Werte.
    """
    from trading_helfer.scanner.cli import parse_params

    grid: dict[str, list[Any]] = {}
    for item in items:
        name, sep, raw = item.partition("=")
        values = [v.strip() for v in raw.split(",") if v.strip()]
        if not sep or not name.strip() or not values:
            raise ValueError(f"Raster {item!r} hat nicht die Form NAME=W1,W2.")
        grid[name.strip()] = [parse_params([f"v={v}"])["v"] for v in values]
    return grid


def default_report_dir(strategy_id: str, timeframe: str, target: str, in_sample_only: bool) -> Path:
    """Standardordner ``research/reports/<strategie>_<zeitrahmen>_<ziel>[_is]``."""
    safe = "".join(ch if ch.isalnum() else "-" for ch in target)[:40].strip("-") or "liste"
    suffix = "_is" if in_sample_only else ""
    return DEFAULT_OUT_DIR / f"{strategy_id}_{timeframe}_{safe}{suffix}"


def main(argv: Sequence[str] | None = None) -> int:
    """Führt die Studie aus und schreibt den Bericht.

    Returns:
        0, wenn der Bericht geschrieben wurde (unabhängig vom Urteil); 2 bei ungültiger
        Eingabe oder fehlenden Daten.
    """
    from trading_helfer.backtest.engine import BacktestError
    from trading_helfer.core.config import load_settings
    from trading_helfer.research.report import write_report
    from trading_helfer.research.study import StudyConfig, StudyError, run_study
    from trading_helfer.scanner.cli import parse_params
    from trading_helfer.strategies.registry import StrategyNotFoundError, StrategyRegistry

    args = build_parser().parse_args(argv)
    settings = load_settings()
    if args.db is not None:
        settings = settings.model_copy(update={"db_path": args.db})
    registry = StrategyRegistry(settings.plugins_path)
    registry.load()
    try:
        strategy = registry.create(args.strategy)
        params = parse_params(args.param)
        grid = parse_grid(args.grid) or None
        config = StudyConfig(
            end=Date.fromisoformat(args.end) if args.end else None,
            include_out_of_sample=not args.in_sample_only,
            mc_simulations=args.simulations,
            mc_seed=args.seed,
            max_positions=args.max_positions,
            workers=args.workers,
        )
        study = run_study(
            strategy,
            params,
            args.tf,
            settings,
            symbol=args.symbol,
            universe=args.index if args.index else args.symbols,
            config=config,
            grid=grid,
            progress=lambda step: print(f"… {step}", flush=True),
        )
    except (StrategyNotFoundError, StudyError, BacktestError, ValueError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"Fehler: {message}", file=sys.stderr)
        return 2
    target = args.symbol or args.index or "liste"
    out_dir = args.out or default_report_dir(
        study.strategy_id, study.timeframe.value, target, args.in_sample_only
    )
    path = write_report(study, out_dir)
    for criterion in study.criteria:
        state = {True: "erfüllt", False: "nicht erfüllt", None: "offen"}[criterion.passed]
        print(f"{criterion.name}: {criterion.value} – {state}")
    print(f"Urteil: {study.verdict.value}")
    print(f"Bericht: {path}")
    return 0
