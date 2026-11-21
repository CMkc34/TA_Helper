"""Selbsttest der App ohne Oberfläche (Teil von ``trading-helfer --selftest``).

Der Selbsttest prüft, ob ein gebautes Paket (PyInstaller) alles mitbringt, was die App
braucht: Datenbank-Migrationen, Indexdaten, Indikatoren, eingebaute Strategien samt
Quelltext, die Strategie-Prüfung in einem eigenen Prozess (``multiprocessing`` im Paket)
und einen Datenordner außerhalb des App-Ordners. Er läuft komplett offline.

Das Modul importiert kein Qt; die Prüfung des Hauptfensters übernimmt
:func:`trading_helfer.gui.application.run_selftest`.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from trading_helfer import __version__
from trading_helfer.core.config import Settings, default_data_dir

__all__ = [
    "SelfTestCheck",
    "app_dir",
    "format_report",
    "is_frozen",
    "run_core_checks",
]


@dataclass(frozen=True, slots=True)
class SelfTestCheck:
    """Ergebnis einer einzelnen Prüfung.

    Attributes:
        name: Kurzer deutscher Name der Prüfung.
        ok: ``True``, wenn die Prüfung bestanden ist.
        detail: Erläuterung (Fundstelle oder Fehlermeldung).
    """

    name: str
    ok: bool
    detail: str


def is_frozen() -> bool:
    """``True``, wenn die App als gebautes Paket (PyInstaller) läuft."""
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """Ordner der installierten App.

    Im gebauten Paket der Ordner der ``.exe``, sonst der Ordner des Pakets
    ``trading_helfer``.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def run_core_checks(
    settings: Settings, *, user_data_dir: Path | None = None
) -> list[SelfTestCheck]:
    """Führt alle Prüfungen ohne Oberfläche aus.

    Jede Prüfung fängt ihre Ausnahmen selbst ab, damit der Bericht vollständig ist.

    Args:
        settings: Einstellungen; Datenbank und Plugins liegen unter ``settings.data_dir``
            (für den Selbsttest ein temporärer Ordner).
        user_data_dir: Datenordner, der außerhalb des App-Ordners liegen muss;
            Standard ist :func:`~trading_helfer.core.config.default_data_dir`.

    Returns:
        Ergebnisse in fester Reihenfolge.
    """
    target = user_data_dir if user_data_dir is not None else default_data_dir()
    checks: list[tuple[str, Callable[[], str]]] = [
        ("Datenordner", lambda: _check_data_dir(target)),
        ("Datenbank", lambda: _check_database(settings)),
        ("Indexdaten", _check_universe),
        ("Indikatoren", _check_indicators),
        ("Strategien", _check_strategies),
        ("Strategie-Prüfung", _check_validator),
    ]
    results: list[SelfTestCheck] = []
    for name, check in checks:
        try:
            results.append(SelfTestCheck(name, True, check()))
        except Exception as exc:
            results.append(SelfTestCheck(name, False, f"{type(exc).__name__}: {exc}"))
    return results


def format_report(checks: Sequence[SelfTestCheck]) -> str:
    """Formatiert die Ergebnisse als mehrzeiligen Text mit Gesamturteil."""
    mode = "Paket" if is_frozen() else "Python"
    lines = [f"Trading Helfer {__version__} – Selbsttest ({mode})"]
    lines += [f"{'OK  ' if c.ok else 'FEHL'} {c.name}: {c.detail}" for c in checks]
    passed = bool(checks) and all(c.ok for c in checks)
    lines.append("Selbsttest bestanden." if passed else "Selbsttest fehlgeschlagen.")
    return "\n".join(lines)


# -- Einzelprüfungen --------------------------------------------------------------------


def _check_data_dir(target: Path) -> str:
    base = app_dir()
    resolved = target.expanduser().resolve()
    if resolved == base or resolved.is_relative_to(base):
        raise RuntimeError(f"Datenordner {resolved} liegt im App-Ordner {base}.")
    return str(resolved)


def _check_database(settings: Settings) -> str:
    from trading_helfer.data.schema import discover_migrations
    from trading_helfer.data.storage import MarketStore

    expected = max(m.version for m in discover_migrations())
    with MarketStore(settings.database_path) as store:
        version = store.schema_version()
    if version != expected:
        raise RuntimeError(f"Schema-Version {version}, erwartet {expected}.")
    return f"Schema-Version {version}"


def _check_universe() -> str:
    from trading_helfer.data.universe import load_history

    counts = []
    for code in ("SP500", "NDX100"):
        history = load_history(code)
        if not history.current:
            raise RuntimeError(f"{code} hat keine Mitglieder.")
        counts.append(f"{code} {len(history.current)}")
    return ", ".join(counts)


def _check_indicators() -> str:
    from trading_helfer.indicators import registry

    count = len(registry.list_all())
    if count == 0:
        raise RuntimeError("keine Indikatoren registriert.")
    return f"{count} registriert"


def _check_strategies() -> str:
    from trading_helfer.backtest.results import strategy_source_hash
    from trading_helfer.strategies.registry import StrategyRegistry

    registry = StrategyRegistry(None)
    report = registry.load()
    if not report.ok:
        raise RuntimeError("; ".join(str(e) for e in report.errors))
    if not report.loaded:
        raise RuntimeError("keine eingebauten Strategien gefunden.")
    missing = [i for i in report.loaded if strategy_source_hash(registry.get(i)) is None]
    if missing:
        raise RuntimeError(f"Quelltext fehlt für {', '.join(missing)}.")
    return f"{len(report.loaded)} eingebaut, Quelltext vorhanden"


def _check_validator() -> str:
    import inspect

    from trading_helfer.strategies.builtin import sma_cross
    from trading_helfer.strategies.validator import validate

    # Läuft wie bei KI-Strategien in einem eigenen Prozess; im Paket startet dieser die
    # .exe erneut und braucht multiprocessing.freeze_support() in app.main.
    report = validate(inspect.getsource(sma_cross), filename="sma_cross.py")
    if not report.ok:
        raise RuntimeError(report.summary())
    return "eigener Prozess, alle Stufen bestanden"
