"""Strategie-Validator: prüft den Quelltext einer Strategie, bevor er geladen wird.

:func:`validate` prüft in sechs Stufen (siehe Spezifikation, Strategie-Schnittstelle):

1. **Syntax** – der Text ist gültiges Python.
2. **Sicherheit** – AST-Prüfung: Importe nur aus :data:`ALLOWED_IMPORTS`; keine Aufrufe
   oder Verweise auf ``open``, ``eval``, ``exec``, ``__import__`` und ähnliche Builtins;
   keine Datei- oder Netzwerkfunktionen von ``numpy``/``pandas`` (``read_csv``,
   ``to_csv``, ``np.load`` …); kein Zugriff auf Dunder-Attribute wie ``__class__``.
3. **Pflichtfelder und Typen** – die Datei lässt sich laden, enthält mindestens eine
   Strategie-Klasse mit gültigen Pflichtfeldern, SemVer-Version, gültigen Parametern
   und den richtigen Methodensignaturen.
4. **Probelauf** – Backtest auf synthetischen Testdaten in jedem unterstützten
   Zeitrahmen ohne Fehler.
5. **Determinismus** – zwei Läufe mit gleichen Daten ergeben dasselbe Ergebnis
   (Zufall nur mit festem Seed).
6. **Lookahead** – Indikatoren und Orders bis zur Kerze ``t`` ändern sich nicht, wenn
   die Kurse nach ``t`` verändert werden (mehrere zufällige, aber reproduzierbare
   Stichtage).

Scheitert eine Stufe, werden die folgenden übersprungen. Die Stufen 3 bis 6 führen
Code der Strategie aus. Das geschieht standardmäßig in einem eigenen Prozess mit
Zeitlimit je Stufe (:data:`DEFAULT_TIMEOUT`), damit Endlosschleifen die App nicht
blockieren. Die AST-Prüfung ist ein Schutz vor versehentlichen oder offensichtlichen
Zugriffen, keine vollständige Sandbox.

Beispiel::

    report = validate(Path("plugins/meine_strategie.py").read_text("utf-8"))
    if not report.ok:
        print(report.format())
"""

from __future__ import annotations

import ast
import inspect
import math
import multiprocessing
import queue
import re
import sys
import time
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import numpy as np
import pandas as pd

__all__ = [
    "ALLOWED_IMPORTS",
    "DEFAULT_TIMEOUT",
    "FORBIDDEN_ATTRIBUTES",
    "FORBIDDEN_NAMES",
    "StageResult",
    "StageStatus",
    "ValidationReport",
    "ValidationStage",
    "synthetic_bars",
    "validate",
]

DEFAULT_TIMEOUT: Final = 30.0
"""Zeitlimit in Sekunden je Stufe, die Code der Strategie ausführt."""

_STARTUP_TIMEOUT: Final = 120.0
"""Zeit, die der Prüfprozess zum Starten (Importe) haben darf; zählt nicht zum Zeitlimit."""

ALLOWED_IMPORTS: Final[tuple[str, ...]] = (
    "math",
    "numpy",
    "pandas",
    "trading_helfer.indicators",
    "trading_helfer.strategies.base",
    "trading_helfer.core.types",
    "typing",
    "__future__",
)
"""Erlaubte Module. ``trading_helfer.indicators`` einschließlich Untermodulen, die
übrigen nur genau so (z. B. nicht ``numpy.ctypeslib``). ``trading_helfer.core.types``
(``Timeframe``, ``OrderType``, ``Side``), ``typing`` und ``__future__`` enthalten nur
Typen und sind deshalb ebenfalls erlaubt."""

_ALLOWED_WITH_SUBMODULES: Final[tuple[str, ...]] = ("trading_helfer.indicators",)

FORBIDDEN_NAMES: Final[frozenset[str]] = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "input",
        "exit",
        "quit",
        "help",
        "memoryview",
    }
)
"""Builtins, die eine Strategie weder aufrufen noch verwenden darf."""

FORBIDDEN_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {
        # Dateien lesen und schreiben (numpy)
        "load",
        "save",
        "savez",
        "savez_compressed",
        "savetxt",
        "loadtxt",
        "genfromtxt",
        "fromfile",
        "tofile",
        "fromregex",
        "memmap",
        "open_memmap",
        "DataSource",
        "npyio",
        "ctypeslib",
        "f2py",
        "distutils",
        "testing",
        "dump",
        "dumps",
        # Dateien und Netzwerk (pandas); read_* wird zusätzlich als Präfix geprüft
        "to_csv",
        "to_pickle",
        "to_excel",
        "to_parquet",
        "to_feather",
        "to_hdf",
        "to_sql",
        "to_json",
        "to_html",
        "to_latex",
        "to_stata",
        "to_xml",
        "to_orc",
        "to_markdown",
        "to_clipboard",
        "HDFStore",
        "ExcelWriter",
        "ExcelFile",
        "io",
        "eval",
        "query",
        # Module, die über Attribute anderer Module erreichbar wären
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "Path",
        "builtins",
        "importlib",
        "requests",
        "urllib",
        "http",
        "pickle",
        "ctypes",
        "multiprocessing",
        "threading",
        "asyncio",
        "logging",
        "tempfile",
        "glob",
    }
)
"""Attribut- und Importnamen, die für Datei-, Netzwerk- oder Systemzugriffe stehen."""

_ALLOWED_DUNDERS: Final[frozenset[str]] = frozenset({"__init__", "__name__", "__doc__"})
_SEMVER: Final = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")

_TIMEOUT_PREFIX: Final = "Zeitlimit"
_LOOKAHEAD_CUTS: Final = 3
_SEED: Final = 20_001_103
_ORDER_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "submitted",
    "side",
    "type",
    "qty",
    "sizing",
    "stop_price",
    "limit_price",
    "tag",
)
_SYMBOL: Final = "TEST"


# ---------------------------------------------------------------------------
# Bericht
# ---------------------------------------------------------------------------


class ValidationStage(StrEnum):
    """Prüfstufen in der Reihenfolge, in der sie laufen."""

    SYNTAX = "syntax"
    SECURITY = "security"
    FIELDS = "fields"
    TRIAL_RUN = "trial_run"
    DETERMINISM = "determinism"
    LOOKAHEAD = "lookahead"

    @property
    def label(self) -> str:
        """Name der Stufe auf Deutsch."""
        return _STAGE_LABELS[self]


_STAGE_LABELS: Final[dict[ValidationStage, str]] = {
    ValidationStage.SYNTAX: "Syntax",
    ValidationStage.SECURITY: "Sicherheit (Importe und Aufrufe)",
    ValidationStage.FIELDS: "Pflichtfelder und Typen",
    ValidationStage.TRIAL_RUN: "Probelauf auf Testdaten",
    ValidationStage.DETERMINISM: "Determinismus",
    ValidationStage.LOOKAHEAD: "Lookahead-Test",
}

_DYNAMIC_STAGES: Final[tuple[ValidationStage, ...]] = (
    ValidationStage.FIELDS,
    ValidationStage.TRIAL_RUN,
    ValidationStage.DETERMINISM,
    ValidationStage.LOOKAHEAD,
)


class StageStatus(StrEnum):
    """Ergebnis einer Stufe."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def label(self) -> str:
        """Status auf Deutsch."""
        return {"passed": "bestanden", "failed": "fehlgeschlagen", "skipped": "übersprungen"}[
            self.value
        ]


@dataclass(frozen=True)
class StageResult:
    """Ergebnis einer Prüfstufe.

    Attributes:
        stage: Die Stufe.
        status: Bestanden, fehlgeschlagen oder übersprungen.
        message: Verständliche Meldung auf Deutsch.
        line: Zeilennummer im Quelltext, falls bekannt.
        details: Weitere Meldungen (z. B. alle Verstöße der Sicherheitsprüfung).
    """

    stage: ValidationStage
    status: StageStatus
    message: str = ""
    line: int | None = None
    details: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """``True``, wenn die Stufe bestanden ist."""
        return self.status is StageStatus.PASSED

    def __str__(self) -> str:
        where = f" (Zeile {self.line})" if self.line is not None else ""
        text = f"{self.stage.label}: {self.status.label}{where}"
        return f"{text} – {self.message}" if self.message else text


@dataclass(frozen=True)
class ValidationReport:
    """Bericht des Validators mit dem Ergebnis jeder Stufe.

    Attributes:
        filename: Name der geprüften Datei (für Meldungen).
        stages: Ergebnisse aller Stufen in Prüfreihenfolge.
        strategy_ids: IDs der gefundenen Strategien (leer, wenn das Laden scheiterte).
    """

    filename: str
    stages: tuple[StageResult, ...]
    strategy_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """``True``, wenn alle Stufen bestanden sind."""
        return all(s.ok for s in self.stages)

    @property
    def failed_stage(self) -> ValidationStage | None:
        """Erste fehlgeschlagene Stufe oder ``None``."""
        for result in self.stages:
            if result.status is StageStatus.FAILED:
                return result.stage
        return None

    def stage(self, stage: ValidationStage) -> StageResult:
        """Ergebnis einer bestimmten Stufe."""
        for result in self.stages:
            if result.stage is stage:
                return result
        raise KeyError(stage)

    def summary(self) -> str:
        """Ein Satz auf Deutsch, z. B. für die Statuszeile."""
        failed = self.failed_stage
        if failed is None:
            ids = ", ".join(self.strategy_ids)
            return f"Strategie geprüft: alle Stufen bestanden ({ids})."
        result = self.stage(failed)
        where = f" in Zeile {result.line}" if result.line is not None else ""
        return f"Strategie abgelehnt – Stufe „{failed.label}“{where}: {result.message}"

    def format(self) -> str:
        """Mehrzeiliger Bericht auf Deutsch (eine Zeile je Stufe plus Details)."""
        lines = [self.summary(), ""]
        for result in self.stages:
            lines.append(str(result))
            lines.extend(f"    {d}" for d in result.details)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Öffentliche Funktion
# ---------------------------------------------------------------------------


def validate(
    source_code: str,
    *,
    filename: str = "<strategie>",
    timeout: float = DEFAULT_TIMEOUT,
    isolate: bool = True,
) -> ValidationReport:
    """Prüft den Quelltext einer Strategie in allen Stufen.

    Args:
        source_code: Inhalt der Python-Datei.
        filename: Name für Meldungen und Tracebacks.
        timeout: Zeitlimit in Sekunden je ausführender Stufe (nur mit ``isolate``).
        isolate: Code der Strategie in einem eigenen Prozess ausführen (Standard). Mit
            ``False`` läuft alles im aufrufenden Prozess ohne Zeitlimit – nur für Tests
            und bereits vertrauenswürdigen Code gedacht.

    Returns:
        Bericht mit dem Ergebnis jeder Stufe.
    """
    results: list[StageResult] = []

    tree, syntax = _check_syntax(source_code, filename)
    results.append(syntax)
    if tree is None:
        return _finish(filename, results, ())

    security = _check_security(tree)
    results.append(security)
    if not security.ok:
        return _finish(filename, results, ())

    hints = _Hints.from_tree(tree)
    if isolate:
        dynamic, ids = _run_isolated(source_code, filename, timeout)
    else:
        collected: list[_Message] = []
        _run_dynamic(source_code, filename, collected.append)
        dynamic, ids = _collect(collected)
    results.extend(hints.apply(r) for r in dynamic)
    return _finish(filename, results, ids)


def synthetic_bars(seed: int = _SEED) -> pd.DataFrame:
    """Synthetische Tageskurse 2000–2014 für Probelauf und Lookahead-Test.

    Die Reihe wechselt zwischen Auf- und Abwärtstrends und schwankt dazwischen, damit
    Trend- und Mean-Reversion-Strategien handeln. Gleicher ``seed`` ergibt gleiche Daten.

    Args:
        seed: Startwert des Zufallsgenerators.

    Returns:
        Tageskurse im Schema aus :mod:`trading_helfer.core.bars` (ohne ``adj_close``).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", "2014-12-31")
    n = len(dates)
    t = np.arange(n, dtype=float)
    drift = 0.0006 * np.sign(np.sin(2 * np.pi * t / 750.0) + 0.3)
    cycle = 0.004 * np.sin(2 * np.pi * t / 90.0)
    log_close = np.log(100.0) + np.cumsum(drift + rng.normal(0.0, 0.012, n)) + cycle
    return _ohlc_from_close(dates, np.exp(log_close), rng)


def _ohlc_from_close(
    dates: pd.DatetimeIndex, close: np.ndarray[Any, Any], rng: np.random.Generator
) -> pd.DataFrame:
    n = len(close)
    prev = np.concatenate([[close[0]], close[:-1]])
    gap = rng.normal(0.0, 0.003, n)
    opens = prev * np.exp(gap)
    spread = np.abs(rng.normal(0.0, 0.008, n))
    high = np.maximum(opens, close) * (1.0 + spread)
    low = np.minimum(opens, close) * (1.0 - spread)
    volume = np.round(rng.uniform(5e5, 2e6, n)).astype("int64")
    return pd.DataFrame(
        {"date": dates, "open": opens, "high": high, "low": low, "close": close, "volume": volume}
    )


# ---------------------------------------------------------------------------
# Stufe 1 und 2: statisch
# ---------------------------------------------------------------------------


def _check_syntax(source: str, filename: str) -> tuple[ast.Module | None, StageResult]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return None, StageResult(
            ValidationStage.SYNTAX,
            StageStatus.FAILED,
            f"Syntaxfehler: {exc.msg}.",
            line=exc.lineno,
        )
    except ValueError as exc:  # z. B. Nullbytes im Text
        return None, StageResult(
            ValidationStage.SYNTAX, StageStatus.FAILED, f"Kein gültiger Quelltext: {exc}."
        )
    return tree, StageResult(ValidationStage.SYNTAX, StageStatus.PASSED)


def _import_allowed(module: str) -> bool:
    if module in ALLOWED_IMPORTS:
        return True
    return any(module.startswith(f"{base}.") for base in _ALLOWED_WITH_SUBMODULES)


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _forbidden_attribute(name: str) -> bool:
    if name in FORBIDDEN_ATTRIBUTES or name.startswith("read_"):
        return True
    return _is_dunder(name) and name not in _ALLOWED_DUNDERS


def _security_issues(tree: ast.Module) -> Iterator[tuple[int, str]]:
    allowed = ", ".join(m for m in ALLOWED_IMPORTS if m not in ("typing", "__future__"))
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _import_allowed(alias.name):
                    yield line, f"Import von „{alias.name}“ ist nicht erlaubt (erlaubt: {allowed})."
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                yield line, "Relative Importe sind nicht erlaubt."
            elif not _import_allowed(module):
                yield line, f"Import aus „{module}“ ist nicht erlaubt (erlaubt: {allowed})."
            else:
                for alias in node.names:
                    if alias.name == "*":
                        yield line, f"„from {module} import *“ ist nicht erlaubt."
                    elif _forbidden_attribute(alias.name):
                        yield line, f"Import von „{alias.name}“ aus „{module}“ ist nicht erlaubt."
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                yield (
                    line,
                    f"„{node.id}“ ist nicht erlaubt (keine Datei-, Code- oder Systemzugriffe).",
                )
            elif _is_dunder(node.id) and node.id not in ("__name__", "__doc__", "__file__"):
                yield line, f"Zugriff auf „{node.id}“ ist nicht erlaubt."
        elif isinstance(node, ast.Attribute) and _forbidden_attribute(node.attr):
            yield (
                line,
                f"Zugriff auf „.{node.attr}“ ist nicht erlaubt (Datei-, Netzwerk- oder "
                "Systemzugriff).",
            )


def _check_security(tree: ast.Module) -> StageResult:
    issues = sorted(set(_security_issues(tree)))
    if not issues:
        return StageResult(ValidationStage.SECURITY, StageStatus.PASSED)
    line, message = issues[0]
    more = f" (und {len(issues) - 1} weitere Verstöße)" if len(issues) > 1 else ""
    return StageResult(
        ValidationStage.SECURITY,
        StageStatus.FAILED,
        message + more,
        line=line,
        details=tuple(f"Zeile {ln}: {msg}" for ln, msg in issues),
    )


@dataclass(frozen=True)
class _Hints:
    """Zeilen im Quelltext, die eine fehlgeschlagene Laufzeit-Stufe erklären können."""

    random_line: int | None = None
    lookahead_line: int | None = None
    loop_line: int | None = None

    @classmethod
    def from_tree(cls, tree: ast.Module) -> _Hints:
        random_line: int | None = None
        lookahead_line: int | None = None
        loop_line: int | None = None
        for node in ast.walk(tree):
            line = getattr(node, "lineno", None)
            if random_line is None and _is_random_use(node):
                random_line = line
            if lookahead_line is None and _is_future_access(node):
                lookahead_line = line
            if loop_line is None and isinstance(node, ast.While):
                loop_line = line
        return cls(random_line, lookahead_line, loop_line)

    def apply(self, result: StageResult) -> StageResult:
        if result.status is not StageStatus.FAILED or result.line is not None:
            return result
        if result.message.startswith(_TIMEOUT_PREFIX) and self.loop_line is not None:
            return StageResult(
                result.stage,
                result.status,
                result.message + f" Verdächtig: Schleife in Zeile {self.loop_line}.",
                line=self.loop_line,
                details=result.details,
            )
        if result.stage is ValidationStage.DETERMINISM and self.random_line is not None:
            return StageResult(
                result.stage,
                result.status,
                result.message + f" Zufall ohne festen Seed? Siehe Zeile {self.random_line}.",
                line=self.random_line,
                details=result.details,
            )
        if result.stage is ValidationStage.LOOKAHEAD and self.lookahead_line is not None:
            return StageResult(
                result.stage,
                result.status,
                result.message + f" Verdächtig: Zeile {self.lookahead_line} (Blick nach vorn).",
                line=self.lookahead_line,
                details=result.details,
            )
        return result


def _is_random_use(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "random":
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "default_rng"
        and not node.args
        and not node.keywords
    )


def _is_negative_constant(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int | float)
        and node.operand.value > 0
    )


def _is_future_access(node: ast.AST) -> bool:
    """``shift(-n)``, ``rolling(..., center=True)`` und ähnliche Blicke nach vorn."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr == "shift":
        if node.args and _is_negative_constant(node.args[0]):
            return True
        return any(kw.arg == "periods" and _is_negative_constant(kw.value) for kw in node.keywords)
    return any(
        kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in node.keywords
    )


# ---------------------------------------------------------------------------
# Stufen 3–6: Ausführung (im Prüfprozess oder im eigenen Prozess)
# ---------------------------------------------------------------------------

_Message = tuple[str, Any]
"""Nachricht der Ausführung: ("ready", None), ("ids", tuple), ("stage", StageResult),
("done", None)."""


class _StageFailedError(Exception):
    """Interne Ausnahme: Stufe fehlgeschlagen."""

    def __init__(self, message: str, line: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.line = line


def _run_dynamic(source: str, filename: str, emit: Callable[[_Message], None]) -> None:
    """Führt die Stufen 3–6 aus und meldet jedes Ergebnis über ``emit``."""
    from trading_helfer.core.config import Settings

    settings = Settings.model_construct()
    state: dict[str, Any] = {}

    def fields() -> None:
        classes = _load_classes(source, filename)
        state["classes"] = classes
        emit(("ids", tuple(cls.id for cls in classes)))

    def trial() -> None:
        state["bars"] = synthetic_bars()
        for cls in state["classes"]:
            _trial_run(cls, state["bars"], settings, filename)

    def determinism() -> None:
        for cls in state["classes"]:
            _determinism(cls, state["bars"], settings, filename)

    def lookahead() -> None:
        for cls in state["classes"]:
            _lookahead(cls, state["bars"], settings, filename)

    runners: list[tuple[ValidationStage, Callable[[], None]]] = [
        (ValidationStage.FIELDS, fields),
        (ValidationStage.TRIAL_RUN, trial),
        (ValidationStage.DETERMINISM, determinism),
        (ValidationStage.LOOKAHEAD, lookahead),
    ]
    failed = False
    for stage, runner in runners:
        if failed:
            emit(("stage", StageResult(stage, StageStatus.SKIPPED)))
            continue
        try:
            runner()
        except _StageFailedError as exc:
            failed = True
            emit(("stage", StageResult(stage, StageStatus.FAILED, exc.message, exc.line)))
        except Exception as exc:  # unerwarteter Fehler der Strategie
            failed = True
            line = _last_line(exc, filename)
            emit(("stage", StageResult(stage, StageStatus.FAILED, _describe(exc), line)))
        else:
            emit(("stage", StageResult(stage, StageStatus.PASSED, _PASSED_TEXT[stage])))
    emit(("done", None))


_PASSED_TEXT: Final[dict[ValidationStage, str]] = {
    ValidationStage.FIELDS: "Pflichtfelder, Parameter und Methoden in Ordnung.",
    ValidationStage.TRIAL_RUN: "Backtest auf Testdaten ohne Fehler.",
    ValidationStage.DETERMINISM: "Zwei Läufe ergeben dasselbe Ergebnis.",
    ValidationStage.LOOKAHEAD: "Kein Blick in die Zukunft gefunden.",
}


def _load_classes(source: str, filename: str) -> list[Any]:
    from trading_helfer.strategies.base import Strategy
    from trading_helfer.strategies.registry import (
        _no_strategy_error,
        _strategy_classes,
        check_fields,
    )

    tree = ast.parse(source, filename=filename)
    class_lines = {n.name: n.lineno for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

    name = f"trading_helfer_validator.m{time.monotonic_ns()}"
    module = types.ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module  # nötig z. B. für @dataclass in der Strategie
    try:
        exec(compile(source, filename, "exec", dont_inherit=True), module.__dict__)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise _StageFailedError(
            f"Fehler beim Laden der Datei: {_describe(exc)}", _last_line(exc, filename)
        ) from exc
    finally:
        sys.modules.pop(name, None)

    classes = _strategy_classes(module)
    if not classes:
        error = _no_strategy_error(filename, module)
        line = class_lines.get(error.strategy) if error.strategy else None
        raise _StageFailedError(error.message, line)

    seen: set[str] = set()
    for cls in classes:
        line = class_lines.get(cls.__name__)
        problems = check_fields(cls)
        if problems:
            raise _StageFailedError(f"Klasse {cls.__name__}: {problems[0][1]}", line)
        if not _SEMVER.match(cls.version):
            raise _StageFailedError(
                f"Klasse {cls.__name__}: version {cls.version!r} ist keine SemVer-Version "
                "(z. B. „1.0.0“).",
                line,
            )
        if cls.id in seen:
            raise _StageFailedError(f"Strategie-ID {cls.id!r} kommt mehrfach vor.", line)
        seen.add(cls.id)
        try:
            cls.make_params()
        except Exception as exc:
            raise _StageFailedError(
                f"Klasse {cls.__name__}: Parameter ungültig: {exc}", line
            ) from exc
        _check_signatures(cls, Strategy, line)
    return classes


_SIGNATURES: Final[dict[str, tuple[str, ...]]] = {
    "on_bar": ("ctx",),
    "indicators": ("bars", "p"),
    "plot": (),
}


def _check_signatures(cls: type, base: type, class_line: int | None) -> None:
    for method, expected in _SIGNATURES.items():
        func = getattr(cls, method, None)
        if func is getattr(base, method):
            continue
        line = getattr(getattr(func, "__code__", None), "co_firstlineno", class_line)
        wanted = f"def {method}(self{''.join(', ' + a for a in expected)})"
        if not callable(func) or isinstance(inspect.getattr_static(cls, method), staticmethod):
            raise _StageFailedError(
                f"{cls.__name__}.{method} muss eine Methode sein: {wanted}.", line
            )
        sig = inspect.signature(func)
        try:
            sig.bind(None, *([None] * len(expected)))
        except TypeError:
            raise _StageFailedError(
                f"{cls.__name__}.{method} hat die falsche Signatur {method}{sig}; "
                f"erwartet: {wanted}.",
                line,
            ) from None
        required = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(required) != len(expected) + 1:
            raise _StageFailedError(
                f"{cls.__name__}.{method} hat die falsche Signatur {method}{sig}; "
                f"erwartet: {wanted}.",
                line,
            )


def _run(cls: Any, bars: pd.DataFrame, settings: Any, timeframe: Any, filename: str) -> Any:
    from trading_helfer.backtest.engine import BacktestError, run_backtest

    first = pd.Timestamp(bars["date"].iloc[0]).date()
    last = pd.Timestamp(bars["date"].iloc[-1]).date()
    try:
        return run_backtest(cls, None, _SYMBOL, timeframe, first, last, settings, bars=bars)
    except BacktestError as exc:
        raise _StageFailedError(
            f"Zeitrahmen {timeframe.value}: {exc}", _last_line(exc, filename)
        ) from exc


def _trial_run(cls: Any, bars: pd.DataFrame, settings: Any, filename: str) -> None:
    from trading_helfer.strategies.base import PlotSpec

    for timeframe in cls.timeframes:
        _run(cls, bars, settings, timeframe, filename)
    specs = cls().plot()
    if not isinstance(specs, list | tuple) or not all(isinstance(s, PlotSpec) for s in specs):
        raise _StageFailedError(f"{cls.__name__}.plot() muss eine Liste von PlotSpec liefern.")
    if specs:
        frame = _frame(bars, cls.timeframes[0])
        columns = set(cls().indicators(frame.copy(), cls.make_params()).columns)
        unknown = sorted({s.column for s in specs} - columns)
        if unknown:
            raise _StageFailedError(
                f"{cls.__name__}.plot() nennt unbekannte Indikator-Spalten: {', '.join(unknown)}."
            )


def _determinism(cls: Any, bars: pd.DataFrame, settings: Any, filename: str) -> None:
    for timeframe in cls.timeframes:
        first = _run(cls, bars, settings, timeframe, filename)
        second = _run(cls, bars, settings, timeframe, filename)
        frame = _frame(bars, timeframe)
        ind_a = cls().indicators(frame.copy(), cls.make_params())
        ind_b = cls().indicators(frame.copy(), cls.make_params())
        diff = _first_difference(ind_a, ind_b, len(frame))
        if diff is not None:
            column, row = diff
            day = pd.Timestamp(frame["date"].iloc[row]).date()
            raise _StageFailedError(
                f"Strategie {cls.id!r}, Zeitrahmen {timeframe.value}: Indikator {column!r} "
                f"liefert am {day} bei gleichen Daten verschiedene Werte."
            )
        if first.fingerprint() != second.fingerprint():
            raise _StageFailedError(
                f"Strategie {cls.id!r}, Zeitrahmen {timeframe.value}: zwei Läufe mit gleichen "
                "Daten und Parametern ergeben verschiedene Ergebnisse."
            )


def _lookahead(cls: Any, bars: pd.DataFrame, settings: Any, filename: str) -> None:
    rng = np.random.default_rng(_SEED)
    for timeframe in cls.timeframes:
        frame = _frame(bars, timeframe)
        n = len(frame)
        if n < 10:
            continue
        base_ind = cls().indicators(frame.copy(), cls.make_params())
        base_orders = _run(cls, bars, settings, timeframe, filename).orders
        cuts = sorted(rng.choice(np.arange(int(n * 0.3), int(n * 0.9)), _LOOKAHEAD_CUTS, False))
        for cut in cuts:
            cut_day = pd.Timestamp(frame["date"].iloc[int(cut)])
            changed = _alter_after(bars, cut_day, rng)
            changed_frame = _frame(changed, timeframe)
            ind = cls().indicators(changed_frame.copy(), cls.make_params())
            diff = _first_difference(base_ind, ind, int(cut) + 1)
            if diff is not None:
                column, row = diff
                day = pd.Timestamp(frame["date"].iloc[row]).date()
                raise _StageFailedError(
                    f"Strategie {cls.id!r}, Zeitrahmen {timeframe.value}: Indikator {column!r} "
                    f"am {day} ändert sich, wenn Kurse nach dem {cut_day.date()} verändert "
                    "werden. Indikatoren dürfen nur Kerzen bis t verwenden."
                )
            orders = _run(cls, changed, settings, timeframe, filename).orders
            day = _first_order_difference(base_orders, orders, cut_day)
            if day is not None:
                raise _StageFailedError(
                    f"Strategie {cls.id!r}, Zeitrahmen {timeframe.value}: die Order vom {day} "
                    f"ändert sich, wenn Kurse nach dem {cut_day.date()} verändert werden. "
                    "on_bar darf nur Daten bis t verwenden."
                )


def _frame(bars: pd.DataFrame, timeframe: Any) -> pd.DataFrame:
    from trading_helfer.data.resample import PARTIAL_COLUMN, resample

    frame = resample(bars, timeframe)
    frame = frame.loc[~frame[PARTIAL_COLUMN].to_numpy()]
    return frame.drop(columns=[PARTIAL_COLUMN]).reset_index(drop=True)


def _alter_after(bars: pd.DataFrame, cut: pd.Timestamp, rng: np.random.Generator) -> pd.DataFrame:
    """Kopie der Tageskurse, deren Verlauf nach ``cut`` durch einen anderen ersetzt ist."""
    after = (pd.to_datetime(bars["date"]) > cut).to_numpy()
    count = int(after.sum())
    if count == 0:
        return bars.copy()
    start = float(bars["close"].to_numpy()[~after][-1])
    trend = rng.choice([-1.0, 1.0]) * 0.002
    close = start * np.exp(np.cumsum(trend + rng.normal(0.0, 0.03, count)))
    tail = _ohlc_from_close(pd.DatetimeIndex(bars["date"].to_numpy()[after]), close, rng)
    head = bars.loc[~after].reset_index(drop=True)
    return pd.concat([head, tail], ignore_index=True)


def _first_difference(a: pd.DataFrame, b: pd.DataFrame, rows: int) -> tuple[str, int] | None:
    """Erste Spalte/Zeile (bis ``rows``), in der sich zwei Indikator-Frames unterscheiden."""
    if list(a.columns) != list(b.columns):
        return (", ".join(map(str, sorted(set(a.columns) ^ set(b.columns)))) or "?", 0)
    for column in a.columns:
        x = a[column].to_numpy()[:rows]
        y = b[column].to_numpy()[:rows]
        for row in range(min(len(x), len(y))):
            if not _same(x[row], y[row]):
                return str(column), row
    return None


def _same(x: Any, y: Any) -> bool:
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        return bool(x == y) or (pd.isna(x) and pd.isna(y))
    if math.isnan(fx) or math.isnan(fy):
        return math.isnan(fx) and math.isnan(fy)
    return math.isclose(fx, fy, rel_tol=1e-9, abs_tol=1e-12)


def _first_order_difference(a: pd.DataFrame, b: pd.DataFrame, cut: pd.Timestamp) -> Any:
    """Datum der ersten Order bis ``cut``, die sich zwischen zwei Läufen unterscheidet."""

    def rows(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
        mask = [pd.Timestamp(d) <= cut for d in frame["submitted"]]
        keep = frame.loc[mask, list(_ORDER_KEY_COLUMNS)]
        return [
            (pd.Timestamp(r[0]).date(), *r[1:]) for r in keep.itertuples(index=False, name=None)
        ]

    ra, rb = rows(a), rows(b)
    for x, y in zip(ra, rb, strict=False):
        if not all(_same(u, v) for u, v in zip(x, y, strict=True)):
            return x[0]
    if len(ra) != len(rb):
        longer = ra if len(ra) > len(rb) else rb
        return longer[min(len(ra), len(rb))][0]
    return None


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _last_line(exc: BaseException, filename: str) -> int | None:
    """Letzte Zeile in ``filename`` im Traceback von ``exc`` oder seinen Ursachen."""
    seen: set[int] = set()
    current: BaseException | None = exc
    line: int | None = None
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        tb = current.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == filename:
                line = tb.tb_lineno
            tb = tb.tb_next
        current = current.__cause__ or current.__context__
    return line


def _collect(messages: list[_Message]) -> tuple[list[StageResult], tuple[str, ...]]:
    results = [payload for kind, payload in messages if kind == "stage"]
    ids: tuple[str, ...] = ()
    for kind, payload in messages:
        if kind == "ids":
            ids = tuple(payload)
    return results, ids


def _finish(filename: str, results: list[StageResult], ids: tuple[str, ...]) -> ValidationReport:
    done = {r.stage for r in results}
    for stage in ValidationStage:
        if stage not in done:
            results.append(StageResult(stage, StageStatus.SKIPPED))
    order = list(ValidationStage)
    results.sort(key=lambda r: order.index(r.stage))
    return ValidationReport(filename, tuple(results), ids)


# ---------------------------------------------------------------------------
# Eigener Prozess mit Zeitlimit
# ---------------------------------------------------------------------------


def _child_main(source: str, filename: str, channel: Any) -> None:
    """Einstiegspunkt des Prüfprozesses."""
    channel.put(("ready", None))
    _run_dynamic(source, filename, channel.put)


def _run_isolated(
    source: str, filename: str, timeout: float
) -> tuple[list[StageResult], tuple[str, ...]]:
    ctx = multiprocessing.get_context("spawn")
    channel = ctx.Queue()
    process = ctx.Process(
        target=_child_main, args=(source, filename, channel), name="strategie-validator"
    )
    process.daemon = True
    process.start()
    messages: list[_Message] = []
    try:
        if _receive(channel, process, _STARTUP_TIMEOUT) is None:
            return [
                StageResult(
                    ValidationStage.FIELDS,
                    StageStatus.FAILED,
                    "Der Prüfprozess konnte nicht gestartet werden.",
                )
            ], ()
        for stage in _DYNAMIC_STAGES:
            while True:
                message = _receive(channel, process, timeout)
                if message is None:
                    return _aborted(messages, stage, process, timeout), _collect(messages)[1]
                messages.append(message)
                if message[0] == "stage":
                    break
        return _collect(messages)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(5)
        channel.close()


def _receive(channel: Any, process: Any, timeout: float) -> _Message | None:
    """Nächste Nachricht oder ``None`` bei Zeitüberschreitung oder Absturz."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            message: _Message = channel.get(timeout=min(0.2, remaining))
            return message
        except queue.Empty:
            if not process.is_alive():
                try:
                    late: _Message = channel.get(timeout=0.5)
                    return late
                except queue.Empty:
                    return None


def _aborted(
    messages: list[_Message], stage: ValidationStage, process: Any, timeout: float
) -> list[StageResult]:
    results, _ = _collect(messages)
    if process.is_alive():
        text = (
            f"{_TIMEOUT_PREFIX} von {timeout:g} s überschritten – Endlosschleife oder zu langsame "
            "Berechnung?"
        )
    else:
        text = f"Der Prüfprozess wurde unerwartet beendet (Exitcode {process.exitcode})."
    results.append(StageResult(stage, StageStatus.FAILED, text))
    return results
