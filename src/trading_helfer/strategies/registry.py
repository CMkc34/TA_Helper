"""Strategie-Registry: lädt eingebaute Strategien und Plugins aus ``plugins/``.

Eingebaute Strategien liegen als Module in :mod:`trading_helfer.strategies.builtin`.
Plugins sind beliebige ``*.py``-Dateien im Plugin-Ordner (Pfad einstellbar über
``Settings.plugins_dir``). Jede Datei wird für sich geladen: ein Fehler in einer
Datei verhindert nicht das Laden der anderen. Dateien, deren Name mit ``_`` beginnt
(z. B. ``__init__.py``), werden übersprungen.

Fehler werden nie stillschweigend übergangen, sondern als :class:`LoadError` im
:class:`LoadReport` gemeldet und protokolliert:

- Syntaxfehler und Fehler beim Ausführen der Datei,
- Datei ohne Strategie-Klasse,
- fehlende oder ungültige Pflichtfelder (``id``, ``name``, ``version``,
  ``description``, ``timeframes``, ``params``),
- doppelte ``id`` (die zuerst geladene Strategie bleibt gültig; Reihenfolge:
  eingebaute Strategien, dann Plugin-Dateien alphabetisch).

:meth:`StrategyRegistry.reload` liest den Plugin-Ordner neu ein, ohne dass die App
neu gestartet werden muss. Die tiefergehende Prüfung (verbotene Importe,
Lookahead-Test, Probelauf) übernimmt der Validator.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import re
import sys
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from trading_helfer.core.types import Timeframe
from trading_helfer.strategies import builtin
from trading_helfer.strategies.base import Param, Strategy

__all__ = [
    "BUILTIN_SOURCE",
    "REQUIRED_FIELDS",
    "LoadError",
    "LoadErrorKind",
    "LoadReport",
    "StrategyInfo",
    "StrategyNotFoundError",
    "StrategyRegistry",
    "check_fields",
]

_log = logging.getLogger(__name__)

BUILTIN_SOURCE = "builtin"
"""Quelle eingebauter Strategien in :attr:`StrategyInfo.source`."""

REQUIRED_FIELDS: tuple[str, ...] = ("id", "name", "version", "description", "timeframes")
"""Klassenattribute, die jede Strategie selbst setzen muss."""

_PLUGIN_PACKAGE = "trading_helfer_plugins"
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class LoadErrorKind(StrEnum):
    """Art eines Ladefehlers."""

    SYNTAX = "syntax"
    """Die Datei ist kein gültiges Python."""
    EXECUTION = "execution"
    """Beim Ausführen der Datei (z. B. Import) trat eine Ausnahme auf."""
    NO_STRATEGY = "no_strategy"
    """Die Datei enthält keine (vollständige) Strategie-Klasse."""
    MISSING_FIELD = "missing_field"
    """Ein Pflichtfeld fehlt."""
    INVALID_FIELD = "invalid_field"
    """Ein Pflichtfeld hat einen ungültigen Wert."""
    DUPLICATE_ID = "duplicate_id"
    """Die ``id`` ist bereits vergeben."""
    IO = "io"
    """Datei oder Ordner konnte nicht gelesen werden."""


@dataclass(frozen=True)
class LoadError:
    """Ein gemeldeter Ladefehler.

    Attributes:
        source: Herkunft: Dateipfad des Plugins oder ``builtin:<modul>``.
        kind: Art des Fehlers.
        message: Verständliche Meldung auf Deutsch.
        strategy: Name der betroffenen Klasse, falls bekannt.
        line: Zeilennummer bei Syntaxfehlern, sonst ``None``.
    """

    source: str
    kind: LoadErrorKind
    message: str
    strategy: str | None = None
    line: int | None = None

    def __str__(self) -> str:
        where = self.source if self.line is None else f"{self.source}:{self.line}"
        return f"{where}: {self.message}"


@dataclass(frozen=True)
class StrategyInfo:
    """Eine geladene Strategie.

    Attributes:
        id: Eindeutige ID der Strategie.
        cls: Die Strategie-Klasse.
        source: ``"builtin"`` oder Pfad der Plugin-Datei.
        path: Pfad der Plugin-Datei; ``None`` bei eingebauten Strategien.
    """

    id: str
    cls: type[Strategy]
    source: str
    path: Path | None = None

    @property
    def is_builtin(self) -> bool:
        """``True`` für mitgelieferte Strategien."""
        return self.path is None

    @property
    def name(self) -> str:
        """Anzeigename der Strategie."""
        return self.cls.name


@dataclass(frozen=True)
class LoadReport:
    """Ergebnis eines Ladevorgangs.

    Attributes:
        loaded: IDs der erfolgreich geladenen Strategien in Ladereihenfolge.
        errors: Alle gemeldeten Fehler.
    """

    loaded: tuple[str, ...] = ()
    errors: tuple[LoadError, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """``True``, wenn keine Fehler gemeldet wurden."""
        return not self.errors

    def summary(self) -> str:
        """Kurze Zusammenfassung auf Deutsch, z. B. für die Statuszeile."""
        text = f"{len(self.loaded)} Strategien geladen"
        if self.errors:
            text += f", {len(self.errors)} Fehler"
        return text + "."


class StrategyNotFoundError(KeyError):
    """Unbekannte Strategie-ID."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class StrategyRegistry:
    """Verzeichnis aller verfügbaren Strategien.

    Beispiel::

        registry = StrategyRegistry(settings.plugins_path)
        report = registry.load()
        for error in report.errors:
            print(error)
        cls = registry.get("trend_donchian_breakout")
    """

    def __init__(
        self,
        plugins_dir: Path | None = None,
        *,
        include_builtin: bool = True,
        builtin_package: str = builtin.__name__,
    ) -> None:
        """Legt die Registry an, ohne schon zu laden (dafür :meth:`load`).

        Args:
            plugins_dir: Ordner mit Plugin-Dateien; ``None`` = keine Plugins.
            include_builtin: Eingebaute Strategien mitladen.
            builtin_package: Paket mit den eingebauten Strategien (für Tests änderbar).
        """
        self._plugins_dir = plugins_dir
        self._include_builtin = include_builtin
        self._builtin_package = builtin_package
        self._strategies: dict[str, StrategyInfo] = {}
        self._errors: tuple[LoadError, ...] = ()
        self._plugin_modules: list[str] = []
        self._generation = 0

    # -- Abfragen -------------------------------------------------------------

    @property
    def plugins_dir(self) -> Path | None:
        """Aktueller Plugin-Ordner."""
        return self._plugins_dir

    @plugins_dir.setter
    def plugins_dir(self, value: Path | None) -> None:
        """Setzt den Plugin-Ordner; wirksam ab dem nächsten :meth:`reload`."""
        self._plugins_dir = value

    @property
    def errors(self) -> tuple[LoadError, ...]:
        """Fehler des letzten Ladevorgangs."""
        return self._errors

    def ids(self) -> list[str]:
        """IDs aller geladenen Strategien in Ladereihenfolge."""
        return list(self._strategies)

    def list_all(self) -> list[StrategyInfo]:
        """Alle geladenen Strategien in Ladereihenfolge."""
        return list(self._strategies.values())

    def info(self, strategy_id: str) -> StrategyInfo:
        """Beschreibung der Strategie ``strategy_id``.

        Raises:
            StrategyNotFoundError: Die ID ist nicht geladen.
        """
        try:
            return self._strategies[strategy_id]
        except KeyError:
            known = ", ".join(sorted(self._strategies)) or "keine"
            raise StrategyNotFoundError(
                f"Unbekannte Strategie: {strategy_id!r}. Verfügbar: {known}."
            ) from None

    def get(self, strategy_id: str) -> type[Strategy]:
        """Klasse der Strategie ``strategy_id``.

        Raises:
            StrategyNotFoundError: Die ID ist nicht geladen.
        """
        return self.info(strategy_id).cls

    def create(self, strategy_id: str) -> Strategy:
        """Neue Instanz der Strategie ``strategy_id``.

        Raises:
            StrategyNotFoundError: Die ID ist nicht geladen.
        """
        return self.get(strategy_id)()

    def __contains__(self, strategy_id: object) -> bool:
        return strategy_id in self._strategies

    def __len__(self) -> int:
        return len(self._strategies)

    # -- Laden ----------------------------------------------------------------

    def load(self) -> LoadReport:
        """Lädt eingebaute Strategien und alle Plugins neu.

        Bisher geladene Strategien werden verworfen. Fehler werden protokolliert und im
        Bericht zurückgegeben, nicht als Ausnahme geworfen.

        Returns:
            Bericht mit geladenen IDs und Fehlern.
        """
        self._unload_plugins()
        self._generation += 1
        strategies: dict[str, StrategyInfo] = {}
        errors: list[LoadError] = []

        if self._include_builtin:
            for source, classes in self._collect_builtin(errors):
                self._register_all(classes, source, None, strategies, errors)

        for path in self._plugin_files(errors):
            module = self._load_plugin(path, errors)
            if module is None:
                continue
            classes = _strategy_classes(module)
            if not classes:
                errors.append(_no_strategy_error(str(path), module))
                continue
            self._register_all(classes, str(path), path, strategies, errors)

        self._strategies = strategies
        self._errors = tuple(errors)
        for error in errors:
            _log.warning("Strategie nicht geladen – %s", error)
        report = LoadReport(loaded=tuple(strategies), errors=self._errors)
        _log.info(report.summary())
        return report

    def reload(self) -> LoadReport:
        """Liest Plugins nach Änderungen erneut ein (ohne Neustart der App).

        Returns:
            Bericht wie bei :meth:`load`.
        """
        return self.load()

    # -- Interna --------------------------------------------------------------

    def _collect_builtin(
        self, errors: list[LoadError]
    ) -> Iterable[tuple[str, list[type[Strategy]]]]:
        package = importlib.import_module(self._builtin_package)
        search = getattr(package, "__path__", None)
        if search is None:
            return
        for module_info in sorted(pkgutil.iter_modules(search), key=lambda m: m.name):
            if module_info.name.startswith("_"):
                continue
            full_name = f"{self._builtin_package}.{module_info.name}"
            source = f"{BUILTIN_SOURCE}:{module_info.name}"
            try:
                module = importlib.import_module(full_name)
            except Exception as exc:
                errors.append(_exception_error(source, exc))
                continue
            yield source, _strategy_classes(module)

    def _plugin_files(self, errors: list[LoadError]) -> list[Path]:
        folder = self._plugins_dir
        if folder is None or not folder.exists():
            return []
        if not folder.is_dir():
            errors.append(LoadError(str(folder), LoadErrorKind.IO, "Plugin-Pfad ist kein Ordner."))
            return []
        return sorted(
            (p for p in folder.glob("*.py") if p.is_file() and not p.name.startswith("_")),
            key=lambda p: p.name,
        )

    def _load_plugin(self, path: Path, errors: list[LoadError]) -> types.ModuleType | None:
        source = str(path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(LoadError(source, LoadErrorKind.IO, f"Datei nicht lesbar: {exc}."))
            return None
        try:
            code = compile(text, source, "exec", dont_inherit=True)
        except SyntaxError as exc:
            errors.append(
                LoadError(
                    source,
                    LoadErrorKind.SYNTAX,
                    f"Syntaxfehler in Zeile {exc.lineno}: {exc.msg}.",
                    line=exc.lineno,
                )
            )
            return None

        # Eigener Modulname je Datei und Ladevorgang: Dateien beeinflussen sich nicht
        # gegenseitig, und ein Reload sieht nie eine veraltete Fassung.
        name = f"{_PLUGIN_PACKAGE}.g{self._generation}_{_safe_identifier(path.stem)}"
        module = types.ModuleType(name)
        module.__file__ = source
        sys.modules[name] = module  # nötig z. B. für @dataclass im Plugin
        try:
            exec(code, module.__dict__)
        except BaseException as exc:  # auch SystemExit aus einem Plugin abfangen
            sys.modules.pop(name, None)
            if isinstance(exc, KeyboardInterrupt):
                raise
            errors.append(_exception_error(source, exc))
            return None
        self._plugin_modules.append(name)
        return module

    def _unload_plugins(self) -> None:
        for name in self._plugin_modules:
            sys.modules.pop(name, None)
        self._plugin_modules.clear()

    def _register_all(
        self,
        classes: list[type[Strategy]],
        source: str,
        path: Path | None,
        strategies: dict[str, StrategyInfo],
        errors: list[LoadError],
    ) -> None:
        for cls in classes:
            problems = check_fields(cls)
            if problems:
                errors.extend(
                    LoadError(source, kind, message, strategy=cls.__name__)
                    for kind, message in problems
                )
                continue
            existing = strategies.get(cls.id)
            if existing is not None:
                errors.append(
                    LoadError(
                        source,
                        LoadErrorKind.DUPLICATE_ID,
                        f"Strategie-ID {cls.id!r} ist bereits vergeben "
                        f"(durch {existing.cls.__name__} aus {existing.source}).",
                        strategy=cls.__name__,
                    )
                )
                continue
            strategies[cls.id] = StrategyInfo(id=cls.id, cls=cls, source=source, path=path)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def check_fields(cls: type[Strategy]) -> list[tuple[LoadErrorKind, str]]:
    """Prüft die Pflichtfelder einer Strategie-Klasse.

    Args:
        cls: Zu prüfende Klasse.

    Returns:
        Liste aus (Fehlerart, Meldung); leer, wenn alles stimmt.
    """
    problems: list[tuple[LoadErrorKind, str]] = []
    missing = [f for f in REQUIRED_FIELDS if not hasattr(cls, f)]
    if missing:
        problems.append((LoadErrorKind.MISSING_FIELD, f"Pflichtfeld fehlt: {', '.join(missing)}."))
    attrs: Mapping[str, Any] = {f: getattr(cls, f) for f in REQUIRED_FIELDS if f not in missing}

    def invalid(message: str) -> None:
        problems.append((LoadErrorKind.INVALID_FIELD, message))

    if "id" in attrs:
        value = attrs["id"]
        if not isinstance(value, str) or not _ID_PATTERN.match(value):
            invalid(
                f"Ungültige id {value!r}: nur Kleinbuchstaben, Ziffern und _, "
                "beginnend mit einem Buchstaben."
            )
    for name in ("name", "version", "description"):
        if name in attrs and (not isinstance(attrs[name], str) or not attrs[name].strip()):
            invalid(f"Feld {name} muss ein nicht leerer Text sein.")
    if "timeframes" in attrs:
        tfs = attrs["timeframes"]
        if (
            not isinstance(tfs, list | tuple)
            or not tfs
            or not all(isinstance(t, Timeframe) for t in tfs)
        ):
            invalid("Feld timeframes muss eine nicht leere Liste von Timeframe-Werten sein.")
    params = getattr(cls, "params", None)
    if not isinstance(params, dict) or not all(
        isinstance(k, str) and isinstance(v, Param) for k, v in params.items()
    ):
        invalid("Feld params muss ein dict aus Name → Param sein.")
    return problems


def _strategy_classes(module: types.ModuleType) -> list[type[Strategy]]:
    """Konkrete Strategie-Klassen, die in ``module`` selbst definiert sind."""
    return [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and issubclass(obj, Strategy)
        and obj is not Strategy
        and obj.__module__ == module.__name__
        and not inspect.isabstract(obj)
    ]


def _no_strategy_error(source: str, module: types.ModuleType) -> LoadError:
    abstract = sorted(
        obj.__name__
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and issubclass(obj, Strategy)
        and obj.__module__ == module.__name__
        and inspect.isabstract(obj)
    )
    if abstract:
        return LoadError(
            source,
            LoadErrorKind.NO_STRATEGY,
            f"Strategie {', '.join(abstract)} ist unvollständig: Methode on_bar fehlt.",
            strategy=abstract[0],
        )
    return LoadError(
        source,
        LoadErrorKind.NO_STRATEGY,
        "Keine Strategie gefunden: Klasse von Strategy ableiten und on_bar implementieren.",
    )


def _exception_error(source: str, exc: BaseException) -> LoadError:
    line = _last_line(exc, source)
    where = f" in Zeile {line}" if line is not None else ""
    return LoadError(
        source,
        LoadErrorKind.EXECUTION,
        f"Fehler beim Laden{where}: {type(exc).__name__}: {exc}",
        line=line,
    )


def _last_line(exc: BaseException, filename: str) -> int | None:
    """Letzte Zeilennummer im Traceback, die zu ``filename`` gehört."""
    line: int | None = None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == filename:
            line = tb.tb_lineno
        tb = tb.tb_next
    return line


def _safe_identifier(text: str) -> str:
    cleaned = re.sub(r"\W", "_", text)
    return cleaned if cleaned and not cleaned[0].isdigit() else f"_{cleaned}"
