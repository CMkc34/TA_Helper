"""Fassade für den Strategien-Tab: Liste, Details, Prüfen und Hinzufügen von Plugins.

:class:`StrategyService` verbindet die GUI mit der :class:`StrategyRegistry` und dem
Validator. Das Modul importiert kein Qt.

Ablauf „Strategie hinzufügen“:

1. Code aus einer Datei lesen (:meth:`StrategyService.read_file`) oder einfügen.
2. Mit :meth:`StrategyService.validate` prüfen (lange Aufgabe, gehört in den
   Hintergrund-Thread) und den Bericht anzeigen.
3. Mit :meth:`StrategyService.save_plugin` als ``plugins/<id>.py`` speichern. Gibt es
   die ID schon als Plugin, wirft der erste Aufruf :class:`StrategyExistsError`; nach
   Rückfrage speichert ``replace=True`` und sichert die alte Datei vorher als
   ``plugins/<id>_v<version>.py.bak``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from trading_helfer.strategies.base import Param
from trading_helfer.strategies.registry import (
    LoadError,
    LoadReport,
    StrategyInfo,
    StrategyRegistry,
)
from trading_helfer.strategies.validator import ValidationReport, validate

__all__ = [
    "BACKUP_SUFFIX",
    "MAX_SOURCE_BYTES",
    "Origin",
    "ParamRow",
    "SaveResult",
    "StrategyDetails",
    "StrategyExistsError",
    "StrategyRow",
    "StrategySaveError",
    "StrategyService",
    "StrategyStatus",
    "backup_path",
    "format_param_value",
]

logger = logging.getLogger(__name__)

BACKUP_SUFFIX: Final = ".py.bak"
"""Endung gesicherter Plugin-Dateien (werden von der Registry nicht geladen)."""

MAX_SOURCE_BYTES: Final = 1_000_000
"""Größte Strategie-Datei, die eingelesen wird (1 MB)."""

Validator = Callable[..., ValidationReport]
"""Signatur wie :func:`trading_helfer.strategies.validator.validate`."""


class Origin(StrEnum):
    """Herkunft einer Strategie."""

    BUILTIN = "eingebaut"
    PLUGIN = "Plugin"


class StrategyStatus(StrEnum):
    """Zustand einer Zeile in der Strategienliste."""

    LOADED = "geladen"
    ERROR = "Fehler"


@dataclass(frozen=True)
class StrategyRow:
    """Eine Zeile der Strategienliste.

    Ladefehler einer Plugin-Datei erscheinen als eigene Zeile mit Status
    :attr:`StrategyStatus.ERROR`; ``key`` ist dann der Dateipfad.

    Attributes:
        key: Strategie-ID oder (bei Fehlern) Pfad der Datei.
        name: Anzeigename; bei Fehlern der Dateiname.
        version: SemVer-Version; leer bei Fehlern.
        timeframes: Zeitrahmen, z. B. ``"3D, 1W"``; leer bei Fehlern.
        origin: Eingebaut oder Plugin.
        status: Geladen oder Fehler.
        message: Fehlermeldungen (leer, wenn geladen).
    """

    key: str
    name: str
    version: str
    timeframes: str
    origin: Origin
    status: StrategyStatus
    message: str = ""

    @property
    def ok(self) -> bool:
        """``True``, wenn die Strategie geladen ist und genutzt werden kann."""
        return self.status is StrategyStatus.LOADED


@dataclass(frozen=True)
class ParamRow:
    """Ein Parameter für die Detailansicht (alle Werte als Text)."""

    name: str
    default: str
    minimum: str
    maximum: str
    step: str
    description: str


@dataclass(frozen=True)
class StrategyDetails:
    """Detailansicht einer Zeile der Strategienliste.

    Attributes:
        row: Die Zeile selbst.
        description: Beschreibung der Strategie (bei Fehlern leer).
        params: Parameter in Definitionsreihenfolge.
        path: Plugin-Datei; ``None`` bei eingebauten Strategien.
        class_name: Name der Python-Klasse (bei Fehlern leer).
    """

    row: StrategyRow
    description: str = ""
    params: tuple[ParamRow, ...] = ()
    path: Path | None = None
    class_name: str = ""


@dataclass(frozen=True)
class SaveResult:
    """Ergebnis von :meth:`StrategyService.save_plugin`.

    Attributes:
        path: Gespeicherte Plugin-Datei.
        strategy_ids: IDs der Strategien in der Datei.
        backups: Sicherungen ersetzter Dateien.
        load_report: Bericht des anschließenden Neuladens.
    """

    path: Path
    strategy_ids: tuple[str, ...]
    backups: tuple[Path, ...] = ()
    load_report: LoadReport = field(default_factory=LoadReport)


class StrategySaveError(ValueError):
    """Die Strategie kann nicht gespeichert werden; die Meldung ist deutsch."""


class StrategyExistsError(StrategySaveError):
    """Eine Strategie mit gleicher ID ist schon als Plugin vorhanden (Rückfrage nötig).

    Attributes:
        existing: Die vorhandenen Plugin-Strategien, die ersetzt würden.
    """

    def __init__(self, existing: tuple[StrategyInfo, ...]) -> None:
        self.existing = existing
        names = ", ".join(f"„{i.id}“ (Version {i.cls.version})" for i in existing)
        super().__init__(f"Strategie {names} ist bereits vorhanden.")


def format_param_value(value: Any) -> str:
    """Formatiert einen Parameterwert für die Anzeige (``None`` → leer, ``True`` → „ja“)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "ja" if value else "nein"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _param_row(name: str, param: Param) -> ParamRow:
    default = format_param_value(param.default)
    if param.choices is not None:
        default += " (Auswahl: " + ", ".join(format_param_value(c) for c in param.choices) + ")"
    return ParamRow(
        name=name,
        default=default,
        minimum=format_param_value(param.min),
        maximum=format_param_value(param.max),
        step=format_param_value(param.step),
        description=param.description,
    )


def backup_path(folder: Path, strategy_id: str, version: str) -> Path:
    """Freier Name für die Sicherung, z. B. ``<id>_v1.0.0.py.bak`` (sonst mit Zähler)."""
    candidate = folder / f"{strategy_id}_v{version}{BACKUP_SUFFIX}"
    counter = 2
    while candidate.exists():
        candidate = folder / f"{strategy_id}_v{version}_{counter}{BACKUP_SUFFIX}"
        counter += 1
    return candidate


class StrategyService:
    """Einzige Schnittstelle des Strategien-Tabs zur Logik.

    Args:
        plugins_dir: Plugin-Ordner (``Settings.plugins_path``); wird beim Speichern
            angelegt.
        registry: Vorhandene Registry; Standard ist eine neue für ``plugins_dir``.
        validator: Prüffunktion; Standard :func:`~trading_helfer.strategies.validator.validate`
            (eigener Prozess mit Zeitlimit).
    """

    def __init__(
        self,
        plugins_dir: Path,
        *,
        registry: StrategyRegistry | None = None,
        validator: Validator = validate,
    ) -> None:
        self._plugins_dir = plugins_dir
        self._registry = registry if registry is not None else StrategyRegistry(plugins_dir)
        self._validator = validator
        self._loaded = False

    @property
    def plugins_dir(self) -> Path:
        """Ordner, in dem neue Strategien gespeichert werden."""
        return self._plugins_dir

    @property
    def registry(self) -> StrategyRegistry:
        """Die Strategie-Registry (z. B. für den Backtest-Tab)."""
        return self._registry

    # -- Liste und Details -----------------------------------------------------------

    def reload(self) -> LoadReport:
        """Lädt eingebaute Strategien und Plugins neu."""
        report = self._registry.reload()
        self._loaded = True
        return report

    def list_strategies(self) -> list[StrategyRow]:
        """Alle Strategien (eingebaute zuerst) und danach Plugin-Dateien mit Ladefehlern.

        Lädt die Registry beim ersten Aufruf.
        """
        self._ensure_loaded()
        rows = [self._row(info) for info in self._registry.list_all()]
        for source, errors in self._errors_by_source().items():
            rows.append(
                StrategyRow(
                    key=source,
                    name=Path(source).name,
                    version="",
                    timeframes="",
                    origin=Origin.PLUGIN,
                    status=StrategyStatus.ERROR,
                    message="\n".join(e.message for e in errors),
                )
            )
        return rows

    def details(self, key: str) -> StrategyDetails:
        """Details zur Zeile ``key`` (Strategie-ID oder Pfad einer fehlerhaften Datei).

        Raises:
            KeyError: ``key`` ist weder geladen noch eine fehlerhafte Datei.
        """
        self._ensure_loaded()
        if key in self._registry:
            info = self._registry.info(key)
            params = getattr(info.cls, "params", {}) or {}
            return StrategyDetails(
                row=self._row(info),
                description=info.cls.description,
                params=tuple(_param_row(n, p) for n, p in params.items()),
                path=info.path,
                class_name=info.cls.__name__,
            )
        for row in self.list_strategies():
            if row.key == key:
                return StrategyDetails(row=row, path=Path(key))
        raise KeyError(key)

    # -- Hinzufügen ------------------------------------------------------------------

    def read_file(self, path: Path) -> str:
        """Liest eine Strategie-Datei (UTF-8).

        Raises:
            StrategySaveError: Datei fehlt, ist zu groß oder kein UTF-8-Text.
        """
        try:
            if path.stat().st_size > MAX_SOURCE_BYTES:
                raise StrategySaveError(
                    f"„{path.name}“ ist größer als {MAX_SOURCE_BYTES // 1_000_000} MB."
                )
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise StrategySaveError(f"„{path.name}“ ist keine UTF-8-Textdatei.") from None
        except OSError as exc:
            raise StrategySaveError(f"„{path.name}“ kann nicht gelesen werden: {exc}") from exc

    def validate(self, source_code: str, filename: str = "<eingefügt>") -> ValidationReport:
        """Prüft den Code mit dem Validator (dauert Sekunden: im Hintergrund aufrufen)."""
        return self._validator(source_code, filename=filename)

    def conflicts(self, report: ValidationReport) -> tuple[StrategyInfo, ...]:
        """Vorhandene Plugin-Strategien mit einer der IDs aus ``report``."""
        self._ensure_loaded()
        return tuple(
            self._registry.info(i)
            for i in report.strategy_ids
            if i in self._registry and not self._registry.info(i).is_builtin
        )

    def save_plugin(
        self, source_code: str, report: ValidationReport, *, replace: bool = False
    ) -> SaveResult:
        """Speichert geprüften Code als ``plugins/<id>.py`` und lädt die Registry neu.

        Args:
            source_code: Der geprüfte Code (genau der, zu dem ``report`` gehört).
            report: Bericht des Validators; muss bestanden sein.
            replace: Vorhandene Plugins mit gleicher ID ersetzen; die alten Dateien
                werden vorher als ``<id>_v<version>.py.bak`` gesichert.

        Raises:
            StrategySaveError: Bericht nicht bestanden, ID gehört einer eingebauten
                Strategie, die alte Datei enthält weitere Strategien oder Schreibfehler.
            StrategyExistsError: Die ID ist schon als Plugin vorhanden und ``replace``
                ist ``False``.
        """
        self._ensure_loaded()
        if not report.ok or not report.strategy_ids:
            raise StrategySaveError(
                "Nur geprüfte Strategien ohne Fehler können gespeichert werden."
            )
        ids = report.strategy_ids
        builtin = [i for i in ids if i in self._registry and self._registry.info(i).is_builtin]
        if builtin:
            raise StrategySaveError(
                f"Die ID „{builtin[0]}“ gehört zu einer eingebauten Strategie. "
                "Bitte im Code eine andere id wählen."
            )
        existing = self.conflicts(report)
        if existing and not replace:
            raise StrategyExistsError(existing)
        self._check_shared_files(existing, ids)

        folder = self._plugins_dir
        target = folder / f"{ids[0]}.py"
        backups: list[Path] = []
        try:
            folder.mkdir(parents=True, exist_ok=True)
            done: set[Path] = set()
            for info in existing:
                assert info.path is not None
                if info.path in done:
                    continue
                done.add(info.path)
                backups.append(self._backup(info.path, info.id, info.cls.version))
            if target.exists():
                # Datei gleichen Namens, die nicht (fehlerfrei) geladen war.
                backups.append(self._backup(target, ids[0], "unbekannt"))
            target.write_text(source_code, encoding="utf-8", newline="\n")
        except OSError as exc:
            raise StrategySaveError(f"Speichern in {folder} fehlgeschlagen: {exc}") from exc

        logger.info("Strategie %s gespeichert: %s", ", ".join(ids), target)
        load_report = self.reload()
        return SaveResult(target, ids, tuple(backups), load_report)

    # -- Interna ---------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()

    def _errors_by_source(self) -> dict[str, list[LoadError]]:
        grouped: dict[str, list[LoadError]] = {}
        for error in self._registry.errors:
            grouped.setdefault(error.source, []).append(error)
        return grouped

    @staticmethod
    def _row(info: StrategyInfo) -> StrategyRow:
        return StrategyRow(
            key=info.id,
            name=info.cls.name,
            version=info.cls.version,
            timeframes=", ".join(str(t) for t in info.cls.timeframes),
            origin=Origin.BUILTIN if info.is_builtin else Origin.PLUGIN,
            status=StrategyStatus.LOADED,
        )

    def _check_shared_files(self, existing: tuple[StrategyInfo, ...], ids: tuple[str, ...]) -> None:
        paths = {info.path for info in existing}
        others = [
            info.id
            for info in self._registry.list_all()
            if info.path in paths and info.id not in ids
        ]
        if others:
            raise StrategySaveError(
                "Die vorhandene Datei enthält weitere Strategien "
                f"({', '.join(others)}), die verloren gingen. Bitte die Datei von Hand anpassen."
            )

    def _backup(self, path: Path, strategy_id: str, version: str) -> Path:
        destination = backup_path(self._plugins_dir, strategy_id, version)
        path.replace(destination)
        logger.info("Alte Fassung gesichert: %s", destination)
        return destination
