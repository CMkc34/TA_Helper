"""Fassade für Indikatoren im Chart: Auswahl, Parameterfelder, Berechnung, Vorlagen.

Das Modul importiert kein Qt. Die Parameterfelder des Dialogs „Indikator hinzufügen“
werden aus den Metadaten der Registry erzeugt (Name, Standardwert, Typ), daher braucht
ein neu registrierter Indikator keinen eigenen GUI-Code. Vorlagen (z. B. „Trend 1W“)
liegen als JSON-Datei neben dem GUI-Zustand und überdauern einen Neustart.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from trading_helfer.core.bars import PRICE_COLUMNS
from trading_helfer.core.markers import Marker
from trading_helfer.indicators import registry
from trading_helfer.indicators.registry import IndicatorError, IndicatorSpec

__all__ = [
    "LINE_COLORS",
    "SOURCE_CHOICES",
    "TEMPLATES_FILE_NAME",
    "IndicatorConfig",
    "IndicatorError",
    "IndicatorService",
    "ParamField",
    "TemplateError",
    "marker_indices",
]

logger = logging.getLogger(__name__)

TEMPLATES_FILE_NAME: Final = "indicator_templates.json"
"""Dateiname der Indikator-Vorlagen (neben ``gui_state.json``)."""

LINE_COLORS: Final[tuple[str, ...]] = (
    "#2962ff",
    "#ff6d00",
    "#ab47bc",
    "#00897b",
    "#f9a825",
    "#e91e63",
    "#6d4c41",
    "#546e7a",
)
"""Standardfarben der Linien in fester Reihenfolge."""

SOURCE_CHOICES: Final[tuple[str, ...]] = tuple(PRICE_COLUMNS)
"""Auswahl für den Parameter ``source`` (Kursspalte)."""

ParamKind = Literal["int", "float", "choice", "text"]

_MAX_TEMPLATE_NAME: Final = 60


class TemplateError(ValueError):
    """Ungültige Vorlagen-Aktion; die Meldung ist deutsch."""


@dataclass(frozen=True)
class ParamField:
    """Beschreibung eines Eingabefelds, abgeleitet aus den Registry-Metadaten.

    Attributes:
        name: Parametername.
        default: Standardwert aus der Registry.
        kind: ``int``, ``float``, ``choice`` (Auswahlliste) oder ``text``.
        choices: Erlaubte Werte bei ``choice``.
    """

    name: str
    default: Any
    kind: ParamKind
    choices: tuple[str, ...] = ()


class IndicatorConfig(BaseModel):
    """Ein Indikator, wie er im Chart steht: Name, Parameter, Farben je Ausgabe."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    colors: dict[str, str] = Field(default_factory=dict)


class _TemplateFile(BaseModel):
    templates: dict[str, list[IndicatorConfig]] = Field(default_factory=dict)


def _param_kind(name: str, default: Any) -> ParamKind:
    if name == "source":
        return "choice"
    if isinstance(default, bool):
        return "text"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    return "text"


def marker_indices(
    markers: Iterable[Marker], dates: npt.NDArray[np.datetime64]
) -> list[tuple[int, Marker]]:
    """Ordnet jedem Marker die Kerze zu, deren Zeitraum sein Datum enthält.

    Kerzendaten sind der letzte Handelstag ihres Zeitraums (siehe
    :mod:`trading_helfer.data.resample`); ein Marker gehört daher zur ersten Kerze mit
    Datum ≥ Markerdatum. Marker vor dem Zeitraum der ersten Kerze (bei Tageskerzen:
    vor deren Datum) oder nach der letzten Kerze werden weggelassen.

    Args:
        markers: Marker in beliebiger Reihenfolge.
        dates: Kerzendaten aufsteigend (``datetime64[D]``).

    Returns:
        Paare ``(Kerzenposition, Marker)`` in der Reihenfolge der Eingabe.
    """
    if len(dates) == 0:
        return []
    days = dates.astype("datetime64[D]")
    # Länge des ersten Zeitraums schätzen: Abstand der ersten beiden Kerzen.
    first_span = days[1] - days[0] if len(days) > 1 else np.timedelta64(0, "D")
    result: list[tuple[int, Marker]] = []
    for marker in markers:
        day = np.datetime64(marker.date, "D")
        index = int(np.searchsorted(days, day, side="left"))
        if index >= len(days):
            continue
        if index == 0 and day < days[0] - first_span:
            continue
        result.append((index, marker))
    return result


class IndicatorService:
    """Liefert Indikatoren aus der Registry, berechnet sie und verwaltet Vorlagen.

    Args:
        templates_path: JSON-Datei für Vorlagen; ``None`` hält Vorlagen nur im Speicher.
    """

    def __init__(self, templates_path: Path | None = None) -> None:
        self._templates_path = templates_path
        self._templates = self._load_templates()

    # -- Registry --------------------------------------------------------------------

    @staticmethod
    def available() -> list[IndicatorSpec]:
        """Alle registrierten Indikatoren, sortiert nach Name."""
        return registry.list_all()

    @staticmethod
    def spec(name: str) -> IndicatorSpec:
        """Registry-Eintrag des Indikators ``name``.

        Raises:
            IndicatorError: Unbekannter Indikator.
        """
        return registry.get(name)

    def param_fields(self, name: str) -> list[ParamField]:
        """Eingabefelder für die Parameter von ``name`` in Registry-Reihenfolge."""
        fields: list[ParamField] = []
        for param, default in self.spec(name).params.items():
            kind = _param_kind(param, default)
            choices = SOURCE_CHOICES if kind == "choice" else ()
            fields.append(ParamField(param, default, kind, choices))
        return fields

    def default_config(self, name: str, color_offset: int = 0) -> IndicatorConfig:
        """Konfiguration mit Standardparametern und Standardfarben.

        Args:
            name: Indikatorname.
            color_offset: Erste Farbe aus :data:`LINE_COLORS` (z. B. Zahl der Linien
                im Chart), damit neue Linien sich unterscheiden.
        """
        spec = self.spec(name)
        colors = {
            output: LINE_COLORS[(color_offset + i) % len(LINE_COLORS)]
            for i, output in enumerate(spec.visible_outputs())
        }
        return IndicatorConfig(name=name, params=dict(spec.params), colors=colors)

    def normalize(self, config: IndicatorConfig, color_offset: int = 0) -> IndicatorConfig:
        """Prüft eine Konfiguration und ergänzt fehlende Parameter und Farben.

        Zahlen werden in den Typ des Standardwerts umgewandelt (``int``/``float``).

        Raises:
            IndicatorError: Unbekannter Indikator, unbekannter Parameter oder Wert vom
                falschen Typ.
        """
        base = self.default_config(config.name, color_offset)
        unknown = sorted(set(config.params) - set(base.params))
        if unknown:
            raise IndicatorError(
                f"Unbekannte(r) Parameter für {config.name!r}: {', '.join(unknown)}."
            )
        params = dict(base.params)
        for field in self.param_fields(config.name):
            if field.name in config.params:
                params[field.name] = self._coerce(config.name, field, config.params[field.name])
        colors = {
            output: config.colors.get(output) or color for output, color in base.colors.items()
        }
        return IndicatorConfig(name=config.name, params=params, colors=colors)

    @staticmethod
    def _coerce(name: str, field: ParamField, value: Any) -> Any:
        try:
            if field.kind == "int":
                number = float(value)
                if not number.is_integer():
                    raise ValueError(value)
                return int(number)
            if field.kind == "float":
                return float(value)
        except (TypeError, ValueError):
            raise IndicatorError(
                f"Parameter {field.name!r} von {name!r} muss eine Zahl sein, nicht {value!r}."
            ) from None
        if field.kind == "choice" and value not in field.choices:
            raise IndicatorError(
                f"Parameter {field.name!r} von {name!r}: {value!r} ist nicht erlaubt "
                f"({', '.join(field.choices)})."
            )
        return value

    def label(self, config: IndicatorConfig) -> str:
        """Anzeigename, z. B. ``SMA(20)`` oder ``MACD(12, 26, 9, close)``."""
        spec = self.spec(config.name)
        values = [config.params.get(name, default) for name, default in spec.params.items()]
        args = ", ".join(_format_param(value) for value in values)
        return f"{config.name.upper()}({args})" if args else config.name.upper()

    def compute(self, config: IndicatorConfig, bars: pd.DataFrame) -> pd.DataFrame:
        """Berechnet den Indikator auf ``bars``.

        Returns:
            DataFrame mit dem Index von ``bars`` und allen Ausgaben als Spalten.

        Raises:
            IndicatorError: Ungültige Parameter oder Fehler bei der Berechnung.
        """
        spec = self.spec(config.name)
        try:
            return spec.compute(bars, **config.params)
        except IndicatorError:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise IndicatorError(
                f"{self.label(config)} konnte nicht berechnet werden: {exc}"
            ) from exc

    # -- Vorlagen --------------------------------------------------------------------

    @property
    def templates_path(self) -> Path | None:
        """Datei der Vorlagen (``None``: nur im Speicher)."""
        return self._templates_path

    def _load_templates(self) -> dict[str, list[IndicatorConfig]]:
        path = self._templates_path
        if path is None or not path.exists():
            return {}
        try:
            data = _TemplateFile.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, ValidationError) as exc:
            logger.warning("Indikator-Vorlagen %s nicht lesbar: %s", path, exc)
            return {}
        return data.templates

    def _save_templates(self) -> None:
        path = self._templates_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                _TemplateFile(templates=self._templates).model_dump_json(indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Indikator-Vorlagen %s können nicht gespeichert werden: %s", path, exc)

    def template_names(self) -> list[str]:
        """Namen aller Vorlagen, alphabetisch."""
        return sorted(self._templates, key=str.casefold)

    def template(self, name: str) -> list[IndicatorConfig]:
        """Indikatoren der Vorlage ``name`` (Kopien).

        Raises:
            TemplateError: Vorlage gibt es nicht.
        """
        try:
            configs = self._templates[name]
        except KeyError:
            raise TemplateError(f"Vorlage „{name}“ gibt es nicht.") from None
        return [config.model_copy(deep=True) for config in configs]

    def save_template(self, name: str, configs: Sequence[IndicatorConfig]) -> str:
        """Speichert ``configs`` unter ``name`` (überschreibt eine gleichnamige Vorlage).

        Returns:
            Bereinigter Name.

        Raises:
            TemplateError: Name leer oder zu lang, oder keine Indikatoren.
        """
        clean = name.strip()
        if not clean:
            raise TemplateError("Bitte einen Namen für die Vorlage eingeben.")
        if len(clean) > _MAX_TEMPLATE_NAME:
            raise TemplateError(
                f"Der Name der Vorlage darf höchstens {_MAX_TEMPLATE_NAME} Zeichen haben."
            )
        if not configs:
            raise TemplateError("Die Vorlage braucht mindestens einen Indikator.")
        self._templates[clean] = [config.model_copy(deep=True) for config in configs]
        self._save_templates()
        return clean

    def delete_template(self, name: str) -> None:
        """Löscht die Vorlage ``name``; unbekannte Namen werden ignoriert."""
        if self._templates.pop(name, None) is not None:
            self._save_templates()


def _format_param(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}".replace(".", ",")
    return str(value)
