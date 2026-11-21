"""Registry der technischen Indikatoren.

Ein Indikator ist eine reine Funktion ``f(df, **params) -> pd.Series | pd.DataFrame``.
Sie erhält eine Kursreihe (Schema siehe :mod:`trading_helfer.core.bars`, bereinigte
Kurse) und liefert ein Ergebnis mit demselben Index wie die Eingabe. Die Anlaufphase
ist ``NaN``; der Wert bei ``t`` hängt nur von Kerzen bis einschließlich ``t`` ab.

Registriert wird mit dem Decorator :func:`indicator`::

    @indicator("sma", params={"period": 20, "source": "close"}, pane="price", outputs=["sma"])
    def sma(df: pd.DataFrame, period: int = 20, source: str = "close") -> pd.Series: ...

Indikatoren mit einer Ausgabe liefern eine ``pd.Series`` mit dem Namen der Ausgabe,
solche mit mehreren Ausgaben ein ``pd.DataFrame`` mit genau den Spalten ``outputs``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import pandas as pd

__all__ = [
    "IndicatorError",
    "IndicatorFunc",
    "IndicatorSpec",
    "Pane",
    "get",
    "indicator",
    "list_all",
]

Pane = Literal["price", "separate"]
"""Anzeigeart: ``price`` im Kurs-Chart, ``separate`` in einem eigenen Panel."""

IndicatorFunc = Callable[..., "pd.Series[float] | pd.DataFrame"]
"""Signatur einer Indikatorfunktion: ``f(df, **params)``."""

_PANES: frozenset[str] = frozenset({"price", "separate"})
_REGISTRY: dict[str, IndicatorSpec] = {}


class IndicatorError(ValueError):
    """Fehler bei Registrierung oder Aufruf eines Indikators; Text auf Deutsch."""


@dataclass(frozen=True)
class IndicatorSpec:
    """Beschreibung eines registrierten Indikators.

    Attributes:
        name: Eindeutiger Name, z. B. ``"sma"``.
        func: Die Indikatorfunktion.
        params: Parameternamen mit Standardwerten (unveränderlich).
        pane: Anzeigeart im Chart.
        outputs: Namen der Ausgaben in fester Reihenfolge.
    """

    name: str
    func: IndicatorFunc
    params: Mapping[str, Any] = field(default_factory=dict)
    pane: Pane = "separate"
    outputs: tuple[str, ...] = ()

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """Berechnet den Indikator und liefert immer ein DataFrame mit den Ausgaben.

        Nicht angegebene Parameter erhalten ihre Standardwerte.

        Args:
            df: Kursreihe (bereinigte Kurse).
            **params: Parameter, die vom Standard abweichen.

        Returns:
            DataFrame mit dem Index von ``df`` und den Spalten :attr:`outputs`.

        Raises:
            IndicatorError: Unbekannter Parameter.
        """
        unknown = sorted(set(params) - set(self.params))
        if unknown:
            raise IndicatorError(
                f"Unbekannte(r) Parameter für {self.name!r}: {', '.join(unknown)}."
            )
        result = self.func(df, **{**self.params, **params})
        if isinstance(result, pd.Series):
            return result.to_frame(self.outputs[0])
        return result.loc[:, list(self.outputs)]


def indicator(
    name: str,
    params: Mapping[str, Any] | None = None,
    pane: Pane = "separate",
    outputs: Iterable[str] | None = None,
) -> Callable[[IndicatorFunc], IndicatorFunc]:
    """Decorator, der eine Indikatorfunktion unter ``name`` registriert.

    Args:
        name: Eindeutiger Name (Kleinbuchstaben empfohlen).
        params: Parameternamen mit Standardwerten.
        pane: ``"price"`` (im Kurs-Chart) oder ``"separate"`` (eigenes Panel).
        outputs: Namen der Ausgaben; Standard ist ``[name]``.

    Returns:
        Decorator, der die Funktion unverändert zurückgibt.

    Raises:
        IndicatorError: Name schon vergeben, ungültige Anzeigeart oder leere Ausgaben.
    """
    if pane not in _PANES:
        raise IndicatorError(f"Ungültige Anzeigeart {pane!r}; erlaubt: price, separate.")
    output_names = tuple(outputs) if outputs is not None else (name,)
    if not output_names:
        raise IndicatorError(f"Indikator {name!r} braucht mindestens eine Ausgabe.")

    def decorator(func: IndicatorFunc) -> IndicatorFunc:
        if name in _REGISTRY:
            raise IndicatorError(f"Indikator {name!r} ist bereits registriert.")
        _REGISTRY[name] = IndicatorSpec(
            name=name,
            func=func,
            params=MappingProxyType(dict(params or {})),
            pane=pane,
            outputs=output_names,
        )
        return func

    return decorator


def get(name: str) -> IndicatorSpec:
    """Liefert den registrierten Indikator ``name``.

    Raises:
        IndicatorError: Kein Indikator mit diesem Namen.
    """
    _load_builtin()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise IndicatorError(f"Unbekannter Indikator {name!r}.") from None


def list_all() -> list[IndicatorSpec]:
    """Alle registrierten Indikatoren, sortiert nach Name."""
    _load_builtin()
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def _unregister(name: str) -> None:
    """Entfernt einen Indikator (nur für Tests)."""
    _REGISTRY.pop(name, None)


def _load_builtin() -> None:
    """Importiert die mitgelieferten Indikator-Module, damit sie registriert sind."""
    from trading_helfer.indicators import trend  # noqa: F401
