"""Strategie-Schnittstelle: ``Strategy``, Parameter, Orders und Kerzen-Kontext.

Eine Strategie leitet von :class:`Strategy` ab, beschreibt ihre Parameter mit
:class:`Param` und liefert je Kerze eine Liste von :class:`Order`\\ s. Die Engine ruft
zuerst :meth:`Strategy.indicators` (vektorisiert über die ganze Reihe) und danach je
Kerze ``t`` :meth:`Strategy.on_bar` mit einem :class:`BarContext` auf.

Der Kontext zeigt nur Daten bis einschließlich ``t`` und ist schreibgeschützt: Kurse
und Indikatoren liegen als nicht beschreibbare ``numpy``-Arrays vor. So kann eine
Strategie weder in die Zukunft schauen noch die Daten der Engine verändern.

Beispiel::

    class SmaCross(Strategy):
        id = "sma_cross"
        name = "SMA-Kreuzung"
        version = "1.0.0"
        description = "Kauft, wenn die schnelle SMA die langsame von unten kreuzt."
        timeframes = [Timeframe.D3, Timeframe.W1]
        params = {
            "fast": Param(10, 2, 50, 1, "Periode der schnellen SMA"),
            "slow": Param(30, 10, 200, 5, "Periode der langsamen SMA"),
        }

        def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
            close = bars["close"]
            return pd.DataFrame(
                {"fast": close.rolling(p.fast).mean(), "slow": close.rolling(p.slow).mean()}
            )

        def on_bar(self, ctx: BarContext) -> list[Order]:
            if ctx.position is None and ctx.crossed_above("fast", "slow"):
                return [ctx.buy(sizing=Sizing.percent_equity(100))]
            if ctx.position is not None and ctx.crossed_below("fast", "slow"):
                return ctx.exit()
            return []
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date as Date  # noqa: N812
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from trading_helfer.core.bars import DATE_COLUMN
from trading_helfer.core.types import OrderType, Side, Timeframe

__all__ = [
    "POSITION_SUFFIX",
    "RANK_SUFFIX",
    "BarContext",
    "BarsView",
    "IndicatorView",
    "Order",
    "OrderError",
    "OrderSide",
    "Param",
    "ParamError",
    "ParamKind",
    "Params",
    "PlotSpec",
    "Position",
    "PreparedSeries",
    "Sizing",
    "SizingMethod",
    "Strategy",
]

RANK_SUFFIX: Final = "_rank"
"""Endung der Spalte mit dem Perzentilrang einer ``rank_columns``-Spalte."""

POSITION_SUFFIX: Final = "_pos"
"""Endung der Spalte mit dem Platz (1 = höchster Wert) einer ``rank_columns``-Spalte."""

ParamKind = Literal["int", "float", "bool", "choice"]
"""Typ eines Parameters."""

Operand = str | float | int
"""Operand für Kreuzungsprüfungen: Spaltenname (Indikator oder Kurs) oder Konstante."""

FloatArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------


class ParamError(ValueError):
    """Ungültige Parameterdefinition oder ungültiger Parameterwert; Text auf Deutsch."""


@dataclass(frozen=True)
class Param:
    """Beschreibung eines Strategie-Parameters.

    Der Typ wird aus ``default`` abgeleitet (``bool``, ``int``, ``float``), sofern
    ``choices`` nicht gesetzt ist; mit ``choices`` ist der Typ ``choice``.

    Attributes:
        default: Standardwert.
        min: Kleinster erlaubter Wert (nur ``int``/``float``).
        max: Größter erlaubter Wert (nur ``int``/``float``).
        step: Schrittweite für Optimierungsraster (nur ``int``/``float``).
        description: Beschreibung für die GUI, auf Deutsch.
        choices: Erlaubte Werte für Auswahlparameter.
    """

    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    description: str = ""
    choices: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if self.choices is not None:
            object.__setattr__(self, "choices", tuple(self.choices))
        kind = self.kind
        if kind in ("bool", "choice"):
            if self.min is not None or self.max is not None or self.step is not None:
                raise ParamError(f"Parameter vom Typ {kind!r} hat kein min, max oder step.")
            if kind == "choice":
                if not self.choices:
                    raise ParamError("Auswahlparameter braucht mindestens eine Auswahl.")
                if self.default not in self.choices:
                    raise ParamError(
                        f"Standardwert {self.default!r} ist keine der Auswahlen {self.choices!r}."
                    )
            return
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ParamError(f"min ({self.min}) ist größer als max ({self.max}).")
        if self.step is not None and not self.step > 0:
            raise ParamError(f"step muss größer als 0 sein, erhalten: {self.step}.")
        self.validate(self.default, "default")

    @property
    def kind(self) -> ParamKind:
        """Typ des Parameters: ``int``, ``float``, ``bool`` oder ``choice``."""
        if self.choices is not None:
            return "choice"
        if isinstance(self.default, bool):
            return "bool"
        if isinstance(self.default, int):
            return "int"
        if isinstance(self.default, float):
            return "float"
        raise ParamError(
            f"Standardwert muss int, float oder bool sein (sonst choices angeben), "
            f"erhalten: {type(self.default).__name__}."
        )

    def validate(self, value: Any, name: str = "Parameter") -> Any:
        """Prüft ``value`` gegen Typ und Grenzen und gibt den normalisierten Wert zurück.

        ``int``-Werte werden für ``float``-Parameter akzeptiert und in ``float``
        umgewandelt; ganzzahlige ``float``-Werte (z. B. ``20.0``) werden für
        ``int``-Parameter zu ``int``.

        Args:
            value: Zu prüfender Wert.
            name: Parametername für die Fehlermeldung.

        Returns:
            Der geprüfte Wert im Typ des Parameters.

        Raises:
            ParamError: Falscher Typ, außerhalb von min/max oder keine erlaubte Auswahl.
        """
        kind = self.kind
        if kind == "choice":
            assert self.choices is not None
            if value not in self.choices:
                raise ParamError(
                    f"{name}: {value!r} ist nicht erlaubt; Auswahl: "
                    f"{', '.join(repr(c) for c in self.choices)}."
                )
            return value
        if kind == "bool":
            if not isinstance(value, bool | np.bool_):
                raise ParamError(f"{name}: erwartet wahr/falsch, erhalten: {value!r}.")
            return bool(value)
        if isinstance(value, bool | np.bool_) or not isinstance(
            value, int | float | np.integer | np.floating
        ):
            raise ParamError(f"{name}: erwartet eine Zahl, erhalten: {value!r}.")
        number: int | float
        if kind == "int":
            if isinstance(value, float | np.floating) and not float(value).is_integer():
                raise ParamError(f"{name}: erwartet eine ganze Zahl, erhalten: {value!r}.")
            number = int(value)
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ParamError(f"{name}: erwartet eine endliche Zahl, erhalten: {value!r}.")
        if self.min is not None and number < self.min:
            raise ParamError(f"{name}: {number} liegt unter dem Minimum {self.min}.")
        if self.max is not None and number > self.max:
            raise ParamError(f"{name}: {number} liegt über dem Maximum {self.max}.")
        return number


class Params(Mapping[str, Any]):
    """Geprüfte, unveränderliche Parameterwerte einer Strategie mit Attributzugriff.

    ``p.fast`` und ``p["fast"]`` liefern denselben Wert.
    """

    __slots__ = ("_values",)

    _values: Mapping[str, Any]

    def __init__(self, spec: Mapping[str, Param], values: Mapping[str, Any] | None = None) -> None:
        """Prüft ``values`` gegen ``spec``; fehlende Werte erhalten den Standardwert.

        Args:
            spec: Parameterdefinitionen der Strategie.
            values: Abweichende Werte.

        Raises:
            ParamError: Unbekannter Parameter oder ungültiger Wert.
        """
        values = dict(values or {})
        unknown = sorted(set(values) - set(spec))
        if unknown:
            raise ParamError(f"Unbekannte(r) Parameter: {', '.join(unknown)}.")
        checked = {
            name: param.validate(values.get(name, param.default), name)
            for name, param in spec.items()
        }
        object.__setattr__(self, "_values", MappingProxyType(checked))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(f"Unbekannter Parameter: {name!r}.") from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Params sind schreibgeschützt.")

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Params):
            return dict(self._values) == dict(other._values)
        if isinstance(other, Mapping):
            return dict(self._values) == dict(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(sorted((k, repr(v)) for k, v in self._values.items())))

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"Params({inner})"

    def to_dict(self) -> dict[str, Any]:
        """Werte als neues ``dict`` (z. B. für ``params_json``)."""
        return dict(self._values)


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


class OrderError(ValueError):
    """Ungültige Order; Text auf Deutsch."""


class OrderSide(StrEnum):
    """Richtung einer Order."""

    BUY = "BUY"
    """Long-Position eröffnen oder aufstocken."""
    SELL = "SELL"
    """Long-Position (teilweise) schließen."""
    SELL_SHORT = "SELL_SHORT"
    """Short-Position eröffnen oder aufstocken."""
    BUY_TO_COVER = "BUY_TO_COVER"
    """Short-Position (teilweise) schließen."""

    @property
    def is_entry(self) -> bool:
        """``True`` für Orders, die eine Position eröffnen oder vergrößern."""
        return self in (OrderSide.BUY, OrderSide.SELL_SHORT)

    @property
    def position_side(self) -> Side:
        """Richtung der Position, auf die sich die Order bezieht."""
        return Side.LONG if self in (OrderSide.BUY, OrderSide.SELL) else Side.SHORT


class SizingMethod(StrEnum):
    """Methoden der Positionsgröße (siehe Spezifikation, Backtest-Engine)."""

    FIXED_AMOUNT = "FIXED_AMOUNT"
    """Fester Betrag in Kontowährung."""
    PERCENT_EQUITY = "PERCENT_EQUITY"
    """Prozent des aktuellen Kapitals."""
    RISK = "RISK"
    """Prozent des Kapitals als Risiko je Trade, bezogen auf den Abstand zum Stop."""
    VOLATILITY_TARGET = "VOLATILITY_TARGET"
    """Ziel-Volatilität der Position in Prozent p. a."""


@dataclass(frozen=True)
class Sizing:
    """Regel für die Positionsgröße, die der Broker beim Ausführen in eine Stückzahl umrechnet.

    Attributes:
        method: Methode der Positionsgröße.
        value: Betrag (``FIXED_AMOUNT``) oder Prozentwert (alle anderen Methoden).
    """

    method: SizingMethod
    value: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value <= 0:
            raise OrderError(f"Positionsgröße muss größer als 0 sein, erhalten: {self.value}.")
        if self.method is SizingMethod.PERCENT_EQUITY and self.value > 100:
            raise OrderError(f"Prozent des Kapitals höchstens 100, erhalten: {self.value}.")

    @classmethod
    def fixed_amount(cls, amount: float) -> Sizing:
        """Fester Betrag in Kontowährung."""
        return cls(SizingMethod.FIXED_AMOUNT, float(amount))

    @classmethod
    def percent_equity(cls, percent: float) -> Sizing:
        """Prozent des aktuellen Kapitals (0 < percent ≤ 100)."""
        return cls(SizingMethod.PERCENT_EQUITY, float(percent))

    @classmethod
    def risk(cls, percent: float) -> Sizing:
        """Prozent des Kapitals als Risiko je Trade (braucht einen Stop)."""
        return cls(SizingMethod.RISK, float(percent))

    @classmethod
    def volatility_target(cls, percent: float) -> Sizing:
        """Ziel-Volatilität der Position in Prozent p. a."""
        return cls(SizingMethod.VOLATILITY_TARGET, float(percent))


@dataclass(frozen=True)
class Order:
    """Auftrag einer Strategie; wird von der Engine zum Open der nächsten Kerze ausgeführt.

    Entweder ``qty`` oder ``sizing`` wird angegeben. Bei schließenden Orders
    (``SELL``, ``BUY_TO_COVER``) ohne beides wird die ganze Position geschlossen.

    Attributes:
        symbol: Wertpapier.
        side: Richtung der Order.
        type: Order-Typ.
        qty: Stückzahl (Bruchstücke erlaubt), > 0.
        sizing: Regel für die Positionsgröße statt fester Stückzahl.
        stop_price: Auslösekurs für ``STOP`` (und Startwert für ``TRAILING_STOP``).
        limit_price: Limitkurs für ``LIMIT``.
        trail_pct: Abstand des Trailing Stops in Prozent.
        trail_atr: Abstand des Trailing Stops in ATR-Vielfachen.
        tag: Freier Text, z. B. Grund des Signals (erscheint in der Trade-Liste).
        strength: Signalstärke für die Rangfolge im Portfolio-Backtest (größer = besser),
            wenn es mehr Einstiegssignale als freie Plätze gibt; ``None`` = keine Angabe.
    """

    symbol: str
    side: OrderSide
    type: OrderType = OrderType.MARKET
    qty: float | None = None
    sizing: Sizing | None = None
    stop_price: float | None = None
    limit_price: float | None = None
    trail_pct: float | None = None
    trail_atr: float | None = None
    tag: str = ""
    strength: float | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise OrderError("Order braucht ein Symbol.")
        object.__setattr__(self, "side", OrderSide(self.side))
        object.__setattr__(self, "type", OrderType(self.type))
        if self.qty is not None and self.sizing is not None:
            raise OrderError("Entweder qty oder sizing angeben, nicht beides.")
        if self.qty is not None and (not math.isfinite(self.qty) or self.qty <= 0):
            raise OrderError(f"Stückzahl muss größer als 0 sein, erhalten: {self.qty}.")
        if self.side.is_entry and self.qty is None and self.sizing is None:
            raise OrderError("Eröffnende Order braucht qty oder sizing.")
        for name in ("stop_price", "limit_price", "trail_pct", "trail_atr"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise OrderError(f"{name} muss größer als 0 sein, erhalten: {value}.")
        if self.strength is not None and not math.isfinite(self.strength):
            raise OrderError(f"strength muss endlich sein, erhalten: {self.strength}.")
        self._check_type_fields()

    def _check_type_fields(self) -> None:
        kind = self.type
        trailing = (self.trail_pct, self.trail_atr)
        if kind is OrderType.TRAILING_STOP:
            if sum(v is not None for v in trailing) != 1:
                raise OrderError("Trailing Stop braucht genau einen von trail_pct oder trail_atr.")
            if self.limit_price is not None:
                raise OrderError("Trailing Stop hat keinen limit_price.")
            return
        if any(v is not None for v in trailing):
            raise OrderError("trail_pct und trail_atr gibt es nur bei TRAILING_STOP.")
        if kind is OrderType.STOP and (self.stop_price is None or self.limit_price is not None):
            raise OrderError("Stop-Order braucht stop_price und keinen limit_price.")
        if kind is OrderType.LIMIT and (self.limit_price is None or self.stop_price is not None):
            raise OrderError("Limit-Order braucht limit_price und keinen stop_price.")
        if kind is OrderType.MARKET and (
            self.stop_price is not None or self.limit_price is not None
        ):
            raise OrderError("Market-Order hat weder stop_price noch limit_price.")


@dataclass(frozen=True)
class Position:
    """Offene Position, wie die Engine sie der Strategie zeigt.

    Attributes:
        symbol: Wertpapier.
        side: Long oder Short.
        qty: Stückzahl (> 0, auch bei Short).
        entry_price: Durchschnittlicher Einstiegskurs.
        entry_date: Datum der Einstiegskerze.
    """

    symbol: str
    side: Side
    qty: float
    entry_price: float
    entry_date: Date


@dataclass(frozen=True)
class PlotSpec:
    """Beschreibt eine Linie oder Marker-Reihe, die der Chart zur Strategie zeigt.

    Attributes:
        column: Spalte aus :meth:`Strategy.indicators`.
        kind: ``line`` (Linie), ``marker`` (Punkte, wo der Wert nicht NaN ist) oder
            ``histogram`` (Balken).
        pane: ``price`` im Kurs-Chart, ``separate`` in eigenem Panel.
        label: Anzeigename; leer bedeutet ``column``.
        color: Farbe als ``#RRGGBB``; leer bedeutet automatische Farbe.
    """

    column: str
    kind: Literal["line", "marker", "histogram"] = "line"
    pane: Literal["price", "separate"] = "price"
    label: str = ""
    color: str = ""


# ---------------------------------------------------------------------------
# Schreibgeschützte Sichten
# ---------------------------------------------------------------------------


def _readonly(values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    arr = np.array(values, copy=True)
    arr.flags.writeable = False
    return arr


class PreparedSeries:
    """Kurse und Indikatoren einer Reihe als schreibgeschützte Arrays.

    Die Engine erzeugt das Objekt einmal je Symbol; :meth:`bars_at` und
    :meth:`ind_at` liefern danach ohne Kopie die Sicht bis zur Kerze ``t``.
    """

    __slots__ = ("_bar_cols", "_dates", "_ind_cols", "length")

    def __init__(self, bars: pd.DataFrame, ind: pd.DataFrame | None = None) -> None:
        """Übernimmt Kopien von ``bars`` und ``ind``.

        Args:
            bars: Kursreihe mit Spalte ``date`` (Schema :mod:`trading_helfer.core.bars`).
            ind: Indikatoren mit gleicher Zeilenzahl wie ``bars``; ``None`` = keine.

        Raises:
            ValueError: ``date`` fehlt oder die Zeilenzahlen passen nicht.
        """
        if DATE_COLUMN not in bars.columns:
            raise ValueError(f"Spalte {DATE_COLUMN!r} fehlt in den Kursdaten.")
        ind = ind if ind is not None else pd.DataFrame(index=bars.index)
        if len(ind) != len(bars):
            raise ValueError(f"Indikatoren haben {len(ind)} Zeilen, Kursdaten {len(bars)} Zeilen.")
        self.length = len(bars)
        self._dates = _readonly(bars[DATE_COLUMN].to_numpy(dtype="datetime64[ns]"))
        self._bar_cols = {
            str(c): _readonly(bars[c].to_numpy()) for c in bars.columns if c != DATE_COLUMN
        }
        self._ind_cols = {str(c): _readonly(ind[c].to_numpy()) for c in ind.columns}

    def with_indicators(self, extra: Mapping[str, npt.NDArray[Any]]) -> PreparedSeries:
        """Neue Reihe mit zusätzlichen Indikatorspalten (gleiche Kurse, ohne Kopie).

        Args:
            extra: Spaltenname → Werte mit genau ``length`` Einträgen; gleichnamige
                Spalten werden ersetzt.

        Raises:
            ValueError: Eine Spalte hat eine andere Länge als die Reihe.
        """
        clone = object.__new__(PreparedSeries)
        clone.length = self.length
        clone._dates = self._dates
        clone._bar_cols = self._bar_cols
        cols = dict(self._ind_cols)
        for name, values in extra.items():
            if len(values) != self.length:
                raise ValueError(
                    f"Indikator {name!r} hat {len(values)} Werte, Kursdaten {self.length}."
                )
            cols[str(name)] = _readonly(np.asarray(values))
        clone._ind_cols = cols
        return clone

    def _check(self, t: int) -> int:
        if not 0 <= t < self.length:
            raise IndexError(f"Kerze {t} außerhalb von 0 … {self.length - 1}.")
        return t + 1

    def bars_at(self, t: int) -> BarsView:
        """Kurse bis einschließlich Kerze ``t``."""
        end = self._check(t)
        return BarsView(self._dates[:end], {k: v[:end] for k, v in self._bar_cols.items()})

    def ind_at(self, t: int) -> IndicatorView:
        """Indikatoren bis einschließlich Kerze ``t``."""
        end = self._check(t)
        return IndicatorView({k: v[:end] for k, v in self._ind_cols.items()})


class _ColumnView:
    """Gemeinsame Basis: Spalten als schreibgeschützte Arrays gleicher Länge."""

    __slots__ = ("_cols", "_len")

    _cols: Mapping[str, npt.NDArray[Any]]
    _len: int

    def __init__(self, cols: Mapping[str, npt.NDArray[Any]], length: int) -> None:
        object.__setattr__(self, "_cols", MappingProxyType(dict(cols)))
        object.__setattr__(self, "_len", length)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} ist schreibgeschützt.")

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError(f"{type(self).__name__} ist schreibgeschützt.")

    def __len__(self) -> int:
        return int(self._len)

    def __contains__(self, name: object) -> bool:
        return name in self._cols

    @property
    def columns(self) -> list[str]:
        """Spaltennamen."""
        return list(self._cols)

    def column(self, name: str) -> npt.NDArray[Any]:
        """Ganze Historie einer Spalte bis ``t`` als schreibgeschütztes Array."""
        try:
            values: npt.NDArray[Any] = self._cols[name]
        except KeyError:
            raise KeyError(f"Unbekannte Spalte: {name!r}.") from None
        return values

    def get(self, name: str, ago: int = 0) -> Any:
        """Wert der Spalte ``ago`` Kerzen vor ``t`` (``0`` = aktuelle Kerze).

        Raises:
            KeyError: Unbekannte Spalte.
            IndexError: ``ago`` negativ oder weiter zurück als die Historie reicht.
        """
        values = self.column(name)
        if ago < 0:
            raise IndexError("ago darf nicht negativ sein (kein Blick in die Zukunft).")
        if ago >= len(values):
            raise IndexError(f"Nur {len(values)} Kerzen Historie, angefragt: {ago} zurück.")
        return values[len(values) - 1 - ago]


class BarsView(_ColumnView):
    """Schreibgeschützte Sicht auf die Kurse bis einschließlich ``t``.

    ``view["close"]`` und ``view.close`` liefern die Historie als Array (letzter Wert =
    Kerze ``t``); :meth:`get` liefert einen einzelnen Wert. :meth:`to_frame` erzeugt
    eine unabhängige Kopie als DataFrame.
    """

    __slots__ = ("_dates",)

    _dates: npt.NDArray[np.datetime64]

    def __init__(self, dates: npt.NDArray[Any], cols: Mapping[str, npt.NDArray[Any]]) -> None:
        super().__init__(cols, len(dates))
        object.__setattr__(self, "_dates", dates)

    def __getitem__(self, name: str) -> npt.NDArray[Any]:
        if name == DATE_COLUMN:
            return self.dates
        return self.column(name)

    @property
    def dates(self) -> npt.NDArray[np.datetime64]:
        """Handelstage bis ``t``."""
        dates: npt.NDArray[np.datetime64] = self._dates
        return dates

    @property
    def open(self) -> FloatArray:
        """Eröffnungskurse bis ``t``."""
        return self.column("open")

    @property
    def high(self) -> FloatArray:
        """Höchstkurse bis ``t``."""
        return self.column("high")

    @property
    def low(self) -> FloatArray:
        """Tiefstkurse bis ``t``."""
        return self.column("low")

    @property
    def close(self) -> FloatArray:
        """Schlusskurse bis ``t``."""
        return self.column("close")

    @property
    def volume(self) -> npt.NDArray[Any]:
        """Volumen bis ``t``."""
        return self.column("volume")

    def to_frame(self) -> pd.DataFrame:
        """Unabhängige, veränderbare Kopie als DataFrame (Schema wie die Kursreihe)."""
        data: dict[str, Any] = {DATE_COLUMN: self._dates.copy()}
        data.update({k: v.copy() for k, v in self._cols.items()})
        return pd.DataFrame(data)


class IndicatorView(_ColumnView):
    """Schreibgeschützte Sicht auf die Indikatoren: Zeile bei ``t`` plus Historie.

    ``ind["sma"]`` liefert den Wert bei ``t``, ``ind.get("sma", ago=1)`` den Wert der
    Vorkerze und ``ind.column("sma")`` die Historie als Array.
    """

    __slots__ = ()

    def __init__(self, cols: Mapping[str, npt.NDArray[Any]]) -> None:
        lengths = {len(v) for v in cols.values()}
        super().__init__(cols, lengths.pop() if lengths else 0)

    def __getitem__(self, name: str) -> Any:
        return self.get(name)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.get(name)
        except KeyError:
            raise AttributeError(f"Unbekannter Indikator: {name!r}.") from None

    def row(self) -> dict[str, Any]:
        """Alle Indikatorwerte bei ``t`` als ``dict``."""
        return {k: self.get(k) for k in self._cols} if len(self) else {}


# ---------------------------------------------------------------------------
# Kontext je Kerze
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarContext:
    """Alles, was eine Strategie auf Kerze ``t`` sieht.

    Attributes:
        symbol: Wertpapier.
        date: Datum der Kerze ``t``.
        timeframe: Zeitrahmen der Kerzen.
        bars: Kurse bis einschließlich ``t`` (schreibgeschützt).
        ind: Indikatoren bei ``t`` plus Historie (schreibgeschützt).
        position: Offene Position im Symbol oder ``None``.
        cash: Freies Kapital.
        equity: Gesamtkapital (Cash plus Wert aller Positionen).
        is_member: ``True``, wenn das Symbol am Tag ``t`` zum Universum gehört; nur dann
            darf neu eröffnet werden.
    """

    symbol: str
    date: Date
    timeframe: Timeframe
    bars: BarsView
    ind: IndicatorView
    position: Position | None = None
    cash: float = 0.0
    equity: float = 0.0
    is_member: bool = True

    @classmethod
    def at(
        cls,
        series: PreparedSeries,
        t: int,
        *,
        symbol: str,
        timeframe: Timeframe,
        position: Position | None = None,
        cash: float = 0.0,
        equity: float = 0.0,
        is_member: bool = True,
    ) -> BarContext:
        """Kontext für Kerze ``t`` einer vorbereiteten Reihe (ohne Kopie der Daten).

        Args:
            series: Vorbereitete Kurse und Indikatoren.
            t: Index der aktuellen Kerze (0-basiert).
            symbol: Wertpapier.
            timeframe: Zeitrahmen.
            position: Offene Position oder ``None``.
            cash: Freies Kapital.
            equity: Gesamtkapital.
            is_member: Mitgliedschaft im Universum am Tag ``t``.

        Returns:
            Neuer Kontext.
        """
        bars = series.bars_at(t)
        day = pd.Timestamp(bars.dates[-1]).date()
        return cls(
            symbol=symbol,
            date=day,
            timeframe=timeframe,
            bars=bars,
            ind=series.ind_at(t),
            position=position,
            cash=cash,
            equity=equity,
            is_member=is_member,
        )

    # -- Order-Helfer -------------------------------------------------------

    def buy(
        self,
        qty: float | None = None,
        *,
        sizing: Sizing | None = None,
        type: OrderType = OrderType.MARKET,
        stop_price: float | None = None,
        limit_price: float | None = None,
        tag: str = "",
        strength: float | None = None,
    ) -> Order:
        """Order, die eine Long-Position eröffnet oder aufstockt."""
        return Order(
            self.symbol,
            OrderSide.BUY,
            type,
            qty=qty,
            sizing=sizing,
            stop_price=stop_price,
            limit_price=limit_price,
            tag=tag,
            strength=strength,
        )

    def sell_short(
        self,
        qty: float | None = None,
        *,
        sizing: Sizing | None = None,
        type: OrderType = OrderType.MARKET,
        stop_price: float | None = None,
        limit_price: float | None = None,
        tag: str = "",
        strength: float | None = None,
    ) -> Order:
        """Order, die eine Short-Position eröffnet oder aufstockt."""
        return Order(
            self.symbol,
            OrderSide.SELL_SHORT,
            type,
            qty=qty,
            sizing=sizing,
            stop_price=stop_price,
            limit_price=limit_price,
            tag=tag,
            strength=strength,
        )

    def exit(
        self,
        qty: float | None = None,
        *,
        type: OrderType = OrderType.MARKET,
        stop_price: float | None = None,
        limit_price: float | None = None,
        trail_pct: float | None = None,
        trail_atr: float | None = None,
        tag: str = "",
    ) -> list[Order]:
        """Orders, die die offene Position schließen (ganz oder ``qty`` Stück).

        Die Richtung ergibt sich aus der Position (``SELL`` für Long,
        ``BUY_TO_COVER`` für Short). Mit ``type`` lassen sich Schutz-Stops und
        Trailing Stops setzen.

        Returns:
            Liste mit einer Order, oder leere Liste, wenn keine Position offen ist.
        """
        if self.position is None:
            return []
        side = OrderSide.SELL if self.position.side is Side.LONG else OrderSide.BUY_TO_COVER
        return [
            Order(
                self.symbol,
                side,
                type,
                qty=qty,
                stop_price=stop_price,
                limit_price=limit_price,
                trail_pct=trail_pct,
                trail_atr=trail_atr,
                tag=tag,
            )
        ]

    # -- Signal-Helfer ------------------------------------------------------

    def value(self, operand: Operand, ago: int = 0) -> float:
        """Wert eines Operanden ``ago`` Kerzen vor ``t``.

        Ein Spaltenname wird zuerst unter den Indikatoren, dann unter den Kursen
        gesucht; eine Zahl ist konstant.

        Raises:
            KeyError: Spalte weder Indikator noch Kurs.
        """
        if isinstance(operand, int | float) and not isinstance(operand, bool):
            return float(operand)
        if isinstance(operand, str):
            view: _ColumnView = self.ind if operand in self.ind else self.bars
            return float(view.get(operand, ago))
        raise TypeError(f"Operand muss Spaltenname oder Zahl sein, erhalten: {operand!r}.")

    def _pair(self, a: Operand, b: Operand) -> tuple[float, float, float, float] | None:
        if len(self.bars) < 2:
            return None
        values = (self.value(a, 1), self.value(b, 1), self.value(a), self.value(b))
        if any(math.isnan(v) for v in values):
            return None
        return values

    def crossed_above(self, a: Operand, b: Operand) -> bool:
        """``True``, wenn ``a`` auf Kerze ``t`` über ``b`` kreuzt.

        Bedingung: ``a[t-1] <= b[t-1]`` und ``a[t] > b[t]``. Fehlt die Vorkerze oder
        ist ein Wert NaN (Anlaufphase), ist das Ergebnis ``False``.

        Args:
            a: Spaltenname (Indikator oder Kurs) oder Zahl.
            b: Spaltenname (Indikator oder Kurs) oder Zahl.
        """
        pair = self._pair(a, b)
        return pair is not None and pair[0] <= pair[1] and pair[2] > pair[3]

    def crossed_below(self, a: Operand, b: Operand) -> bool:
        """``True``, wenn ``a`` auf Kerze ``t`` unter ``b`` kreuzt.

        Bedingung: ``a[t-1] >= b[t-1]`` und ``a[t] < b[t]``; sonst wie
        :meth:`crossed_above`.
        """
        pair = self._pair(a, b)
        return pair is not None and pair[0] >= pair[1] and pair[2] < pair[3]


# ---------------------------------------------------------------------------
# Strategie
# ---------------------------------------------------------------------------


class Strategy(ABC):
    """Basisklasse aller Strategien (eingebaut und aus ``plugins/``).

    Unterklassen setzen die Klassenattribute und implementieren :meth:`on_bar`;
    :meth:`indicators` und :meth:`plot` sind optional.

    Attributes:
        id: Eindeutige ID, z. B. ``"trend_donchian_breakout"``.
        name: Anzeigename auf Deutsch.
        version: Version nach SemVer.
        description: Beschreibung auf Deutsch.
        timeframes: Unterstützte Zeitrahmen.
        params: Parameterdefinitionen.
        rank_columns: Indikatorspalten, die die Portfolio-Engine je Termin über das
            Universum rangiert (Querschnitt). Für jede Spalte ``c`` stehen danach
            ``c_rank`` (Perzentilrang 0 bis 100, 100 = höchster Wert, wie beim RS-Rang)
            und ``c_pos`` (Platz 1 = höchster Wert) in ``ctx.ind``. Es zählen nur Werte,
            die am Termin Indexmitglied sind und einen gültigen Wert haben; sonst NaN.
            Im Einzelwert-Backtest gilt der Wert als einziger: Rang 100, Platz 1.
    """

    id: ClassVar[str]
    name: ClassVar[str]
    version: ClassVar[str]
    description: ClassVar[str]
    timeframes: ClassVar[list[Timeframe]]
    params: ClassVar[dict[str, Param]] = {}
    rank_columns: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def make_params(cls, values: Mapping[str, Any] | None = None) -> Params:
        """Geprüfte Parameter aus Standardwerten und ``values``.

        Raises:
            ParamError: Unbekannter Parameter oder Wert außerhalb von min/max.
        """
        return Params(cls.params, values)

    def indicators(self, bars: pd.DataFrame, p: Params) -> pd.DataFrame:
        """Vektorisiert alle Indikatoren vorab berechnen.

        Der Wert in Zeile ``t`` darf nur von Kerzen bis ``t`` abhängen.

        Args:
            bars: Ganze Kursreihe (bereinigte Kurse).
            p: Geprüfte Parameter.

        Returns:
            DataFrame mit gleichem Index wie ``bars``; Standard: keine Spalten.
        """
        return pd.DataFrame(index=bars.index)

    @abstractmethod
    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Je Kerze: Kontext mit Kursen bis t, Indikatoren, Position, Kapital.

        Args:
            ctx: Kontext der aktuellen Kerze.

        Returns:
            Orders zur Ausführung am Open der nächsten Kerze (leer = nichts tun).
        """

    def plot(self) -> list[PlotSpec]:
        """Optional: welche Linien und Marker im Chart erscheinen."""
        return []

    def prepare(self, bars: pd.DataFrame, p: Params) -> PreparedSeries:
        """Berechnet die Indikatoren und bereitet die Reihe für :meth:`BarContext.at` vor.

        Raises:
            ValueError: Indikatoren haben eine andere Zeilenzahl als ``bars``.
        """
        return PreparedSeries(bars, self.indicators(bars, p))
