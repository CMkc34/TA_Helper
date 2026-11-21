"""Dialog „Indikator hinzufügen/bearbeiten“; Felder werden aus der Registry erzeugt."""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from trading_helfer.gui import texts
from trading_helfer.services.indicator_service import (
    IndicatorConfig,
    IndicatorService,
    ParamField,
)

__all__ = ["ColorButton", "IndicatorDialog"]

_INT_MAX = 100_000
_FLOAT_MAX = 1_000_000.0


class ColorButton(QPushButton):
    """Knopf, der eine Farbe zeigt und beim Klick einen Farbdialog öffnet."""

    def __init__(self, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setToolTip(texts.INDICATOR_COLOR_TOOLTIP)
        self.setMaximumWidth(80)
        self._color = ""
        self.set_color(color)
        self.clicked.connect(self._choose)

    @property
    def color(self) -> str:
        """Gewählte Farbe als ``#rrggbb``."""
        return self._color

    def set_color(self, color: str) -> None:
        """Setzt die Farbe (``#rrggbb``)."""
        self._color = QColor(color).name()
        self.setText(self._color)
        self.setStyleSheet(f"background-color: {self._color};")

    def _choose(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._color), self, texts.INDICATOR_COLOR_TITLE)
        if chosen.isValid():
            self.set_color(chosen.name())


class IndicatorDialog(QDialog):
    """Auswahl eines Indikators mit Parameter- und Farbfeldern.

    Args:
        service: Liefert Indikatoren, Parameterfelder und Standardwerte.
        config: Bestehende Konfiguration zum Bearbeiten; ``None`` zum Hinzufügen.
        color_offset: Erste Standardfarbe für neue Indikatoren.
    """

    def __init__(
        self,
        service: IndicatorService,
        config: IndicatorConfig | None = None,
        color_offset: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._color_offset = color_offset
        self.param_widgets: dict[str, QWidget] = {}
        self.color_buttons: dict[str, ColorButton] = {}
        editing = config is not None
        self.setWindowTitle(texts.INDICATOR_EDIT_TITLE if editing else texts.INDICATOR_ADD_TITLE)

        self.indicator_combo = QComboBox()
        for spec in service.available():
            pane = texts.INDICATOR_PANE_PRICE if spec.pane == "price" else texts.INDICATOR_PANE_OWN
            self.indicator_combo.addItem(f"{spec.name.upper()} · {pane}", spec.name)
        self.indicator_combo.setEnabled(not editing)

        self.form = QFormLayout()
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        top = QFormLayout()
        top.addRow(texts.INDICATOR_LABEL, self.indicator_combo)
        layout.addLayout(top)
        layout.addLayout(self.form)
        layout.addWidget(self.hint_label)
        layout.addWidget(buttons)

        if config is not None:
            self.indicator_combo.setCurrentIndex(self.indicator_combo.findData(config.name))
            self._build_fields(service.normalize(config, color_offset))
        else:
            self._build_fields(None)
        self.indicator_combo.currentIndexChanged.connect(lambda _i: self._build_fields(None))

    @property
    def indicator_name(self) -> str:
        """Gewählter Indikator."""
        return str(self.indicator_combo.currentData())

    def select_indicator(self, name: str) -> None:
        """Wählt den Indikator ``name`` und erzeugt seine Felder neu."""
        self.indicator_combo.setCurrentIndex(self.indicator_combo.findData(name))

    def _build_fields(self, config: IndicatorConfig | None) -> None:
        while self.form.rowCount():
            self.form.removeRow(0)
        self.param_widgets.clear()
        self.color_buttons.clear()
        name = self.indicator_name
        if config is None:
            config = self._service.default_config(name, self._color_offset)
        for field in self._service.param_fields(name):
            widget = _param_widget(field, config.params.get(field.name, field.default))
            self.param_widgets[field.name] = widget
            self.form.addRow(field.name, widget)
        for output, color in config.colors.items():
            button = ColorButton(color)
            self.color_buttons[output] = button
            self.form.addRow(texts.INDICATOR_COLOR_ROW.format(output=output), button)
        spec = self._service.spec(name)
        doc = (spec.func.__doc__ or "").strip().splitlines()
        self.hint_label.setText(doc[0] if doc else "")

    def config(self) -> IndicatorConfig:
        """Konfiguration aus den aktuellen Eingaben."""
        params = {name: _widget_value(widget) for name, widget in self.param_widgets.items()}
        colors = {output: button.color for output, button in self.color_buttons.items()}
        return IndicatorConfig(name=self.indicator_name, params=params, colors=colors)


def _param_widget(field: ParamField, value: Any) -> QWidget:
    if field.kind == "int":
        spin = QSpinBox()
        spin.setRange(1, _INT_MAX)
        spin.setValue(int(value))
        return spin
    if field.kind == "float":
        dspin = QDoubleSpinBox()
        dspin.setDecimals(4)
        dspin.setRange(0.0, _FLOAT_MAX)
        dspin.setSingleStep(0.01 if abs(float(field.default)) < 1 else 0.1)
        dspin.setValue(float(value))
        return dspin
    if field.kind == "choice":
        combo = QComboBox()
        combo.addItems(list(field.choices))
        combo.setCurrentText(str(value))
        return combo
    return QLineEdit(str(value))


def _widget_value(widget: QWidget) -> Any:
    if isinstance(widget, QSpinBox):
        return widget.value()
    if isinstance(widget, QDoubleSpinBox):
        return widget.value()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QLineEdit):
        return widget.text()
    raise TypeError(type(widget).__name__)
