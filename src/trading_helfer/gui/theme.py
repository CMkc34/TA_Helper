"""Hell- und Dunkelmodus über eine Qt-Palette im Fusion-Stil."""

from __future__ import annotations

from typing import Final

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from trading_helfer.services.app_service import ThemeName

__all__ = ["THEMES", "apply_theme", "build_palette"]

THEMES: Final[tuple[ThemeName, ...]] = ("light", "dark")

_DARK: Final[dict[QPalette.ColorRole, str]] = {
    QPalette.ColorRole.Window: "#2b2b2b",
    QPalette.ColorRole.WindowText: "#e6e6e6",
    QPalette.ColorRole.Base: "#1e1e1e",
    QPalette.ColorRole.AlternateBase: "#323232",
    QPalette.ColorRole.ToolTipBase: "#3c3c3c",
    QPalette.ColorRole.ToolTipText: "#e6e6e6",
    QPalette.ColorRole.PlaceholderText: "#8c8c8c",
    QPalette.ColorRole.Text: "#e6e6e6",
    QPalette.ColorRole.Button: "#353535",
    QPalette.ColorRole.ButtonText: "#e6e6e6",
    QPalette.ColorRole.BrightText: "#ff5555",
    QPalette.ColorRole.Link: "#5aa0ff",
    QPalette.ColorRole.Highlight: "#2f6fbf",
    QPalette.ColorRole.HighlightedText: "#ffffff",
}


def build_palette(theme: ThemeName) -> QPalette:
    """Erzeugt die Palette für ``theme`` (``light`` = Standardpalette des Fusion-Stils)."""
    if theme == "light":
        return QPalette()
    palette = QPalette()
    for role, color in _DARK.items():
        palette.setColor(role, QColor(color))
    disabled = QColor("#7f7f7f")
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.WindowText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return palette


def apply_theme(app: QApplication, theme: ThemeName) -> None:
    """Setzt Fusion-Stil und die Palette für ``theme`` für die ganze Anwendung."""
    app.setStyle("Fusion")
    app.setPalette(build_palette(theme))
    app.setProperty("theme", theme)
