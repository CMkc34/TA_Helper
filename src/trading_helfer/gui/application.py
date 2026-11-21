"""Start der GUI: QApplication anlegen, Logging, globale Fehlerbehandlung, Hauptfenster."""

from __future__ import annotations

import logging
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType

from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from trading_helfer.core.config import ConfigError, load_settings
from trading_helfer.core.logging import setup_logging
from trading_helfer.gui import texts
from trading_helfer.gui.main_window import MainWindow
from trading_helfer.services.app_service import AppService

__all__ = ["get_application", "install_exception_hook", "run_gui"]

logger = logging.getLogger(__name__)

ExceptHook = Callable[[type[BaseException], BaseException, TracebackType | None], object]


def get_application(argv: Sequence[str] | None = None) -> QApplication:
    """Gibt die laufende ``QApplication`` zurück oder legt sie an."""
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    app = QApplication(list(argv) if argv is not None else sys.argv[:1])
    app.setApplicationName(texts.APP_TITLE)
    app.setOrganizationName("trading-helfer")
    return app


def install_exception_hook(parent: QWidget | None = None) -> ExceptHook:
    """Fängt unbehandelte Ausnahmen ab, loggt sie und zeigt einen Dialog statt abzustürzen.

    Returns:
        Der vorherige ``sys.excepthook`` (zum Wiederherstellen).
    """
    previous = sys.excepthook

    def hook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logger.error("Unbehandelter Fehler:\n%s", text)
        QMessageBox.critical(
            parent,
            texts.ERROR_TITLE,
            texts.ERROR_UNEXPECTED.format(message=exc or exc_type.__name__),
        )

    sys.excepthook = hook
    return previous


def run_gui(
    argv: Sequence[str] | None = None,
    *,
    config_path: Path | None = None,
    state_path: Path | None = None,
    exec_loop: bool = True,
) -> int:
    """Startet die Oberfläche und gibt den Exit-Code zurück.

    Args:
        argv: Argumente für ``QApplication``.
        config_path: Pfad der ``config.toml`` (Standard: Benutzer-Konfigordner).
        state_path: Pfad der GUI-Zustandsdatei (Standard: neben der ``config.toml``).
        exec_loop: Ereignisschleife starten; ``False`` nur für Tests.
    """
    app = get_application(argv)
    try:
        settings = load_settings(config_path)
    except ConfigError as exc:
        QMessageBox.critical(None, texts.ERROR_CONFIG_TITLE, str(exc))
        return 1
    setup_logging(settings)
    logger.info("Trading Helfer startet")
    service = AppService(settings, state_path)
    window = MainWindow(service)
    previous_hook = install_exception_hook(window)
    try:
        window.show()
        if not exec_loop:
            window.close()
            return 0
        return app.exec()
    finally:
        sys.excepthook = previous_hook
