"""Start der GUI: QApplication anlegen, Logging, globale Fehlerbehandlung, Hauptfenster."""

from __future__ import annotations

import logging
import sys
import tempfile
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from trading_helfer.core.config import ConfigError, default_config_path, load_settings
from trading_helfer.core.logging import reset_logging, setup_logging
from trading_helfer.gui import texts
from trading_helfer.gui.main_window import MainWindow
from trading_helfer.services.app_service import AppService
from trading_helfer.services.chart_service import ChartService, store_loader
from trading_helfer.services.data_service import DataService
from trading_helfer.services.fundamentals_service import FundamentalsService
from trading_helfer.services.scanner_service import ScannerService
from trading_helfer.services.selftest import SelfTestCheck, format_report, run_core_checks

__all__ = [
    "SELFTEST_TIMEOUT_MS",
    "get_application",
    "install_exception_hook",
    "run_gui",
    "run_selftest",
]

SELFTEST_TIMEOUT_MS = 500
"""So lange zeigt ``--selftest`` das Hauptfenster, bevor sich die App beendet."""

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
    quit_after_ms: int | None = None,
) -> int:
    """Startet die Oberfläche und gibt den Exit-Code zurück.

    Args:
        argv: Argumente für ``QApplication``.
        config_path: Pfad der ``config.toml`` (Standard: Benutzer-Konfigordner).
        state_path: Pfad der GUI-Zustandsdatei (Standard: neben der ``config.toml``).
        exec_loop: Ereignisschleife starten; ``False`` nur für Tests.
        quit_after_ms: Wenn gesetzt, beendet sich die App so viele Millisekunden nach dem
            Start; Exit-Code 0, wenn das Hauptfenster da noch sichtbar ist, sonst 1
            (für ``--selftest``).
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
    chart_service = ChartService(store_loader(settings.database_path))
    fundamentals_service = FundamentalsService(settings.database_path)
    data_service = DataService(
        settings, config_path if config_path is not None else default_config_path()
    )
    window = MainWindow(
        service,
        chart_service=chart_service,
        fundamentals_service=fundamentals_service,
        data_service=data_service,
        scanner_service=ScannerService(settings),
    )
    previous_hook = install_exception_hook(window)
    try:
        window.show()
        if not exec_loop:
            window.close()
            return 0
        if quit_after_ms is not None:

            def finish() -> None:
                visible = window.isVisible()
                logger.info("Selbsttest: Hauptfenster sichtbar = %s", visible)
                window.close()
                app.exit(0 if visible else 1)

            QTimer.singleShot(quit_after_ms, finish)
        return app.exec()
    finally:
        sys.excepthook = previous_hook


def run_selftest(
    argv: Sequence[str] | None = None,
    *,
    report_path: Path | None = None,
    quit_after_ms: int = SELFTEST_TIMEOUT_MS,
) -> int:
    """Prüft das Paket und zeigt kurz das Hauptfenster (``trading-helfer --selftest``).

    Der Test nutzt einen temporären Konfigurations- und Datenordner, verändert also
    keine Benutzerdaten, und braucht kein Netzwerk. Der Bericht geht auf ``stdout`` und,
    wenn ``report_path`` gesetzt ist, zusätzlich in diese Datei (das Fenster-Paket hat
    keine Konsole).

    Args:
        argv: Argumente für ``QApplication``.
        report_path: Datei für den Bericht.
        quit_after_ms: Anzeigedauer des Hauptfensters.

    Returns:
        0, wenn alle Prüfungen bestanden sind und das Hauptfenster erschien, sonst 1.
    """
    with tempfile.TemporaryDirectory(
        prefix="trading-helfer-selftest-", ignore_cleanup_errors=True
    ) as tmp:
        root = Path(tmp)
        config_path = root / "config.toml"
        config_path.write_text(f'data_dir = "{(root / "data").as_posix()}"\n', encoding="utf-8")
        checks = run_core_checks(load_settings(config_path))
        try:
            code = run_gui(
                argv,
                config_path=config_path,
                state_path=root / "gui_state.json",
                quit_after_ms=quit_after_ms,
            )
            window_check = SelfTestCheck(
                "Hauptfenster", code == 0, "angezeigt" if code == 0 else f"Exit-Code {code}"
            )
        except Exception as exc:
            window_check = SelfTestCheck("Hauptfenster", False, f"{type(exc).__name__}: {exc}")
        finally:
            reset_logging()  # Logdatei schließen, damit der Ordner gelöscht werden kann
    checks.append(window_check)
    report = format_report(checks)
    print(report, flush=True)
    if report_path is not None:
        report_path.write_text(report + "\n", encoding="utf-8")
    return 0 if all(c.ok for c in checks) else 1
