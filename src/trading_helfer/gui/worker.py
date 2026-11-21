"""Generischer Hintergrund-Worker für den ``QThreadPool``.

Ein :class:`Worker` führt eine Funktion in einem Pool-Thread aus und meldet über
:class:`WorkerSignals` Fortschritt, Ergebnis oder Fehler zurück in den GUI-Thread.
Ausnahmen der Funktion werden gefangen und geloggt; sie beenden nie die App.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal

__all__ = ["ProgressCallback", "Worker", "WorkerError", "WorkerSignals"]

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]
"""Rückruf für Fortschritt: Prozent (0–100) und kurze Meldung."""


@dataclass(frozen=True)
class WorkerError:
    """Fehler aus einem Worker: die Ausnahme und ihr formatierter Traceback."""

    exception: BaseException
    traceback: str

    @property
    def message(self) -> str:
        """Kurze Fehlermeldung für den Dialog."""
        return str(self.exception) or type(self.exception).__name__


class WorkerSignals(QObject):
    """Signale eines :class:`Worker` (werden im GUI-Thread zugestellt)."""

    progress = Signal(int, str)
    result = Signal(object)
    error = Signal(object)
    finished = Signal()


class Worker(QRunnable):
    """Führt ``fn(*args, **kwargs)`` im ``QThreadPool`` aus.

    Mit ``report_progress=True`` erhält ``fn`` zusätzlich das Schlüsselwortargument
    ``progress`` (ein :data:`ProgressCallback`).

    Args:
        fn: Auszuführende Funktion.
        *args: Positionsargumente für ``fn``.
        report_progress: ``progress``-Rückruf an ``fn`` übergeben.
        name: Name der Aufgabe für Log und Statusleiste.
        **kwargs: Schlüsselwortargumente für ``fn``.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        report_progress: bool = False,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.name = name or getattr(fn, "__name__", "Aufgabe")
        self.signals = WorkerSignals()
        if report_progress:
            self.kwargs["progress"] = self._emit_progress

    def _emit_progress(self, percent: int, message: str = "") -> None:
        self.signals.progress.emit(max(0, min(100, int(percent))), message)

    def run(self) -> None:
        """Führt die Funktion aus; wird vom Pool-Thread aufgerufen."""
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:
            logger.exception("Hintergrundaufgabe „%s“ fehlgeschlagen", self.name)
            self.signals.error.emit(WorkerError(exc, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
