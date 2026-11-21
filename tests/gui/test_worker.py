"""Worker im QThreadPool: Ergebnis, Fortschritt und Fehler."""

from __future__ import annotations

import logging

import pytest
from PySide6.QtCore import QThreadPool
from pytestqt.qtbot import QtBot

from trading_helfer.gui.main_window import MainWindow
from trading_helfer.gui.worker import ProgressCallback, Worker, WorkerError


def _add(a: int, b: int) -> int:
    return a + b


def _fail() -> None:
    raise RuntimeError("Datenquelle nicht erreichbar")


def _steps(n: int, *, progress: ProgressCallback) -> str:
    for i in range(n):
        progress((i + 1) * 100 // n, f"Schritt {i + 1}")
    progress(150, "")
    return "fertig"


def test_worker_reports_result(qtbot: QtBot) -> None:
    worker = Worker(_add, 2, b=3)
    with qtbot.waitSignal(worker.signals.result) as blocker:
        QThreadPool.globalInstance().start(worker)
    assert blocker.args == [5]


def test_worker_run_synchronously(qtbot: QtBot) -> None:
    """Direkter Aufruf von run(): gleiche Signale wie im Pool (und messbar für die Abdeckung)."""
    ok = Worker(_add, 1, 1)
    results: list[object] = []
    finished: list[bool] = []
    ok.signals.result.connect(results.append)
    ok.signals.finished.connect(lambda: finished.append(True))
    ok.run()

    bad = Worker(_fail)
    errors: list[WorkerError] = []
    bad.signals.error.connect(errors.append)
    bad.run()

    assert results == [2]
    assert finished == [True]
    assert errors[0].message == "Datenquelle nicht erreichbar"
    assert bad.name == "_fail"


def test_worker_reports_error_and_logs(qtbot: QtBot, caplog: pytest.LogCaptureFixture) -> None:
    worker = Worker(_fail, name="Update")
    with caplog.at_level(logging.ERROR), qtbot.waitSignal(worker.signals.error) as blocker:
        QThreadPool.globalInstance().start(worker)
    error = blocker.args[0]
    assert isinstance(error, WorkerError)
    assert error.message == "Datenquelle nicht erreichbar"
    assert "RuntimeError" in error.traceback
    assert "Update" in caplog.text


def test_worker_error_message_falls_back_to_type() -> None:
    assert WorkerError(KeyError(), "").message == "KeyError"


def test_worker_reports_progress(qtbot: QtBot) -> None:
    worker = Worker(_steps, 2, report_progress=True)
    seen: list[tuple[int, str]] = []
    worker.signals.progress.connect(lambda p, m: seen.append((p, m)))
    with qtbot.waitSignal(worker.signals.result) as blocker:
        QThreadPool.globalInstance().start(worker)
    qtbot.waitUntil(lambda: len(seen) == 3)
    assert blocker.args == ["fertig"]
    assert seen == [(50, "Schritt 1"), (100, "Schritt 2"), (100, "")]


def test_main_window_task_result(window: MainWindow, qtbot: QtBot) -> None:
    results: list[object] = []
    worker = window.run_task(_steps, 4, on_result=results.append, name="Test", report_progress=True)
    qtbot.waitUntil(lambda: results == ["fertig"])
    qtbot.waitUntil(lambda: worker not in window._active_workers)
    assert window.statusBar().currentMessage() == "Test abgeschlossen"


def test_main_window_task_error_shows_dialog(
    window: MainWindow,
    qtbot: QtBot,
    dialogs: list[tuple[str, str]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR):
        window.run_task(_fail, name="Datenupdate")
        qtbot.waitUntil(lambda: bool(dialogs))
    kind, text = dialogs[0]
    assert kind == "critical"
    assert "Datenupdate" in text
    assert "Datenquelle nicht erreichbar" in text
    assert "Datenupdate" in caplog.text
    assert window.statusBar().currentMessage() == "Datenupdate fehlgeschlagen"
