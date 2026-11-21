"""Gemeinsame Fixtures für GUI-Tests (pytest-qt, ohne Bildschirm)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox
from pytestqt.qtbot import QtBot

from trading_helfer.core.config import Settings
from trading_helfer.gui.main_window import MainWindow
from trading_helfer.gui.theme import apply_theme
from trading_helfer.services.app_service import AppService


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "gui_state.json"


@pytest.fixture
def service(tmp_path: Path, state_path: Path) -> AppService:
    return AppService(Settings(data_dir=tmp_path / "data"), state_path)


@pytest.fixture
def dialogs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Fängt Meldungsdialoge ab, statt sie modal anzuzeigen."""
    shown: list[tuple[str, str]] = []

    def record(kind: str) -> object:
        def show(_parent: object, title: str, text: str, *_a: object) -> object:
            shown.append((kind, f"{title}: {text}"))
            return QMessageBox.StandardButton.Ok

        return show

    for kind in ("critical", "warning", "information", "about"):
        monkeypatch.setattr(QMessageBox, kind, record(kind))
    return shown


@pytest.fixture
def window(
    qtbot: QtBot, service: AppService, dialogs: list[tuple[str, str]]
) -> Iterator[MainWindow]:
    win = MainWindow(service)
    qtbot.addWidget(win)
    yield win
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    apply_theme(app, "light")
