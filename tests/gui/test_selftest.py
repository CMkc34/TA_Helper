"""Selbsttest ``--selftest``: Hauptfenster kurz zeigen und beenden (P-01)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from trading_helfer.gui import application
from trading_helfer.gui.application import run_gui, run_selftest
from trading_helfer.gui.main_window import MainWindow
from trading_helfer.gui.theme import build_palette
from trading_helfer.services.selftest import SelfTestCheck


@pytest.fixture(autouse=True)
def core_checks_seen(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Path]]:
    """Ersetzt die Kernprüfungen (eigene Tests in tests/unit) und merkt sich den Datenordner."""
    seen: list[Path] = []

    def checks(settings: object) -> list[SelfTestCheck]:
        seen.append(settings.data_dir)  # type: ignore[attr-defined]
        return [SelfTestCheck("Datenbank", True, "Schema-Version 5")]

    monkeypatch.setattr(application, "run_core_checks", checks)
    yield seen
    QApplication.instance().setPalette(build_palette("light"))  # type: ignore[union-attr]


def test_selftest_shows_window_and_quits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    dialogs: list[tuple[str, str]],
    core_checks_seen: list[Path],
) -> None:
    report = tmp_path / "bericht.txt"
    assert run_selftest(report_path=report, quit_after_ms=10) == 0
    text = report.read_text(encoding="utf-8")
    assert "OK   Hauptfenster: angezeigt" in text
    assert text.rstrip().endswith("Selbsttest bestanden.")
    assert "Selbsttest bestanden." in capsys.readouterr().out
    # temporärer Datenordner, danach gelöscht
    data_dir = core_checks_seen[0]
    assert "trading-helfer-selftest-" in str(data_dir)
    assert not data_dir.parent.exists()
    assert not dialogs


def test_selftest_fails_when_core_check_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        application, "run_core_checks", lambda _s: [SelfTestCheck("Indexdaten", False, "fehlt")]
    )
    assert run_selftest(quit_after_ms=10) == 1
    out = capsys.readouterr().out
    assert "FEHL Indexdaten: fehlt" in out
    assert "Selbsttest fehlgeschlagen." in out


def test_selftest_reports_gui_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*_a: object, **_k: object) -> int:
        raise RuntimeError("kein Bildschirm")

    monkeypatch.setattr(application, "run_gui", broken)
    assert run_selftest() == 1
    assert "FEHL Hauptfenster: RuntimeError: kein Bildschirm" in capsys.readouterr().out


def test_selftest_reports_gui_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(application, "run_gui", lambda *_a, **_k: 1)
    assert run_selftest() == 1
    assert "FEHL Hauptfenster: Exit-Code 1" in capsys.readouterr().out


def test_quit_after_fails_when_window_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dialogs: list[tuple[str, str]]
) -> None:
    monkeypatch.setattr(application, "setup_logging", lambda _s: tmp_path)
    monkeypatch.setattr(MainWindow, "isVisible", lambda _self: False)
    config = tmp_path / "config.toml"
    config.write_text(f'data_dir = "{(tmp_path / "d").as_posix()}"\n', encoding="utf-8")
    code = run_gui(config_path=config, state_path=tmp_path / "s.json", quit_after_ms=10)
    assert code == 1
