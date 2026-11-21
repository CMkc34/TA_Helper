"""Version und Einstiegspunkt."""

from __future__ import annotations

import socket
import tomllib
from pathlib import Path

import pytest

import trading_helfer
from trading_helfer.app import main

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert trading_helfer.__version__ == data["project"]["version"]


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert trading_helfer.__version__ in capsys.readouterr().out


def test_cli_without_command_starts_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    import trading_helfer.gui.application as application

    calls: list[bool] = []
    monkeypatch.setattr(application, "run_gui", lambda: calls.append(True) or 7)
    assert main([]) == 7
    assert calls == [True]


def test_network_is_blocked_in_tests() -> None:
    with socket.socket() as s, pytest.raises(RuntimeError, match="Netzwerkzugriff"):
        s.connect(("example.com", 80))


def test_cli_selftest_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import trading_helfer.gui.application as application

    calls: list[Path | None] = []

    def fake(*, report_path: Path | None = None) -> int:
        calls.append(report_path)
        return 7

    monkeypatch.setattr(application, "run_selftest", fake)
    assert main(["--selftest"]) == 7
    assert main(["--selftest", str(tmp_path / "r.txt")]) == 7
    assert calls == [None, tmp_path / "r.txt"]


def test_missing_std_streams_are_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from trading_helfer.app import ensure_std_streams

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    ensure_std_streams()
    assert sys.stdout is not None
    assert sys.stderr is not None
    print("geht ins Leere")
    sys.stdout.close()
    sys.stderr.close()
