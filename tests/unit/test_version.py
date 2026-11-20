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


def test_cli_without_command_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert trading_helfer.__version__ in capsys.readouterr().out


def test_network_is_blocked_in_tests() -> None:
    with socket.socket() as s, pytest.raises(RuntimeError, match="Netzwerkzugriff"):
        s.connect(("example.com", 80))
