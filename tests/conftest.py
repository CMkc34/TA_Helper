"""Gemeinsame Test-Einstellungen.

Tests laufen komplett offline: Verbindungen zu anderen Hosts als localhost schlagen fehl.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

_ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost"}
_original_connect = socket.socket.connect


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, str) and host not in _ALLOWED_HOSTS:
        raise RuntimeError(f"Netzwerkzugriff in Tests verboten: {address!r}")
    return _original_connect(self, address)


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
