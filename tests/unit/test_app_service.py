"""AppService: Watchlists, Theme und Fensterzustand ohne Qt."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from trading_helfer.core.config import Settings
from trading_helfer.services.app_service import (
    DEFAULT_WATCHLISTS,
    AppService,
    WatchlistError,
    default_gui_state_path,
    normalize_symbol,
)


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "gui_state.json"


@pytest.fixture
def service(tmp_path: Path, state_path: Path) -> AppService:
    return AppService(Settings(data_dir=tmp_path / "data"), state_path)


def test_service_module_does_not_import_qt() -> None:
    code = (
        "import sys, trading_helfer.services.app_service; "
        "print(any(m.startswith('PySide6') for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_defaults(service: AppService, tmp_path: Path) -> None:
    assert service.watchlist_names() == list(DEFAULT_WATCHLISTS)
    assert service.active_watchlist == "Eigene"
    assert service.theme == "light"
    assert service.window_layout() == (None, None)
    assert service.settings.data_dir == tmp_path / "data"


def test_default_state_path_is_next_to_config() -> None:
    assert default_gui_state_path().name == "gui_state.json"


@pytest.mark.parametrize(
    ("raw", "clean"), [(" aapl ", "AAPL"), ("brk.b", "BRK.B"), ("^gspc", "^GSPC")]
)
def test_normalize_symbol(raw: str, clean: str) -> None:
    assert normalize_symbol(raw) == clean


@pytest.mark.parametrize("raw", ["", "   ", "AB CD", "A" * 20, "ÄPFEL"])
def test_normalize_symbol_rejects(raw: str) -> None:
    with pytest.raises(WatchlistError):
        normalize_symbol(raw)


def test_add_remove_and_persist(service: AppService, state_path: Path, tmp_path: Path) -> None:
    assert service.add_symbol("Eigene", "msft") == "MSFT"
    service.add_symbol("Eigene", "AAPL")
    service.add_symbol("S&P 500", "AAPL")
    with pytest.raises(WatchlistError, match="bereits"):
        service.add_symbol("Eigene", "aapl")
    service.remove_symbol("Eigene", "msft")
    service.remove_symbol("Eigene", "NICHTDA")
    service.set_theme("dark")
    service.set_active_watchlist("S&P 500")
    service.set_window_layout("Z2Vv", "c3Q=")

    reloaded = AppService(Settings(data_dir=tmp_path / "data"), state_path)
    assert reloaded.symbols("Eigene") == ["AAPL"]
    assert reloaded.symbols("S&P 500") == ["AAPL"]
    assert reloaded.theme == "dark"
    assert reloaded.active_watchlist == "S&P 500"
    assert reloaded.window_layout() == ("Z2Vv", "c3Q=")


def test_search(service: AppService) -> None:
    for symbol in ("AAPL", "AMZN", "MSFT"):
        service.add_symbol("Eigene", symbol)
    assert service.search_symbols("Eigene", "a") == ["AAPL", "AMZN"]
    assert service.search_symbols("Eigene", "") == ["AAPL", "AMZN", "MSFT"]


def test_unknown_list(service: AppService) -> None:
    with pytest.raises(WatchlistError, match="gibt es nicht"):
        service.symbols("Gibtsnicht")
    with pytest.raises(WatchlistError):
        service.set_active_watchlist("Gibtsnicht")


def test_broken_state_file_falls_back(
    tmp_path: Path, state_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{kaputt", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        svc = AppService(Settings(data_dir=tmp_path), state_path)
    assert svc.watchlist_names() == list(DEFAULT_WATCHLISTS)
    assert "nicht lesbar" in caplog.text


def test_state_file_is_completed(tmp_path: Path, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True)
    data = {"watchlists": {"Tech": ["NVDA"]}, "active_watchlist": "weg"}
    state_path.write_text(json.dumps(data), encoding="utf-8")
    svc = AppService(Settings(data_dir=tmp_path), state_path)
    assert svc.watchlist_names() == ["Tech", *DEFAULT_WATCHLISTS]
    assert svc.symbols("Tech") == ["NVDA"]
    assert svc.active_watchlist == "Eigene"


def test_save_error_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "datei"
    blocker.write_text("x", encoding="utf-8")
    svc = AppService(Settings(data_dir=tmp_path), blocker / "gui_state.json")
    with caplog.at_level(logging.ERROR):
        svc.set_theme("dark")
    assert "kann nicht gespeichert" in caplog.text
    assert svc.theme == "dark"
