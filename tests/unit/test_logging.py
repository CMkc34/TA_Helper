"""Logging: Handler, Idempotenz, Rotation und Maskierung von Geheimnissen."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest
from pydantic import SecretStr

from trading_helfer.core import logging as th_logging
from trading_helfer.core.config import Settings
from trading_helfer.core.logging import (
    LOG_BACKUP_COUNT,
    LOG_FILE_NAME,
    LOG_MAX_BYTES,
    MASK,
    SecretMaskingFilter,
    register_secret,
    reset_logging,
    setup_logging,
)

API_KEY = "tiingo-geheim-0123456789"


@pytest.fixture(autouse=True)
def clean_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stellt Root-Level und Geheimnis-Registry nach jedem Test wieder her."""
    monkeypatch.setattr(th_logging, "_registered_secrets", set())
    root = logging.getLogger()
    level = root.level
    yield
    reset_logging()
    root.setLevel(level)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data", log_level="DEBUG", tiingo_api_key=SecretStr(API_KEY)
    )


def own_handlers() -> list[logging.Handler]:
    return [
        h for h in logging.getLogger().handlers if (h.get_name() or "").startswith("trading_helfer")
    ]


def flush_all() -> None:
    for handler in own_handlers():
        handler.flush()


def test_setup_creates_file_and_console_handler(settings: Settings) -> None:
    log_file = setup_logging(settings)
    assert log_file == settings.log_path / LOG_FILE_NAME
    assert settings.log_path.is_dir()
    handlers = own_handlers()
    assert len(handlers) == 2
    file_handler = next(h for h in handlers if isinstance(h, RotatingFileHandler))
    assert file_handler.maxBytes == LOG_MAX_BYTES == 5 * 1024 * 1024
    assert file_handler.backupCount == LOG_BACKUP_COUNT == 5
    console = next(h for h in handlers if not isinstance(h, RotatingFileHandler))
    assert isinstance(console, logging.StreamHandler)
    assert console.stream is sys.stderr
    assert logging.getLogger().level == logging.DEBUG


def test_messages_reach_log_file(settings: Settings) -> None:
    log_file = setup_logging(settings, console=False)
    logging.getLogger("trading_helfer.test").info("Hallo Logdatei äöü")
    flush_all()
    text = log_file.read_text(encoding="utf-8")
    assert "INFO" in text
    assert "trading_helfer.test: Hallo Logdatei äöü" in text


def test_setup_is_idempotent(settings: Settings, tmp_path: Path) -> None:
    log_file = setup_logging(settings, console=False)
    setup_logging(settings, console=False)
    setup_logging(settings)
    assert len(own_handlers()) == 2
    logging.getLogger("trading_helfer.test").warning("einmal")
    flush_all()
    assert log_file.read_text(encoding="utf-8").count("einmal") == 1


def test_setup_applies_new_settings(settings: Settings, tmp_path: Path) -> None:
    setup_logging(settings, console=False)
    other = Settings(data_dir=tmp_path / "anders", log_level="warning")
    log_file = setup_logging(other, console=False)
    assert log_file == tmp_path / "anders" / "logs" / LOG_FILE_NAME
    assert logging.getLogger().level == logging.WARNING
    assert len(own_handlers()) == 1


def test_reset_logging_keeps_foreign_handlers(settings: Settings) -> None:
    foreign = logging.NullHandler()
    root = logging.getLogger()
    root.addHandler(foreign)
    try:
        setup_logging(settings)
        reset_logging()
        assert own_handlers() == []
        assert foreign in root.handlers
    finally:
        root.removeHandler(foreign)


def test_api_key_never_in_log_file(settings: Settings) -> None:
    log_file = setup_logging(settings, console=False)
    log = logging.getLogger("trading_helfer.data.providers.tiingo")
    log.info("GET https://api.tiingo.com/daily?token=%s", API_KEY)
    log.debug(f"Key direkt in der Nachricht: {API_KEY}")
    try:
        raise RuntimeError(f"Fehler mit {API_KEY}")
    except RuntimeError:
        log.exception("Anfrage fehlgeschlagen")
    log.warning("Stack", stack_info=True, extra={"k": API_KEY})
    flush_all()
    text = log_file.read_text(encoding="utf-8")
    assert API_KEY not in text
    assert f"token={MASK}" in text
    assert f"Fehler mit {MASK}" in text


def test_api_key_never_on_console(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging(settings)
    logging.getLogger("trading_helfer").error("Key=%s", API_KEY)
    err = capsys.readouterr().err
    assert API_KEY not in err
    assert f"Key={MASK}" in err


def test_masking_with_caplog(caplog: pytest.LogCaptureFixture) -> None:
    caplog.handler.addFilter(SecretMaskingFilter([API_KEY]))
    with caplog.at_level(logging.INFO):
        logging.getLogger("trading_helfer.x").info("token=%s Ende", API_KEY)
    assert API_KEY not in caplog.text
    assert caplog.records[0].getMessage() == f"token={MASK} Ende"


def test_registered_secret_is_masked_with_caplog(caplog: pytest.LogCaptureFixture) -> None:
    caplog.handler.addFilter(SecretMaskingFilter())
    register_secret("spaeter-eingegeben-42")
    with caplog.at_level(logging.INFO):
        logging.getLogger("trading_helfer.gui").info("neuer Key spaeter-eingegeben-42")
    assert "spaeter-eingegeben-42" not in caplog.text
    assert f"neuer Key {MASK}" in caplog.text


def test_setup_registers_settings_key(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    setup_logging(settings, console=False)
    caplog.handler.addFilter(SecretMaskingFilter())
    with caplog.at_level(logging.INFO):
        logging.getLogger("trading_helfer").info("Key %s", API_KEY)
    assert API_KEY not in caplog.text


def test_longer_secret_masked_first() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "abcd-abcdef", None, None)
    SecretMaskingFilter(["abcd", "abcd-abcdef"]).filter(record)
    assert record.getMessage() == MASK


@pytest.mark.parametrize("value", [None, "", "abc"])
def test_short_secrets_are_ignored(value: str | None) -> None:
    register_secret(value)
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "abc %s", ("abc",), None)
    assert SecretMaskingFilter(["ab"]).filter(record) is True
    assert record.getMessage() == "abc abc"
    assert record.args == ("abc",)


def test_unrelated_message_is_untouched() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "Wert %d", (5,), None)
    SecretMaskingFilter([API_KEY]).filter(record)
    assert record.msg == "Wert %d"
    assert record.args == (5,)
