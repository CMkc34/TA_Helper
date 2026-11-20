"""Logging-Einrichtung: rotierende Logdatei, Konsole und Maskierung von Geheimnissen.

:func:`setup_logging` hängt zwei Handler an den Root-Logger (Datei und Konsole).
Beide tragen einen :class:`SecretMaskingFilter`, der bekannte Geheimnisse
(z. B. API-Keys) in allen Lognachrichten durch ``***`` ersetzt. Weitere Geheimnisse,
etwa ein später in der GUI eingegebener Key, lassen sich mit :func:`register_secret`
nachmelden. Module loggen wie üblich über ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

from trading_helfer.core.config import Settings

__all__ = [
    "LOG_BACKUP_COUNT",
    "LOG_FILE_NAME",
    "LOG_MAX_BYTES",
    "MASK",
    "SecretMaskingFilter",
    "register_secret",
    "reset_logging",
    "setup_logging",
]

LOG_FILE_NAME: Final = "trading-helfer.log"
LOG_MAX_BYTES: Final = 5 * 1024 * 1024
LOG_BACKUP_COUNT: Final = 5
LOG_FORMAT: Final = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
MASK: Final = "***"
MIN_SECRET_LENGTH: Final = 4
"""Kürzere Werte werden nicht maskiert, damit normale Texte lesbar bleiben."""

_HANDLER_NAMES: Final = ("trading_helfer.file", "trading_helfer.console")

_registry_lock = threading.Lock()
_registered_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Meldet ein Geheimnis an, das in allen Lognachrichten maskiert wird.

    Leere und sehr kurze Werte (unter ``MIN_SECRET_LENGTH`` Zeichen) werden ignoriert.
    """
    if value is None or len(value) < MIN_SECRET_LENGTH:
        return
    with _registry_lock:
        _registered_secrets.add(value)


def _known_secrets(extra: Iterable[str]) -> list[str]:
    with _registry_lock:
        secrets = set(_registered_secrets)
    secrets.update(s for s in extra if len(s) >= MIN_SECRET_LENGTH)
    # Längste zuerst, damit ein Geheimnis, das ein anderes enthält, ganz maskiert wird.
    return sorted(secrets, key=len, reverse=True)


def _mask(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = text.replace(secret, MASK)
    return text


class SecretMaskingFilter(logging.Filter):
    """Ersetzt bekannte Geheimnisse in Nachricht, Traceback und Stack durch ``***``.

    Maskiert werden die übergebenen ``secrets`` sowie alle per :func:`register_secret`
    angemeldeten Werte. Der Filter verändert den Datensatz an Ort und Stelle und lässt
    ihn immer passieren.
    """

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        """Maskiert ``record`` und gibt immer ``True`` zurück."""
        secrets = _known_secrets(self._secrets)
        if not secrets:
            return True
        message = record.getMessage()
        masked = _mask(message, secrets)
        if masked != message:
            record.msg = masked
            record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = _mask(record.exc_text, secrets)
        if record.stack_info:
            record.stack_info = _mask(record.stack_info, secrets)
        return True


def reset_logging() -> None:
    """Entfernt und schließt die von :func:`setup_logging` angelegten Handler."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() in _HANDLER_NAMES:
            root.removeHandler(handler)
            handler.close()


def setup_logging(settings: Settings, *, console: bool = True) -> Path:
    """Richtet Logdatei (rotierend, 5 × 5 MB) und Konsolenausgabe ein.

    Mehrfacher Aufruf ersetzt die zuvor angelegten Handler, statt neue hinzuzufügen.
    Der API-Key aus ``settings`` wird als Geheimnis registriert und nie geloggt.

    Args:
        settings: Einstellungen; verwendet werden ``log_dir`` und ``log_level``.
        console: Zusätzlich auf ``stderr`` loggen.

    Returns:
        Pfad der Logdatei.
    """
    reset_logging()
    for secret in settings.secret_values():
        register_secret(secret)

    log_dir = settings.log_path
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME

    formatter = logging.Formatter(LOG_FORMAT)
    masking = SecretMaskingFilter()

    handlers: list[logging.Handler] = [
        RotatingFileHandler(
            log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
        )
    ]
    if console:
        handlers.append(logging.StreamHandler(sys.stderr))

    root = logging.getLogger()
    for handler, name in zip(handlers, _HANDLER_NAMES, strict=False):
        handler.set_name(name)
        handler.setFormatter(formatter)
        handler.addFilter(masking)
        root.addHandler(handler)
    root.setLevel(settings.log_level)
    return log_file
