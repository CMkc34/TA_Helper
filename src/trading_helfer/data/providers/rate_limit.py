"""Rate-Limit (Token-Bucket) und Wiederholungen für Datenquellen.

Laut Spezifikation stellt jeder Provider höchstens 2 Anfragen pro Sekunde und
wiederholt eine fehlgeschlagene Anfrage bis zu dreimal, mit 2, 4 und 8 Sekunden Pause.
Uhr und Warten sind injizierbar, damit Tests ohne echte Wartezeit laufen.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

__all__ = [
    "DEFAULT_RATE_PER_SECOND",
    "DEFAULT_RETRY_DELAYS",
    "RetryPolicy",
    "TokenBucket",
    "call_with_retry",
]

DEFAULT_RATE_PER_SECOND: Final = 2.0
"""Höchstens so viele Anfragen pro Sekunde je Provider."""

DEFAULT_RETRY_DELAYS: Final[tuple[float, ...]] = (2.0, 4.0, 8.0)
"""Pausen in Sekunden vor der 1., 2. und 3. Wiederholung."""

_log = logging.getLogger(__name__)


class TokenBucket:
    """Thread-sicherer Token-Bucket.

    Der Eimer fasst ``capacity`` Token und füllt sich mit ``rate`` Token pro Sekunde.
    Jede Anfrage verbraucht ein Token; ist keins da, wartet :meth:`acquire`. Mit
    ``capacity = 1`` (Standard) liegen zwischen zwei Anfragen mindestens ``1 / rate``
    Sekunden, sodass in keinem Ein-Sekunden-Fenster mehr als ``rate`` Anfragen liegen.
    """

    def __init__(
        self,
        rate: float = DEFAULT_RATE_PER_SECOND,
        capacity: float = 1.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Legt einen vollen Eimer an.

        Args:
            rate: Nachfüllrate in Token pro Sekunde (> 0).
            capacity: Größe des Eimers in Token (≥ 1).
            clock: Monotone Uhr in Sekunden.
            sleep: Wartefunktion in Sekunden.

        Raises:
            ValueError: ``rate`` oder ``capacity`` ist ungültig.
        """
        if rate <= 0:
            raise ValueError("Die Rate muss größer als 0 sein.")
        if capacity < 1:
            raise ValueError("Die Kapazität muss mindestens 1 sein.")
        self.rate = float(rate)
        self.capacity = float(capacity)
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._updated = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._updated = now

    def try_acquire(self) -> bool:
        """Nimmt ein Token, falls vorhanden, ohne zu warten.

        Returns:
            ``True``, wenn ein Token genommen wurde.
        """
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def acquire(self) -> float:
        """Nimmt ein Token und wartet dafür, falls nötig.

        Returns:
            Gesamte Wartezeit in Sekunden.
        """
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return waited
                wait = (1.0 - self._tokens) / self.rate
            self._sleep(wait)
            waited += wait


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Wann und wie oft eine Anfrage wiederholt wird."""

    delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS
    """Pause vor jeder Wiederholung; ihre Anzahl ist die Zahl der Wiederholungen."""
    retry_on: tuple[type[BaseException], ...] = field(default=(OSError,))
    """Ausnahmen, bei denen wiederholt wird; alle anderen werden sofort weitergereicht."""

    @property
    def max_attempts(self) -> int:
        """Höchstzahl der Versuche (erster Versuch plus Wiederholungen)."""
        return len(self.delays) + 1


def call_with_retry[T](
    func: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = time.sleep,
    limiter: TokenBucket | None = None,
) -> T:
    """Ruft ``func`` auf und wiederholt bei Fehlern aus ``policy.retry_on``.

    Vor jedem Versuch wird ein Token aus ``limiter`` genommen.

    Args:
        func: Aufruf ohne Argumente.
        policy: Pausen und wiederholbare Ausnahmen.
        sleep: Wartefunktion in Sekunden.
        limiter: Optionales Rate-Limit.

    Returns:
        Ergebnis des ersten erfolgreichen Versuchs.

    Raises:
        BaseException: Die letzte Ausnahme, wenn alle Versuche scheitern, oder sofort
            eine Ausnahme, die nicht in ``policy.retry_on`` steht.
    """
    attempt = 0
    while True:
        if limiter is not None:
            limiter.acquire()
        try:
            return func()
        except policy.retry_on as exc:
            if attempt >= len(policy.delays):
                raise
            delay = policy.delays[attempt]
            attempt += 1
            _log.warning(
                "Anfrage fehlgeschlagen (%s), Wiederholung %d von %d in %.0f s.",
                exc,
                attempt,
                len(policy.delays),
                delay,
            )
            sleep(delay)
