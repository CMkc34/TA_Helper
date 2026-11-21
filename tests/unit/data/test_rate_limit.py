"""Token-Bucket und Wiederholungen mit gefälschter Uhr."""

from __future__ import annotations

import threading

import pytest

from trading_helfer.data.providers.rate_limit import (
    DEFAULT_RETRY_DELAYS,
    RetryPolicy,
    TokenBucket,
    call_with_retry,
)


class FakeClock:
    """Uhr, die nur durch ``sleep`` vorrückt."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_bucket(clock: FakeClock, rate: float = 2.0, capacity: float = 1.0) -> TokenBucket:
    return TokenBucket(rate, capacity, clock=clock, sleep=clock.sleep)


def test_defaults_match_spec() -> None:
    bucket = TokenBucket()
    assert bucket.rate == 2.0
    assert bucket.capacity == 1.0
    assert DEFAULT_RETRY_DELAYS == (2.0, 4.0, 8.0)
    assert RetryPolicy().max_attempts == 4


def test_first_request_does_not_wait() -> None:
    clock = FakeClock()
    assert make_bucket(clock).acquire() == 0.0
    assert clock.sleeps == []


def test_at_most_two_requests_per_second() -> None:
    clock = FakeClock()
    bucket = make_bucket(clock)
    times = []
    for _ in range(10):
        bucket.acquire()
        times.append(clock.now)
    assert times == pytest.approx([i * 0.5 for i in range(10)])
    # In keinem Ein-Sekunden-Fenster mehr als zwei Anfragen.
    for t in times:
        assert sum(t <= other < t + 1.0 for other in times) <= 2


def test_tokens_refill_while_idle() -> None:
    clock = FakeClock()
    bucket = make_bucket(clock)
    bucket.acquire()
    assert not bucket.try_acquire()
    clock.now += 0.5
    assert bucket.try_acquire()
    clock.now += 100.0
    assert bucket.try_acquire()
    assert not bucket.try_acquire()  # Kapazität begrenzt den Vorrat


def test_capacity_allows_burst() -> None:
    clock = FakeClock()
    bucket = make_bucket(clock, rate=1.0, capacity=3.0)
    assert [bucket.try_acquire() for _ in range(4)] == [True, True, True, False]
    assert bucket.acquire() == pytest.approx(1.0)


def test_clock_going_backwards_is_ignored() -> None:
    clock = FakeClock()
    bucket = make_bucket(clock)
    bucket.acquire()
    clock.now -= 10.0
    assert not bucket.try_acquire()


@pytest.mark.parametrize(("rate", "capacity"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.5)])
def test_invalid_parameters(rate: float, capacity: float) -> None:
    with pytest.raises(ValueError, match="muss"):
        TokenBucket(rate, capacity)


def test_thread_safe_token_count() -> None:
    clock = FakeClock()
    bucket = TokenBucket(1.0, 50.0, clock=clock, sleep=clock.sleep)
    results: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(10):
            ok = bucket.try_acquire()
            with lock:
                results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 50


class Flaky:
    """Scheitert ``failures``-mal mit ``error``, danach Erfolg."""

    def __init__(self, failures: int, error: BaseException) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"


def test_retry_succeeds_after_transient_errors() -> None:
    clock = FakeClock()
    func = Flaky(2, ConnectionError("weg"))
    assert call_with_retry(func, RetryPolicy(), sleep=clock.sleep) == "ok"
    assert func.calls == 3
    assert clock.sleeps == [2.0, 4.0]


def test_retry_gives_up_after_all_attempts() -> None:
    clock = FakeClock()
    func = Flaky(99, TimeoutError("zu langsam"))
    with pytest.raises(TimeoutError):
        call_with_retry(func, RetryPolicy(), sleep=clock.sleep)
    assert func.calls == 4
    assert clock.sleeps == [2.0, 4.0, 8.0]


def test_non_retryable_error_is_raised_immediately() -> None:
    clock = FakeClock()
    func = Flaky(1, KeyError("kaputt"))
    with pytest.raises(KeyError):
        call_with_retry(func, RetryPolicy(), sleep=clock.sleep)
    assert func.calls == 1
    assert clock.sleeps == []


def test_retry_uses_limiter_before_each_attempt() -> None:
    clock = FakeClock()
    bucket = make_bucket(clock)
    func = Flaky(1, OSError("weg"))
    call_with_retry(func, RetryPolicy(delays=(0.1,)), sleep=clock.sleep, limiter=bucket)
    # 1. Versuch sofort, 0,1 s Pause, dann wartet der Bucket bis 0,5 s.
    assert clock.sleeps == pytest.approx([0.1, 0.4])
