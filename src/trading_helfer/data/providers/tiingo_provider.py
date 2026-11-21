"""Datenquelle Tiingo über die REST-API (Endpunkt ``/tiingo/daily/<symbol>/prices``).

Liefert rohe Tageskurse plus bereinigten Schlusskurs sowie Splits und Dividenden, die
Tiingo in derselben Antwort mitschickt (``splitFactor``, ``divCash``). Der API-Key kommt
aus den Einstellungen (``Settings.tiingo_api_key``, Umgebungsvariable
``TH_TIINGO_API_KEY``).

Kontingent (Starter-Plan): 50 Anfragen pro Stunde und 1.000 pro Tag. Jede Anfrage wird
in einem :class:`~trading_helfer.data.providers.usage.RequestCounter` vermerkt,
standardmäßig in der Datenbank. Ist ein Kontingent erschöpft, pausiert der Provider:
er schickt keine Anfrage mehr ab, sondern meldet :class:`TiingoQuotaExceededError` mit dem
Zeitpunkt, ab dem es weitergeht.

Fehlerbehandlung:

- fehlender API-Key → :class:`TiingoApiKeyMissingError` („Tiingo-API-Key fehlt“),
- unbekanntes Symbol (HTTP 404) → leeres Ergebnis,
- Netzwerkfehler und HTTP 5xx → bis zu drei Wiederholungen, danach
  :class:`~trading_helfer.data.providers.base.ProviderError`,
- HTTP 429 (Tiingo meldet selbst ein erschöpftes Kontingent) → sofort
  :class:`TiingoQuotaExceededError`,
- HTTP 401/403 → :class:`~trading_helfer.data.providers.base.ProviderError`
  „API-Key ungültig“.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final

import pandas as pd
import requests
from pydantic import SecretStr

from trading_helfer.core.bars import BAR_COLUMNS, empty_bars, validate_bars
from trading_helfer.core.config import Settings
from trading_helfer.data.providers.base import (
    ACTION_COLUMNS,
    ActionType,
    ProviderError,
    empty_actions,
)
from trading_helfer.data.providers.rate_limit import (
    DEFAULT_RETRY_DELAYS,
    RetryPolicy,
    TokenBucket,
    call_with_retry,
)
from trading_helfer.data.providers.usage import (
    MemoryRequestCounter,
    QuotaWindow,
    RequestCounter,
    next_free_slot,
)

__all__ = [
    "BASE_URL",
    "MISSING_KEY_MESSAGE",
    "PROVIDER_NAME",
    "STARTER_QUOTAS",
    "TiingoApiKeyMissingError",
    "TiingoProvider",
    "TiingoQuotaExceededError",
    "to_tiingo_symbol",
]

PROVIDER_NAME: Final = "tiingo"
BASE_URL: Final = "https://api.tiingo.com"
MISSING_KEY_MESSAGE: Final = "Tiingo-API-Key fehlt"

STARTER_QUOTAS: Final[tuple[QuotaWindow, ...]] = (
    QuotaWindow(limit=50, period=timedelta(hours=1), label="pro Stunde"),
    QuotaWindow(limit=1000, period=timedelta(days=1), label="pro Tag"),
)
"""Kontingente des kostenlosen Starter-Plans."""

HISTORY_START: Final = date(1900, 1, 1)
"""Startdatum, um für Corporate Actions die ganze Historie abzufragen."""

DEFAULT_TIMEOUT: Final = 30.0
"""Zeitlimit je HTTP-Anfrage in Sekunden."""

_PRICE_FIELDS: Final[Mapping[str, str]] = {
    "date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "adjClose": "adj_close",
}

_log = logging.getLogger(__name__)


class TiingoApiKeyMissingError(ProviderError):
    """In den Einstellungen ist kein Tiingo-API-Key hinterlegt."""

    def __init__(self, symbol: str | None = None) -> None:
        super().__init__(PROVIDER_NAME, symbol, MISSING_KEY_MESSAGE)


class TiingoQuotaExceededError(ProviderError):
    """Das Anfragekontingent ist erschöpft; der Provider pausiert bis ``resume_at``.

    Attributes:
        resume_at: Zeitpunkt (UTC, ohne Zone), ab dem wieder Anfragen möglich sind,
            oder ``None``, wenn Tiingo selbst abgelehnt hat und der Zeitpunkt unbekannt ist.
    """

    def __init__(self, symbol: str | None, reason: str, resume_at: datetime | None) -> None:
        super().__init__(PROVIDER_NAME, symbol, reason)
        self.resume_at = resume_at


class _ServerError(OSError):
    """HTTP 5xx; wird wie ein Netzwerkfehler wiederholt."""


def to_tiingo_symbol(symbol: str) -> str:
    """Übersetzt ein Symbol in Tiingo-Schreibweise (``brk.b`` → ``BRK-B``).

    Args:
        symbol: Symbol in App-Schreibweise.

    Returns:
        Symbol für Tiingo.

    Raises:
        ValueError: Das Symbol ist leer.
    """
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("Das Symbol darf nicht leer sein.")
    return cleaned.replace(".", "-")


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class TiingoProvider:
    """Tageskurse und Corporate Actions von Tiingo.

    Erfüllt :class:`~trading_helfer.data.providers.base.PriceProvider`.
    """

    name: str = PROVIDER_NAME

    def __init__(
        self,
        api_key: str | SecretStr | None,
        *,
        counter: RequestCounter | None = None,
        quotas: tuple[QuotaWindow, ...] = STARTER_QUOTAS,
        session: requests.Session | None = None,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        limiter: TokenBucket | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Legt den Provider an.

        Ein fehlender Key ist hier noch kein Fehler; erst eine Anfrage meldet
        :class:`TiingoApiKeyMissingError`. So kann die App den Provider immer anlegen.

        Args:
            api_key: Tiingo-API-Key oder ``None``.
            counter: Speicher der Anfragen für die Kontingente; in der App der
                Datenbank-Zähler, Standard ist ein Zähler im Arbeitsspeicher.
            quotas: Kontingente; Standard ist der Starter-Plan.
            session: HTTP-Session; Standard ist eine neue ``requests.Session``.
            base_url: Basisadresse der API.
            timeout: Zeitlimit je Anfrage in Sekunden.
            limiter: Rate-Limit; Standard 2 Anfragen pro Sekunde.
            retry_policy: Wiederholungen; Standard 3 Wiederholungen mit 2, 4, 8 s.
            sleep: Wartefunktion für Rate-Limit und Wiederholungen.
            clock: Liefert die aktuelle Zeit in UTC ohne Zone.
        """
        if isinstance(api_key, SecretStr):
            api_key = api_key.get_secret_value()
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self._counter: RequestCounter = counter if counter is not None else MemoryRequestCounter()
        self._quotas = quotas
        self._session = session if session is not None else requests.Session()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._limiter = limiter if limiter is not None else TokenBucket(sleep=sleep)
        self._retry_policy = retry_policy or RetryPolicy(
            delays=DEFAULT_RETRY_DELAYS,
            retry_on=(requests.ConnectionError, requests.Timeout, _ServerError),
        )
        self._sleep = sleep
        self._clock = clock

    @classmethod
    def from_settings(
        cls, settings: Settings, counter: RequestCounter | None = None, **kwargs: Any
    ) -> TiingoProvider:
        """Legt den Provider mit dem API-Key aus den Einstellungen an.

        Args:
            settings: Einstellungen der App (``tiingo_api_key``).
            counter: Zähler für die Kontingente, in der App der Datenbank-Zähler.
            **kwargs: Weitere Argumente für den Konstruktor.

        Returns:
            Neuer Provider.
        """
        return cls(settings.tiingo_api_key, counter=counter, **kwargs)

    @property
    def has_api_key(self) -> bool:
        """``True``, wenn ein API-Key hinterlegt ist."""
        return self._api_key is not None

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Rohe Tageskurse plus ``adj_close`` von ``start`` bis einschließlich ``end``.

        Args:
            symbol: Symbol in App-Schreibweise (z. B. ``BRK.B``).
            start: Erster Tag.
            end: Letzter Tag (einschließlich).

        Returns:
            Spalten ``date, open, high, low, close, volume, adj_close``; ``date`` ohne
            Zeitzone, aufsteigend sortiert. Leer, wenn Tiingo das Symbol nicht kennt
            oder im Zeitraum keine Kurse hat.

        Raises:
            ValueError: ``start`` liegt nach ``end`` oder das Symbol ist leer.
            TiingoApiKeyMissingError: Kein API-Key hinterlegt.
            TiingoQuotaExceededError: Kontingent erschöpft, der Provider pausiert.
            ProviderError: Tiingo war nach allen Versuchen nicht erreichbar.
        """
        if start > end:
            raise ValueError("Der Start darf nicht nach dem Ende liegen.")
        rows = self._fetch_prices(symbol, start, end)
        if not rows:
            return empty_bars()
        bars = self._normalize_prices(symbol, rows)
        in_range = (bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))
        return bars.loc[in_range].reset_index(drop=True)

    def fetch_actions(self, symbol: str) -> pd.DataFrame:
        """Alle Splits und Dividenden des Symbols aus der ganzen Kurshistorie.

        Args:
            symbol: Symbol in App-Schreibweise.

        Returns:
            Spalten ``date, type, value``, sortiert nach Datum und Art; leer, wenn
            nichts da ist.

        Raises:
            ValueError: Das Symbol ist leer.
            TiingoApiKeyMissingError: Kein API-Key hinterlegt.
            TiingoQuotaExceededError: Kontingent erschöpft, der Provider pausiert.
            ProviderError: Tiingo war nach allen Versuchen nicht erreichbar.
        """
        rows = self._fetch_prices(symbol, HISTORY_START, self._clock().date())
        if not rows:
            return empty_actions()
        return self._normalize_actions(symbol, rows)

    # -- Anfrage --------------------------------------------------------------------

    def _fetch_prices(self, symbol: str, start: date, end: date) -> list[dict[str, Any]]:
        """Fragt den Preis-Endpunkt ab; leere Liste bei unbekanntem Symbol."""
        tiingo_symbol = to_tiingo_symbol(symbol)
        if self._api_key is None:
            raise TiingoApiKeyMissingError(symbol)
        url = f"{self._base_url}/tiingo/daily/{tiingo_symbol.lower()}/prices"
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "format": "json",
            "resampleFreq": "daily",
        }
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/json",
        }

        def request() -> requests.Response:
            self._check_quota(symbol)
            self._counter.record(self.name, self._clock())
            response = self._session.get(url, params=params, headers=headers, timeout=self._timeout)
            if response.status_code >= 500:
                raise _ServerError(f"HTTP {response.status_code}")
            return response

        try:
            response = call_with_retry(
                request, self._retry_policy, sleep=self._sleep, limiter=self._limiter
            )
        except self._retry_policy.retry_on as exc:
            attempts = self._retry_policy.max_attempts
            raise ProviderError(
                self.name, symbol, f"Anfrage nach {attempts} Versuchen fehlgeschlagen: {exc}"
            ) from exc
        except requests.RequestException as exc:
            raise ProviderError(self.name, symbol, f"Unerwarteter Fehler: {exc!r}") from exc
        return self._parse_response(symbol, response)

    def _check_quota(self, symbol: str) -> None:
        """Wirft :class:`TiingoQuotaExceededError`, wenn ein Kontingent erschöpft ist."""
        blocked = next_free_slot(self._counter, self.name, self._quotas, self._clock())
        if blocked is None:
            return
        resume_at, window = blocked
        reason = (
            f"Tiingo-Limit erreicht ({window.limit} Anfragen {window.label}). "
            f"Der Provider pausiert bis {resume_at:%d.%m.%Y %H:%M} UTC."
        )
        _log.warning("%s", reason)
        raise TiingoQuotaExceededError(symbol, reason, resume_at)

    def _parse_response(self, symbol: str, response: requests.Response) -> list[dict[str, Any]]:
        status = response.status_code
        if status == 404:
            _log.info("%s: keine Daten für %s (%s).", self.name, symbol, _detail(response))
            return []
        if status == 429:
            raise TiingoQuotaExceededError(
                symbol,
                f"Tiingo-Limit erreicht, Tiingo lehnt weitere Anfragen ab: {_detail(response)}",
                None,
            )
        if status in (401, 403):
            raise ProviderError(
                self.name, symbol, f"Tiingo-API-Key ungültig (HTTP {status}): {_detail(response)}"
            )
        if status != 200:
            raise ProviderError(
                self.name, symbol, f"Unerwartete Antwort HTTP {status}: {_detail(response)}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(self.name, symbol, "Antwort ist kein gültiges JSON.") from exc
        if not isinstance(payload, list) or not all(isinstance(r, dict) for r in payload):
            raise ProviderError(
                self.name, symbol, f"Unerwartete Antwort vom Typ {type(payload).__name__}."
            )
        return payload

    # -- Umwandlung -----------------------------------------------------------------

    def _normalize_prices(self, symbol: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
        raw = pd.DataFrame.from_records(rows)
        missing = [field for field in _PRICE_FIELDS if field not in raw.columns]
        if missing:
            raise ProviderError(
                self.name, symbol, f"Feld(er) fehlen in der Antwort: {', '.join(missing)}."
            )
        df = raw.loc[:, list(_PRICE_FIELDS)].rename(columns=dict(_PRICE_FIELDS))
        df["date"] = _parse_dates(df["date"])
        try:
            bars = validate_bars(df.loc[:, [*BAR_COLUMNS, "adj_close"]])
        except ValueError as exc:
            raise ProviderError(self.name, symbol, f"Ungültige Kursdaten: {exc}") from exc
        return bars

    def _normalize_actions(self, symbol: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
        raw = pd.DataFrame.from_records(rows)
        if "date" not in raw.columns:
            raise ProviderError(self.name, symbol, "Feld(er) fehlen in der Antwort: date.")
        dates = _parse_dates(raw["date"])
        frames = []
        for column, action_type, neutral in (
            ("splitFactor", ActionType.SPLIT, 1.0),
            ("divCash", ActionType.DIVIDEND, 0.0),
        ):
            if column not in raw.columns:
                continue
            values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype="float64")
            part = pd.DataFrame({"date": dates, "type": action_type.value, "value": values})
            frames.append(part[part["value"].notna() & (part["value"] != neutral)])
        if not frames:
            return empty_actions()
        result = pd.concat(frames, ignore_index=True)
        if result.empty:
            return empty_actions()
        result = result.sort_values(["date", "type"], kind="mergesort")
        return result.loc[:, list(ACTION_COLUMNS)].reset_index(drop=True)


def _parse_dates(values: pd.Series) -> pd.Series:
    """Tiingo liefert ``2024-01-02T00:00:00.000Z``; daraus wird der Tag ohne Zone."""
    parsed = pd.to_datetime(values, utc=True).dt.tz_localize(None).dt.normalize()
    return parsed.astype("datetime64[ns]")


def _detail(response: requests.Response) -> str:
    """Fehlertext aus der Antwort (Feld ``detail`` oder Rohtext, gekürzt)."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and payload.get("detail"):
        return str(payload["detail"])
    return response.text[:200] or response.reason or ""
