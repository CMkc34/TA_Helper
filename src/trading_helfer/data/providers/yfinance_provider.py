"""Datenquelle Yahoo Finance über das Paket ``yfinance``.

Liefert rohe Tageskurse plus bereinigten Schlusskurs, Splits und Dividenden sowie
Stammdaten. Symbole werden in Yahoo-Schreibweise übersetzt (``BRK.B`` → ``BRK-B``).

Fehlerbehandlung: ``yfinance`` verschluckt Fehler standardmäßig und liefert dann ein
leeres Ergebnis. Damit Netzwerkfehler von unbekannten Symbolen unterscheidbar sind,
schaltet der Provider ``yf.config.debug.hide_exceptions`` ab. Danach gilt:

- unbekanntes Symbol oder keine Kurse im Zeitraum → leeres Ergebnis,
- Netzwerkfehler, Rate-Limit oder „Yahoo ist down“ → bis zu drei Wiederholungen,
  danach :class:`ProviderError`,
- jede andere Ausnahme → sofort :class:`ProviderError`.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from datetime import date, timedelta
from typing import Any, Final, Protocol

import pandas as pd
import yfinance as yf
from yfinance.exceptions import (
    YFDataException,
    YFRateLimitError,
    YFTickerMissingError,
)

from trading_helfer.core.bars import BAR_COLUMNS, PRICE_COLUMNS, empty_bars, validate_bars
from trading_helfer.data.fundamentals import empty_fundamentals, normalize_fundamentals
from trading_helfer.data.providers.base import (
    ACTION_COLUMNS,
    ActionType,
    InstrumentInfo,
    ProviderError,
    empty_actions,
)
from trading_helfer.data.providers.rate_limit import (
    DEFAULT_RETRY_DELAYS,
    RetryPolicy,
    TokenBucket,
    call_with_retry,
)

__all__ = ["PROVIDER_NAME", "TickerLike", "YFinanceProvider", "to_yahoo_symbol"]

PROVIDER_NAME: Final = "yfinance"

RETRYABLE_ERRORS: Final[tuple[type[BaseException], ...]] = (
    OSError,  # requests, curl_cffi, Socket- und Zeitüberschreitungsfehler
    YFRateLimitError,
    YFDataException,
)
"""Vorübergehende Fehler, bei denen die Anfrage wiederholt wird."""

_HISTORY_COLUMNS: Final[Mapping[str, str]] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
_ACTION_SOURCE_COLUMNS: Final[Mapping[str, ActionType]] = {
    "Stock Splits": ActionType.SPLIT,
    "Dividends": ActionType.DIVIDEND,
}

_STATEMENT_METRICS: Final[Mapping[str, Mapping[str, tuple[str, ...]]]] = {
    "income": {
        "revenue": ("TotalRevenue", "OperatingRevenue"),
        "cost_of_revenue": ("CostOfRevenue", "ReconciledCostOfRevenue"),
        "gross_profit": ("GrossProfit",),
        "operating_income": ("OperatingIncome", "TotalOperatingIncomeAsReported"),
        "net_income": ("NetIncome", "NetIncomeCommonStockholders"),
        "ebitda": ("EBITDA", "NormalizedEBITDA"),
        "eps_diluted": ("DilutedEPS",),
        "interest_expense": ("InterestExpense", "InterestExpenseNonOperating"),
    },
    "balance": {
        "total_assets": ("TotalAssets",),
        "total_liabilities": ("TotalLiabilitiesNetMinorityInterest",),
        "total_equity": (
            "StockholdersEquity",
            "CommonStockEquity",
            "TotalEquityGrossMinorityInterest",
        ),
        "total_debt": ("TotalDebt",),
        "long_term_debt": ("LongTermDebt", "LongTermDebtAndCapitalLeaseObligation"),
        "cash": ("CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments"),
        "current_assets": ("CurrentAssets",),
        "current_liabilities": ("CurrentLiabilities",),
        "inventory": ("Inventory",),
        "retained_earnings": ("RetainedEarnings",),
        "shares_outstanding": ("OrdinarySharesNumber", "ShareIssued"),
    },
    "cashflow": {
        "operating_cashflow": (
            "OperatingCashFlow",
            "CashFlowFromContinuingOperatingActivities",
        ),
        "capex": ("CapitalExpenditure",),
        "dividends_paid": ("CashDividendsPaid", "CommonStockDividendPaid"),
    },
}
"""Je Abschluss: fester Metrikname → Yahoo-Zeilen in Vorrangreihenfolge.

Fehlt die erste Zeile (oder ihr Wert in einer Periode), gilt die nächste.
"""

_STATEMENT_METHODS: Final[Mapping[str, str]] = {
    "income": "get_income_stmt",
    "balance": "get_balance_sheet",
    "cashflow": "get_cash_flow",
}
_FREQUENCIES: Final[Mapping[str, str]] = {"quarterly": "Q", "yearly": "FY"}

_log = logging.getLogger(__name__)


class TickerLike(Protocol):
    """Der Teil von ``yfinance.Ticker``, den der Provider nutzt."""

    def history(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Kursverlauf wie ``yfinance.Ticker.history``."""
        ...

    @property
    def actions(self) -> pd.DataFrame:
        """Dividenden und Splits wie ``yfinance.Ticker.actions``."""
        ...

    @property
    def info(self) -> Mapping[str, Any]:
        """Stammdaten wie ``yfinance.Ticker.info``."""
        ...

    def get_income_stmt(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Gewinn- und Verlustrechnung wie ``yfinance.Ticker.get_income_stmt``."""
        ...

    def get_balance_sheet(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Bilanz wie ``yfinance.Ticker.get_balance_sheet``."""
        ...

    def get_cash_flow(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Kapitalflussrechnung wie ``yfinance.Ticker.get_cash_flow``."""
        ...


def to_yahoo_symbol(symbol: str) -> str:
    """Übersetzt ein Symbol in Yahoo-Schreibweise.

    Leerzeichen werden entfernt, Buchstaben großgeschrieben, Punkte werden zu
    Bindestrichen (``brk.b`` → ``BRK-B``). Indizes mit ``^`` bleiben unverändert.

    Args:
        symbol: Symbol in App-Schreibweise.

    Returns:
        Symbol für Yahoo.

    Raises:
        ValueError: Das Symbol ist leer.
    """
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("Das Symbol darf nicht leer sein.")
    return cleaned.replace(".", "-")


def _default_ticker_factory(symbol: str) -> TickerLike:
    ticker: TickerLike = yf.Ticker(symbol)
    return ticker


class YFinanceProvider:
    """Kurse, Corporate Actions und Stammdaten von Yahoo Finance.

    Erfüllt :class:`~trading_helfer.data.providers.base.PriceProvider`,
    :class:`~trading_helfer.data.providers.base.InstrumentProvider` und
    :class:`~trading_helfer.data.providers.base.FundamentalsProvider`.
    """

    name: str = PROVIDER_NAME

    def __init__(
        self,
        *,
        ticker_factory: Callable[[str], TickerLike] | None = None,
        limiter: TokenBucket | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Legt den Provider an.

        Args:
            ticker_factory: Erzeugt ein Ticker-Objekt zu einem Yahoo-Symbol;
                Standard ist ``yfinance.Ticker``. Tests übergeben hier Attrappen.
            limiter: Rate-Limit; Standard 2 Anfragen pro Sekunde.
            retry_policy: Wiederholungen; Standard 3 Wiederholungen mit 2, 4, 8 s.
            sleep: Wartefunktion für die Pausen zwischen Wiederholungen.
        """
        if ticker_factory is None:
            # Fehler sichtbar machen, statt sie als leeres Ergebnis zu verschlucken.
            yf.config.debug.hide_exceptions = False
            ticker_factory = _default_ticker_factory
        self._ticker_factory = ticker_factory
        self._limiter = limiter if limiter is not None else TokenBucket(sleep=sleep)
        self._retry_policy = retry_policy or RetryPolicy(
            delays=DEFAULT_RETRY_DELAYS, retry_on=RETRYABLE_ERRORS
        )
        self._sleep = sleep

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Rohe Tageskurse plus ``adj_close`` von ``start`` bis einschließlich ``end``.

        Args:
            symbol: Symbol in App-Schreibweise (z. B. ``BRK.B``).
            start: Erster Tag.
            end: Letzter Tag (einschließlich).

        Returns:
            Spalten ``date, open, high, low, close, volume, adj_close``; ``date`` ohne
            Zeitzone, aufsteigend sortiert. Leer, wenn Yahoo das Symbol nicht kennt
            oder im Zeitraum keine Kurse hat.

        Raises:
            ValueError: ``start`` liegt nach ``end`` oder das Symbol ist leer.
            ProviderError: Yahoo war nach allen Versuchen nicht erreichbar.
        """
        if start > end:
            raise ValueError("Der Start darf nicht nach dem Ende liegen.")
        yahoo_symbol = to_yahoo_symbol(symbol)

        def request() -> pd.DataFrame:
            return self._ticker_factory(yahoo_symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # Yahoo: Ende exklusiv
                interval="1d",
                auto_adjust=False,
                actions=False,
                repair=False,
            )

        raw = self._call(symbol, request)
        if raw is None:
            return empty_bars()
        return self._normalize_history(symbol, raw, start, end)

    def fetch_actions(self, symbol: str) -> pd.DataFrame:
        """Alle Splits und Dividenden des Symbols.

        Args:
            symbol: Symbol in App-Schreibweise.

        Returns:
            Spalten ``date, type, value``, sortiert nach Datum und Art; leer, wenn
            nichts da ist.

        Raises:
            ValueError: Das Symbol ist leer.
            ProviderError: Yahoo war nach allen Versuchen nicht erreichbar.
        """
        yahoo_symbol = to_yahoo_symbol(symbol)
        raw = self._call(symbol, lambda: self._ticker_factory(yahoo_symbol).actions)
        if raw is None:
            return empty_actions()
        return self._normalize_actions(symbol, raw)

    def fetch_instrument(self, symbol: str) -> InstrumentInfo | None:
        """Stammdaten (Name, Börse, Sektor, Branche, Währung).

        Args:
            symbol: Symbol in App-Schreibweise.

        Returns:
            Stammdaten oder ``None``, wenn Yahoo das Symbol nicht kennt.

        Raises:
            ValueError: Das Symbol ist leer.
            ProviderError: Yahoo war nach allen Versuchen nicht erreichbar.
        """
        yahoo_symbol = to_yahoo_symbol(symbol)
        raw = self._call(symbol, lambda: self._ticker_factory(yahoo_symbol).info)
        if not raw:
            return None
        if not isinstance(raw, Mapping):
            raise ProviderError(
                self.name, symbol, f"Unerwartete Stammdaten vom Typ {type(raw).__name__}."
            )
        name = _text(raw.get("longName")) or _text(raw.get("shortName"))
        if name is None and _text(raw.get("quoteType")) is None:
            return None
        return InstrumentInfo(
            symbol=symbol.strip().upper(),
            name=name,
            exchange=_text(raw.get("exchange")),
            sector=_text(raw.get("sector")),
            industry=_text(raw.get("industry")),
            currency=_text(raw.get("currency")),
            source=self.name,
        )

    def fetch_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Quartals- und Jahreszahlen aus GuV, Bilanz und Kapitalflussrechnung.

        Die Yahoo-Zeilen werden auf die festen Namen aus
        :data:`~trading_helfer.data.fundamentals.METRICS` abgebildet. Was Yahoo nicht
        meldet, fehlt im Ergebnis (nie ``0``). Yahoo liefert kein
        Veröffentlichungsdatum, daher ist ``reported_date`` immer das konservative
        Datum (Periodenende + 45 bzw. 75 Tage).

        Args:
            symbol: Symbol in App-Schreibweise.

        Returns:
            Spalten ``period_end, period_type, metric, value, reported_date``; leer,
            wenn Yahoo das Symbol nicht kennt.

        Raises:
            ValueError: Das Symbol ist leer.
            ProviderError: Yahoo war nach allen Versuchen nicht erreichbar oder
                antwortete fehlerhaft.
        """
        yahoo_symbol = to_yahoo_symbol(symbol)
        frames: list[pd.DataFrame] = []
        for statement, method in _STATEMENT_METHODS.items():
            for freq, period_type in _FREQUENCIES.items():

                def request(method: str = method, freq: str = freq) -> pd.DataFrame:
                    ticker = self._ticker_factory(yahoo_symbol)
                    result: pd.DataFrame = getattr(ticker, method)(freq=freq, pretty=False)
                    return result

                raw = self._call(symbol, request)
                if raw is None:
                    return empty_fundamentals()
                frames.append(
                    self._normalize_statement(
                        symbol, raw, _STATEMENT_METRICS[statement], period_type
                    )
                )
        combined = pd.concat([f for f in frames if not f.empty] or [empty_fundamentals()])
        return normalize_fundamentals(combined)

    def _normalize_statement(
        self,
        symbol: str,
        raw: pd.DataFrame,
        mapping: Mapping[str, tuple[str, ...]],
        period_type: str,
    ) -> pd.DataFrame:
        """Ein Abschluss (Zeilen = Yahoo-Posten, Spalten = Periodenende) ins Langformat."""
        if not isinstance(raw, pd.DataFrame):
            raise ProviderError(
                self.name, symbol, f"Unerwartete Antwort vom Typ {type(raw).__name__}."
            )
        if raw.empty:
            return empty_fundamentals()
        try:
            period_ends = pd.DatetimeIndex(pd.to_datetime(raw.columns))
        except (ValueError, TypeError) as exc:
            raise ProviderError(
                self.name, symbol, f"Unerwartete Perioden im Abschluss: {exc}"
            ) from exc
        table = raw.copy()
        table.columns = period_ends
        table = table.apply(pd.to_numeric, errors="coerce")
        rows: list[pd.DataFrame] = []
        for metric, candidates in mapping.items():
            values = pd.Series(float("nan"), index=period_ends, dtype="float64")
            for candidate in candidates:
                if candidate in table.index:
                    row = table.loc[candidate]
                    if isinstance(row, pd.DataFrame):  # doppelte Zeile: erste zählt
                        row = row.iloc[0]
                    values = values.combine_first(row.astype("float64"))
            values = values.dropna()
            if values.empty:
                continue
            rows.append(
                pd.DataFrame(
                    {
                        "period_end": values.index,
                        "period_type": period_type,
                        "metric": metric,
                        "value": values.to_numpy(),
                    }
                )
            )
        if not rows:
            return empty_fundamentals()
        return pd.concat(rows, ignore_index=True)

    def _call[T](self, symbol: str, request: Callable[[], T]) -> T | None:
        """Führt eine Anfrage mit Rate-Limit und Wiederholungen aus.

        Returns:
            Antwort oder ``None``, wenn Yahoo das Symbol nicht kennt.
        """
        try:
            return call_with_retry(
                request, self._retry_policy, sleep=self._sleep, limiter=self._limiter
            )
        except YFTickerMissingError as exc:
            _log.info("%s: keine Daten für %s (%s).", self.name, symbol, exc)
            return None
        except self._retry_policy.retry_on as exc:
            attempts = self._retry_policy.max_attempts
            raise ProviderError(
                self.name,
                symbol,
                f"Anfrage nach {attempts} Versuchen fehlgeschlagen: {exc}",
            ) from exc
        except Exception as exc:
            raise ProviderError(self.name, symbol, f"Unerwarteter Fehler: {exc!r}") from exc

    def _normalize_history(
        self, symbol: str, raw: pd.DataFrame, start: date, end: date
    ) -> pd.DataFrame:
        if not isinstance(raw, pd.DataFrame):
            raise ProviderError(
                self.name, symbol, f"Unerwartete Antwort vom Typ {type(raw).__name__}."
            )
        if raw.empty:
            return empty_bars()
        missing = [c for c in _HISTORY_COLUMNS if c not in raw.columns and c != "Adj Close"]
        if missing:
            raise ProviderError(
                self.name, symbol, f"Spalte(n) fehlen in der Antwort: {', '.join(missing)}."
            )
        df = raw.rename(columns=dict(_HISTORY_COLUMNS))
        if "adj_close" not in df.columns:
            _log.warning("%s: %s ohne Adj Close, verwende Close.", self.name, symbol)
            df["adj_close"] = df["close"]
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "date"})
        df = df.loc[:, [*BAR_COLUMNS, "adj_close"]]
        # Yahoo liefert gelegentlich leere Zeilen (z. B. reine Dividendentage).
        df = df.dropna(subset=[*PRICE_COLUMNS, "adj_close"], how="any")
        df["volume"] = df["volume"].fillna(0)
        try:
            bars = validate_bars(df)
        except ValueError as exc:
            raise ProviderError(self.name, symbol, f"Ungültige Kursdaten: {exc}") from exc
        in_range = (bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))
        return bars.loc[in_range].reset_index(drop=True)

    def _normalize_actions(self, symbol: str, raw: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(raw, pd.DataFrame):
            raise ProviderError(
                self.name, symbol, f"Unerwartete Antwort vom Typ {type(raw).__name__}."
            )
        if raw.empty:
            return empty_actions()
        index = pd.DatetimeIndex(pd.to_datetime(raw.index))
        if index.tz is not None:
            index = index.tz_localize(None)
        dates = index.normalize().astype("datetime64[ns]")
        frames = []
        for column, action_type in _ACTION_SOURCE_COLUMNS.items():
            if column not in raw.columns:
                continue
            values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype="float64")
            part = pd.DataFrame({"date": dates, "type": action_type.value, "value": values})
            frames.append(part[part["value"].notna() & (part["value"] != 0)])
        if not frames:
            return empty_actions()
        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(["date", "type"], kind="mergesort")
        return result.loc[:, list(ACTION_COLUMNS)].reset_index(drop=True)


def _text(value: object) -> str | None:
    """Nicht leerer Text oder ``None``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None
