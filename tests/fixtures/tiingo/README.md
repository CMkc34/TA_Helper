# Tiingo-Fixtures

Synthetische Antworten im Format von `GET /tiingo/daily/<symbol>/prices?format=json`
(keine echten Kursdaten). Symbol `SYNTH` wie bei den yfinance-Fixtures: steigender Trend,
Split 4:1 am 2024-01-08 (`splitFactor`), Dividende 0,25 USD am 2024-01-04 (`divCash`).

- `SYNTH_prices.json`: Liste der Tageskurse mit allen Feldern, die Tiingo liefert
  (`date` als `2024-01-02T00:00:00.000Z`, roh und bereinigt).
- `not_found.json`: Antwort mit HTTP 404 für ein unbekanntes Symbol.
