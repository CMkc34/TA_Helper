# yfinance-Fixtures

Synthetische Antworten im Format von `yfinance.Ticker` (keine echten Kursdaten).
Symbol `SYNTH`: steigender Trend, Split 4:1 am 2024-01-08, Dividende 0,25 USD am 2024-01-04.

- `SYNTH_history.csv`: `Ticker.history(auto_adjust=False, actions=False)`, Index `Date`
  mit Zeitzone America/New_York; enthält eine leere Zeile (2024-01-05) wie Yahoo sie
  gelegentlich liefert.
- `SYNTH_actions.csv`: `Ticker.actions` (Spalten `Dividends`, `Stock Splits`).
- `SYNTH_info.json`: Auszug aus `Ticker.info`.
