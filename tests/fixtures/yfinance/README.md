# yfinance-Fixtures

Synthetische Antworten im Format von `yfinance.Ticker` (keine echten Kursdaten).
Symbol `SYNTH`: steigender Trend, Split 4:1 am 2024-01-08, Dividende 0,25 USD am 2024-01-04.

- `SYNTH_history.csv`: `Ticker.history(auto_adjust=False, actions=False)`, Index `Date`
  mit Zeitzone America/New_York; enthält eine leere Zeile (2024-01-05) wie Yahoo sie
  gelegentlich liefert.
- `SYNTH_actions.csv`: `Ticker.actions` (Spalten `Dividends`, `Stock Splits`).
- `SYNTH_info.json`: Auszug aus `Ticker.info`.
- `SYNTH_<abschluss>_<freq>.csv`: `Ticker.get_income_stmt`, `get_balance_sheet` und
  `get_cash_flow` mit `pretty=False` und `freq` = `yearly` bzw. `quarterly`. Zeilen
  sind Yahoo-Posten, Spalten Periodenenden (neueste zuerst). Absichtlich enthalten:
  eine leere Jahresspalte 2021, ein fehlender Wert (`InterestExpense` 2022), Posten
  nur unter Ausweichnamen (Quartal: `OperatingRevenue`, `CommonStockEquity`,
  `ShareIssued`, Kasse 2024-03 nur als `CashCashEquivalentsAndShortTermInvestments`),
  ein nicht abgebildeter Posten (`ResearchAndDevelopment`) und Metriken, die ganz
  fehlen (`ebitda` im Quartal, `inventory` im Quartal, `dividends_paid` im Quartal).
