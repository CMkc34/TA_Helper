# CSV-Fixtures (D-04)

Synthetische Kursdaten für den CSV-Import, keine echten Kurse.

- `per_symbol/`: eine Datei je Symbol (Symbol = Dateiname), mit und ohne `Adj Close`.
- `combined.csv`: eine Datei mit Symbolspalte `Ticker` (Kleinschreibung `synb` absichtlich).
- `custom.csv` + `custom_mapping.toml`: eigene Spaltennamen, deutsches Datumsformat.
- `bad_rows.csv`: 2 gültige Zeilen, 1 exaktes Duplikat und 9 fehlerhafte Zeilen.
