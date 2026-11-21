-- 004_backtest_provenance: Herkunft und Details gespeicherter Backtest-Läufe (Issue B-06).
-- strategy_source_hash: SHA-256 des Strategie-Quellcodes zum Zeitpunkt des Laufs.
-- data_last_date / data_symbol_count: Datenstand der Kursdatenbank (letztes Datum, Anzahl Symbole).
-- details_json: übrige Ergebnisteile (Orders, Protokoll, offene Positionen, Spaltentypen).
ALTER TABLE backtest_runs ADD COLUMN strategy_source_hash VARCHAR;
ALTER TABLE backtest_runs ADD COLUMN data_last_date DATE;
ALTER TABLE backtest_runs ADD COLUMN data_symbol_count INTEGER;
ALTER TABLE backtest_runs ADD COLUMN details_json VARCHAR;
CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy ON backtest_runs (strategy_id);
