-- 005_scans: Kopfdaten gespeicherter Scans für den Verlauf im Scanner-Tab (Issue G-09).
-- Die Signale selbst stehen in der Tabelle signals (gleiche scan_id); ein Scan ohne
-- Signal erscheint nur hier.
CREATE TABLE scans (
    scan_id      VARCHAR PRIMARY KEY,
    created_at   TIMESTAMP NOT NULL,
    strategy_id  VARCHAR   NOT NULL,
    params_json  VARCHAR,
    universe     VARCHAR   NOT NULL,
    timeframe    VARCHAR   NOT NULL,
    as_of        DATE,
    filters_json VARCHAR,
    signal_count INTEGER   NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scans_created ON scans (created_at);
