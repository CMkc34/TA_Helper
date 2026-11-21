-- 003_option_snapshots: tägliche Schnappschüsse der Optionsketten (Issue O-01).
-- Kostenlose historische Optionsdaten gibt es nicht; die App speichert deshalb ab
-- Start jeden Tag die aktuellen Ketten. Ein Schnappschuss je Symbol und Tag; ein
-- zweiter Lauf am selben Tag ersetzt ihn.
CREATE TABLE IF NOT EXISTS option_snapshots (
    symbol            VARCHAR NOT NULL,
    snapshot_date     DATE    NOT NULL,
    expiry            DATE    NOT NULL,
    strike            DOUBLE  NOT NULL,
    type              VARCHAR NOT NULL,
    bid               DOUBLE,
    ask               DOUBLE,
    last              DOUBLE,
    volume            BIGINT,
    open_interest     BIGINT,
    iv                DOUBLE,
    underlying_price  DOUBLE,
    source            VARCHAR,
    PRIMARY KEY (symbol, snapshot_date, expiry, strike, type)
);
