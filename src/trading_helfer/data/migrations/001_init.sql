-- 001_init: alle Tabellen des Datenmodells (docs/SPEC.md, Abschnitt „Datenmodell“).

CREATE TABLE instruments (
    symbol      VARCHAR PRIMARY KEY,
    name        VARCHAR,
    exchange    VARCHAR,
    sector      VARCHAR,
    industry    VARCHAR,
    currency    VARCHAR,
    is_active   BOOLEAN,
    first_date  DATE,
    last_date   DATE,
    source      VARCHAR
);

CREATE TABLE index_membership (
    index_code  VARCHAR NOT NULL,
    symbol      VARCHAR NOT NULL,
    start_date  DATE    NOT NULL,
    end_date    DATE,
    PRIMARY KEY (index_code, symbol, start_date)
);

CREATE TABLE bars_daily (
    symbol      VARCHAR NOT NULL,
    date        DATE    NOT NULL,
    open        DOUBLE  NOT NULL,
    high        DOUBLE  NOT NULL,
    low         DOUBLE  NOT NULL,
    close       DOUBLE  NOT NULL,
    volume      BIGINT  NOT NULL,
    adj_close   DOUBLE  NOT NULL,
    adj_factor  DOUBLE  NOT NULL,
    source      VARCHAR,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE corporate_actions (
    symbol      VARCHAR NOT NULL,
    date        DATE    NOT NULL,
    type        VARCHAR NOT NULL,
    value       DOUBLE  NOT NULL,
    PRIMARY KEY (symbol, date, type)
);

CREATE TABLE fundamentals (
    symbol         VARCHAR NOT NULL,
    period_end     DATE    NOT NULL,
    period_type    VARCHAR NOT NULL,
    metric         VARCHAR NOT NULL,
    value          DOUBLE,
    reported_date  DATE,
    source         VARCHAR,
    PRIMARY KEY (symbol, period_end, period_type, metric)
);

CREATE SEQUENCE update_log_id_seq START 1;

CREATE TABLE update_log (
    id              BIGINT PRIMARY KEY DEFAULT nextval('update_log_id_seq'),
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    provider        VARCHAR,
    symbols_ok      INTEGER,
    symbols_failed  INTEGER,
    message         VARCHAR
);

CREATE TABLE backtest_runs (
    run_id            UUID PRIMARY KEY,
    strategy_id       VARCHAR NOT NULL,
    strategy_version  VARCHAR,
    params_json       VARCHAR,
    universe          VARCHAR,
    timeframe         VARCHAR,
    start             DATE,
    "end"             DATE,
    settings_json     VARCHAR,
    metrics_json      VARCHAR,
    created_at        TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE backtest_trades (
    run_id       UUID    NOT NULL,
    trade_id     INTEGER NOT NULL,
    symbol       VARCHAR NOT NULL,
    side         VARCHAR NOT NULL,
    entry_date   DATE,
    entry_price  DOUBLE,
    exit_date    DATE,
    exit_price   DOUBLE,
    qty          DOUBLE,
    pnl          DOUBLE,
    pnl_pct      DOUBLE,
    fees         DOUBLE,
    exit_reason  VARCHAR,
    PRIMARY KEY (run_id, trade_id)
);

CREATE TABLE backtest_equity (
    run_id    UUID   NOT NULL,
    date      DATE   NOT NULL,
    equity    DOUBLE,
    cash      DOUBLE,
    exposure  DOUBLE,
    drawdown  DOUBLE,
    PRIMARY KEY (run_id, date)
);

CREATE TABLE signals (
    scan_id      VARCHAR NOT NULL,
    symbol       VARCHAR NOT NULL,
    date         DATE    NOT NULL,
    strategy_id  VARCHAR,
    signal       VARCHAR NOT NULL,
    strength     DOUBLE,
    note         VARCHAR,
    PRIMARY KEY (scan_id, symbol, date)
);
