-- Anfragen je Datenquelle, damit Tageslimits (z. B. Tiingo: 50/Stunde, 1.000/Tag)
-- über Neustarts hinweg eingehalten werden (Issue D-03).
CREATE TABLE IF NOT EXISTS provider_requests (
    provider     VARCHAR   NOT NULL,
    requested_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_provider_requests ON provider_requests (provider, requested_at);
