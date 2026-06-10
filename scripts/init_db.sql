-- Scored transactions audit trail
CREATE TABLE IF NOT EXISTS scored_transactions (
    id              SERIAL PRIMARY KEY,
    transaction_id  TEXT NOT NULL UNIQUE,
    card_id         TEXT NOT NULL,
    amount          NUMERIC(12, 2),
    fraud_prob      NUMERIC(6, 4),
    is_fraud        BOOLEAN,
    model           TEXT,
    top_feature     TEXT,
    top_shap        NUMERIC(8, 4),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_st_card_id    ON scored_transactions(card_id);
CREATE INDEX IF NOT EXISTS idx_st_is_fraud   ON scored_transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_st_created_at ON scored_transactions(created_at DESC);

-- Drift reports log
CREATE TABLE IF NOT EXISTS drift_reports (
    id                   SERIAL PRIMARY KEY,
    drift_detected       BOOLEAN,
    n_drifted_features   INT,
    drifted_features     TEXT[],
    report_path          TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);
