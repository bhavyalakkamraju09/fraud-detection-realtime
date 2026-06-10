"""
create_mock_model.py
--------------------
Generates 50K synthetic transactions, trains XGBoost + LightGBM on them,
and saves models to models/ so the streaming pipeline can start immediately.

Run: python scripts/create_mock_model.py
Time: ~2 minutes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from src.features.engineering import FEATURES, TARGET
from src.models.threshold import find_optimal_threshold


def generate_synthetic_dataset(n: int = 50_000, fraud_rate: float = 0.015, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def make_rows(count, is_fraud):
        if is_fraud:
            amount = rng.lognormal(6.0, 0.8, count)
            hour = rng.choice([1, 2, 3, 4, 22, 23], count)
            tx_count_1h = rng.integers(5, 20, count).astype(float)
            zscore = rng.uniform(2.0, 6.0, count)
            email_rate = rng.uniform(0.15, 0.5, count)
        else:
            amount = rng.lognormal(4.5, 1.2, count)
            hour = rng.integers(6, 22, count)
            tx_count_1h = rng.integers(1, 5, count).astype(float)
            zscore = rng.uniform(-1.0, 1.0, count)
            email_rate = rng.uniform(0.01, 0.08, count)

        mean_amt = rng.lognormal(4.5, 0.5, count)
        return pd.DataFrame({
            "log_amount":      np.log1p(amount),
            "amount_cents":    amount % 1,
            "hour":            hour,
            "day_of_week":     rng.integers(0, 7, count),
            "is_weekend":      rng.integers(0, 2, count),
            "amount_zscore":   zscore,
            "email_fraud_rate": email_rate,
            "card_tx_count":   rng.integers(1, 200, count).astype(float),
            "card_mean_amt":   mean_amt,
            "is_mobile":       rng.integers(0, 2, count),
            "tx_count_1h":     tx_count_1h,
            "tx_count_24h":    tx_count_1h * rng.uniform(2, 6, count),
            "avg_amount_24h":  mean_amt,
            "max_amount_24h":  mean_amt * rng.uniform(1.0, 3.0, count),
            TARGET:            int(is_fraud),
        })

    df = pd.concat([make_rows(n_legit, False), make_rows(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(f"Generated {len(df):,} rows | Fraud rate: {df[TARGET].mean():.4f}")
    return df


def train_and_save(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score
    import xgboost as xgb
    import lightgbm as lgb

    X = df[FEATURES].fillna(-999)
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    Path("models").mkdir(exist_ok=True)

    # ── XGBoost champion ─────────────────────────────────────────────────────
    logger.info("Training XGBoost champion...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=67,
        tree_method="hist",
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric="aucpr",
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    xgb_preds = xgb_model.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_preds)
    xgb_pr  = average_precision_score(y_val, xgb_preds)
    threshold = find_optimal_threshold(y_val.values, xgb_preds)
    threshold = min(threshold, 0.5)  # cap for synthetic data

    logger.info(f"XGBoost  →  AUC-ROC={xgb_auc:.4f}  AUC-PR={xgb_pr:.4f}  threshold={threshold:.4f}")

    joblib.dump(xgb_model, "models/xgb_champion.pkl")
    joblib.dump(threshold, "models/xgb_threshold.pkl")

    # ── LightGBM challenger ───────────────────────────────────────────────────
    logger.info("Training LightGBM challenger...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=63,
        scale_pos_weight=67,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)

    lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
    lgb_auc = roc_auc_score(y_val, lgb_preds)
    lgb_pr  = average_precision_score(y_val, lgb_preds)
    lgb_threshold = find_optimal_threshold(y_val.values, lgb_preds)
    lgb_threshold = min(lgb_threshold, 0.5)

    logger.info(f"LightGBM →  AUC-ROC={lgb_auc:.4f}  AUC-PR={lgb_pr:.4f}  threshold={lgb_threshold:.4f}")

    joblib.dump(lgb_model, "models/lgbm_challenger.pkl")
    joblib.dump(lgb_threshold, "models/lgbm_threshold.pkl")

    logger.info("\n✅ Models saved to models/")
    logger.info("   models/xgb_champion.pkl")
    logger.info("   models/lgbm_challenger.pkl")
    logger.info("\nNext steps:")
    logger.info("   make stream      ← start Kafka producer + consumer")
    logger.info("   make dashboard   ← open http://localhost:8501")


if __name__ == "__main__":
    logger.info("Generating 50K synthetic transactions...")
    df = generate_synthetic_dataset(n=50_000)
    train_and_save(df)
