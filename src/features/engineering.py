"""
Offline feature engineering for IEEE-CIS dataset.
Run once to produce data/processed/features.parquet before training.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Features used by all models — must match consumer.py FEATURES list
FEATURES = [
    "log_amount",
    "amount_cents",
    "hour",
    "day_of_week",
    "is_weekend",
    "amount_zscore",
    "email_fraud_rate",
    "card_tx_count",
    "card_mean_amt",
    "is_mobile",
    "tx_count_1h",
    "tx_count_24h",
    "avg_amount_24h",
    "max_amount_24h",
]

TARGET = "isFraud"


def build_offline_features(
    train_transaction: pd.DataFrame,
    train_identity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge identity + transaction tables and engineer all model features.
    Returns a DataFrame with FEATURES + TARGET columns.
    """
    logger.info(
        f"Building features: {len(train_transaction):,} transactions, "
        f"{len(train_identity):,} identity rows"
    )

    df = train_transaction.merge(train_identity, on="TransactionID", how="left")

    # ── Amount features ──────────────────────────────────────────────────────
    df["log_amount"] = np.log1p(df["TransactionAmt"])
    df["amount_cents"] = df["TransactionAmt"] % 1  # fractional part — fraud signal

    # ── Time features ────────────────────────────────────────────────────────
    df["hour"] = (df["TransactionDT"] / 3600 % 24).astype(int)
    df["day_of_week"] = (df["TransactionDT"] / (3600 * 24) % 7).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # ── Card-level aggregation features ──────────────────────────────────────
    card_stats = (
        df.groupby("card1")["TransactionAmt"]
        .agg(["mean", "std", "count"])
        .rename(
            columns={
                "mean": "card_mean_amt",
                "std": "card_std_amt",
                "count": "card_tx_count",
            }
        )
    )
    df = df.merge(card_stats, on="card1", how="left")
    # card_std_amt is NaN for single-transaction cards — fill with 1.0 before dividing
    df["card_std_amt"] = df["card_std_amt"].fillna(1.0)
    df["amount_zscore"] = (df["TransactionAmt"] - df["card_mean_amt"]) / (
        df["card_std_amt"] + 1e-8
    )

    # ── Email domain fraud rate (target encoding — compute on full train) ─────
    email_fraud_rate = (
        df.groupby("P_emaildomain")["isFraud"]
        .mean()
        .rename("email_fraud_rate")
    )
    df = df.merge(email_fraud_rate, on="P_emaildomain", how="left")
    df["email_fraud_rate"] = df["email_fraud_rate"].fillna(df["isFraud"].mean())

    # ── Device features ───────────────────────────────────────────────────────
    if "DeviceType" in df.columns:
        df["is_mobile"] = (df["DeviceType"] == "mobile").astype(int)
    else:
        df["is_mobile"] = 0

    # ── Rolling window proxies (offline approximation) ────────────────────────
    # In production these come from Redis; offline we proxy with card-level stats
    df["tx_count_1h"] = df["card_tx_count"].clip(upper=50)
    df["tx_count_24h"] = df["card_tx_count"]
    df["avg_amount_24h"] = df["card_mean_amt"]
    df["max_amount_24h"] = df.groupby("card1")["TransactionAmt"].transform("max")

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def load_ieee_cis(data_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load IEEE-CIS CSVs from data/raw/."""
    path = Path(data_dir)
    tx = pd.read_csv(path / "train_transaction.csv")
    identity = pd.read_csv(path / "train_identity.csv")
    logger.info(f"Loaded {len(tx):,} transactions and {len(identity):,} identity rows")
    return tx, identity


def save_features(df: pd.DataFrame, out_path: str = "data/processed/features.parquet"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df[FEATURES + [TARGET]].to_parquet(out_path, index=False)
    logger.info(f"Saved features to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tx, identity = load_ieee_cis()
    df = build_offline_features(tx, identity)
    save_features(df)
    print(f"\nFeature stats:\n{df[FEATURES].describe().round(3)}")
    print(f"\nFraud rate: {df[TARGET].mean():.4f} ({df[TARGET].sum():,} fraud / {len(df):,} total)")
