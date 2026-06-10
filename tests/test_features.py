"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import FEATURES, TARGET, build_offline_features


def make_mock_transaction(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            "TransactionAmt": rng.lognormal(4.5, 1.2, n),
            "TransactionDT": rng.integers(0, 86400 * 7, n),
            "card1": rng.integers(1000, 2000, n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], n),
            "isFraud": rng.integers(0, 2, n),
        }
    )


def make_mock_identity(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            "DeviceType": rng.choice(["desktop", "mobile", None], n),
        }
    )


class TestFeatureEngineering:
    def test_build_offline_features_returns_all_features(self):
        tx = make_mock_transaction()
        identity = make_mock_identity()
        df = build_offline_features(tx, identity)

        for feat in FEATURES:
            assert feat in df.columns, f"Missing feature: {feat}"

    def test_log_amount_is_positive(self):
        tx = make_mock_transaction()
        df = build_offline_features(tx, make_mock_identity())
        assert (df["log_amount"] >= 0).all()

    def test_hour_in_range(self):
        tx = make_mock_transaction()
        df = build_offline_features(tx, make_mock_identity())
        assert df["hour"].between(0, 23).all()

    def test_is_weekend_binary(self):
        tx = make_mock_transaction()
        df = build_offline_features(tx, make_mock_identity())
        assert set(df["is_weekend"].unique()).issubset({0, 1})

    def test_amount_zscore_finite(self):
        tx = make_mock_transaction(n=200)
        df = build_offline_features(tx, make_mock_identity(n=50))
        assert np.isfinite(df["amount_zscore"]).all()

    def test_email_fraud_rate_in_0_1(self):
        tx = make_mock_transaction()
        df = build_offline_features(tx, make_mock_identity())
        assert df["email_fraud_rate"].between(0, 1).all()

    def test_no_inf_values(self):
        tx = make_mock_transaction()
        df = build_offline_features(tx, make_mock_identity())
        numeric = df[FEATURES].select_dtypes(include="number")
        assert not np.isinf(numeric.values).any()
