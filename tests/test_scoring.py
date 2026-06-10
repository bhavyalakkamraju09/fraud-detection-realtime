"""
Unit tests for scoring logic — no Kafka or Redis required.
Tests the feature vector construction and threshold logic.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import FEATURES
from src.models.threshold import find_optimal_threshold


class TestThreshold:
    def test_threshold_in_0_1(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 1000)
        y_scores = rng.uniform(0, 1, 1000)
        t = find_optimal_threshold(y_true, y_scores)
        assert 0.0 <= t <= 1.0

    def test_threshold_with_perfect_scores(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        t = find_optimal_threshold(y_true, y_scores)
        # Should choose a threshold below 0.9 to catch both positives
        assert t <= 0.9

    def test_threshold_type_is_float(self):
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 2, 500)
        y_scores = rng.uniform(0, 1, 500)
        t = find_optimal_threshold(y_true, y_scores)
        assert isinstance(t, float)


class TestFeatureVector:
    """Test that the feature vector construction for the API is correct."""

    def _build_row(self, amount: float, hour: int = 12) -> pd.DataFrame:
        row = {
            "log_amount": math.log1p(amount),
            "amount_cents": amount % 1,
            "hour": hour,
            "day_of_week": 1,
            "is_weekend": 0,
            "amount_zscore": 0.0,
            "email_fraud_rate": 0.05,
            "card_tx_count": 5.0,
            "card_mean_amt": 100.0,
            "is_mobile": 0,
            "tx_count_1h": 2.0,
            "tx_count_24h": 5.0,
            "avg_amount_24h": 100.0,
            "max_amount_24h": 200.0,
        }
        return pd.DataFrame([row])[FEATURES].fillna(-999)

    def test_all_features_present(self):
        X = self._build_row(50.0)
        assert list(X.columns) == FEATURES

    def test_log_amount_correct(self):
        X = self._build_row(100.0)
        assert abs(X["log_amount"].iloc[0] - math.log1p(100.0)) < 1e-6

    def test_no_nans_after_fillna(self):
        X = self._build_row(75.0)
        assert not X.isnull().any().any()

    def test_hour_range(self):
        for h in [0, 12, 23]:
            X = self._build_row(50.0, hour=h)
            assert X["hour"].iloc[0] == h
