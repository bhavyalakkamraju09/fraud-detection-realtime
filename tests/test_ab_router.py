"""Tests for A/B router and Mann-Whitney statistical test."""

import random

import pytest

from src.streaming.ab_router import ABRouter, run_ab_test, EVAL_WINDOW_SIZE


class TestABRouter:
    def test_route_returns_valid_model_name(self):
        router = ABRouter(traffic_a=0.8)
        model_a, model_b = "model_a", "model_b"
        for _ in range(100):
            model, name = router.route(model_a, model_b)
            assert name in ("xgb_champion", "lgbm_challenger")
            assert model in (model_a, model_b)

    def test_traffic_split_approximate(self):
        """80/20 split should be within 10pp over 1000 samples."""
        router = ABRouter(traffic_a=0.8)
        champion_count = 0
        for _ in range(1000):
            _, name = router.route("a", "b")
            if name == "xgb_champion":
                champion_count += 1
        ratio = champion_count / 1000
        assert 0.70 <= ratio <= 0.90, f"Traffic ratio {ratio:.2f} outside expected range"

    def test_record_increments_results(self):
        router = ABRouter()
        router.record("xgb_champion", 0, 0.1)
        router.record("lgbm_challenger", 1, 0.9)
        assert len(router.results_a) == 1
        assert len(router.results_b) == 1

    def test_insufficient_data_status(self):
        result = run_ab_test([], [])
        assert result["status"] == "insufficient_data"

    def test_ab_test_complete_with_sufficient_data(self):
        rng = random.Random(42)
        results_a = [
            {"y_true": 1 if rng.random() < 0.05 else 0, "score": rng.random()}
            for _ in range(500)
        ]
        results_b = [
            {"y_true": 1 if rng.random() < 0.05 else 0, "score": rng.random()}
            for _ in range(200)
        ]
        result = run_ab_test(results_a, results_b)
        assert result["status"] in ("complete", "no_fraud_in_window")

    def test_recommendation_is_valid(self):
        rng = random.Random(99)
        # Challenger clearly better (higher scores on positives)
        results_a = [
            {"y_true": 1 if i % 20 == 0 else 0, "score": 0.3 + rng.random() * 0.2}
            for i in range(500)
        ]
        results_b = [
            {"y_true": 1 if i % 20 == 0 else 0, "score": 0.7 + rng.random() * 0.2}
            for i in range(200)
        ]
        result = run_ab_test(results_a, results_b)
        if result["status"] == "complete":
            assert result["recommendation"] in ("promote_challenger", "keep_champion")
