"""
Kafka consumer: reads raw transactions, enriches with Redis features,
scores with the A/B-routed model, explains fraud with SHAP,
publishes to scored_transactions topic, and updates Prometheus metrics.
"""

import json
import logging
MODEL_FEATURES = []
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from confluent_kafka import Consumer, KafkaException, Producer

from src.features.engineering import FEATURES
from src.features.redis_store import update_user_features
from src.explainability.shap_explainer import FraudExplainer
from src.monitoring.metrics import (
    FRAUD_COUNTER,
    LATENCY_HISTOGRAM,
    SCORE_HISTOGRAM,
    TRANSACTIONS_TOTAL,
    start_metrics_server,
)
from src.streaming.ab_router import ABRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")

CONSUMER_CONFIG = {
    "bootstrap.servers": KAFKA_BROKER,
    "group.id": "fraud-scorer-v1",
    "auto.offset.reset": "latest",
    "enable.auto.commit": True,
    "session.timeout.ms": 30000,
}

PRODUCER_CONFIG = {"bootstrap.servers": KAFKA_BROKER}


def load_models():
    model_a = joblib.load("models/xgb_champion.pkl")
    threshold_a = joblib.load("models/xgb_threshold.pkl")

    # Challenger is optional — fall back to champion if not built yet
    challenger_path = Path("models/lgbm_challenger.pkl")
    if challenger_path.exists():
        model_b = joblib.load(challenger_path)
        threshold_b = joblib.load("models/lgbm_threshold.pkl")
    else:
        logger.warning("LightGBM challenger not found — using XGBoost for both A/B arms")
        model_b = model_a
        threshold_b = threshold_a

    return model_a, threshold_a, model_b, threshold_b


def run_consumer():
    start_metrics_server(port=8000)

    global MODEL_FEATURES
    MODEL_FEATURES = joblib.load("models/feature_cols.pkl")
    model_a, threshold_a, model_b, threshold_b = load_models()
    explainer = FraudExplainer(model_a, FEATURES)
    router = ABRouter(traffic_a=float(os.getenv("AB_TRAFFIC_A", 0.80)))

    consumer = Consumer(CONSUMER_CONFIG)
    producer = Producer(PRODUCER_CONFIG)
    consumer.subscribe(["transactions"])

    logger.info("Consumer started — waiting for transactions...")
    processed = 0

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                raise KafkaException(msg.error())

            t0 = time.perf_counter()

            # ── 1. Parse incoming transaction ─────────────────────────────
            tx = json.loads(msg.value().decode())
            card_id = tx["card_id"]

            # ── 2. Enrich with Redis rolling features ─────────────────────
            rolling = update_user_features(card_id, tx)
            tx.update(rolling)

            # ── 3. Build feature vector ───────────────────────────────────
            row = {f: -999 for f in MODEL_FEATURES}
            row.update({
                "log_amount": __import__("math").log1p(tx["amount"]),
                "amount_cents": tx["amount"] % 1,
                "hour": tx.get("hour_of_day", 12),
                "day_of_week": 1,
                "is_weekend": int(tx.get("is_weekend", False)),
                "amount_zscore": 0.0,
                "email_fraud_rate": 0.05,
                "card_tx_count": rolling.get("tx_count_24h", 1),
                "card_mean_amt": rolling.get("avg_amount_24h", tx["amount"]),
                "is_mobile": 0,
                **rolling,
            })
            X = pd.DataFrame([row])[MODEL_FEATURES].fillna(-999)

            # ── 4. A/B routing + scoring ──────────────────────────────────
            model, model_name = router.route(model_a, model_b)
            threshold = threshold_a if model_name == "xgb_champion" else threshold_b

            fraud_prob = float(model.predict_proba(X)[0, 1])
            is_fraud = fraud_prob >= threshold

            # ── 5. SHAP (only for flagged — cost control) ─────────────────
            explanation = None
            if is_fraud:
                try:
                    explanation = explainer.explain_transaction(X)
                except Exception as e:
                    logger.warning(f"SHAP failed for tx {tx['transaction_id']}: {e}")

            # ── 6. Publish scored result ──────────────────────────────────
            result = {
                **tx,
                "fraud_prob": round(fraud_prob, 4),
                "is_fraud": is_fraud,
                "model": model_name,
                "explanation": explanation,
            }
            producer.produce(
                "scored_transactions",
                key=card_id,
                value=json.dumps(result).encode(),
            )
            producer.poll(0)

            # ── 7. Record for A/B evaluation ──────────────────────────────
            router.record(model_name, tx.get("label", 0), fraud_prob)

            # ── 8. Prometheus metrics ─────────────────────────────────────
            latency = time.perf_counter() - t0
            TRANSACTIONS_TOTAL.labels(model=model_name).inc()
            FRAUD_COUNTER.labels(model=model_name).inc(int(is_fraud))
            SCORE_HISTOGRAM.labels(model=model_name).observe(fraud_prob)
            LATENCY_HISTOGRAM.labels(model=model_name).observe(latency)

            processed += 1
            if processed % 500 == 0:
                logger.info(
                    f"Processed {processed:,} | Latest: {tx['transaction_id']} "
                    f"prob={fraud_prob:.3f} fraud={is_fraud} latency={latency*1000:.1f}ms"
                )

    except KeyboardInterrupt:
        logger.info("Consumer shutting down")
    finally:
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    run_consumer()
