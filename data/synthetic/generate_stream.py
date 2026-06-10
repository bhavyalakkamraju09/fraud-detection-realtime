"""
Kafka producer: generates synthetic credit card transactions
following a Poisson process at a configurable rate.

Fraud rate ~1.5% to match IEEE-CIS dataset distribution.
Amount distributions differ between legit and fraud (log-normal).
"""

import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from confluent_kafka import Producer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = "transactions"
FRAUD_RATE = 0.015  # 1.5% — matches IEEE-CIS

MERCHANT_CATEGORIES = ["grocery", "gas", "online", "restaurant", "atm", "travel", "pharmacy"]
LEGIT_COUNTRIES = ["US", "US", "US", "US", "CA", "GB", "MX"]
FRAUD_COUNTRIES = ["NG", "RU", "CN", "UA", "BR"]


def generate_transaction(is_fraud: bool = False) -> dict:
    if is_fraud:
        # Fraud: higher amounts, unusual countries, late night hours
        amount = float(np.random.lognormal(mean=6.0, sigma=0.8))
        country = random.choice(FRAUD_COUNTRIES)
        hour = random.choice([1, 2, 3, 4, 22, 23])
    else:
        amount = float(np.random.lognormal(mean=4.5, sigma=1.2))
        country = random.choice(LEGIT_COUNTRIES)
        hour = datetime.now(timezone.utc).hour

    return {
        "transaction_id": str(uuid.uuid4()),
        "card_id": f"card_{random.randint(1000, 9999)}",
        "amount": round(amount, 2),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "hour_of_day": hour,
        "is_weekend": datetime.now(timezone.utc).weekday() >= 5,
        "country_code": country,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": int(is_fraud),
    }


def delivery_report(err, msg):
    if err:
        logger.error(f"Delivery failed: {err}")


def run_producer(tx_per_second: int = int(os.getenv("TX_PER_SECOND", 50))):
    producer = Producer({"bootstrap.servers": KAFKA_BROKER})
    logger.info(f"Streaming {tx_per_second} tx/sec → topic: {TOPIC}")

    total = 0
    while True:
        batch_start = time.perf_counter()
        for _ in range(tx_per_second):
            is_fraud = random.random() < FRAUD_RATE
            tx = generate_transaction(is_fraud)
            producer.produce(
                TOPIC,
                key=tx["card_id"],
                value=json.dumps(tx).encode(),
                callback=delivery_report,
            )
            total += 1

        producer.poll(0)          # trigger callbacks
        producer.flush(timeout=1)

        elapsed = time.perf_counter() - batch_start
        sleep_time = max(0.0, 1.0 - elapsed)
        if total % 1000 == 0:
            logger.info(f"Produced {total:,} transactions")
        time.sleep(sleep_time)


if __name__ == "__main__":
    run_producer()
