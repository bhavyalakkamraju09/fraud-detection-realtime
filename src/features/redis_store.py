"""
Redis-backed rolling feature store.
Maintains sorted-set transaction history per card and computes
1h / 24h rolling window aggregations in sub-millisecond latency.
"""

import json
import logging
import os
from datetime import datetime, timezone

import redis

logger = logging.getLogger(__name__)

WINDOW_1H = 3600
WINDOW_24H = 86400

_redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True,
        )
        _redis_client.ping()  # fail fast if not available
        logger.info("Redis connection established")
    return _redis_client


def update_user_features(card_id: str, tx: dict) -> dict:
    """
    Push a new transaction into the card's sorted-set history,
    prune events older than 24h, and return fresh rolling features.
    """
    r = get_redis()
    key = f"user:{card_id}"
    now = datetime.now(timezone.utc).timestamp()

    # Append to sorted set: score = UNIX timestamp
    r.zadd(
        f"{key}:tx_history",
        {json.dumps({"amt": tx["amount"], "ts": now}): now},
    )
    # Prune >24h-old records
    r.zremrangebyscore(f"{key}:tx_history", 0, now - WINDOW_24H)

    # Fetch rolling windows
    recent_1h_raw = r.zrangebyscore(f"{key}:tx_history", now - WINDOW_1H, now)
    recent_24h_raw = r.zrangebyscore(f"{key}:tx_history", now - WINDOW_24H, now)

    amounts_1h = [json.loads(x)["amt"] for x in recent_1h_raw]
    amounts_24h = [json.loads(x)["amt"] for x in recent_24h_raw]

    features = {
        "tx_count_1h": float(len(amounts_1h)),
        "tx_amount_1h": float(sum(amounts_1h)),
        "tx_count_24h": float(len(amounts_24h)),
        "avg_amount_24h": float(sum(amounts_24h) / max(len(amounts_24h), 1)),
        "max_amount_24h": float(max(amounts_24h, default=0.0)),
    }

    # Write back for dashboard consumption
    r.hset(f"{key}:features", mapping={k: str(v) for k, v in features.items()})
    r.expire(f"{key}:features", WINDOW_24H)

    return features


def get_user_features(card_id: str) -> dict:
    """Retrieve last-written feature hash for a card (for monitoring)."""
    r = get_redis()
    raw = r.hgetall(f"user:{card_id}:features")
    return {k: float(v) for k, v in raw.items()}


def flush_test_data():
    """Clear all user keys — used in tests only."""
    r = get_redis()
    for key in r.scan_iter("user:*"):
        r.delete(key)
