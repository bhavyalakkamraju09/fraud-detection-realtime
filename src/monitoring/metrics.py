"""
Prometheus metrics exposed on :8000/metrics via prometheus-client.
Scraped by the Prometheus container defined in docker-compose.yml.
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging

logger = logging.getLogger(__name__)

# ── Counters ────────────────────────────────────────────────────────────────

TRANSACTIONS_TOTAL = Counter(
    "fraud_transactions_total",
    "Total transactions processed",
    ["model"],
)

FRAUD_COUNTER = Counter(
    "fraud_flagged_total",
    "Total transactions flagged as fraud",
    ["model"],
)

# ── Histograms ───────────────────────────────────────────────────────────────

SCORE_HISTOGRAM = Histogram(
    "fraud_score",
    "Distribution of fraud probability scores",
    ["model"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

LATENCY_HISTOGRAM = Histogram(
    "scoring_latency_seconds",
    "End-to-end per-transaction scoring latency",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# ── Gauges ────────────────────────────────────────────────────────────────────

FRAUD_RATE_GAUGE = Gauge(
    "fraud_rate_rolling",
    "Rolling fraud rate over last 1000 transactions",
    ["model"],
)

AB_WIN_COUNT = Counter(
    "ab_test_wins_total",
    "Number of A/B test windows where challenger won",
    ["winner"],
)


def start_metrics_server(port: int = 8000):
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on :{port}")
