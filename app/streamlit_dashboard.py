"""
Real-Time Fraud Detection Dashboard
Streams from scored_transactions Kafka topic and updates every 200ms.
"""

import json
import os
import time
from collections import deque

import pandas as pd
import streamlit as st
from confluent_kafka import Consumer

st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🛡️ Real-Time Fraud Detection Monitor")

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
MAX_HISTORY = 1000  # rolling window for charts


# ── Sidebar config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    refresh_ms = st.slider("Refresh interval (ms)", 100, 2000, 200, step=100)
    window_size = st.slider("Chart window (transactions)", 50, 500, 200)
    show_raw = st.checkbox("Show raw transaction feed", value=False)

# ── Metric placeholders ───────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
total_ph = col1.empty()
fraud_ph = col2.empty()
rate_ph = col3.empty()
latency_ph = col4.empty()
ab_status_ph = col5.empty()

# ── Charts ────────────────────────────────────────────────────────────────────
st.divider()
chart_col, shap_col = st.columns([3, 2])

with chart_col:
    st.subheader("📈 Fraud Rate (rolling 50-tx window)")
    chart_ph = st.empty()

with shap_col:
    st.subheader("🔍 SHAP — Latest Flagged Transaction")
    shap_ph = st.empty()

st.divider()
ab_col, drift_col = st.columns(2)

with ab_col:
    st.subheader("🧪 A/B Test Results")
    ab_ph = st.empty()

with drift_col:
    st.subheader("📊 Score Distribution")
    score_dist_ph = st.empty()

if show_raw:
    st.divider()
    st.subheader("📋 Live Transaction Feed (last 20)")
    feed_ph = st.empty()


# ── Kafka consumer (cached for session) ──────────────────────────────────────
@st.cache_resource
def get_consumer():
    c = Consumer({
        "bootstrap.servers": KAFKA_BROKER,
        "group.id": "streamlit-dashboard",
        "auto.offset.reset": "latest",
    })
    c.subscribe(["scored_transactions"])
    return c


history: deque = deque(maxlen=MAX_HISTORY)
consumer = None

try:
    consumer = get_consumer()
    connected = True
except Exception as e:
    st.error(f"Kafka connection failed: {e}\nStart `make up` and `make stream` first.")
    connected = False

if connected:
    while True:
        # Poll for new messages
        for _ in range(50):  # batch up to 50 per refresh
            msg = consumer.poll(0.01)
            if msg and not msg.error():
                try:
                    tx = json.loads(msg.value().decode())
                    history.append(tx)
                except Exception:
                    pass

        if not history:
            st.info("Waiting for transactions… run `make stream` to start the producer.")
            time.sleep(1)
            continue

        df = pd.DataFrame(list(history))

        # ── Top metrics ───────────────────────────────────────────────────
        total = len(history)
        n_fraud = int(df["is_fraud"].sum())
        fraud_rate = df["is_fraud"].mean() * 100

        total_ph.metric("Total Transactions", f"{total:,}")
        fraud_ph.metric("Fraud Flagged", f"{n_fraud:,}")
        rate_ph.metric("Fraud Rate", f"{fraud_rate:.2f}%")

        # Model split
        n_champion = int((df["model"] == "xgb_champion").sum())
        n_challenger = int((df["model"] == "lgbm_challenger").sum())
        ab_status_ph.metric("Champion / Challenger", f"{n_champion} / {n_challenger}")

        # ── Fraud rate chart ──────────────────────────────────────────────
        recent = df.tail(window_size).copy()
        recent["rolling_rate"] = (
            recent["is_fraud"].rolling(50, min_periods=1).mean() * 100
        )
        chart_ph.line_chart(
            recent.set_index(pd.RangeIndex(len(recent)))["rolling_rate"],
            height=220,
        )

        # ── SHAP waterfall (latest fraud) ─────────────────────────────────
        frauds = df[df["is_fraud"] == True]
        if len(frauds) > 0:
            latest_fraud = frauds.iloc[-1]
            exp = latest_fraud.get("explanation")
            if exp and isinstance(exp, dict) and "shap_values" in exp:
                shap_df = (
                    pd.DataFrame(
                        list(exp["shap_values"].items()),
                        columns=["Feature", "SHAP Value"],
                    )
                    .sort_values("SHAP Value", key=abs, ascending=False)
                    .head(8)
                )
                shap_ph.bar_chart(
                    shap_df.set_index("Feature")["SHAP Value"],
                    height=220,
                )
            else:
                shap_ph.info("No SHAP data yet for flagged transactions")
        else:
            shap_ph.info("No fraud flagged yet")

        # ── A/B results table ─────────────────────────────────────────────
        ab_data = {
            "Model": ["XGBoost Champion", "LightGBM Challenger"],
            "Transactions": [n_champion, n_challenger],
            "Fraud Flagged": [
                int(df[df["model"] == "xgb_champion"]["is_fraud"].sum()),
                int(df[df["model"] == "lgbm_challenger"]["is_fraud"].sum()),
            ],
            "Avg Score": [
                round(df[df["model"] == "xgb_champion"]["fraud_prob"].mean(), 4),
                round(df[df["model"] == "lgbm_challenger"]["fraud_prob"].mean(), 4),
            ],
        }
        ab_ph.dataframe(pd.DataFrame(ab_data), use_container_width=True, hide_index=True)

        # ── Score distribution ────────────────────────────────────────────
        score_dist_ph.bar_chart(
            df["fraud_prob"]
            .apply(lambda x: pd.cut(df["fraud_prob"], bins=10).value_counts().sort_index().rename(index=str)),
            height=220,
        )

        # ── Raw feed ──────────────────────────────────────────────────────
        if show_raw:
            feed_cols = ["transaction_id", "card_id", "amount", "is_fraud", "fraud_prob", "model"]
            available = [c for c in feed_cols if c in df.columns]
            feed_ph.dataframe(
                df[available].tail(20).sort_index(ascending=False),
                use_container_width=True,
                hide_index=True,
            )

        time.sleep(refresh_ms / 1000)
