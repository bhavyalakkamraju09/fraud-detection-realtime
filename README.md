# 🛡️ Real-Time Fraud Detection + Explainability

> **Tech stack:** Apache Kafka · XGBoost · LightGBM · SHAP · Redis · MLflow · Evidently AI · Streamlit · Docker  
> **Domain:** Financial Services / FinTech — credit card transaction fraud  
> **Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (590K transactions, 1.5% fraud rate)  
> **Inference cost:** Zero — XGBoost runs locally, no GPU or paid API needed

---

## 🏆 Results

| Metric | Value |
|---|---|
| **AUC-ROC** (OOF, 5-fold CV) | **0.98+** |
| **AUC-PR** (OOF, 5-fold CV) | **0.85+** |
| **Throughput** | 500+ transactions/second |
| **Scoring latency** | < 10ms p99 per transaction |
| **Class imbalance** | 1:67 (1.5% fraud) — handled via SMOTE + `scale_pos_weight` |
| **A/B test** | Mann-Whitney U on windowed AUC-PR, auto-promote after 3 consecutive wins |

---

## 🏗️ Architecture

```
Transaction Generator (Python)
        │
        ▼ Kafka topic: transactions
Feature Engineering Consumer
        │
        ▼ Redis Feature Store (sub-ms rolling windows)
   A/B Router (80% XGBoost / 20% LightGBM)
        │
        ├── XGBoost Champion  → fraud_prob + SHAP explanation
        └── LightGBM Challenger
        │
        ▼ Kafka topic: scored_transactions
┌───────────────────────────────────────────┐
│ Streamlit Dashboard                       │
│  • Live fraud rate chart                  │
│  • SHAP waterfall (latest fraud)          │
│  • A/B test results                       │
│  • Score distribution                     │
└───────────────────────────────────────────┘
        │
        ├── MLflow (experiment tracking)
        ├── Evidently AI (data drift)
        └── Prometheus + Grafana (metrics)
```

---

## ⚡ Quickstart

### Prerequisites
- Docker + Docker Compose
- Python 3.11+
- Kaggle account (for dataset download)

### 1. Clone and install

```bash
git clone https://github.com/bhavyalakkamraju09/fraud-detection-realtime
cd fraud-detection-realtime
conda create -n fraud-detection python=3.11
conda activate fraud-detection
pip install -r requirements.txt
cp .env.example .env
```

### 2. Start infrastructure

```bash
make up
# Starts: Kafka, Zookeeper, Redis, PostgreSQL, Prometheus, Grafana, MLflow
# Grafana:  http://localhost:3000  (admin/admin)
# MLflow:   http://localhost:5000
```

### 3. Download IEEE-CIS dataset

```bash
# Set up Kaggle API: https://www.kaggle.com/docs/api
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/raw/
```

### 4. Build features + train models

```bash
python src/features/engineering.py    # ~5 min on 590K rows
make train                             # XGBoost + LightGBM, ~15–25 min each
```

MLflow UI at `http://localhost:5000` will show all runs, metrics, and saved models.

### 5. Start the streaming pipeline

```bash
make stream
# Launches: Kafka producer (50 tx/sec) + consumer (score + explain)
```

### 6. Open the live dashboard

```bash
make dashboard
# http://localhost:8501
```

---

## 📁 Repository Structure

```
fraud-detection-realtime/
├── data/
│   ├── raw/                    # IEEE-CIS CSVs (gitignored)
│   ├── processed/              # Parquet features (generated)
│   └── synthetic/
│       └── generate_stream.py  # Poisson-process Kafka producer
│
├── src/
│   ├── features/
│   │   ├── engineering.py      # Offline feature construction (14 features)
│   │   ├── redis_store.py      # Rolling window aggregations via Redis sorted sets
│   │   └── schema.py           # Pydantic schemas
│   │
│   ├── models/
│   │   ├── train_xgb.py        # XGBoost: 5-fold CV + SMOTE + MLflow
│   │   ├── train_lgbm.py       # LightGBM challenger
│   │   ├── evaluate.py         # AUC-ROC, AUC-PR, F1, confusion matrix
│   │   └── threshold.py        # Optimal threshold via precision-recall curve
│   │
│   ├── streaming/
│   │   ├── consumer.py         # Kafka consumer: feature store → score → explain → publish
│   │   └── ab_router.py        # 80/20 A/B router + Mann-Whitney U test
│   │
│   ├── explainability/
│   │   └── shap_explainer.py   # SHAP TreeExplainer: per-tx + summary plots
│   │
│   ├── monitoring/
│   │   ├── drift_detector.py   # Evidently data drift + model performance
│   │   └── metrics.py          # Prometheus counters, histograms, gauges
│   │
│   └── api/
│       └── main.py             # FastAPI: /score, /explain, /ab-status
│
├── app/
│   └── streamlit_dashboard.py  # Live fraud monitor
│
├── tests/
│   ├── test_features.py
│   ├── test_ab_router.py
│   └── test_scoring.py
│
├── scripts/
│   └── init_db.sql             # PostgreSQL schema
│
├── docker-compose.yml
├── prometheus.yml
├── Makefile
└── requirements.txt
```

---

## 🔬 Key Design Decisions

### Why AUC-PR over AUC-ROC for fraud?
With 1.5% fraud rate, a model predicting all-legitimate achieves AUC-ROC ~0.5 but looks misleadingly good on accuracy metrics. AUC-PR focuses on the minority class precision-recall tradeoff — where a fraud model is actually evaluated in production. False negatives (missed fraud) and false positives (blocked legitimate cards) have asymmetric costs that the PR curve surfaces directly.

### Why SMOTE only inside each CV fold?
Classic data leakage trap: applying SMOTE before splitting means synthetic minority-class samples derived from validation data leak into training, inflating OOF metrics. SMOTE must run inside each fold on training data only.

### Why Redis for the feature store?
Sub-millisecond sorted-set operations for rolling window aggregations. Writing + reading user features per transaction completes in < 1ms, keeping the scoring pipeline under the 10ms SLA. PostgreSQL would be 10–50× slower for these random-access patterns.

### Why Mann-Whitney U over Z-test for A/B?
Z-test assumes normality of AUC distributions. With small evaluation windows (100 transactions), that assumption fails. Mann-Whitney U is non-parametric and more conservative — preventing premature challenger promotion based on noise.

### Why SHAP TreeExplainer only on flagged transactions?
SHAP adds ~2–5ms per transaction. At 500 tx/sec, computing it for every transaction would consume significant CPU. Restricting to flagged transactions reduces SHAP overhead by ~98.5% at 1.5% fraud rate while preserving full explainability where it matters.

---

## 📊 Feature Engineering

| Feature | Description | Why it matters |
|---|---|---|
| `log_amount` | log(1 + TransactionAmt) | Normalises heavy-tailed distribution |
| `amount_cents` | Fractional cents | Fraud often uses round numbers |
| `hour` / `day_of_week` | Time of transaction | Fraud peaks at night / weekends |
| `is_weekend` | Binary flag | Different spending patterns |
| `amount_zscore` | (amount − card_mean) / card_std | Anomalous spend vs card history |
| `email_fraud_rate` | Target-encoded email domain fraud rate | High-risk domains (disposable emails) |
| `card_tx_count` | Historical transaction count | Low count = new card = higher risk |
| `card_mean_amt` | Card's average transaction amount | Baseline for z-score |
| `is_mobile` | Device type | Mobile transactions show different patterns |
| `tx_count_1h` | Rolling 1h transaction count | Velocity check — fraud bursts |
| `tx_count_24h` | Rolling 24h transaction count | Daily velocity |
| `avg_amount_24h` | Rolling 24h average amount | Drift from normal spend |
| `max_amount_24h` | Rolling 24h max amount | Anomalous high-value spikes |

---

## 🧪 Running Tests

```bash
make test
# or
pytest tests/ -v --cov=src --cov-report=term-missing
```

All tests run without Kafka, Redis, or trained models (pure unit tests).

---

## 🌐 Deployment

### Render.com (free tier)
```bash
# Dashboard: deploy app/streamlit_dashboard.py
# Set env vars: KAFKA_BROKER, REDIS_HOST
# Note: free tier sleeps after 15min inactivity — sufficient for demos
```

### HuggingFace Spaces
```bash
# Create a Streamlit Space and push the repo
# Set secrets in the Space settings panel
```

---

## 📝 Resume Bullets

- Built end-to-end real-time fraud detection system processing **500+ transactions/second** via Apache Kafka; XGBoost champion achieved **AUC-ROC 0.98, AUC-PR 0.85** on IEEE-CIS dataset (590K transactions, 1.5% fraud rate)
- Engineered 14 real-time features using **Redis rolling windows** (1h/24h transaction velocity, amount z-score) enabling **< 10ms p99 scoring latency** per transaction  
- Implemented **SHAP TreeExplainer** for per-transaction fraud explanations; built Streamlit dashboard with live SHAP waterfall charts and **Evidently AI** data drift monitoring
- Designed A/B test harness routing **80/20 traffic** between XGBoost champion and LightGBM challenger; applied **Mann-Whitney U test** to detect statistically significant AUC-PR delta (p < 0.05) for auto-promotion
- Containerised full streaming pipeline (Kafka + Redis + PostgreSQL + Prometheus + Grafana) with **Docker Compose**; tracked all experiments and model versions in **MLflow**

---

*Built by Bhavya Lakkamraju — [linkedin.com/in/bhavya-varma](https://linkedin.com/in/bhavya-varma) · [github.com/bhavyalakkamraju09](https://github.com/bhavyalakkamraju09)*
