"""
FastAPI service exposing:
  POST /score        — score a single transaction
  POST /explain      — score + SHAP explanation
  GET  /ab-status    — current A/B test results
  GET  /health       — liveness probe
"""

import logging
import math
import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features.engineering import FEATURES
from src.explainability.shap_explainer import FraudExplainer
from src.streaming.ab_router import ABRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time scoring + SHAP explainability for credit card transactions",
    version="1.0.0",
)

# ── Startup: load models ─────────────────────────────────────────────────────

_model_a = None
_model_b = None
_threshold = None
_explainer: Optional[FraudExplainer] = None
_router: Optional[ABRouter] = None


@app.on_event("startup")
def load_models():
    global _model_a, _model_b, _threshold, _explainer, _router

    model_path = Path("models/xgb_champion.pkl")
    if not model_path.exists():
        logger.warning("Model not found — /score will return 503 until trained")
        return

    _model_a = joblib.load(model_path)
    _threshold = joblib.load("models/xgb_threshold.pkl")

    challenger = Path("models/lgbm_challenger.pkl")
    _model_b = joblib.load(challenger) if challenger.exists() else _model_a

    _explainer = FraudExplainer(_model_a, FEATURES)
    _router = ABRouter(traffic_a=float(os.getenv("AB_TRAFFIC_A", 0.80)))
    logger.info("Models loaded successfully")


# ── Request / Response schemas ────────────────────────────────────────────────

class TransactionIn(BaseModel):
    transaction_id: str
    card_id: str
    amount: float
    hour_of_day: int = 12
    is_weekend: bool = False
    tx_count_1h: float = 1.0
    tx_amount_1h: float = 0.0
    tx_count_24h: float = 1.0
    avg_amount_24h: float = 100.0
    max_amount_24h: float = 100.0


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_prob: float
    is_fraud: bool
    model: str
    threshold: float


class ExplainResponse(ScoreResponse):
    explanation: dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_feature_row(tx: TransactionIn) -> pd.DataFrame:
    row = {
        "log_amount": math.log1p(tx.amount),
        "amount_cents": tx.amount % 1,
        "hour": tx.hour_of_day,
        "day_of_week": 1,
        "is_weekend": int(tx.is_weekend),
        "amount_zscore": 0.0,
        "email_fraud_rate": 0.05,
        "card_tx_count": tx.tx_count_24h,
        "card_mean_amt": tx.avg_amount_24h,
        "is_mobile": 0,
        "tx_count_1h": tx.tx_count_1h,
        "tx_count_24h": tx.tx_count_24h,
        "avg_amount_24h": tx.avg_amount_24h,
        "max_amount_24h": tx.max_amount_24h,
    }
    return pd.DataFrame([row])[FEATURES].fillna(-999)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_a is not None}


@app.post("/score", response_model=ScoreResponse)
def score(tx: TransactionIn):
    if _model_a is None:
        raise HTTPException(status_code=503, detail="Models not loaded — run make train first")

    X = build_feature_row(tx)
    model, model_name = _router.route(_model_a, _model_b)
    fraud_prob = float(model.predict_proba(X)[0, 1])
    is_fraud = fraud_prob >= _threshold

    return ScoreResponse(
        transaction_id=tx.transaction_id,
        fraud_prob=round(fraud_prob, 4),
        is_fraud=is_fraud,
        model=model_name,
        threshold=_threshold,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(tx: TransactionIn):
    if _model_a is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    X = build_feature_row(tx)
    fraud_prob = float(_model_a.predict_proba(X)[0, 1])
    is_fraud = fraud_prob >= _threshold
    explanation = _explainer.explain_transaction(X)

    return ExplainResponse(
        transaction_id=tx.transaction_id,
        fraud_prob=round(fraud_prob, 4),
        is_fraud=is_fraud,
        model="xgb_champion",
        threshold=_threshold,
        explanation=explanation,
    )


@app.get("/ab-status")
def ab_status():
    if _router is None:
        return {"status": "no data yet"}
    result = _router.evaluate()
    result["summary"] = _router.summary()
    return result
