from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class Transaction(BaseModel):
    transaction_id: str
    card_id: str
    amount: float = Field(gt=0)
    merchant_category: str
    hour_of_day: int = Field(ge=0, le=23)
    is_weekend: bool
    country_code: str
    timestamp: str
    label: Optional[int] = None  # ground truth (delayed in real life)


class ScoredTransaction(Transaction):
    fraud_prob: float
    is_fraud: bool
    model: str
    explanation: Optional[dict] = None
    tx_count_1h: Optional[float] = None
    tx_amount_1h: Optional[float] = None
    tx_count_24h: Optional[float] = None
    avg_amount_24h: Optional[float] = None
    max_amount_24h: Optional[float] = None
