"""
Evidently-based data drift + model performance monitor.
Runs on a schedule (e.g., hourly) comparing a reference window
(first week of training data) against the current production window.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.engineering import FEATURES

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "log_amount", "amount_cents", "hour", "day_of_week",
    "amount_zscore", "card_tx_count", "card_mean_amt",
    "tx_count_1h", "tx_count_24h", "avg_amount_24h", "max_amount_24h",
]
CATEGORICAL_FEATURES = ["is_weekend", "is_mobile"]


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: str = "isFraud",
    prediction_col: str = "fraud_prob",
    output_path: str = "outputs/drift_report.html",
) -> dict:
    """
    Compare reference vs current data distributions.
    Returns a summary dict and writes an HTML report.
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import ClassificationPreset, DataDriftPreset
        from evidently.report import Report
    except ImportError:
        logger.error("evidently not installed — run: pip install evidently")
        return {"error": "evidently not installed"}

    column_mapping = ColumnMapping(
        target=target_col,
        prediction=prediction_col,
        numerical_features=[f for f in NUMERIC_FEATURES if f in reference.columns],
        categorical_features=[f for f in CATEGORICAL_FEATURES if f in reference.columns],
    )

    metrics = [DataDriftPreset()]
    if target_col in current.columns and prediction_col in current.columns:
        metrics.append(ClassificationPreset())

    report = Report(metrics=metrics)
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)

    result = report.as_dict()
    drift_summary = result["metrics"][0]["result"]
    drift_detected = drift_summary.get("dataset_drift", False)

    drifted_features = [
        col
        for col, v in drift_summary.get("drift_by_columns", {}).items()
        if v.get("drift_detected", False)
    ]

    summary = {
        "drift_detected": drift_detected,
        "n_drifted_features": len(drifted_features),
        "drifted_features": drifted_features,
        "report_path": output_path,
    }

    level = logging.WARNING if drift_detected else logging.INFO
    logger.log(level, f"Drift report: {summary}")
    return summary
