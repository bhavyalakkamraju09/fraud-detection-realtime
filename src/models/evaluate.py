"""
Evaluation helpers — AUC-ROC, AUC-PR, F1, confusion matrix.
Used by training scripts and the monitoring module.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def full_report(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> dict:
    """Return a dict of all key metrics given predicted probabilities."""
    y_pred = (y_scores >= threshold).astype(int)

    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = {
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "f1": round(f1, 4),
        "precision": round(tp / (tp + fp + 1e-8), 4),
        "recall": round(tp / (tp + fn + 1e-8), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": threshold,
    }

    print("\n── Evaluation Report ──────────────────────────────────")
    for k, v in report.items():
        print(f"  {k:<18}: {v}")
    print("───────────────────────────────────────────────────────\n")
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))

    return report
