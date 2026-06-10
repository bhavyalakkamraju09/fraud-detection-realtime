"""
Threshold selection via precision-recall curve.
Finds the probability cutoff that maximises F1 on the validation set,
then allows a precision-floor override for operational use.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    min_precision: float = 0.70,
) -> float:
    """
    Return the threshold maximising F1, subject to precision >= min_precision.

    If no threshold satisfies the precision floor (rare with a well-trained
    model), fall back to the raw F1-optimal threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (
        precisions[:-1] + recalls[:-1] + 1e-8
    )

    # Apply precision floor
    mask = precisions[:-1] >= min_precision
    if mask.any():
        best_idx = np.argmax(f1_scores * mask)
    else:
        best_idx = np.argmax(f1_scores)

    threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    best_prec = float(precisions[best_idx])
    best_rec = float(recalls[best_idx])

    print(
        f"Optimal threshold: {threshold:.4f} | "
        f"F1={best_f1:.4f} | Precision={best_prec:.4f} | Recall={best_rec:.4f}"
    )
    return threshold
