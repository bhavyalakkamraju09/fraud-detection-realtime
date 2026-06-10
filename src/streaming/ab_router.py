"""
A/B traffic router + Mann-Whitney U statistical test.

Router sends 80% traffic to Model A (XGBoost champion) and 20% to
Model B (LightGBM challenger). Every EVAL_WINDOW_SIZE transactions,
runs a Mann-Whitney U test on windowed AUC-PR scores and auto-promotes
the challenger if it wins 3 consecutive windows.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)

EVAL_WINDOW_SIZE = 100      # transactions per A/B evaluation window
PROMOTE_THRESHOLD = 3       # consecutive challenger wins before auto-promotion


@dataclass
class ABRouter:
    traffic_a: float = 0.80
    results_a: list = field(default_factory=list)
    results_b: list = field(default_factory=list)
    challenger_wins: int = 0
    promoted: bool = False

    def route(self, model_a: Any, model_b: Any) -> tuple[Any, str]:
        if random.random() < self.traffic_a:
            return model_a, "xgb_champion"
        return model_b, "lgbm_challenger"

    def record(self, model_name: str, y_true: int, score: float):
        entry = {"y_true": y_true, "score": score}
        if model_name == "xgb_champion":
            self.results_a.append(entry)
        else:
            self.results_b.append(entry)

        # Trigger evaluation after each window
        if len(self.results_a) % EVAL_WINDOW_SIZE == 0:
            result = self.evaluate()
            if result.get("recommendation") == "promote_challenger":
                self.challenger_wins += 1
                logger.info(
                    f"Challenger wins window {self.challenger_wins}/{PROMOTE_THRESHOLD}"
                )
                if self.challenger_wins >= PROMOTE_THRESHOLD:
                    self.promoted = True
                    logger.warning("CHALLENGER AUTO-PROMOTED — swap model_a reference!")
            else:
                self.challenger_wins = 0  # reset streak

    def evaluate(self) -> dict:
        return run_ab_test(self.results_a, self.results_b)

    def summary(self) -> dict:
        return {
            "n_champion": len(self.results_a),
            "n_challenger": len(self.results_b),
            "challenger_win_streak": self.challenger_wins,
            "promoted": self.promoted,
        }


def run_ab_test(
    results_a: list[dict],
    results_b: list[dict],
    window: int = EVAL_WINDOW_SIZE,
) -> dict:
    """
    Compare windowed AUC-PR distributions between champion and challenger
    using a non-parametric Mann-Whitney U test (no normality assumption).
    """
    if len(results_a) < window or len(results_b) < window:
        return {
            "status": "insufficient_data",
            "n_a": len(results_a),
            "n_b": len(results_b),
        }

    def windowed_auc_pr(results: list[dict]) -> list[float]:
        scores = []
        for i in range(0, len(results) - window, window):
            chunk = results[i : i + window]
            y_true = [r["y_true"] for r in chunk]
            y_score = [r["score"] for r in chunk]
            if sum(y_true) > 0:  # need at least one positive
                scores.append(average_precision_score(y_true, y_score))
        return scores

    auc_pr_a = windowed_auc_pr(results_a)
    auc_pr_b = windowed_auc_pr(results_b)

    if not auc_pr_a or not auc_pr_b:
        return {"status": "no_fraud_in_window"}

    stat, p_value = stats.mannwhitneyu(auc_pr_b, auc_pr_a, alternative="greater")

    mean_a = float(np.mean(auc_pr_a))
    mean_b = float(np.mean(auc_pr_b))
    significant = bool(p_value < 0.05)
    challenger_better = bool(significant and mean_b > mean_a)

    result = {
        "status": "complete",
        "mean_auc_pr_champion": round(mean_a, 4),
        "mean_auc_pr_challenger": round(mean_b, 4),
        "delta": round(mean_b - mean_a, 4),
        "mann_whitney_stat": round(float(stat), 2),
        "p_value": round(float(p_value), 4),
        "significant": significant,
        "recommendation": "promote_challenger" if challenger_better else "keep_champion",
    }

    logger.info(
        f"A/B test | Champion AUC-PR={mean_a:.4f} | Challenger={mean_b:.4f} | "
        f"p={p_value:.4f} | {result['recommendation']}"
    )
    return result
