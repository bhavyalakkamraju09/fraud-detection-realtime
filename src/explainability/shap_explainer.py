"""
SHAP TreeExplainer wrapper.
- explain_transaction(): per-transaction SHAP dict for the API + dashboard
- plot_waterfall(): saves a waterfall PNG for flagged transactions
- plot_summary(): global feature importance over a batch
"""

import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.features.engineering import FEATURES

logger = logging.getLogger(__name__)


class FraudExplainer:
    def __init__(self, model, feature_names: list[str] = FEATURES):
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names
        logger.info(f"FraudExplainer initialised with {len(feature_names)} features")

    def explain_transaction(self, X_row: pd.DataFrame) -> dict:
        """
        Returns a dict with base_value, per-feature SHAP values,
        and ranked top fraud / legitimacy drivers.
        """
        shap_values = self.explainer.shap_values(X_row)

        # For binary classifiers, shap_values is a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1])
        else:
            base_value = float(base_value)

        per_feature = {
            feat: float(val)
            for feat, val in zip(self.feature_names, shap_values[0])
        }

        sorted_shap = sorted(per_feature.items(), key=lambda x: abs(x[1]), reverse=True)

        explanation = {
            "base_value": base_value,
            "shap_values": per_feature,
            "top_fraud_drivers": [
                {"feature": f, "shap": v} for f, v in sorted_shap[:5] if v > 0
            ],
            "top_legit_drivers": [
                {"feature": f, "shap": v} for f, v in sorted_shap[:5] if v < 0
            ],
        }
        return explanation

    def plot_waterfall(self, X_row: pd.DataFrame, save_path: Optional[str] = None):
        """Waterfall chart for a single transaction prediction."""
        shap_vals = self.explainer(X_row)
        shap.plots.waterfall(shap_vals[0], show=False, max_display=12)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info(f"Waterfall saved to {save_path}")
        else:
            plt.show()

    def plot_summary(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """Global SHAP summary (beeswarm) over a batch."""
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap.summary_plot(
            shap_vals,
            X,
            feature_names=self.feature_names,
            show=False,
            max_display=14,
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info(f"Summary plot saved to {save_path}")
        else:
            plt.show()
