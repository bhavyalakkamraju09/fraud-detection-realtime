"""
Train LightGBM challenger model (A/B test counterpart to XGBoost champion).
Intentionally lighter hyperparameter tuning — used as a live challenger
to the XGBoost champion, not as a replacement.
"""

import logging
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd

from .threshold import find_optimal_threshold
from src.features.engineering import FEATURES, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "scale_pos_weight": 67,
    "metric": "average_precision",
    "early_stopping_round": 50,
    "random_state": 42,
    "verbose": -1,
}

MODEL_OUT = Path("models/lgbm_challenger.pkl")
THRESHOLD_OUT = Path("models/lgbm_threshold.pkl")


def train(
    data_path: str = "data/processed/features.parquet",
    experiment_name: str = "fraud_lgbm",
) -> tuple:
    df = pd.read_parquet(data_path)
    X = df[FEATURES].fillna(-999)
    y = df[TARGET]

    logger.info(f"Training LightGBM on {len(df):,} samples")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment_name)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    models = []

    with mlflow.start_run(run_name="lgbm_5fold_smote"):
        mlflow.log_params(LGBM_PARAMS)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            sm = SMOTE(random_state=42, k_neighbors=5)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(
                X_tr_res,
                y_tr_res,
                eval_set=[(X_val, y_val)],
            )
            fold_preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = fold_preds

            fold_auc = roc_auc_score(y_val, fold_preds)
            fold_pr = average_precision_score(y_val, fold_preds)
            logger.info(f"  Fold {fold+1}/5 | AUC-ROC={fold_auc:.4f} | AUC-PR={fold_pr:.4f}")
            models.append(model)

        oof_auc = roc_auc_score(y, oof_preds)
        oof_pr = average_precision_score(y, oof_preds)
        threshold = find_optimal_threshold(y.values, oof_preds)

        mlflow.log_metric("oof_auc_roc", oof_auc)
        mlflow.log_metric("oof_auc_pr", oof_pr)
        mlflow.log_metric("optimal_threshold", threshold)

        logger.info(f"\nOOF AUC-ROC: {oof_auc:.4f} | OOF AUC-PR: {oof_pr:.4f}")

        challenger = models[-1]
        mlflow.lightgbm.log_model(challenger, "lgbm_model")

        MODEL_OUT.parent.mkdir(exist_ok=True)
        joblib.dump(challenger, MODEL_OUT)
        joblib.dump(threshold, THRESHOLD_OUT)
        logger.info(f"Saved challenger to {MODEL_OUT}")

    return challenger, threshold


if __name__ == "__main__":
    train()
