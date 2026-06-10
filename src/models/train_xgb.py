"""
Train XGBoost champion model.
- 5-fold stratified CV
- SMOTE applied inside each fold (no data leakage)
- MLflow experiment tracking
- Saves model + threshold to models/
"""

import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .threshold import find_optimal_threshold
from src.features.engineering import FEATURES, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 67,   # ~1.5% fraud rate → 98.5/1.5 ≈ 67
    "eval_metric": "aucpr",
    "early_stopping_rounds": 50,
    "tree_method": "hist",    # fast CPU; switch to "gpu_hist" with CUDA
    "random_state": 42,
    "use_label_encoder": False,
    "verbosity": 0,
}

MODEL_OUT = Path("models/xgb_champion.pkl")
THRESHOLD_OUT = Path("models/xgb_threshold.pkl")


def train(
    data_path: str = "data/processed/features.parquet",
    experiment_name: str = "fraud_xgb",
) -> tuple:
    df = pd.read_parquet(data_path)
    X = df[FEATURES].fillna(-999)
    y = df[TARGET]

    logger.info(
        f"Training on {len(df):,} samples | "
        f"Fraud rate: {y.mean():.4f} ({y.sum():,} positives)"
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(experiment_name)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    models = []

    with mlflow.start_run(run_name="xgb_5fold_smote"):
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("n_folds", 5)
        mlflow.log_param("smote_k_neighbors", 5)

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            # SMOTE on training fold only — never touch validation
            sm = SMOTE(random_state=42, k_neighbors=5)
            X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

            model = xgb.XGBClassifier(**XGB_PARAMS)
            model.fit(
                X_tr_res,
                y_tr_res,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            fold_preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = fold_preds

            fold_auc = roc_auc_score(y_val, fold_preds)
            fold_pr = average_precision_score(y_val, fold_preds)
            logger.info(f"  Fold {fold + 1}/5 | AUC-ROC={fold_auc:.4f} | AUC-PR={fold_pr:.4f}")
            mlflow.log_metric(f"fold{fold+1}_auc_roc", fold_auc, step=fold)
            mlflow.log_metric(f"fold{fold+1}_auc_pr", fold_pr, step=fold)
            models.append(model)

        # OOF metrics
        oof_auc = roc_auc_score(y, oof_preds)
        oof_pr = average_precision_score(y, oof_preds)
        threshold = find_optimal_threshold(y.values, oof_preds)

        mlflow.log_metric("oof_auc_roc", oof_auc)
        mlflow.log_metric("oof_auc_pr", oof_pr)
        mlflow.log_metric("optimal_threshold", threshold)

        logger.info(
            f"\nOOF AUC-ROC: {oof_auc:.4f} | OOF AUC-PR: {oof_pr:.4f} | "
            f"Threshold: {threshold:.4f}"
        )

        # Use last-fold model as champion (or ensemble via averaging)
        champion = models[-1]
        mlflow.xgboost.log_model(champion, "xgb_model")

        MODEL_OUT.parent.mkdir(exist_ok=True)
        joblib.dump(champion, MODEL_OUT)
        joblib.dump(threshold, THRESHOLD_OUT)
        logger.info(f"Saved champion to {MODEL_OUT}")

    return champion, threshold


if __name__ == "__main__":
    train()
