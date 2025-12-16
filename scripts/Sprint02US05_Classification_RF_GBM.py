# -*- coding: utf-8 -*-
"""
Sprint 02 — US-05: Random Forest and Gradient Boosting classification.

Trains two classifiers to predict next-month direction (`target_direction`)
using the preprocessed dataset from Sprint01US03. Prints Accuracy/Precision/
Recall/F1 and saves the trained models to `models/rf.pkl` and `models/gbm.pkl`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------------------------------
# Resolve paths so the script works from repo root or scripts/ folder.
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

PROCESSED_FILE = ROOT / "data" / "processed" / "processed_data.csv"
MODEL_DIR = ROOT / "models"
TARGET_COL = "target_direction"
TARGET_DROP = ["target_price_next", "target_direction"]  # drop regression target and direction from features
TRAIN_RATIO = 0.8  # chronological split to avoid leakage


def load_processed() -> pd.DataFrame:
    """Load the processed dataset and ensure Date index is parsed."""
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            "processed_data.csv not found. Run Sprint01US03_Preprocessing_FeatureEngineering.py first."
        )
    df = pd.read_csv(PROCESSED_FILE, parse_dates=["Date"], index_col="Date")
    return df.dropna(subset=[TARGET_COL])


def train_test_split_time(df: pd.DataFrame, ratio: float = TRAIN_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split to keep future rows out of training."""
    split_idx = max(int(len(df) * ratio), 1)
    split_idx = min(split_idx, len(df) - 1)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute core classification metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def fit_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train a small RF with a tiny grid search (kept minimal for speed)."""
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    }
    search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    return search.best_estimator_


def fit_gradient_boosting(X_train, y_train) -> GradientBoostingClassifier:
    """Train a Gradient Boosting classifier with a tiny grid search."""
    gb = GradientBoostingClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
    }
    search = GridSearchCV(gb, param_grid, cv=3, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    return search.best_estimator_


def main() -> None:
    # Load and split data ----------------------------------------------------
    df = load_processed()
    feature_cols = [c for c in df.columns if c not in TARGET_DROP]
    X_train, X_test = train_test_split_time(df[feature_cols])
    y_train, y_test = train_test_split_time(df[[TARGET_COL]])
    y_train = y_train[TARGET_COL].values
    y_test = y_test[TARGET_COL].values

    # Fit models ------------------------------------------------------------
    rf_model = fit_random_forest(X_train, y_train)
    gb_model = fit_gradient_boosting(X_train, y_train)

    # Predict ---------------------------------------------------------------
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)

    # Metrics ---------------------------------------------------------------
    rf_metrics = evaluate(y_test, rf_pred)
    gb_metrics = evaluate(y_test, gb_pred)

    print("=== Sprint 02 US-05: RF vs GBM (direction classification) ===")
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,} | Split ratio: {TRAIN_RATIO}")

    for name, metrics, preds in [
        ("RandomForest", rf_metrics, rf_pred),
        ("GradientBoosting", gb_metrics, gb_pred),
    ]:
        print(f"\n{name} metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:0.4f}")
        print("  Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_test, preds))

    print("\nClassification report (RandomForest):")
    print(classification_report(y_test, rf_pred, zero_division=0))
    print("Classification report (GradientBoosting):")
    print(classification_report(y_test, gb_pred, zero_division=0))

    # Save models -----------------------------------------------------------
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump(rf_model, MODEL_DIR / "rf.pkl")
    joblib.dump(gb_model, MODEL_DIR / "gbm.pkl")
    print(f"\nSaved models to: {MODEL_DIR / 'rf.pkl'} and {MODEL_DIR / 'gbm.pkl'}")


if __name__ == "__main__":
    main()
