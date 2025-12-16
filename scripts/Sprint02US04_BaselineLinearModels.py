# -*- coding: utf-8 -*-
"""
Sprint 02 — US-04: Baseline linear models (Linear, Ridge, Lasso).

This script trains three regression models to predict the next-month S&P 500
price (`target_price_next`) using the preprocessed dataset generated in
Sprint01US03. It prints MAE, MSE and RMSE for a simple train/test temporal
split. Keep it lightweight for classroom demos.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# Resolve project paths so the script works when run from repo root or scripts/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

PROCESSED_FILE = ROOT / "data" / "processed" / "processed_data.csv"
TARGET_COL = "target_price_next"
TARGET_DROP = ["target_price_next", "target_direction"]  # drop direction for regression
TRAIN_RATIO = 0.8  # simple chronological split


def load_processed() -> pd.DataFrame:
    """Load the processed dataset and ensure the Date index is parsed."""
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            "processed_data.csv not found. Run Sprint01US03_Preprocessing_FeatureEngineering.py first."
        )
    df = pd.read_csv(PROCESSED_FILE, parse_dates=["Date"], index_col="Date")
    return df


def train_test_split_time(df: pd.DataFrame, ratio: float = TRAIN_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split to avoid leakage from future data into training."""
    split_idx = max(int(len(df) * ratio), 1)
    split_idx = min(split_idx, len(df) - 1)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate_model(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    """Fit the model and return standard regression metrics."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def main() -> None:
    # Load data --------------------------------------------------------------
    df = load_processed()
    df = df.dropna(subset=[TARGET_COL])

    feature_cols = [c for c in df.columns if c not in TARGET_DROP]
    X_train, X_test = train_test_split_time(df[feature_cols])
    y_train, y_test = train_test_split_time(df[[TARGET_COL]])

    # Convert targets to 1D arrays for scikit-learn regressors
    y_train = y_train[TARGET_COL].values
    y_test = y_test[TARGET_COL].values

    # Define models with simple, explainable settings -----------------------
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0),
        # Slightly higher alpha + iterations to avoid convergence warnings for class demo.
        "Lasso(alpha=0.01)": Lasso(alpha=0.01, max_iter=10000),
    }

    # Train/evaluate --------------------------------------------------------
    rows = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        rows.append({"Model": name, **metrics})

    results = pd.DataFrame(rows)
    print("=== Sprint 02 US-04: Baseline Linear Models ===")
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,} | Split ratio: {TRAIN_RATIO}")
    print(results.to_string(index=False, float_format=lambda v: f"{v:0.4f}"))


if __name__ == "__main__":
    main()
