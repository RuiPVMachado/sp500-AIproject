# -*- coding: utf-8 -*-
"""
Sprint 02 — US-06: LSTM time series model for next-month S&P 500 price.

This script builds a small LSTM to predict `target_price_next` using the
preprocessed dataset from Sprint01US03. It keeps the architecture and epochs
minimal so it can train quickly during a classroom demo.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------------------------
# Resolve paths so the script works from repo root or scripts/ folder.
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

PROCESSED_FILE = ROOT / "data" / "processed" / "processed_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "lstm_model.h5"
TARGET_COL = "target_price_next"
TARGET_DROP = ["target_price_next", "target_direction"]
TRAIN_RATIO = 0.8
SEQ_LEN = 12  # use past 12 months to predict next month
EPOCHS = 5   # keep small for quick demo; bump if you want better fit
BATCH_SIZE = 32


def load_processed() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            "processed_data.csv not found. Run Sprint01US03_Preprocessing_FeatureEngineering.py first."
        )
    df = pd.read_csv(PROCESSED_FILE, parse_dates=["Date"], index_col="Date")
    return df.dropna(subset=[TARGET_COL])


def make_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int = SEQ_LEN) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a time-indexed dataframe into (X, y) sequences for LSTM."""
    data = df[feature_cols].values.astype(np.float32)
    targets = df[TARGET_COL].values.astype(np.float32)

    X_list = []
    y_list = []
    for i in range(seq_len, len(df)):
        X_list.append(data[i - seq_len : i])
        y_list.append(targets[i])
    return np.stack(X_list), np.array(y_list)


def train_test_split_sequences(X: np.ndarray, y: np.ndarray, ratio: float = TRAIN_RATIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_idx = max(int(len(X) * ratio), 1)
    split_idx = min(split_idx, len(X) - 1)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def build_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def main() -> None:
    # Load and prep data ----------------------------------------------------
    df = load_processed()
    feature_cols = [c for c in df.columns if c not in TARGET_DROP]
    X, y = make_sequences(df, feature_cols, SEQ_LEN)

    X_train, X_test, y_train, y_test = train_test_split_sequences(X, y, TRAIN_RATIO)

    # Build and train -------------------------------------------------------
    model = build_model(input_shape=(SEQ_LEN, X.shape[-1]))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # Evaluate --------------------------------------------------------------
    preds = model.predict(X_test, verbose=0)
    mse = tf.keras.losses.MSE(y_test, preds[:, 0]).numpy().mean()
    rmse = float(np.sqrt(mse))

    print("=== Sprint 02 US-06: LSTM Next-Month Price ===")
    print(f"Train sequences: {len(X_train):,} | Test sequences: {len(X_test):,} | Seq len: {SEQ_LEN}")
    print(f"Test RMSE: {rmse:0.4f}")

    # Save model ------------------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
