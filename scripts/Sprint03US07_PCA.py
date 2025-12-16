# -*- coding: utf-8 -*-
"""
Sprint 03 — US-07: PCA Dimensionality Reduction.

Generates an explained variance plot and a 2D PCA scatter using the processed
dataset (features only). Saves plots to `docs/pca_explained_variance.png` and
`docs/pca_scatter.png` for quick inclusion in slides or the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

PROCESSED_FILE = ROOT / "data" / "processed" / "processed_data.csv"
EXPLAINED_VAR_PNG = ROOT / "docs" / "pca_explained_variance.png"
SCATTER_PNG = ROOT / "docs" / "pca_scatter.png"
TARGET_DROP = ["target_price_next", "target_direction"]


def load_features() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            "processed_data.csv not found. Run Sprint01US03_Preprocessing_FeatureEngineering.py first."
        )
    df = pd.read_csv(PROCESSED_FILE, parse_dates=["Date"], index_col="Date")
    feature_cols = [c for c in df.columns if c not in TARGET_DROP]
    return df[feature_cols]


def plot_explained_variance(pca: PCA) -> None:
    ratios = pca.explained_variance_ratio_
    cum = ratios.cumsum()
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(ratios) + 1), ratios, alpha=0.6, label="Individual")
    plt.plot(range(1, len(cum) + 1), cum, marker="o", color="red", label="Cumulative")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EXPLAINED_VAR_PNG, dpi=150)
    plt.close()


def plot_scatter(pca: PCA, X: pd.DataFrame) -> None:
    components = pca.transform(X)[:, :2]
    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1], s=10, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D Scatter (PC1 vs PC2)")
    plt.tight_layout()
    plt.savefig(SCATTER_PNG, dpi=150)
    plt.close()


def main() -> None:
    X = load_features()
    # Limit components to a small number so plots stay readable and quick.
    pca = PCA(n_components=min(10, X.shape[1]))
    pca.fit(X)

    plot_explained_variance(pca)
    plot_scatter(pca, X)

    top2_var = pca.explained_variance_ratio_[:2].sum()
    print("=== Sprint 03 US-07: PCA ===")
    print(f"Input features: {X.shape[1]} | Samples: {X.shape[0]}")
    print(f"Explained variance by PC1+PC2: {top2_var:0.4f}")
    print(f"Saved plots: {EXPLAINED_VAR_PNG} and {SCATTER_PNG}")


if __name__ == "__main__":
    main()
