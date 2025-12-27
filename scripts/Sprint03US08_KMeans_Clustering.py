# -*- coding: utf-8 -*-
"""
Sprint 03 — US-08: K-Means Market Regime Clustering.

Identifies historical market regimes using K-Means. Saves cluster labels
to processed_data_with_clusters.csv and generates regime transition plots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

PROCESSED_FILE = ROOT / "data" / "processed" / "processed_data.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "processed_data_with_clusters.csv"
DOCS_DIR = ROOT / "docs"
TARGET_DROP = ["target_price_next", "target_direction"]
OPTIMAL_K = 4


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and return (full df, features only)."""
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError("processed_data.csv not found.")
    df = pd.read_csv(PROCESSED_FILE, parse_dates=["Date"], index_col="Date")
    feature_cols = [c for c in df.columns if c not in TARGET_DROP]
    return df, df[feature_cols].dropna()


def main() -> None:
    # Load and scale
    df, X = load_features()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method
    inertias = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(2, 11), inertias, 'bo-')
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig(DOCS_DIR / "kmeans_elbow.png", dpi=150)
    plt.close()

    # Train final model
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Add labels to dataset
    df_out = df.loc[X.index].copy()
    df_out["cluster"] = labels
    df_out.to_csv(OUTPUT_FILE)

    # PCA scatter
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=10, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("K-Means Clusters (PCA)")
    plt.colorbar(label="Cluster")
    plt.savefig(DOCS_DIR / "kmeans_pca.png", dpi=150)
    plt.close()

    # Regime transitions over time
    plt.figure(figsize=(14, 4))
    plt.scatter(df_out.index, df_out["cluster"], c=df_out["cluster"], cmap="tab10", s=5)
    plt.xlabel("Date")
    plt.ylabel("Regime")
    plt.title("Market Regime Transitions")
    plt.savefig(DOCS_DIR / "kmeans_transitions.png", dpi=150)
    plt.close()

    # Print summary
    sil = silhouette_score(X_scaled, labels)
    print("=== Sprint 03 US-08: K-Means Clustering ===")
    print(f"Samples: {len(X)} | Features: {X.shape[1]} | Clusters: {OPTIMAL_K}")
    print(f"Silhouette Score: {sil:.4f}")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Plots: kmeans_elbow.png, kmeans_pca.png, kmeans_transitions.png")


if __name__ == "__main__":
    main()
