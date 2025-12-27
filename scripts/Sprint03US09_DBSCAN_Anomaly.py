# -*- coding: utf-8 -*-
"""
Sprint 03 — US-09: DBSCAN Anomaly Detection.

Detects market anomalies (crashes/spikes) using DBSCAN. Saves results to
processed_data_with_anomalies.csv and generates docs/anomalies.md report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

PROCESSED_FILE = ROOT / "data" / "processed" / "processed_data.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "processed_data_with_anomalies.csv"
DOCS_DIR = ROOT / "docs"
TARGET_DROP = ["target_price_next", "target_direction"]
EPS = 2.5
MIN_SAMPLES = 14


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

    # Run DBSCAN
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
    labels = dbscan.fit_predict(X_scaled)

    # Add labels to dataset
    df_out = df.loc[X.index].copy()
    df_out["dbscan_cluster"] = labels
    df_out["is_anomaly"] = labels == -1
    df_out.to_csv(OUTPUT_FILE)

    # Stats
    n_anomalies = (labels == -1).sum()
    pct = n_anomalies / len(labels) * 100
    anomalies = df_out[df_out["is_anomaly"]]
    normal = labels != -1

    # PCA scatter with anomalies
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[normal, 0], X_pca[normal, 1], c="steelblue", s=10, alpha=0.5, label="Normal")
    plt.scatter(X_pca[~normal, 0], X_pca[~normal, 1], c="red", s=50, marker="X", label="Anomaly")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("DBSCAN Anomalies (PCA)")
    plt.legend()
    plt.savefig(DOCS_DIR / "dbscan_pca.png", dpi=150)
    plt.close()

    # Timeline plot
    plt.figure(figsize=(14, 5))
    plt.scatter(df_out.index[normal], df_out.loc[normal, "SP500"], c="steelblue", s=5, alpha=0.5)
    plt.scatter(anomalies.index, anomalies["SP500"], c="red", s=30, marker="X", label="Anomaly")
    plt.xlabel("Date")
    plt.ylabel("SP500")
    plt.title("S&P 500 with Detected Anomalies")
    plt.legend()
    plt.savefig(DOCS_DIR / "dbscan_timeline.png", dpi=150)
    plt.close()

    # Generate report
    crashes = anomalies[anomalies.get("return_1m", 0) < 0].sort_values("return_1m").head(5)
    spikes = anomalies[anomalies.get("return_1m", 0) > 0].sort_values("return_1m", ascending=False).head(5)

    report = f"""# DBSCAN Anomaly Detection

## Parameters
- eps: {EPS}
- min_samples: {MIN_SAMPLES}

## Results
- Total samples: {len(labels)}
- Anomalies: {n_anomalies} ({pct:.2f}%)

## Top Crashes Detected
| Date | Return 1m |
|------|-----------|
"""
    for date, row in crashes.iterrows():
        report += f"| {date.strftime('%Y-%m')} | {row.get('return_1m', 0):.4f} |\n"

    report += """
## Top Spikes Detected
| Date | Return 1m |
|------|-----------|
"""
    for date, row in spikes.iterrows():
        report += f"| {date.strftime('%Y-%m')} | {row.get('return_1m', 0):.4f} |\n"

    with open(DOCS_DIR / "anomalies.md", "w") as f:
        f.write(report)

    # Print summary
    print("=== Sprint 03 US-09: DBSCAN Anomaly Detection ===")
    print(f"Samples: {len(X)} | eps={EPS} | min_samples={MIN_SAMPLES}")
    print(f"Anomalies: {n_anomalies} ({pct:.2f}%)")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Report: {DOCS_DIR / 'anomalies.md'}")


if __name__ == "__main__":
    main()
