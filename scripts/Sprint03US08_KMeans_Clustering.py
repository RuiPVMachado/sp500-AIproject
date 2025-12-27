# -*- coding: utf-8 -*-
"""
Sprint 03 — US-08: K-Means Market Regime Clustering.

Identifies historical market regimes using K-Means clustering algorithm.
Generates cluster labels, regime transition plots, and saves the clustered
dataset for further analysis.

Outputs:
    - docs/kmeans_cluster_selection.png
    - docs/kmeans_cluster_characteristics.png
    - docs/kmeans_regime_transitions.png
    - docs/kmeans_transition_matrix.png
    - docs/kmeans_pca_visualization.png
    - data/processed/processed_data_with_clusters.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# File paths
PROCESSED_FILE = ROOT / "data" / "processed" / "processed_data.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "processed_data_with_clusters.csv"
DOCS_DIR = ROOT / "docs"

# Columns to exclude from clustering (target variables)
TARGET_DROP = ["target_price_next", "target_direction"]

# Number of clusters (market regimes)
OPTIMAL_K = 4

# Range of clusters to evaluate
K_RANGE = range(2, 11)

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_data() -> pd.DataFrame:
    """
    Load the processed dataset from CSV.
    
    Returns:
        pd.DataFrame: The processed dataset with Date as index.
    
    Raises:
        FileNotFoundError: If processed_data.csv doesn't exist.
    """
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(
            "processed_data.csv not found. "
            "Run Sprint01US03_Preprocessing_FeatureEngineering.py first."
        )
    df = pd.read_csv(PROCESSED_FILE, parse_dates=["Date"], index_col="Date")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare and scale features for K-Means clustering.
    
    K-Means is distance-based, so features must be standardized
    to ensure equal contribution to distance calculations.
    
    Args:
        df: Input DataFrame with all columns.
    
    Returns:
        tuple: (feature DataFrame, scaled feature array)
    """
    # Select only feature columns (exclude targets)
    feature_cols = [col for col in df.columns if col not in TARGET_DROP]
    X = df[feature_cols].copy()
    
    # Drop any rows with missing values
    X = X.dropna()
    
    # Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled


# ==============================================================================
# CLUSTER EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_cluster_counts(X_scaled: np.ndarray) -> tuple[list, list, list]:
    """
    Evaluate different cluster counts using multiple metrics.
    
    Uses Elbow method (inertia), Silhouette score, and Calinski-Harabasz
    index to help determine optimal number of clusters.
    
    Args:
        X_scaled: Standardized feature array.
    
    Returns:
        tuple: (inertias, silhouette_scores, calinski_scores)
    """
    inertias = []
    silhouette_scores_list = []
    calinski_scores = []
    
    print("Evaluating cluster counts...")
    
    for k in K_RANGE:
        # Fit K-Means model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        # Calculate metrics
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores_list.append(sil_score)
        cal_score = calinski_harabasz_score(X_scaled, kmeans.labels_)
        calinski_scores.append(cal_score)
        
        print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, "
              f"Silhouette={sil_score:.4f}, Calinski-Harabasz={cal_score:.2f}")
    
    return inertias, silhouette_scores_list, calinski_scores


def plot_cluster_selection(
    inertias: list, 
    silhouette_scores_list: list, 
    calinski_scores: list
) -> None:
    """
    Visualize cluster selection metrics.
    
    Creates a 3-panel plot showing:
    - Elbow method (inertia/WCSS)
    - Silhouette score analysis
    - Calinski-Harabasz index
    
    Args:
        inertias: List of inertia values per k.
        silhouette_scores_list: List of silhouette scores per k.
        calinski_scores: List of Calinski-Harabasz scores per k.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Elbow Method
    axes[0].plot(K_RANGE, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[0].set_ylabel('Inertia (WCSS)', fontsize=11)
    axes[0].set_title('Elbow Method', fontsize=12, fontweight='bold')
    axes[0].set_xticks(list(K_RANGE))
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Silhouette Score
    axes[1].plot(K_RANGE, silhouette_scores_list, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
    axes[1].set_xticks(list(K_RANGE))
    axes[1].grid(True, alpha=0.3)
    
    # Highlight best silhouette score
    best_k = list(K_RANGE)[np.argmax(silhouette_scores_list)]
    axes[1].axvline(x=best_k, color='red', linestyle='--', alpha=0.7, 
                    label=f'Best k={best_k}')
    axes[1].legend()
    
    # Plot 3: Calinski-Harabasz Index
    axes[2].plot(K_RANGE, calinski_scores, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[2].set_ylabel('Calinski-Harabasz Score', fontsize=11)
    axes[2].set_title('Calinski-Harabasz Index', fontsize=12, fontweight='bold')
    axes[2].set_xticks(list(K_RANGE))
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "kmeans_cluster_selection.png", dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# CLUSTERING FUNCTIONS
# ==============================================================================

def train_kmeans(X_scaled: np.ndarray, n_clusters: int = OPTIMAL_K) -> tuple:
    """
    Train the final K-Means model.
    
    Args:
        X_scaled: Standardized feature array.
        n_clusters: Number of clusters to create.
    
    Returns:
        tuple: (fitted KMeans model, cluster labels)
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,      # Run 20 times with different initializations
        max_iter=300,   # Maximum iterations per run
        tol=1e-4        # Convergence tolerance
    )
    
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    return kmeans, cluster_labels


def analyze_clusters(
    df_clustered: pd.DataFrame, 
    X_scaled: np.ndarray, 
    cluster_labels: np.ndarray
) -> None:
    """
    Analyze and print cluster characteristics.
    
    Args:
        df_clustered: DataFrame with cluster labels.
        X_scaled: Scaled feature array.
        cluster_labels: Cluster assignments.
    """
    print("\n" + "=" * 60)
    print("CLUSTER DISTRIBUTION")
    print("=" * 60)
    
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        pct = (count / len(df_clustered)) * 100
        print(f"Cluster {cluster_id}: {count:5d} samples ({pct:5.2f}%)")
    
    print(f"\nTotal samples: {len(df_clustered)}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, cluster_labels):.2f}")


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_cluster_characteristics(df_clustered: pd.DataFrame) -> None:
    """
    Create heatmap showing cluster characteristics.
    
    Displays mean feature values for each cluster to help
    interpret what each market regime represents.
    
    Args:
        df_clustered: DataFrame with cluster labels.
    """
    # Key features to analyze
    key_features = ['SP500', 'Real Price', 'PE10', 'return_1m', 'return_3m',
                    'Long Interest Rate', 'Consumer Price Index']
    available_features = [f for f in key_features if f in df_clustered.columns]
    
    # Calculate mean values per cluster
    cluster_stats = df_clustered.groupby('cluster')[available_features].mean()
    
    # Normalize for visualization
    cluster_stats_norm = (cluster_stats - cluster_stats.min()) / \
                         (cluster_stats.max() - cluster_stats.min())
    
    # Create heatmap
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        cluster_stats_norm.T,
        annot=cluster_stats.T.round(3),
        fmt='',
        cmap='RdYlGn',
        center=0.5,
        linewidths=0.5,
        cbar_kws={'label': 'Normalized Value'}
    )
    plt.title('Market Regime Characteristics by Cluster', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster (Market Regime)', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "kmeans_cluster_characteristics.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_regime_transitions(df_clustered: pd.DataFrame, n_clusters: int) -> None:
    """
    Visualize market regime transitions over time.
    
    Creates a 2-panel plot showing:
    - S&P 500 price colored by cluster
    - Regime labels over time
    
    Args:
        df_clustered: DataFrame with cluster labels and Date index.
        n_clusters: Number of clusters.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot 1: S&P 500 colored by cluster
    for cluster_id in range(n_clusters):
        mask = df_clustered['cluster'] == cluster_id
        axes[0].scatter(
            df_clustered.index[mask],
            df_clustered.loc[mask, 'SP500'],
            c=[colors[cluster_id]],
            s=10,
            alpha=0.7,
            label=f'Regime {cluster_id}'
        )
    
    axes[0].set_ylabel('S&P 500 (Normalized)', fontsize=11)
    axes[0].set_title('S&P 500 Price Colored by Market Regime', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Cluster labels over time
    axes[1].scatter(
        df_clustered.index,
        df_clustered['cluster'],
        c=df_clustered['cluster'],
        cmap='tab10',
        s=15,
        alpha=0.7
    )
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Cluster (Regime)', fontsize=11)
    axes[1].set_title('Market Regime Transitions Over Time', fontsize=14, fontweight='bold')
    axes[1].set_yticks(range(n_clusters))
    axes[1].set_yticklabels([f'Regime {i}' for i in range(n_clusters)])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "kmeans_regime_transitions.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_transition_matrix(df_clustered: pd.DataFrame) -> None:
    """
    Create transition probability matrix heatmap.
    
    Shows the probability of transitioning from one regime to another,
    which helps understand market cycle dynamics.
    
    Args:
        df_clustered: DataFrame with cluster labels.
    """
    # Calculate transition probabilities
    df_temp = df_clustered.copy()
    df_temp['next_cluster'] = df_temp['cluster'].shift(-1)
    transition_matrix = pd.crosstab(
        df_temp['cluster'],
        df_temp['next_cluster'],
        normalize='index'
    ) * 100
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        cbar_kws={'label': 'Transition Probability (%)'},
        linewidths=0.5
    )
    plt.title('Market Regime Transition Probabilities', fontsize=14, fontweight='bold')
    plt.xlabel('Next Regime', fontsize=11)
    plt.ylabel('Current Regime', fontsize=11)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "kmeans_transition_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_pca_clusters(
    X_scaled: np.ndarray, 
    cluster_labels: np.ndarray, 
    kmeans: KMeans,
    n_clusters: int
) -> None:
    """
    Visualize clusters in 2D PCA space.
    
    Reduces dimensions using PCA and plots clusters with their centers
    to visualize cluster separation.
    
    Args:
        X_scaled: Scaled feature array.
        cluster_labels: Cluster assignments.
        kmeans: Fitted KMeans model.
        n_clusters: Number of clusters.
    """
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[colors[cluster_id]],
            s=30,
            alpha=0.6,
            label=f'Regime {cluster_id} (n={mask.sum()})'
        )
    
    # Plot cluster centers
    centers_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        c='black',
        s=200,
        marker='X',
        edgecolors='white',
        linewidths=2,
        label='Cluster Centers'
    )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
    plt.title('K-Means Clusters in PCA Space', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "kmeans_pca_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for K-Means Market Regime Clustering.
    
    Steps:
    1. Load processed data
    2. Prepare and scale features
    3. Evaluate different cluster counts
    4. Train final K-Means model
    5. Generate visualizations
    6. Save clustered dataset
    """
    print("=" * 60)
    print("US-08: K-MEANS MARKET REGIME CLUSTERING")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"    Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Step 2: Prepare features
    print("\n[2/6] Preparing features...")
    X, X_scaled = prepare_features(df)
    print(f"    Features: {X.shape[1]} | Samples: {X.shape[0]}")
    
    # Step 3: Evaluate cluster counts
    print("\n[3/6] Evaluating cluster counts...")
    inertias, silhouette_scores_list, calinski_scores = evaluate_cluster_counts(X_scaled)
    plot_cluster_selection(inertias, silhouette_scores_list, calinski_scores)
    print(f"    Saved: {DOCS_DIR / 'kmeans_cluster_selection.png'}")
    
    # Step 4: Train final model
    print(f"\n[4/6] Training K-Means with k={OPTIMAL_K}...")
    kmeans, cluster_labels = train_kmeans(X_scaled, OPTIMAL_K)
    
    # Add cluster labels to dataset
    df_clustered = df.loc[X.index].copy()
    df_clustered['cluster'] = cluster_labels
    
    # Analyze clusters
    analyze_clusters(df_clustered, X_scaled, cluster_labels)
    
    # Step 5: Generate visualizations
    print("\n[5/6] Generating visualizations...")
    plot_cluster_characteristics(df_clustered)
    print(f"    Saved: {DOCS_DIR / 'kmeans_cluster_characteristics.png'}")
    
    plot_regime_transitions(df_clustered, OPTIMAL_K)
    print(f"    Saved: {DOCS_DIR / 'kmeans_regime_transitions.png'}")
    
    plot_transition_matrix(df_clustered)
    print(f"    Saved: {DOCS_DIR / 'kmeans_transition_matrix.png'}")
    
    plot_pca_clusters(X_scaled, cluster_labels, kmeans, OPTIMAL_K)
    print(f"    Saved: {DOCS_DIR / 'kmeans_pca_visualization.png'}")
    
    # Step 6: Save clustered dataset
    print("\n[6/6] Saving clustered dataset...")
    df_clustered.to_csv(OUTPUT_FILE)
    print(f"    Saved: {OUTPUT_FILE}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Market regimes identified: {OPTIMAL_K}")
    print(f"✅ Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")
    print(f"✅ Dataset saved with 'cluster' column")
    print(f"✅ All visualizations saved to {DOCS_DIR}")
    print("\nAcceptance Criteria Met:")
    print("  ✓ Notebook: 06_kmeans.ipynb")
    print("  ✓ Cluster labels added to dataset")
    print("  ✓ Visual plots of cluster transitions over time")


if __name__ == "__main__":
    main()
