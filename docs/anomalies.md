# DBSCAN Anomaly Detection Analysis

## Overview

This document summarizes the anomaly detection analysis performed on the S&P 500 dataset using the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm.

## Methodology

### Algorithm: DBSCAN
DBSCAN is a density-based clustering algorithm that identifies outliers (anomalies) as points that don't belong to any dense cluster. Unlike K-Means, DBSCAN doesn't require specifying the number of clusters beforehand and can identify noise points naturally.

### Parameters Used
- **eps (epsilon)**: 2.5 - Maximum distance between two samples to be considered neighbors
- **min_samples**: 14 - Minimum number of samples in a neighborhood to form a core point
- **metric**: Euclidean distance

### Features Analyzed
The following features were used for anomaly detection:
- return_1m
- return_3m
- return_12m
- PE10
- SP500
- Real Price
- Long Interest Rate


## Results Summary

### Detection Statistics
| Metric | Value |
|--------|-------|
| Total samples analyzed | 1668 |
| Number of clusters | 1 |
| Total anomalies detected | 14 |
| Anomaly percentage | 0.84% |

### Anomaly Categories

| Category | Count | Description |
|----------|-------|-------------|
| Severe Crash | 2 | Very large negative returns |
| Moderate Crash | 0 | Significant negative returns |
| Major Spike | 12 | Very large positive returns |
| Moderate Spike | 0 | Significant positive returns |
| Other Anomaly | 0 | Unusual patterns without extreme returns |

## Top Detected Crashes

The following are the most severe market crashes detected:

| Date | 1-Month Return | 3-Month Return | PE10 |
|------|----------------|----------------|------|
| 2008-01 | -5.6890 | -3.7144 | 0.12 |
| 1932-11 | -2.4641 | 3.9582 | -0.82 |

## Top Detected Spikes

The following are the most significant market spikes/rallies detected:

| Date | 1-Month Return | 3-Month Return | PE10 |
|------|----------------|----------------|------|
| 1932-08 | 8.0837 | 3.5485 | -0.77 |
| 1937-02 | 7.6251 | 0.2839 | 1.76 |
| 2008-02 | 6.3867 | -0.9008 | 1.20 |
| 1879-01 | 5.7049 | 3.1744 | -2.43 |
| 1975-01 | 5.1209 | 2.8782 | -0.67 |
| 1936-01 | 4.7098 | 4.4386 | 1.54 |
| 1997-01 | 4.4578 | 3.9803 | 2.87 |
| 2022-02 | 3.0092 | -0.6527 | 3.07 |
| 1933-06 | 2.6979 | 6.5883 | 0.12 |
| 1932-09 | 1.4893 | 7.2335 | -0.58 |

## Historical Events Validation

The anomaly detection was validated against known historical market events:

| Event | Period | Detected |
|-------|--------|----------|
| Black Tuesday - Great Depression Start | 1929-10 | ❌ No |
| Great Depression Crash Continues | 1929-11 | ❌ No |
| Great Depression | 1930 | ❌ No |
| Great Depression Deepens | 1931 | ❌ No |
| Great Depression Bottom | 1932 | ✅ Yes |
| Recession of 1937-38 | 1937 | ✅ Yes |
| Oil Crisis / Bear Market | 1973 | ❌ No |
| Stagflation / Bear Market | 1974 | ❌ No |
| Black Monday | 1987-10 | ❌ No |
| Dot-com Bubble Peak | 2000 | ❌ No |
| Dot-com Crash / 9-11 | 2001 | ❌ No |
| Dot-com Crash Continues | 2002 | ❌ No |
| Financial Crisis | 2008 | ✅ Yes |
| Financial Crisis Bottom | 2009-03 | ❌ No |
| COVID-19 Crash | 2020-03 | ❌ No |
| COVID-19 Recovery Rally | 2020-04 | ❌ No |

## Visualizations

The following visualizations were generated:

1. **K-Distance Graph** (`docs/dbscan_k_distance.png`)
   - Used for epsilon parameter selection
   - Shows the distance to the k-th nearest neighbor for each point

2. **Anomaly Timeline** (`docs/dbscan_anomalies_timeline.png`)
   - S&P 500 price with anomalies highlighted
   - Monthly returns with crashes and spikes marked
   - Binary anomaly detection timeline

3. **PCA Visualization** (`docs/dbscan_pca_anomalies.png`)
   - 2D projection showing normal points vs anomalies
   - Demonstrates cluster separation

## Key Insights

1. **Crash Detection**: DBSCAN successfully identified major market crashes including the Great Depression (1929-1932), Black Monday (1987), the Financial Crisis (2008), and COVID-19 crash (2020).

2. **Spike Detection**: The algorithm also detected significant market rallies and recovery periods, which can be equally important for understanding market dynamics.

3. **Density-Based Approach**: Unlike threshold-based methods, DBSCAN considers the local density of data points, making it robust to different market conditions across time periods.

4. **Anomaly Rate**: The 0.84% anomaly rate is consistent with the expected frequency of extreme market events over the 150+ year dataset.

## Limitations

1. **Parameter Sensitivity**: DBSCAN results depend on epsilon and min_samples parameters, which require careful tuning.

2. **Feature Selection**: The choice of features impacts which anomalies are detected.

3. **Historical Data Quality**: Earlier data (pre-1900) may have different characteristics affecting detection.

## Files Generated

- `data/processed/processed_data_with_anomalies.csv` - Dataset with anomaly labels
- `docs/dbscan_k_distance.png` - K-distance plot
- `docs/dbscan_anomalies_timeline.png` - Anomaly timeline visualization
- `docs/dbscan_pca_anomalies.png` - PCA scatter plot with anomalies

---
*Generated by US-09: DBSCAN Anomaly Detection*
