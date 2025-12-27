# -*- coding: utf-8 -*-
"""
Sprint 04 — US-10: API Test Script.

Simple tests for the Flask API endpoints.
Run with: python src/api/test_api.py
"""
import requests

BASE_URL = "http://localhost:5000"


def test_metrics():
    """Test /api/metrics endpoint."""
    r = requests.get(f"{BASE_URL}/api/metrics")
    print(f"GET /api/metrics - Status: {r.status_code}")
    print(f"  Response: {r.json()}\n")
    return r.status_code == 200


def test_clusters():
    """Test /api/clusters endpoint."""
    r = requests.get(f"{BASE_URL}/api/clusters")
    print(f"GET /api/clusters - Status: {r.status_code}")
    data = r.json()
    print(f"  Clusters: {data.get('cluster_distribution', 'N/A')}")
    print(f"  Total samples: {data.get('total_samples', 'N/A')}\n")
    return r.status_code == 200


def test_anomalies():
    """Test /api/anomalies endpoint."""
    r = requests.get(f"{BASE_URL}/api/anomalies")
    print(f"GET /api/anomalies - Status: {r.status_code}")
    data = r.json()
    print(f"  Total anomalies: {data.get('total_anomalies', 'N/A')}")
    print(f"  Anomaly %: {data.get('anomaly_pct', 'N/A'):.2f}%\n")
    return r.status_code == 200


def test_data():
    """Test /api/data endpoint."""
    r = requests.get(f"{BASE_URL}/api/data?page=1&per_page=5")
    print(f"GET /api/data - Status: {r.status_code}")
    data = r.json()
    print(f"  Page: {data.get('page')} | Total: {data.get('total')}\n")
    return r.status_code == 200


def test_predict_rf():
    """Test /api/predict_rf endpoint."""
    # Sample features (adjust based on your model)
    payload = {"features": [0.1] * 33}  # 33 features
    r = requests.post(f"{BASE_URL}/api/predict_rf", json=payload)
    print(f"POST /api/predict_rf - Status: {r.status_code}")
    print(f"  Response: {r.json()}\n")
    return r.status_code == 200


def main():
    print("=" * 50)
    print("API Test Suite")
    print("=" * 50)
    print(f"Testing: {BASE_URL}\n")
    
    tests = [
        ("Metrics", test_metrics),
        ("Clusters", test_clusters),
        ("Anomalies", test_anomalies),
        ("Data", test_data),
        ("Predict RF", test_predict_rf),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "✓ PASS" if passed else "✗ FAIL"))
        except Exception as e:
            results.append((name, f"✗ ERROR: {e}"))
    
    print("=" * 50)
    print("Results:")
    for name, status in results:
        print(f"  {name}: {status}")
    print("=" * 50)


if __name__ == "__main__":
    main()
