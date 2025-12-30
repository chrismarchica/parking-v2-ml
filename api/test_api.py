"""Test script for the parking prediction API."""

import requests
from datetime import datetime

API_BASE = "http://localhost:5000"


def test_health():
    """Test health check endpoint."""
    response = requests.get(f"{API_BASE}/health")
    print("Health Check:")
    print(response.json())
    print()


def test_predict():
    """Test prediction endpoint."""
    payload = {
        "latitude": 40.7580,
        "longitude": -73.9855,
        "datetime": datetime.now().isoformat(),
        "precinct": "19",
        "county": "NY"
    }
    
    response = requests.post(f"{API_BASE}/predict", json=payload)
    result = response.json()
    
    print("Prediction:")
    print(f"  Location: {payload['latitude']}, {payload['longitude']}")
    print(f"  Predicted Violation: {result['predicted_violation']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Top 3 Predictions:")
    for pred in result['top_predictions'][:3]:
        print(f"    - Code {pred['violation_code']}: {pred['probability']:.2%}")
    print()


def test_violation_stats():
    """Test violation statistics endpoint."""
    response = requests.get(f"{API_BASE}/violations/stats")
    result = response.json()
    
    print("Top 5 Violations:")
    for violation in result['violations'][:5]:
        print(f"  - Code {violation['violation_code']}: {violation['ticket_count']:,} tickets")
    print()


def test_location_hotspots():
    """Test location hotspots endpoint."""
    response = requests.get(f"{API_BASE}/locations/hotspots")
    result = response.json()
    
    print("Top 5 Hotspots:")
    for hotspot in result['hotspots'][:5]:
        print(f"  - Precinct {hotspot['precinct']} ({hotspot['county']}): {hotspot['ticket_count']:,} tickets")
    print()


def test_heatmap_data():
    """Test heatmap data endpoint."""
    response = requests.get(f"{API_BASE}/heatmap/data?limit=100")
    result = response.json()
    
    print(f"Heatmap Data:")
    print(f"  Points returned: {result['count']}")
    if result['points']:
        print(f"  Sample point: {result['points'][0]}")
    print()


def test_model_info():
    """Test model info endpoint."""
    response = requests.get(f"{API_BASE}/model/info")
    result = response.json()
    
    print("Model Info:")
    print(f"  Target: {result['target']}")
    print(f"  Number of classes: {result['num_classes']}")
    print(f"  Top 5 Features:")
    for feat in result['top_features'][:5]:
        print(f"    - {feat['feature']}: {feat['importance']:.4f}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Parking Prediction API")
    print("=" * 60)
    print()
    
    try:
        test_health()
        test_model_info()
        test_violation_stats()
        test_location_hotspots()
        test_predict()
        test_heatmap_data()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API.")
        print("Make sure the API is running: python api/app.py")
    except Exception as e:
        print(f"Error: {e}")

