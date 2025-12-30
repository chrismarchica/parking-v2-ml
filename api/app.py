"""Flask API for NYC Parking Ticket Predictions."""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_xgb import ParkingTicketModel
from features import FeaturePipeline
from data import ParkingDataLoader

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for model and data loader
model: ParkingTicketModel = None
pipeline: FeaturePipeline = None
data_loader: ParkingDataLoader = None


def load_model(model_path: str = None):
    """Load the trained model."""
    global model, pipeline, data_loader
    
    if model_path is None:
        # Find the latest model
        model_dir = Path(__file__).parent.parent / "model"
        if not model_dir.exists():
            raise FileNotFoundError("No models found. Train a model first.")
        
        model_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            raise FileNotFoundError("No trained models found.")
        
        model_path = max(model_dirs, key=lambda p: p.name)
    
    print(f"Loading model from: {model_path}")
    model = ParkingTicketModel.load(model_path)
    pipeline = FeaturePipeline()
    data_loader = ParkingDataLoader()
    print("Model loaded successfully!")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict parking violation likelihood.
    
    Request body:
    {
        "latitude": 40.7580,
        "longitude": -73.9855,
        "datetime": "2024-01-15T14:30:00",  # optional, defaults to now
        "precinct": "19",  # optional
        "county": "NY"     # optional
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        
        # Parse datetime
        if "datetime" in data:
            dt = pd.to_datetime(data["datetime"])
        else:
            dt = pd.Timestamp.now()
        
        # Build feature dict
        features = {
            "issue_date": dt,
            "violation_time": dt.strftime("%I%M%p").lstrip("0"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "precinct": data.get("precinct", "UNKNOWN"),
            "county": data.get("county", "NY"),
            "street_name": data.get("street_name", ""),
            "issuing_agency": data.get("agency", "P"),
            "plate_type": data.get("plate_type", "PAS"),
            "fine_amount": 0,  # Not used for prediction
        }
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Apply feature transformations
        df = pipeline.add_temporal_features(df)
        df = pipeline.add_location_features(df, has_coordinates=True)
        df = pipeline.add_violation_features(df)
        df = pipeline.clean_and_encode(df)
        
        # Prepare features for model
        X, _ = pipeline.prepare_for_training(df, target="violation_code")
        X_encoded = model.prepare_features(X)
        
        # Get predictions
        pred_class = model.predict(X_encoded)[0]
        pred_proba = model.predict_proba(X_encoded)[0]
        
        # Decode prediction
        label_encoder = model.label_encoders["violation_code"]
        predicted_code = label_encoder.inverse_transform([pred_class])[0]
        
        # Get top 5 predictions
        top_5_idx = np.argsort(pred_proba)[-5:][::-1]
        top_5_predictions = [
            {
                "violation_code": label_encoder.inverse_transform([idx])[0],
                "probability": float(pred_proba[idx])
            }
            for idx in top_5_idx
        ]
        
        return jsonify({
            "predicted_violation": predicted_code,
            "confidence": float(pred_proba[pred_class]),
            "top_predictions": top_5_predictions,
            "input": {
                "latitude": features["latitude"],
                "longitude": features["longitude"],
                "datetime": dt.isoformat(),
                "precinct": features["precinct"],
                "county": features["county"]
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/violations/stats', methods=['GET'])
def violation_stats():
    """Get violation statistics."""
    if data_loader is None:
        return jsonify({"error": "Data loader not initialized"}), 500
    
    try:
        stats = data_loader.get_violation_stats()
        
        # Convert to JSON-friendly format
        result = stats.head(50).to_dict(orient='records')
        
        return jsonify({
            "violations": result,
            "total_count": len(stats)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/locations/hotspots', methods=['GET'])
def location_hotspots():
    """Get high-risk parking locations."""
    if data_loader is None:
        return jsonify({"error": "Data loader not initialized"}), 500
    
    try:
        stats = data_loader.get_location_stats()
        
        # Get top 50 locations
        result = stats.head(50).to_dict(orient='records')
        
        return jsonify({
            "hotspots": result,
            "total_locations": len(stats)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/temporal/distribution', methods=['GET'])
def temporal_distribution():
    """Get ticket distribution by time."""
    if data_loader is None:
        return jsonify({"error": "Data loader not initialized"}), 500
    
    try:
        dist = data_loader.get_temporal_distribution()
        
        # Pivot for easier frontend consumption
        result = dist.to_dict(orient='records')
        
        return jsonify({
            "distribution": result,
            "total_records": len(dist)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/heatmap/data', methods=['GET'])
def heatmap_data():
    """
    Get aggregated location data for heatmap visualization.
    
    Query params:
    - start_date: YYYY-MM-DD (optional)
    - end_date: YYYY-MM-DD (optional)
    - violation_code: filter by code (optional)
    - limit: max results (default 1000)
    """
    if data_loader is None:
        return jsonify({"error": "Data loader not initialized"}), 500
    
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        violation_code = request.args.get('violation_code')
        limit = int(request.args.get('limit', 1000))
        
        # Build WHERE clause
        conditions = []
        params = []
        
        if start_date:
            conditions.append("issue_date >= %s")
            params.append(start_date)
        
        if end_date:
            conditions.append("issue_date < %s")
            params.append(end_date)
        
        if violation_code:
            conditions.append("violation_code = %s")
            params.append(violation_code)
        
        where_clause = " AND ".join(conditions) if conditions else None
        
        # Query with coordinates
        df = data_loader.load_data_with_coordinates(
            columns=["summons_number", "issue_date", "violation_code", "precinct"],
            limit=limit,
            where_clause=where_clause,
            params=tuple(params) if params else None
        )
        
        # Format for frontend
        points = df[['latitude', 'longitude', 'violation_code']].to_dict(orient='records')
        
        return jsonify({
            "points": points,
            "count": len(points),
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "violation_code": violation_code
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        feature_importance = model.get_feature_importance().head(15)
        
        return jsonify({
            "target": model.target,
            "features": model.feature_columns,
            "num_classes": len(model.label_encoders.get(model.target, {}).classes_) if model.target in model.label_encoders else None,
            "top_features": feature_importance.to_dict(orient='records')
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the parking prediction API")
    parser.add_argument('--model', type=str, help='Path to model directory')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Load model
    try:
        load_model(args.model)
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("API will start but predictions will not be available.")
    
    # Run app
    app.run(host=args.host, port=args.port, debug=args.debug)

