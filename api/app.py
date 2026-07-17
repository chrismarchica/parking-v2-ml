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
from features.risk_features import build_features, dow_sunday0
from data import ParkingDataLoader

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for model and data loader
model: ParkingTicketModel = None
pipeline: FeaturePipeline = None
data_loader: ParkingDataLoader = None


def load_model(model_path: str = None):
    """Load the trained model."""
    global model
    
    if model_path is None:
        # Find the latest model
        model_dir = Path(__file__).parent.parent / "model"
        if not model_dir.exists():
            raise FileNotFoundError("No models found. Train a model first.")
        
        model_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            raise FileNotFoundError("No trained models found.")

        # Prefer the risk model (dirs named risk_*); fall back to newest overall.
        risk_dirs = [d for d in model_dirs if d.name.startswith("risk_")]
        candidates = risk_dirs or model_dirs
        model_path = max(candidates, key=lambda p: p.name)
    
    print(f"Loading model from: {model_path}")
    model = ParkingTicketModel.load(model_path)
    print("Model loaded successfully!")


def init_data_loader():
    """Initialize data loader and pipeline (independent of model)."""
    global pipeline, data_loader
    
    if pipeline is None:
        pipeline = FeaturePipeline()
        print("Feature pipeline initialized!")
    
    if data_loader is None:
        data_loader = ParkingDataLoader()
        print("Data loader initialized!")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat()
    })


def _risk_level(score: float) -> str:
    if score >= 75:
        return "Very High"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Moderate"
    return "Low"


def _risk_factors(hour: int, dow: int, score: float) -> list:
    """Human-readable drivers behind a risk score."""
    factors = []
    if 8 <= hour <= 11 and dow not in (0, 6):
        factors.append("Weekday morning — prime street-cleaning hours")
    if (7 <= hour <= 9) or (16 <= hour <= 19):
        factors.append("Rush-hour enforcement")
    if dow in (0, 6):
        factors.append("Weekend — lighter enforcement")
    if hour <= 5 or hour >= 22:
        factors.append("Overnight — very little enforcement")
    if score >= 60:
        factors.append("High-activity ticketing area")
    elif score <= 20:
        factors.append("Low-activity area for this time")
    return factors


@app.route('/risk', methods=['POST'])
def risk():
    """
    Estimate parking-ticket RISK for a location and time.

    Request body:
    {
        "latitude": 40.7580,
        "longitude": -73.9855,
        "datetime": "2024-06-11T09:00:00"  # optional, defaults to now
    }

    Returns a 0-100 relative risk score for the requested moment, plus the
    risk across all 24 hours of that day for the same location.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json or {}
        lat = float(data["latitude"])
        lon = float(data["longitude"])
        dt = pd.to_datetime(data["datetime"]) if data.get("datetime") else pd.Timestamp.now()
        dow = dow_sunday0(dt)

        # Score all 24 hours of this day at this location in one batch.
        hours_df = pd.DataFrame({
            "lat": [lat] * 24,
            "lon": [lon] * 24,
            "hour": list(range(24)),
            "dow": [dow] * 24,
        })
        X = build_features(hours_df).values
        proba = model.predict_proba(X)[:, 1]
        hourly = [{"hour": h, "risk": round(float(proba[h]) * 100, 1)} for h in range(24)]

        score = round(float(proba[dt.hour]) * 100, 1)

        return jsonify({
            "risk_score": score,
            "level": _risk_level(score),
            "factors": _risk_factors(dt.hour, dow, score),
            "hourly": hourly,
            "input": {
                "latitude": lat,
                "longitude": lon,
                "datetime": dt.isoformat(),
                "hour": dt.hour,
                "day_of_week": dow,
            },
        })

    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid request: {e}"}), 400
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
        conditions = ["geom IS NOT NULL"]
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
        
        # First, check if we have any data with geom
        with data_loader.db.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM parking_ticket WHERE geom IS NOT NULL")
            geom_count = cursor.fetchone()[0]
            print(f"DEBUG: Found {geom_count} rows with geom data")
            
            cursor.execute("SELECT COUNT(*) FROM parking_ticket")
            total_count = cursor.fetchone()[0]
            print(f"DEBUG: Total rows in table: {total_count}")
        
        # Query with coordinates
        df = data_loader.load_data_with_coordinates(
            columns=["summons_number", "issue_date", "violation_code", "precinct"],
            limit=limit,
            where_clause=where_clause,
            params=tuple(params) if params else None
        )
        
        print(f"DEBUG: Query returned {len(df)} rows")
        
        # Format for frontend
        if len(df) > 0:
            points = df[['latitude', 'longitude', 'violation_code']].to_dict(orient='records')
        else:
            points = []
        
        return jsonify({
            "points": points,
            "count": len(points),
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "violation_code": violation_code
            },
            "debug": {
                "total_rows_in_table": total_count,
                "rows_with_geom": geom_count
            }
        })
        
    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
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


def initialize_app():
    """Initialize the application (load model and data loader on startup)."""
    # Always initialize data loader (needed for heatmap, stats, etc.)
    try:
        init_data_loader()
    except Exception as e:
        print(f"Warning: Could not initialize data loader: {e}")
        print("Data endpoints will not be available.")
    
    # Try to load model (optional - predictions won't work without it)
    model_path = os.environ.get('MODEL_PATH')
    try:
        load_model(model_path)
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("API will start but predictions will not be available.")


# Auto-initialize when imported (for gunicorn with preload_app=True)
# Skip if running directly with __main__
_initialized = False


@app.before_request
def lazy_init():
    """Lazy initialization on first request if not already initialized."""
    global _initialized
    if not _initialized:
        initialize_app()
        _initialized = True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the parking prediction API")
    parser.add_argument('--model', type=str, help='Path to model directory')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)), help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Initialize data loader
    try:
        init_data_loader()
    except Exception as e:
        print(f"Warning: Could not initialize data loader: {e}")
    
    # Load model
    try:
        load_model(args.model)
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("API will start but predictions will not be available.")
    
    _initialized = True
    
    # Run app
    app.run(host=args.host, port=args.port, debug=args.debug)

