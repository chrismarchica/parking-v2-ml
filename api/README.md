# Parking Prediction API

Flask REST API for querying the trained XGBoost model and parking ticket data.

## Setup

```bash
pip install -r requirements.txt
```

## Running the API

```bash
# Load the latest trained model automatically
python api/app.py

# Or specify a specific model
python api/app.py --model model/20241230_143000

# Run on different port
python api/app.py --port 8080

# Enable debug mode
python api/app.py --debug
```

The API will run on `http://localhost:5000` by default.

## API Endpoints

### Health Check
```bash
GET /health
```

### Get Parking Violation Prediction
```bash
POST /predict
Content-Type: application/json

{
  "latitude": 40.7580,
  "longitude": -73.9855,
  "datetime": "2024-01-15T14:30:00",
  "precinct": "19",
  "county": "NY"
}
```

**Response:**
```json
{
  "predicted_violation": "21",
  "confidence": 0.75,
  "top_predictions": [
    {"violation_code": "21", "probability": 0.75},
    {"violation_code": "38", "probability": 0.12},
    {"violation_code": "14", "probability": 0.08}
  ],
  "input": {...}
}
```

### Get Violation Statistics
```bash
GET /violations/stats
```

Returns top 50 violation codes with counts and statistics.

### Get Location Hotspots
```bash
GET /locations/hotspots
```

Returns precincts/counties with highest ticket counts.

### Get Temporal Distribution
```bash
GET /temporal/distribution
```

Returns ticket counts by hour and day of week.

### Get Heatmap Data
```bash
GET /heatmap/data?start_date=2024-01-01&end_date=2024-02-01&limit=5000
```

**Query Parameters:**
- `start_date`: Filter from date (YYYY-MM-DD)
- `end_date`: Filter to date (YYYY-MM-DD)
- `violation_code`: Filter by violation code
- `limit`: Max results (default 1000)

**Response:**
```json
{
  "points": [
    {"latitude": 40.758, "longitude": -73.985, "violation_code": "21"},
    ...
  ],
  "count": 1000,
  "filters": {...}
}
```

### Get Model Information
```bash
GET /model/info
```

Returns model metadata and top features.

## Frontend Integration

### Example: Get prediction for a location

```javascript
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    latitude: 40.7580,
    longitude: -73.9855,
    datetime: new Date().toISOString(),
    precinct: "19",
    county: "NY"
  })
});

const prediction = await response.json();
console.log(`Most likely violation: ${prediction.predicted_violation}`);
console.log(`Confidence: ${(prediction.confidence * 100).toFixed(1)}%`);
```

### Example: Get heatmap data

```javascript
const response = await fetch(
  'http://localhost:5000/heatmap/data?limit=5000&start_date=2024-01-01'
);
const data = await response.json();

// data.points = [{latitude, longitude, violation_code}, ...]
// Plot on map using Leaflet, Mapbox, Google Maps, etc.
```

## CORS

CORS is enabled for all origins. For production, restrict in `app.py`:

```python
CORS(app, origins=["https://your-frontend.com"])
```

