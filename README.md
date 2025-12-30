# NYC Parking Ticket Prediction

XGBoost ML model for predicting parking ticket likelihood in NYC based on public violation data.

## Setup

1. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env` file with database credentials:
```
DB_HOST=localhost
DB_PORT=5433
DB_NAME=parking
DB_USER=your_username
DB_PASSWORD=your_password
```

For local development, only `DB_USER` and `DB_PASSWORD` are required—the others default to `localhost:5433/parking`.

## Database

The project expects a `parking_ticket` table:

```sql
CREATE TABLE parking_ticket (
  summons_number        text PRIMARY KEY,
  source_dataset        text NOT NULL,
  issue_date            date,
  violation_time        text,                    -- e.g. '0932A'
  violation_code        text,
  violation_desc        text,
  issuing_agency        text,
  county                text,
  precinct              text,
  street_name           text,
  intersecting_street   text,
  geom                  geography(Point, 4326),  -- PostGIS
  fine_amount           numeric,
  plate_id              text,
  registration_state    text,
  plate_type            text,
  -- Socrata fields for incremental ingestion
  soda_row_id           text,
  soda_created_at       timestamptz,
  soda_updated_at       timestamptz,
  ingested_at           timestamptz NOT NULL DEFAULT now()
);
```

## Usage

### Test Database Connection
```python
from data import DatabaseConnection

db = DatabaseConnection()
print(db.test_connection())  # Should print True
```

### Load Data
```python
from data import ParkingDataLoader

loader = ParkingDataLoader()

# Basic stats
print(f"Total rows: {loader.get_row_count()}")
print(f"Date range: {loader.get_date_range()}")

# Load sample for training
df = loader.load_training_data(start_date="2023-01-01", sample_frac=0.1)

# Load with coordinates extracted from PostGIS geom
df = loader.load_training_data(sample_frac=0.1, include_coordinates=True)
```

### Train Model
```bash
python training/train_xgb.py --start-date 2023-01-01 --sample 0.1

# With coordinates from geom column
python training/train_xgb.py --sample 0.1 --with-coords
```

### Evaluate Model
```bash
python training/evaluate.py model/20231215_143000 --sample 0.1
```

## Project Structure

```
├── api/                  # API for serving predictions
├── config/
│   └── db.yaml          # Database configuration
├── data/
│   ├── db_connection.py # PostgreSQL connection manager
│   └── data_loader.py   # Data loading utilities
├── features/
│   ├── feature_pipeline.py  # Feature engineering
│   └── build_features.sql   # SQL for materialized views
├── model/               # Saved models
├── notebooks/           # Jupyter notebooks
└── training/
    ├── train_xgb.py     # Training script
    └── evaluate.py      # Evaluation utilities
```

## Features

The pipeline extracts these features for training:

**Temporal:**
- Hour, day of week, month, year
- Is weekend, is rush hour
- Alternate side parking day indicator

**Location:**
- Borough (normalized from county codes)
- Precinct
- Street type (avenue, street, broadway)
- Lat/lon grid cells (if geom available)

**Violation:**
- Violation code
- Issuing agency
- Fine amount
- Plate type (grouped)
