"""
Feature engineering for the parking-ticket RISK model.

Deliberately self-contained and dependency-free (no database, no other pipeline)
so the EXACT same transformation runs at training time and at inference time in
the API. Given a location (lat/lon) and a moment (day-of-week + hour) it produces
the numeric feature matrix the XGBoost risk model expects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered feature columns the model is trained on. Inference MUST build the
# matrix in this exact order.
FEATURE_COLUMNS = [
    "lat",
    "lon",
    "hour",
    "dow",          # 0 = Sunday .. 6 = Saturday (matches Postgres EXTRACT(DOW))
    "is_weekend",
    "is_rush_hour",
    "is_morning",   # street-cleaning tickets cluster on weekday mornings
]


def parse_violation_hour(raw: object) -> float:
    """Parse an NYC violation_time string like '0932A' / '0415P' to hour 0-23."""
    if not isinstance(raw, str):
        return np.nan
    t = raw.strip().upper()
    if len(t) < 4:
        return np.nan
    try:
        hour = int(t[:2])
    except ValueError:
        return np.nan
    if t.endswith("P") and hour != 12:
        hour += 12
    elif t.endswith("A") and hour == 12:
        hour = 0
    return hour if 0 <= hour <= 23 else np.nan


def dow_sunday0(ts: pd.Timestamp) -> int:
    """Day of week with Sunday=0..Saturday=6 (matches Postgres EXTRACT(DOW))."""
    # pandas/py: Monday=0..Sunday=6 -> shift so Sunday=0
    return (int(ts.weekday()) + 1) % 7


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the model feature matrix.

    Args:
        df: DataFrame with columns: lat, lon, hour (0-23), dow (0-6, Sunday=0).

    Returns:
        DataFrame with exactly FEATURE_COLUMNS, in order.
    """
    out = pd.DataFrame(index=df.index)
    out["lat"] = df["lat"].astype(float)
    out["lon"] = df["lon"].astype(float)
    out["hour"] = df["hour"].astype(int)
    out["dow"] = df["dow"].astype(int)
    out["is_weekend"] = out["dow"].isin([0, 6]).astype(int)
    out["is_rush_hour"] = (
        ((out["hour"] >= 7) & (out["hour"] <= 9))
        | ((out["hour"] >= 16) & (out["hour"] <= 19))
    ).astype(int)
    out["is_morning"] = ((out["hour"] >= 8) & (out["hour"] <= 11)).astype(int)
    return out[FEATURE_COLUMNS]


def features_for_point(lat: float, lon: float, hour: int, dow: int) -> np.ndarray:
    """Build a single-row feature matrix for inference. Returns shape (1, n)."""
    df = pd.DataFrame([{"lat": lat, "lon": lon, "hour": hour, "dow": dow}])
    return build_features(df).values
