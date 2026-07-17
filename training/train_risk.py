"""
Train the parking-ticket RISK model.

Frames "how likely am I to get a parking ticket here, at this day/time?" as a
presence/background problem, because the data contains only issued tickets (no
"parked and didn't get a ticket" records):

  * PRESENCE (label 1): real geocoded tickets -> (location, hour, day-of-week).
    Busy cells contribute many presence points, so risk concentrates where
    tickets actually happen.
  * BACKGROUND (label 0): points sampled uniformly across the populated NYC
    cells at random times. Quiet cells/times are mostly background -> low risk.

XGBoost then estimates P(presence) which we surface as a 0-100% risk score.
This is an honest RELATIVE risk index, not a calibrated probability.

Usage:
  PGHOST=localhost PGPORT=5434 PGUSER=postgres PGPASSWORD=postgres \
  PGDATABASE=parking_predictor venv/bin/python training/train_risk.py
"""

from __future__ import annotations

import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.risk_features import build_features, parse_violation_hour, FEATURE_COLUMNS  # noqa: E402

GRID = 200.0  # 1/0.005deg ~ 500m cells
RANDOM_STATE = 42


def load_presence(conn) -> pd.DataFrame:
    """Load geocoded tickets as presence points with parsed hour and dow."""
    sql = """
        SELECT ST_Y(geom::geometry) AS lat,
               ST_X(geom::geometry) AS lon,
               EXTRACT(DOW FROM issue_date)::int AS dow,
               violation_time
        FROM parking_ticket
        WHERE geom IS NOT NULL AND issue_date IS NOT NULL
    """
    df = pd.read_sql(sql, conn)
    df["hour"] = df["violation_time"].apply(parse_violation_hour)
    df = df.dropna(subset=["lat", "lon", "hour", "dow"])
    df["hour"] = df["hour"].astype(int)
    df["dow"] = df["dow"].astype(int)
    return df[["lat", "lon", "hour", "dow"]].reset_index(drop=True)


def make_background(presence: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Sample background points uniformly over populated cells x random time.

    Sampling per distinct cell (not per ticket) keeps background spatially even,
    so genuinely busy cells stand out as high-risk against it.
    """
    cell = pd.DataFrame({
        "lat_cell": (presence["lat"] * GRID).round() / GRID,
        "lon_cell": (presence["lon"] * GRID).round() / GRID,
    })
    cells = cell.drop_duplicates().reset_index(drop=True)

    idx = rng.integers(0, len(cells), size=n)
    jitter = (rng.random(size=(n, 2)) - 0.5) / GRID  # spread within the cell
    lat = cells["lat_cell"].values[idx] + jitter[:, 0]
    lon = cells["lon_cell"].values[idx] + jitter[:, 1]
    hour = rng.integers(0, 24, size=n)
    dow = rng.integers(0, 7, size=n)
    return pd.DataFrame({"lat": lat, "lon": lon, "hour": hour, "dow": dow})


def main() -> None:
    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", "5434"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
        dbname=os.environ.get("PGDATABASE", "parking_predictor"),
    )
    print("[1/5] Loading presence (ticket) points...")
    presence = load_presence(conn)
    conn.close()
    print(f"      {len(presence):,} presence points, {presence[['lat','lon']].round(3).drop_duplicates().shape[0]:,} distinct ~100m spots")

    print("[2/5] Sampling background points...")
    rng = np.random.default_rng(RANDOM_STATE)
    background = make_background(presence, n=len(presence), rng=rng)

    print("[3/5] Building features...")
    X_pres = build_features(presence)
    X_bg = build_features(background)
    X = pd.concat([X_pres, X_bg], ignore_index=True)
    y = np.concatenate([np.ones(len(X_pres)), np.zeros(len(X_bg))]).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("[4/5] Training XGBoost (binary:logistic)...")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=7,
        learning_rate=0.1,
        n_estimators=300,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print("[5/5] Evaluating...")
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    print(f"      ROC-AUC: {auc:.4f}   PR-AUC: {ap:.4f}")
    print("      Feature importances:")
    for f, imp in sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda t: -t[1]):
        print(f"        {f:14s} {imp:.4f}")

    out_dir = Path(__file__).parent.parent / "model" / f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target": "is_ticket",
            "label_encoders": {},
            "kind": "risk",
        }, f)
    print(f"\nSaved risk model -> {out_dir}")


if __name__ == "__main__":
    main()
