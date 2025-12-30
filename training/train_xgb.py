"""XGBoost training script for parking ticket prediction."""

import sys
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features import FeaturePipeline


class ParkingTicketModel:
    """XGBoost classifier for parking ticket prediction."""

    def __init__(self, target: str = "violation_code"):
        """
        Initialize the model trainer.

        Args:
            target: Target variable to predict.
        """
        self.target = target
        self.model: xgb.XGBClassifier = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_columns: list[str] = []

    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Encode categorical features and prepare feature matrix.

        Args:
            X: Feature DataFrame.

        Returns:
            Numpy array ready for XGBoost.
        """
        X_encoded = X.copy()

        # Identify categorical columns
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in cat_cols:
            # Convert to string first (handles categorical dtypes)
            X_encoded[col] = X_encoded[col].astype(str).replace("nan", "UNKNOWN")
            
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                # Fit on data, ensuring UNKNOWN is always included
                unique_vals = X_encoded[col].unique().tolist()
                if "UNKNOWN" not in unique_vals:
                    unique_vals.append("UNKNOWN")
                self.label_encoders[col].fit(unique_vals)
            
            # Handle unseen labels by mapping to UNKNOWN
            known_labels = set(self.label_encoders[col].classes_)
            X_encoded[col] = X_encoded[col].apply(
                lambda x: x if x in known_labels else "UNKNOWN"
            )
            X_encoded[col] = self.label_encoders[col].transform(X_encoded[col])

        self.feature_columns = X_encoded.columns.tolist()
        return X_encoded.values

    def encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target variable."""
        if self.target not in self.label_encoders:
            self.label_encoders[self.target] = LabelEncoder()
            self.label_encoders[self.target].fit(y.astype(str))

        return self.label_encoders[self.target].transform(y.astype(str))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        params: dict = None,
    ) -> None:
        """
        Train the XGBoost model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            params: XGBoost parameters.
        """
        default_params = {
            "objective": "multi:softmax",
            "num_class": len(np.unique(y_train)),
            "max_depth": 8,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 1,
        }

        if params:
            default_params.update(params)

        self.model = xgb.XGBClassifier(**default_params)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=True,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def save(self, path: str) -> None:
        """Save model and encoders to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save entire model with pickle (simpler and more reliable)
        model_data = {
            "model": self.model,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
            "target": self.target,
        }
        
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, path: str) -> "ParkingTicketModel":
        """Load model from disk."""
        path = Path(path)

        # Load pickled model
        with open(path / "model.pkl", "rb") as f:
            model_data = pickle.load(f)

        instance = cls(target=model_data["target"])
        instance.model = model_data["model"]
        instance.label_encoders = model_data["label_encoders"]
        instance.feature_columns = model_data["feature_columns"]

        return instance


def run_training(
    start_date: str = None,
    end_date: str = None,
    sample_frac: float = None,
    output_dir: str = None,
    include_coordinates: bool = False,
):
    """
    Run the full training pipeline.

    Args:
        start_date: Filter data from this date.
        end_date: Filter data until this date.
        sample_frac: Sample fraction for development.
        output_dir: Directory to save model.
        include_coordinates: Include lat/lon from geom column.
    """
    print("=" * 60)
    print("NYC Parking Ticket Prediction - Training Pipeline")
    print("=" * 60)

    # Initialize pipeline
    pipeline = FeaturePipeline()

    # Load and transform data
    print("\n[1/5] Loading and transforming data...")
    df = pipeline.load_and_transform(
        start_date=start_date,
        end_date=end_date,
        sample_frac=sample_frac,
        include_coordinates=include_coordinates,
    )
    print(f"      Loaded {len(df):,} records")

    # Prepare features
    print("\n[2/5] Preparing features...")
    X, y = pipeline.prepare_for_training(df, target="violation_code")
    print(f"      Features: {X.shape[1]}, Samples: {X.shape[0]:,}")
    print(f"      Target classes (before filtering): {y.nunique()}")

    # Filter out rare classes (need at least 10 samples for stratified split)
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]
    
    removed = len(mask) - mask.sum()
    if removed > 0:
        print(f"      Removed {removed:,} samples from rare classes (< 10 samples)")
    print(f"      Final samples: {X.shape[0]:,}, Target classes: {y.nunique()}")

    # Split data
    print("\n[3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"      Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Initialize and train model
    print("\n[4/5] Training XGBoost model...")
    model = ParkingTicketModel(target="violation_code")

    X_train_enc = model.prepare_features(X_train)
    X_val_enc = model.prepare_features(X_val)
    X_test_enc = model.prepare_features(X_test)

    y_train_enc = model.encode_target(y_train)
    y_val_enc = model.encode_target(y_val)
    y_test_enc = model.encode_target(y_test)

    model.train(X_train_enc, y_train_enc, X_val_enc, y_val_enc)

    # Evaluate
    print("\n[5/5] Evaluating model...")
    y_pred = model.predict(X_test_enc)
    accuracy = accuracy_score(y_test_enc, y_pred)
    print(f"\n      Test Accuracy: {accuracy:.4f}")

    # Feature importance
    print("\n      Top 10 Features:")
    importance = model.get_feature_importance()
    for _, row in importance.head(10).iterrows():
        print(f"        - {row['feature']}: {row['importance']:.4f}")

    # Save model
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "model" / datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n      Saving model to {output_dir}")
    model.save(output_dir)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train parking ticket prediction model")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--sample", type=float, help="Sample fraction (0-1)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--with-coords", action="store_true", help="Include coordinates from geom")

    args = parser.parse_args()

    run_training(
        start_date=args.start_date,
        end_date=args.end_date,
        sample_frac=args.sample,
        output_dir=args.output,
        include_coordinates=args.with_coords,
    )
