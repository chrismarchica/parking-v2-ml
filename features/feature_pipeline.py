"""Feature engineering pipeline for parking ticket prediction."""

from typing import Optional
import pandas as pd
import numpy as np

from data import ParkingDataLoader


class FeaturePipeline:
    """Transforms raw parking data into ML-ready features."""

    # County/borough code mapping
    BOROUGH_MAP = {
        "NY": "Manhattan",
        "MAN": "Manhattan",
        "MH": "Manhattan",
        "MN": "Manhattan",
        "NEWY": "Manhattan",
        "NEW Y": "Manhattan",
        "K": "Brooklyn",
        "BK": "Brooklyn",
        "KINGS": "Brooklyn",
        "Q": "Queens",
        "QN": "Queens",
        "QNS": "Queens",
        "QUEEN": "Queens",
        "BX": "Bronx",
        "BRONX": "Bronx",
        "R": "Staten Island",
        "ST": "Staten Island",
        "RICH": "Staten Island",
    }

    def __init__(self, data_loader: Optional[ParkingDataLoader] = None):
        """
        Initialize the feature pipeline.

        Args:
            data_loader: ParkingDataLoader instance. Creates new one if None.
        """
        self.data_loader = data_loader or ParkingDataLoader()

    def load_and_transform(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sample_frac: Optional[float] = None,
        include_coordinates: bool = False,
    ) -> pd.DataFrame:
        """
        Load data and apply all feature transformations.

        Args:
            start_date: Filter violations on or after this date.
            end_date: Filter violations before this date.
            sample_frac: Random sample fraction for development.
            include_coordinates: Whether to include lat/lon from geom.

        Returns:
            DataFrame with engineered features.
        """
        df = self.data_loader.load_training_data(
            start_date=start_date,
            end_date=end_date,
            sample_frac=sample_frac,
            include_coordinates=include_coordinates,
        )

        df = self.add_temporal_features(df)
        df = self.add_location_features(df, has_coordinates=include_coordinates)
        df = self.add_violation_features(df)
        df = self.clean_and_encode(df)

        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from issue_date and violation_time."""
        df = df.copy()

        # Parse issue_date
        df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")

        # Date-based features
        df["year"] = df["issue_date"].dt.year
        df["month"] = df["issue_date"].dt.month
        df["day_of_week"] = df["issue_date"].dt.dayofweek
        df["day_of_month"] = df["issue_date"].dt.day
        df["week_of_year"] = df["issue_date"].dt.isocalendar().week.astype(int)

        # Is weekend
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Parse violation_time (format: HHMMA or HHMMP, e.g. '0932A')
        df["hour"] = self._parse_violation_time(df["violation_time"])

        # Time of day buckets
        df["time_of_day"] = pd.cut(
            df["hour"],
            bins=[-1, 6, 12, 17, 21, 24],
            labels=["night", "morning", "afternoon", "evening", "late_night"],
        )

        # Rush hour indicator (weekdays 7-9am, 4-7pm)
        df["is_rush_hour"] = (
            (df["is_weekend"] == 0)
            & (
                ((df["hour"] >= 7) & (df["hour"] <= 9))
                | ((df["hour"] >= 16) & (df["hour"] <= 19))
            )
        ).astype(int)

        # Alternate side parking days (Mon/Thu or Tue/Fri depending on street)
        df["is_asp_day"] = df["day_of_week"].isin([0, 1, 3, 4]).astype(int)

        return df

    def _parse_violation_time(self, time_series: pd.Series) -> pd.Series:
        """Parse violation time strings to hour of day (0-23)."""

        def parse_time(t):
            if pd.isna(t) or not isinstance(t, str):
                return np.nan
            t = t.strip().upper()
            if len(t) < 4:
                return np.nan

            try:
                hour = int(t[:2])
                is_pm = t.endswith("P")
                is_am = t.endswith("A")

                if is_pm and hour != 12:
                    hour += 12
                elif is_am and hour == 12:
                    hour = 0

                return hour if 0 <= hour <= 23 else np.nan
            except (ValueError, IndexError):
                return np.nan

        return time_series.apply(parse_time)

    def add_location_features(
        self, df: pd.DataFrame, has_coordinates: bool = False
    ) -> pd.DataFrame:
        """Add location-based features."""
        df = df.copy()

        # Normalize borough names from county
        df["borough"] = (
            df["county"].str.upper().str.strip().map(self.BOROUGH_MAP).fillna("Unknown")
        )

        # Clean precinct
        df["precinct_clean"] = df["precinct"].astype(str).str.strip()

        # Street name features
        df["street_name_clean"] = df["street_name"].str.upper().str.strip()

        # Common street types
        df["is_avenue"] = df["street_name_clean"].str.contains(
            r"\bAVE\b|\bAVENUE\b", na=False, regex=True
        ).astype(int)
        df["is_street"] = df["street_name_clean"].str.contains(
            r"\bST\b|\bSTREET\b", na=False, regex=True
        ).astype(int)
        df["is_broadway"] = df["street_name_clean"].str.contains(
            r"\bBROADWAY\b", na=False, regex=True
        ).astype(int)

        if has_coordinates:
            # Discretize lat/lon into grid cells (roughly 0.005 deg ≈ 500m)
            df["lat_grid"] = (df["latitude"] * 200).round() / 200
            df["lon_grid"] = (df["longitude"] * 200).round() / 200
            df["location_cell"] = (
                df["lat_grid"].astype(str) + "_" + df["lon_grid"].astype(str)
            )

        return df

    def add_violation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add violation-related features."""
        df = df.copy()

        # Clean violation code
        df["violation_code_clean"] = df["violation_code"].astype(str).str.strip()

        # Issuing agency cleaned
        df["agency"] = df["issuing_agency"].str.upper().str.strip().fillna("UNKNOWN")

        # Fine amount features
        df["has_fine"] = df["fine_amount"].notna().astype(int)
        df["fine_amount_clean"] = df["fine_amount"].fillna(0)

        # Plate type grouping (common types vs other)
        common_plate_types = ["PAS", "COM", "OMT", "OMS", "SRF", "MED", "PHS"]
        df["plate_type_clean"] = df["plate_type"].str.upper().str.strip()
        df["plate_type_grouped"] = df["plate_type_clean"].where(
            df["plate_type_clean"].isin(common_plate_types), "OTHER"
        )

        return df

    def clean_and_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and prepare for encoding."""
        df = df.copy()

        # Drop rows with missing critical features
        critical_cols = ["issue_date", "violation_code"]
        df = df.dropna(subset=critical_cols)

        # Violation code as string for categorical encoding
        df["violation_code"] = df["violation_code"].astype(str)

        # Fill missing values for non-critical features
        df["hour"] = df["hour"].fillna(12)  # Default to noon
        df["precinct_clean"] = df["precinct_clean"].fillna("UNKNOWN")
        df["agency"] = df["agency"].fillna("UNKNOWN")

        return df

    def get_feature_columns(self) -> dict:
        """Return column groupings for model training."""
        return {
            "numeric": [
                "hour",
                "day_of_week",
                "day_of_month",
                "month",
                "week_of_year",
                "year",
                "is_weekend",
                "is_rush_hour",
                "is_asp_day",
                "is_avenue",
                "is_street",
                "is_broadway",
                "fine_amount_clean",
            ],
            "categorical": [
                "borough",
                "precinct_clean",
                "time_of_day",
                "agency",
                "plate_type_grouped",
            ],
            "target_candidates": [
                "violation_code",  # Multi-class: predict violation type
                "borough",  # Multi-class: predict borough
                "fine_amount_clean",  # Regression: predict fine
            ],
            "id_columns": [
                "summons_number",
            ],
            "drop_columns": [
                "issue_date",
                "violation_time",
                "violation_desc",
                "issuing_agency",
                "county",
                "precinct",
                "street_name",
                "street_name_clean",
                "fine_amount",
                "plate_type",
                "plate_type_clean",
                "violation_code_clean",
                "has_fine",
            ],
        }

    def prepare_for_training(
        self, df: pd.DataFrame, target: str = "violation_code"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final feature matrix and target vector.

        Args:
            df: DataFrame with engineered features.
            target: Name of target column.

        Returns:
            Tuple of (feature DataFrame, target Series).
        """
        feature_cols = self.get_feature_columns()

        # Get all features except target
        X_cols = feature_cols["numeric"] + feature_cols["categorical"]
        X_cols = [c for c in X_cols if c != target and c in df.columns]

        X = df[X_cols].copy()
        y = df[target].copy()

        return X, y
