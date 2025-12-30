"""Data loader for NYC parking ticket data."""

from typing import Optional, Iterator
import pandas as pd
from psycopg2.extras import RealDictCursor

from .db_connection import DatabaseConnection


class ParkingDataLoader:
    """Loads parking ticket data from PostgreSQL for ML training."""

    # All columns available in parking_ticket table
    ALL_COLUMNS = [
        "summons_number",
        "source_dataset",
        "issue_date",
        "violation_time",
        "violation_code",
        "violation_desc",
        "issuing_agency",
        "county",
        "precinct",
        "street_name",
        "intersecting_street",
        "fine_amount",
        "plate_id",
        "registration_state",
        "plate_type",
        "soda_row_id",
        "soda_created_at",
        "soda_updated_at",
        "ingested_at",
    ]

    # Columns for training (excludes system fields)
    TRAINING_COLUMNS = [
        "summons_number",
        "issue_date",
        "violation_time",
        "violation_code",
        "violation_desc",
        "issuing_agency",
        "county",
        "precinct",
        "street_name",
        "intersecting_street",
        "fine_amount",
        "plate_type",
        "registration_state",
    ]

    def __init__(
        self,
        db_connection: Optional[DatabaseConnection] = None,
        table_name: str = "parking_ticket",
    ):
        """
        Initialize the parking data loader.

        Args:
            db_connection: Database connection manager. Creates new one if None.
            table_name: Name of the parking tickets table.
        """
        self.db = db_connection or DatabaseConnection()
        self.table_name = table_name

    def get_table_info(self) -> pd.DataFrame:
        """Get column information for the parking_ticket table."""
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """
        with self.db.get_cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (self.table_name,))
            results = cursor.fetchall()

        return pd.DataFrame(results)

    def get_row_count(self) -> int:
        """Get total number of rows in the table."""
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        result = self.db.execute_query(query)
        return result[0][0]

    def get_date_range(self) -> tuple[str, str]:
        """Get min and max issue_date in the table."""
        query = f"SELECT MIN(issue_date), MAX(issue_date) FROM {self.table_name}"
        result = self.db.execute_query(query)
        return result[0][0], result[0][1]

    def load_data(
        self,
        columns: Optional[list[str]] = None,
        limit: Optional[int] = None,
        where_clause: Optional[str] = None,
        params: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """
        Load parking ticket data into a DataFrame.

        Args:
            columns: List of columns to select. Uses TRAINING_COLUMNS if None.
            limit: Maximum number of rows to return.
            where_clause: Optional WHERE clause (without 'WHERE' keyword).
            params: Parameters for the WHERE clause.

        Returns:
            DataFrame with parking ticket data.
        """
        cols = columns or self.TRAINING_COLUMNS
        col_str = ", ".join(cols)

        query = f"SELECT {col_str} FROM {self.table_name}"

        if where_clause:
            query += f" WHERE {where_clause}"

        if limit:
            query += f" LIMIT {limit}"

        with self.db.get_cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]

        return pd.DataFrame(results, columns=column_names)

    def load_data_with_coordinates(
        self,
        columns: Optional[list[str]] = None,
        limit: Optional[int] = None,
        where_clause: Optional[str] = None,
        params: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """
        Load data with lat/lon extracted from geom column.

        Args:
            columns: Additional columns to select.
            limit: Maximum number of rows.
            where_clause: Optional WHERE clause.
            params: Parameters for the WHERE clause.

        Returns:
            DataFrame with lat/lon columns from geom.
        """
        cols = columns or self.TRAINING_COLUMNS
        col_str = ", ".join(cols)

        # Extract lat/lon from PostGIS geography
        query = f"""
            SELECT {col_str},
                   ST_Y(geom::geometry) as latitude,
                   ST_X(geom::geometry) as longitude
            FROM {self.table_name}
            WHERE geom IS NOT NULL
        """

        if where_clause:
            query += f" AND {where_clause}"

        if limit:
            query += f" LIMIT {limit}"

        with self.db.get_cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]

        return pd.DataFrame(results, columns=column_names)

    def load_data_chunked(
        self,
        columns: Optional[list[str]] = None,
        chunk_size: int = 50000,
        where_clause: Optional[str] = None,
        params: Optional[tuple] = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Load data in chunks using server-side cursor for memory efficiency.

        Args:
            columns: List of columns to select.
            chunk_size: Number of rows per chunk.
            where_clause: Optional WHERE clause.
            params: Parameters for the WHERE clause.

        Yields:
            DataFrames with chunk_size rows each.
        """
        cols = columns or self.TRAINING_COLUMNS
        col_str = ", ".join(cols)

        query = f"SELECT {col_str} FROM {self.table_name}"

        if where_clause:
            query += f" WHERE {where_clause}"

        with self.db.get_connection() as conn:
            # Use server-side cursor for large datasets
            with conn.cursor(name="parking_data_cursor") as cursor:
                cursor.itersize = chunk_size
                cursor.execute(query, params)

                column_names = [desc[0] for desc in cursor.description]

                while True:
                    rows = cursor.fetchmany(chunk_size)
                    if not rows:
                        break
                    yield pd.DataFrame(rows, columns=column_names)

    def load_training_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        counties: Optional[list[str]] = None,
        sample_frac: Optional[float] = None,
        include_coordinates: bool = False,
    ) -> pd.DataFrame:
        """
        Load data specifically formatted for model training.

        Args:
            start_date: Filter violations on or after this date (YYYY-MM-DD).
            end_date: Filter violations before this date (YYYY-MM-DD).
            counties: List of county codes to include (e.g., ['NY', 'K', 'Q']).
            sample_frac: Random sample fraction (0-1) using TABLESAMPLE.
            include_coordinates: Whether to extract lat/lon from geom.

        Returns:
            DataFrame ready for feature engineering.
        """
        training_cols = [
            "summons_number",
            "issue_date",
            "violation_time",
            "violation_code",
            "violation_desc",
            "issuing_agency",
            "county",
            "precinct",
            "street_name",
            "fine_amount",
            "plate_type",
        ]

        col_str = ", ".join(training_cols)

        # Add coordinate extraction if requested
        if include_coordinates:
            col_str += ", ST_Y(geom::geometry) as latitude, ST_X(geom::geometry) as longitude"

        conditions = []
        params = []

        if start_date:
            conditions.append("issue_date >= %s")
            params.append(start_date)

        if end_date:
            conditions.append("issue_date < %s")
            params.append(end_date)

        if counties:
            placeholders = ", ".join(["%s"] * len(counties))
            conditions.append(f"county IN ({placeholders})")
            params.extend(counties)

        if include_coordinates:
            conditions.append("geom IS NOT NULL")

        # Build query with optional TABLESAMPLE
        if sample_frac:
            pct = sample_frac * 100
            query = f"SELECT {col_str} FROM {self.table_name} TABLESAMPLE BERNOULLI({pct})"
        else:
            query = f"SELECT {col_str} FROM {self.table_name}"

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self.db.get_cursor() as cursor:
            cursor.execute(query, tuple(params) if params else None)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]

        return pd.DataFrame(results, columns=column_names)

    def get_violation_stats(self) -> pd.DataFrame:
        """Get summary statistics by violation code."""
        query = f"""
            SELECT
                violation_code,
                violation_desc,
                COUNT(*) as ticket_count,
                COUNT(DISTINCT precinct) as precinct_count,
                AVG(fine_amount) as avg_fine,
                MIN(issue_date) as first_ticket,
                MAX(issue_date) as last_ticket
            FROM {self.table_name}
            GROUP BY violation_code, violation_desc
            ORDER BY ticket_count DESC;
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]

        return pd.DataFrame(results, columns=column_names)

    def get_location_stats(self) -> pd.DataFrame:
        """Get ticket counts aggregated by precinct/county."""
        query = f"""
            SELECT
                precinct,
                county,
                COUNT(*) as ticket_count,
                COUNT(DISTINCT violation_code) as violation_types,
                AVG(fine_amount) as avg_fine,
                SUM(fine_amount) as total_fines
            FROM {self.table_name}
            GROUP BY precinct, county
            ORDER BY ticket_count DESC;
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]

        return pd.DataFrame(results, columns=column_names)

    def get_temporal_distribution(self) -> pd.DataFrame:
        """Get ticket distribution by hour of day and day of week."""
        query = f"""
            SELECT
                EXTRACT(DOW FROM issue_date)::int as day_of_week,
                CASE
                    WHEN violation_time ~ '^[0-9]{{4}}[AP]$' THEN
                        CASE
                            WHEN RIGHT(violation_time, 1) = 'A' THEN
                                CASE
                                    WHEN LEFT(violation_time, 2)::int = 12 THEN 0
                                    ELSE LEFT(violation_time, 2)::int
                                END
                            ELSE
                                CASE
                                    WHEN LEFT(violation_time, 2)::int = 12 THEN 12
                                    ELSE LEFT(violation_time, 2)::int + 12
                                END
                        END
                    ELSE NULL
                END as hour_of_day,
                COUNT(*) as ticket_count
            FROM {self.table_name}
            WHERE violation_time IS NOT NULL
            GROUP BY day_of_week, hour_of_day
            ORDER BY day_of_week, hour_of_day;
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]

        return pd.DataFrame(results, columns=column_names)

    def get_fine_distribution(self) -> pd.DataFrame:
        """Get fine amount distribution by violation code."""
        query = f"""
            SELECT
                violation_code,
                violation_desc,
                MIN(fine_amount) as min_fine,
                AVG(fine_amount) as avg_fine,
                MAX(fine_amount) as max_fine,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fine_amount) as median_fine,
                COUNT(*) as ticket_count
            FROM {self.table_name}
            WHERE fine_amount IS NOT NULL
            GROUP BY violation_code, violation_desc
            ORDER BY avg_fine DESC;
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]

        return pd.DataFrame(results, columns=column_names)
