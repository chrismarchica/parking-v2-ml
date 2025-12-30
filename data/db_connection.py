"""Database connection manager for PostgreSQL."""

import os
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Optional

import yaml
import psycopg2
from psycopg2 import pool


class DatabaseConnection:
    """Manages PostgreSQL database connections with connection pooling."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize database connection manager.

        Args:
            config_path: Path to db.yaml config file. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "db.yaml"

        self.config = self._load_config(config_path)
        self._pool: Optional[pool.ThreadedConnectionPool] = None

    def _load_config(self, config_path: str | Path) -> dict:
        """Load and parse database configuration."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Resolve environment variables
        db_config = config["database"]
        db_config["user"] = self._resolve_env_var(db_config["user"])
        db_config["password"] = self._resolve_env_var(db_config["password"])

        return config

    def _resolve_env_var(self, value: str) -> str:
        """Resolve ${VAR_NAME} style environment variables."""
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            resolved = os.environ.get(env_var)
            if resolved is None:
                raise ValueError(f"Environment variable {env_var} not set")
            return resolved
        return value

    def _get_connection_params(self) -> dict:
        """Get connection parameters for psycopg2."""
        db = self.config["database"]
        return {
            "host": db["host"],
            "port": db["port"],
            "dbname": db["name"],
            "user": db["user"],
            "password": db["password"],
        }

    def init_pool(self) -> None:
        """Initialize the connection pool."""
        if self._pool is not None:
            return

        pool_config = self.config["pool"]
        self._pool = pool.ThreadedConnectionPool(
            minconn=pool_config["min_connections"],
            maxconn=pool_config["max_connections"],
            **self._get_connection_params(),
        )

    def close_pool(self) -> None:
        """Close all connections in the pool."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None

    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Get a connection from the pool.

        Yields:
            A database connection that will be returned to the pool on exit.
        """
        if self._pool is None:
            self.init_pool()

        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def get_cursor(
        self, cursor_factory=None
    ) -> Generator[psycopg2.extensions.cursor, None, None]:
        """
        Get a cursor with automatic connection management.

        Args:
            cursor_factory: Optional cursor factory (e.g., RealDictCursor)

        Yields:
            A database cursor.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def execute_query(self, query: str, params: tuple = None) -> list:
        """
        Execute a query and return all results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of result tuples
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if cursor.description is not None:
                return cursor.fetchall()
            return []

    def test_connection(self) -> bool:
        """Test if the database connection works."""
        try:
            result = self.execute_query("SELECT 1")
            return result == [(1,)]
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

