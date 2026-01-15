# Use Python 3.11 slim image for smaller container size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies required for psycopg2 and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies + gunicorn for production
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY api/ ./api/
COPY config/ ./config/
COPY data/ ./data/
COPY features/ ./features/
COPY training/ ./training/
COPY model/ ./model/
COPY gunicorn.conf.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Cloud Run uses PORT environment variable (default 8080)
ENV PORT=8080

# Expose the port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Run with gunicorn for production using config file
CMD exec gunicorn --config gunicorn.conf.py "api.app:app"
