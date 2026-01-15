"""Gunicorn configuration for Cloud Run."""

import os

# Bind to PORT env var (Cloud Run provides this)
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

# Workers configuration
# Cloud Run recommends 1 worker per container instance
# Use threads for concurrent requests within the worker
workers = 1
threads = 8

# Timeout - 0 means no timeout (Cloud Run handles timeouts)
timeout = 0

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Preload app to load model once before forking workers
preload_app = True


def on_starting(server):
    """Called just before the master process is initialized."""
    print("Starting Parking ML API server...")


def when_ready(server):
    """Called just after the server is started."""
    print("Parking ML API server is ready to accept connections")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    print(f"Worker {worker.pid} spawned")
