# Telemetry module — disabled for mid-sem evaluation.
# OTel, Prometheus, and Redis instrumentation have been removed.
# Training logs are now streamed directly from PostgreSQL via SSE.

from fastapi import FastAPI

def setup_telemetry(app: FastAPI, engine=None):
    """No-op: telemetry is disabled."""
    pass

def get_tracer():
    """Returns None — no tracing active."""
    return None
