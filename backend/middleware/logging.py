"""
Structured logging + request tracing middleware.
Every request gets a trace_id; all logs are JSON-formatted.
"""

import time
import uuid
import logging
import os
from contextvars import ContextVar

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# ── Context variable for trace ID ─────────────────────────────────────────────
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    return _trace_id_var.get()


# ── Configure structlog ───────────────────────────────────────────────────────

def configure_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    is_prod = os.getenv("APP_ENV", "development") == "production"

    shared_processors = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.StackInfoRenderer(),
]

    if is_prod:
        # JSON output for log aggregators (Datadog, CloudWatch, etc.)
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        # Human-friendly console output for development
        processors = shared_processors + [structlog.dev.ConsoleRenderer()]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to go through structlog
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level, logging.INFO),
    )


# ── Middleware ────────────────────────────────────────────────────────────────

class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Assigns a trace_id to every request, logs request + response details,
    and records latency. Compatible with Prometheus + any log aggregator.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = str(uuid.uuid4())[:8]
        _trace_id_var.set(trace_id)

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
        )

        logger = structlog.get_logger("http")
        t0 = time.perf_counter()

        logger.info(
            "request_received",
            query=str(request.url.query) or None,
            client_ip=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error("request_failed", error=str(exc), exc_info=True)
            raise

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "request_completed",
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        # Propagate trace ID in response headers for client-side correlation
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Latency-MS"] = str(latency_ms)

        return response


# ── Health check helpers ──────────────────────────────────────────────────────

_startup_time = time.time()


def get_uptime_seconds() -> float:
    return round(time.time() - _startup_time, 1)
