"""
Prometheus metrics — Water Tracker AI
Exposes /metrics endpoint for scraping.
"""

import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# ── Metric definitions ────────────────────────────────────────────────────────

water_logged_total = Counter(
    "water_logged_ml_total",
    "Total millilitres of water logged across all users",
    ["source"],           # manual | reminder | api
)

log_requests_total = Counter(
    "log_requests_total",
    "Total number of water log API calls",
    ["status"],           # success | error
)

chat_requests_total = Counter(
    "chat_requests_total",
    "Total AI coach chat requests",
)

ml_predictions_total = Counter(
    "ml_predictions_total",
    "ML inference calls",
    ["prediction_type"],  # daily_intake | goal_met | reminder_urgency
)

goal_achievement_gauge = Gauge(
    "goal_achievement_ratio",
    "Fraction of users who met their daily goal today (0–1)",
)

active_users_gauge = Gauge(
    "active_users_today",
    "Number of distinct users who logged water today",
)

api_latency_histogram = Histogram(
    "api_request_latency_seconds",
    "API endpoint latency",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

ml_inference_latency = Histogram(
    "ml_inference_latency_seconds",
    "ML model inference latency",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
)

reminder_fired_total = Counter(
    "reminder_fired_total",
    "Number of hydration reminders fired",
)

reminder_response_rate_gauge = Gauge(
    "reminder_response_rate_pct",
    "% of reminders that resulted in a logged entry within 30 min",
)

# ── Metrics endpoint ──────────────────────────────────────────────────────────

def metrics_endpoint() -> Response:
    """FastAPI route handler for /metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ── Helper recorders ─────────────────────────────────────────────────────────

def record_water_log(amount_ml: float, source: str = "manual"):
    water_logged_total.labels(source=source).inc(amount_ml)
    log_requests_total.labels(status="success").inc()


def record_chat():
    chat_requests_total.inc()


def record_prediction(prediction_type: str):
    ml_predictions_total.labels(prediction_type=prediction_type).inc()


def record_reminder():
    reminder_fired_total.inc()
