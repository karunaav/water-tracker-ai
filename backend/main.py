"""
Water Tracker AI — FastAPI Backend (v2)
Upgraded with: PostgreSQL/SQLite ORM, ML predictions, Prometheus metrics,
structured logging, request tracing, and smart reminder scoring.
"""

import os, time, json
from datetime import datetime, date, timedelta
from typing import Optional
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db import init_db, get_db
from db import crud
from agent import get_water_ai_response
from scheduler import start_reminder_scheduler
from analytics import compute_analytics
from ml import HydrationPredictor, train_models, load_metrics
from middleware.logging import configure_logging, RequestTracingMiddleware, get_uptime_seconds
from middleware.metrics import (
    metrics_endpoint, record_water_log, record_chat,
    record_prediction, api_latency_histogram, ml_inference_latency,
    goal_achievement_gauge, active_users_gauge,
)

# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger = structlog.get_logger("startup")
    logger.info("initialising_database")
    init_db()
    logger.info("training_ml_models")
    train_models(force=False)
    logger.info("starting_reminder_scheduler")
    start_reminder_scheduler()
    logger.info("startup_complete")
    yield
    logger.info("shutdown")

app = FastAPI(
    title="Water Tracker AI",
    description="AI-powered hydration tracking · ML predictions · Observability",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestTracingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = structlog.get_logger("api")

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class WaterLogIn(BaseModel):
    amount_ml: float = Field(..., gt=0, le=5000, description="ml of water consumed")
    note: Optional[str] = Field(None, max_length=256)
    timestamp: Optional[datetime] = None
    source: str = Field("manual", pattern="^(manual|reminder|api)$")

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    user_name: Optional[str] = "User"
    user_id: Optional[str] = "default"

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    daily_goal_ml: Optional[float] = Field(None, gt=0, le=10000)
    reminder_interval_min: Optional[int] = Field(None, ge=15, le=480)
    weight_kg: Optional[float] = Field(None, gt=0, le=500)
    activity_level: Optional[str] = None
    climate: Optional[str] = None

# ── Root / health ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "version": "2.0.0", "message": "Water Tracker AI 💧"}


@app.get("/health", tags=["Health"])
def health(db: Session = Depends(get_db)):
    """Detailed health check — used by load balancers and monitoring."""
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {e}"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "uptime_seconds": get_uptime_seconds(),
        "database": db_status,
        "ml_models_loaded": (
            __import__("pathlib").Path("data/models/intake_predictor.joblib").exists()
        ),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics", tags=["Observability"], include_in_schema=False)
def prometheus_metrics():
    """Prometheus scrape endpoint — /metrics."""
    return metrics_endpoint()


# ── Water logging ─────────────────────────────────────────────────────────────

@app.post("/log", tags=["Hydration"])
def log_water(entry: WaterLogIn, db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    record = crud.create_log(
        db,
        amount_ml=entry.amount_ml,
        note=entry.note or "",
        timestamp=entry.timestamp,
        source=entry.source,
    )
    record_water_log(entry.amount_ml, source=entry.source)
    api_latency_histogram.labels(endpoint="/log").observe(time.perf_counter() - t0)
    logger.info("water_logged", amount_ml=entry.amount_ml, source=entry.source)
    return {"status": "logged", "record": record.to_dict()}


@app.get("/logs", tags=["Hydration"])
def get_logs(
    date_filter: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    logs = crud.get_logs(db, date_filter=date_filter, limit=limit)
    return {"count": len(logs), "logs": [l.to_dict() for l in logs]}


@app.get("/today", tags=["Hydration"])
def today_summary(db: Session = Depends(get_db)):
    stats = crud.today_stats(db)
    # Update gauge for monitoring
    goal_achievement_gauge.set(stats["progress_pct"] / 100)
    return stats


@app.delete("/log/{log_id}", tags=["Hydration"])
def delete_log(log_id: str, db: Session = Depends(get_db)):
    deleted = crud.delete_log(db, log_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Log not found")
    return {"status": "deleted", "id": log_id}


@app.delete("/logs/clear-today", tags=["Hydration"])
def clear_today(db: Session = Depends(get_db)):
    today_str = date.today().strftime("%Y-%m-%d")
    n = crud.clear_day(db, "default", today_str)
    return {"status": "cleared", "date": today_str, "entries_removed": n}


# ── Analytics ─────────────────────────────────────────────────────────────────

@app.get("/analytics", tags=["Analytics"])
def analytics(period: str = Query("week", pattern="^(week|month)$"), db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    profile = crud.get_or_create_profile(db)
    days = 7 if period == "week" else 30
    totals = crud.daily_totals(db, "default", days=days + 7)
    data = compute_analytics(totals, period, goal_ml=profile.daily_goal_ml)
    api_latency_histogram.labels(endpoint="/analytics").observe(time.perf_counter() - t0)
    return data


# ── ML Predictions ────────────────────────────────────────────────────────────

@app.get("/predict/intake", tags=["ML"])
def predict_intake(db: Session = Depends(get_db)):
    """
    Predict tomorrow's water intake and probability of meeting daily goal.
    Uses GBM regression trained on historical patterns.
    """
    t0 = time.perf_counter()
    profile = crud.get_or_create_profile(db)
    totals = crud.daily_totals(db, "default", days=30)
    hourly = crud.get_hourly_pattern(db, "default", days=14)

    result = HydrationPredictor.predict_daily_intake(
        daily_totals=totals,
        goal_ml=profile.daily_goal_ml,
        hourly_pattern=hourly,
        target_date=date.today() + timedelta(days=1),
    )

    latency = time.perf_counter() - t0
    ml_inference_latency.labels(model="intake_predictor").observe(latency)
    record_prediction("daily_intake")

    # Persist prediction for audit trail
    crud.store_prediction(
        db,
        prediction_type="daily_intake",
        input_features=json.dumps(result["features_used"]),
        predicted_value=result["predicted_intake_ml"],
        confidence=result["goal_met_probability"],
    )

    return {
        **result,
        "inference_latency_ms": round(latency * 1000, 2),
    }


@app.get("/predict/reminder", tags=["ML"])
def predict_reminder_timing(db: Session = Depends(get_db)):
    """
    Score urgency of sending a reminder right now.
    Uses RandomForest classifier trained on reminder-response patterns.
    """
    t0 = time.perf_counter()
    profile = crud.get_or_create_profile(db)
    stats = crud.today_stats(db)
    hourly = crud.get_hourly_pattern(db, "default", days=14)

    # Find minutes since last log
    logs_today = crud.get_logs(db, date_filter=date.today().strftime("%Y-%m-%d"), limit=1)
    if logs_today:
        last_ts = logs_today[0].timestamp
        last_log_min = int((datetime.utcnow() - last_ts).total_seconds() / 60)
    else:
        last_log_min = 999

    result = HydrationPredictor.score_reminder_urgency(
        current_hour=datetime.now().hour,
        intake_so_far_ml=stats["total_ml"],
        goal_ml=profile.daily_goal_ml,
        last_log_minutes_ago=last_log_min,
        hourly_pattern=hourly,
    )

    latency = time.perf_counter() - t0
    ml_inference_latency.labels(model="reminder_scorer").observe(latency)
    record_prediction("reminder_urgency")

    return {
        **result,
        "inference_latency_ms": round(latency * 1000, 2),
    }


@app.get("/ml/metrics", tags=["ML"])
def ml_metrics():
    """Return training metrics and model version for the current models."""
    return load_metrics()


@app.post("/ml/retrain", tags=["ML"])
def retrain_models():
    """Trigger model retraining. In prod this would be async/background."""
    logger.info("manual_retrain_triggered")
    metrics = train_models(force=True)
    # Reload models on next inference call
    HydrationPredictor._intake_model = None
    HydrationPredictor._goal_model = None
    HydrationPredictor._reminder_model = None
    return {"status": "retrained", "metrics": metrics}


# ── AI Coach ──────────────────────────────────────────────────────────────────

@app.post("/chat", tags=["AI Coach"])
def chat(msg: ChatMessage, db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    today = crud.today_stats(db, msg.user_id)
    record_chat()

    # Enrich context with ML prediction
    try:
        profile = crud.get_or_create_profile(db, msg.user_id)
        totals = crud.daily_totals(db, msg.user_id, days=14)
        hourly = crud.get_hourly_pattern(db, msg.user_id, days=14)
        prediction = HydrationPredictor.predict_daily_intake(totals, profile.daily_goal_ml, hourly)
        ml_context = prediction
    except Exception:
        ml_context = None

    reply = get_water_ai_response(msg.message, today, msg.user_name, ml_context=ml_context)
    api_latency_histogram.labels(endpoint="/chat").observe(time.perf_counter() - t0)
    return {"reply": reply}


# ── User Profile ──────────────────────────────────────────────────────────────

@app.get("/profile", tags=["Profile"])
def get_profile(db: Session = Depends(get_db)):
    return crud.get_or_create_profile(db).to_dict()


@app.patch("/profile", tags=["Profile"])
def update_profile(updates: ProfileUpdate, db: Session = Depends(get_db)):
    kwargs = {k: v for k, v in updates.model_dump().items() if v is not None}
    profile = crud.update_profile(db, "default", **kwargs)
    return profile.to_dict()


# ── Reminder ──────────────────────────────────────────────────────────────────

@app.get("/reminder/status", tags=["Reminders"])
def reminder_status(db: Session = Depends(get_db)):
    profile = crud.get_or_create_profile(db)
    rate = crud.get_reminder_response_rate(db, days=14)
    return {
        "interval_minutes": profile.reminder_interval_min,
        "active": True,
        "response_rate_pct": rate,
    }
