"""
CRUD repository — clean data access layer between routes and ORM.
All business queries live here; endpoints stay thin.
"""

from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
import uuid

from db.models import HydrationLog, UserProfile, ReminderLog, MLPrediction


# ── HydrationLog ──────────────────────────────────────────────────────────────

def create_log(
    db: Session,
    amount_ml: float,
    note: str = "",
    timestamp: datetime | None = None,
    user_id: str = "default",
    source: str = "manual",
) -> HydrationLog:
    ts = timestamp or datetime.utcnow()
    log = HydrationLog(
        id=str(uuid.uuid4()),
        amount_ml=amount_ml,
        note=note,
        timestamp=ts,
        date_str=ts.strftime("%Y-%m-%d"),
        user_id=user_id,
        source=source,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_logs(
    db: Session,
    user_id: str = "default",
    date_filter: str | None = None,
    limit: int = 500,
) -> list[HydrationLog]:
    q = db.query(HydrationLog).filter(HydrationLog.user_id == user_id)
    if date_filter:
        q = q.filter(HydrationLog.date_str == date_filter)
    return q.order_by(HydrationLog.timestamp.desc()).limit(limit).all()


def get_logs_range(
    db: Session,
    user_id: str,
    start_date: date,
    end_date: date,
) -> list[HydrationLog]:
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")
    return (
        db.query(HydrationLog)
        .filter(
            HydrationLog.user_id == user_id,
            HydrationLog.date_str >= start_str,
            HydrationLog.date_str <= end_str,
        )
        .order_by(HydrationLog.timestamp.asc())
        .all()
    )


def daily_totals(
    db: Session,
    user_id: str,
    days: int = 30,
) -> dict[str, float]:
    """Return {date_str: total_ml} for the last N days."""
    start_str = (date.today() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    rows = (
        db.query(HydrationLog.date_str, func.sum(HydrationLog.amount_ml))
        .filter(HydrationLog.user_id == user_id, HydrationLog.date_str >= start_str)
        .group_by(HydrationLog.date_str)
        .all()
    )
    return {r[0]: float(r[1]) for r in rows}


def delete_log(db: Session, log_id: str, user_id: str = "default") -> bool:
    log = (
        db.query(HydrationLog)
        .filter(HydrationLog.id == log_id, HydrationLog.user_id == user_id)
        .first()
    )
    if not log:
        return False
    db.delete(log)
    db.commit()
    return True


def clear_day(db: Session, user_id: str, date_str: str) -> int:
    n = (
        db.query(HydrationLog)
        .filter(HydrationLog.user_id == user_id, HydrationLog.date_str == date_str)
        .delete()
    )
    db.commit()
    return n


def today_stats(db: Session, user_id: str = "default") -> dict:
    today_str = date.today().strftime("%Y-%m-%d")
    rows = (
        db.query(HydrationLog)
        .filter(HydrationLog.user_id == user_id, HydrationLog.date_str == today_str)
        .all()
    )
    total = sum(r.amount_ml for r in rows)
    profile = get_or_create_profile(db, user_id)
    goal = profile.daily_goal_ml
    return {
        "date":           today_str,
        "total_ml":       round(total, 1),
        "total_glasses":  round(total / 250, 1),
        "daily_goal_ml":  goal,
        "progress_pct":   round((total / goal) * 100, 1) if goal else 0,
        "remaining_ml":   max(0, round(goal - total, 1)),
        "entries":        len(rows),
    }


def get_hourly_pattern(db: Session, user_id: str, days: int = 14) -> dict[int, float]:
    """Return avg intake by hour-of-day over last N days (for ML features)."""
    start_str = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    logs = (
        db.query(HydrationLog)
        .filter(HydrationLog.user_id == user_id, HydrationLog.date_str >= start_str)
        .all()
    )
    hourly: dict[int, list[float]] = {h: [] for h in range(24)}
    for log in logs:
        hourly[log.timestamp.hour].append(log.amount_ml)
    return {h: round(sum(v) / len(v), 1) if v else 0.0 for h, v in hourly.items()}


# ── UserProfile ───────────────────────────────────────────────────────────────

def get_or_create_profile(db: Session, user_id: str = "default") -> UserProfile:
    profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not profile:
        profile = UserProfile(id=user_id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile


def update_profile(db: Session, user_id: str, **kwargs) -> UserProfile:
    profile = get_or_create_profile(db, user_id)
    for k, v in kwargs.items():
        if hasattr(profile, k):
            setattr(profile, k, v)
    db.commit()
    db.refresh(profile)
    return profile


# ── ReminderLog ───────────────────────────────────────────────────────────────

def log_reminder(db: Session, message: str) -> ReminderLog:
    reminder = ReminderLog(message=message)
    db.add(reminder)
    db.commit()
    db.refresh(reminder)
    return reminder


def mark_reminder_responded(db: Session, reminder_id: int, response_ml: float):
    reminder = db.query(ReminderLog).filter(ReminderLog.id == reminder_id).first()
    if reminder:
        reminder.responded = True
        reminder.response_ml = response_ml
        db.commit()


def get_reminder_response_rate(db: Session, days: int = 14) -> float:
    """% of reminders that resulted in a log within 30 minutes."""
    start = datetime.utcnow() - timedelta(days=days)
    total = db.query(ReminderLog).filter(ReminderLog.timestamp >= start).count()
    responded = db.query(ReminderLog).filter(
        ReminderLog.timestamp >= start,
        ReminderLog.responded == True,
    ).count()
    return round((responded / total) * 100, 1) if total else 0.0


# ── MLPrediction ──────────────────────────────────────────────────────────────

def store_prediction(
    db: Session,
    prediction_type: str,
    input_features: str,
    predicted_value: float,
    confidence: float | None = None,
    model_version: str = "v1.0",
) -> MLPrediction:
    pred = MLPrediction(
        prediction_type=prediction_type,
        input_features=input_features,
        predicted_value=predicted_value,
        confidence=confidence,
        model_version=model_version,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred
