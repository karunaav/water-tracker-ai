"""
Smart Reminder Scheduler (v2)
Uses ML urgency scorer to decide whether to fire a reminder each cycle.
"""

import os
import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

log = logging.getLogger(__name__)
scheduler = BackgroundScheduler()
_reminders_log: list[dict] = []


def hydration_reminder():
    now = datetime.now()
    hour = now.hour
    if not (7 <= hour <= 22):
        return

    # Try ML-based smart scoring
    try:
        from db.session import SessionLocal
        from db import crud
        from ml import HydrationPredictor

        db = SessionLocal()
        try:
            stats = crud.today_stats(db)
            hourly = crud.get_hourly_pattern(db, "default", days=14)
            logs_today = crud.get_logs(db, date_filter=now.strftime("%Y-%m-%d"), limit=1)
            last_log_min = 999
            if logs_today:
                last_log_min = int((datetime.utcnow() - logs_today[0].timestamp).total_seconds() / 60)

            profile = crud.get_or_create_profile(db)
            score = HydrationPredictor.score_reminder_urgency(
                current_hour=hour,
                intake_so_far_ml=stats["total_ml"],
                goal_ml=profile.daily_goal_ml,
                last_log_minutes_ago=last_log_min,
                hourly_pattern=hourly,
            )

            if not score["should_remind"]:
                log.info(f"Smart reminder suppressed (urgency={score['urgency_pct']}%)")
                return

            # Log reminder to DB
            reminder = crud.log_reminder(db, _pick_message(hour))
        finally:
            db.close()

    except Exception as e:
        log.warning(f"ML scoring failed, firing reminder anyway: {e}")

    message = _pick_message(hour)
    entry = {"timestamp": now.isoformat(), "message": message}
    _reminders_log.append(entry)
    if len(_reminders_log) > 50:
        _reminders_log.pop(0)

    from middleware.metrics import record_reminder
    record_reminder()
    log.info(f"[REMINDER {now.strftime('%H:%M')}] {message}")


def _pick_message(hour: int) -> str:
    if hour < 10:
        return "Good morning! Start strong — a glass of water wakes you up faster than coffee ☀️"
    elif hour < 13:
        return "Mid-morning check — staying hydrated keeps your focus sharp 💧"
    elif hour < 15:
        return "Post-lunch dip? Hydration helps more than you think 🍃"
    elif hour < 18:
        return "Afternoon slump — water over caffeine, your future self will thank you 💧"
    elif hour < 21:
        return "Evening push — finish strong and hit that daily goal! 🌙"
    else:
        return "Almost bedtime — a small glass supports overnight recovery 💤"


def start_reminder_scheduler():
    interval = int(os.getenv("REMINDER_INTERVAL_MINUTES", 60))
    if not scheduler.running:
        scheduler.add_job(
            hydration_reminder,
            trigger=IntervalTrigger(minutes=interval),
            id="hydration_reminder",
            replace_existing=True,
            max_instances=1,
        )
        scheduler.start()
        log.info(f"Reminder scheduler started — every {interval} min")
