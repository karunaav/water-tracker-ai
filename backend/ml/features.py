"""
Feature engineering for hydration ML models.
Converts raw log history into structured feature vectors.
"""

from datetime import datetime, date, timedelta
from collections import defaultdict
import numpy as np


def build_daily_features(
    daily_totals: dict[str, float],
    target_date: date,
    goal_ml: float = 2500.0,
    hourly_pattern: dict[int, float] | None = None,
) -> dict:
    """
    Build a feature vector for predicting intake on target_date.

    Features:
    - Rolling averages (3d, 7d, 14d)
    - Goal attainment rates (3d, 7d)
    - Day-of-week encoding (sin/cos for cyclical)
    - Trend slope over 7d
    - Consistency score (std deviation)
    - Peak-hour signal from hourly pattern
    - Days since last goal-met
    """
    today = target_date
    dates_sorted = sorted(daily_totals.keys(), reverse=True)

    def avg_over(n_days: int) -> float:
        vals = []
        for i in range(1, n_days + 1):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            vals.append(daily_totals.get(d, 0.0))
        return float(np.mean(vals)) if vals else 0.0

    def goal_rate(n_days: int) -> float:
        met = 0
        for i in range(1, n_days + 1):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            if daily_totals.get(d, 0.0) >= goal_ml:
                met += 1
        return met / n_days

    # 7-day trend slope via linear regression
    vals_7d = [
        daily_totals.get((today - timedelta(days=i)).strftime("%Y-%m-%d"), 0.0)
        for i in range(6, -1, -1)
    ]
    x = np.arange(len(vals_7d))
    slope = float(np.polyfit(x, vals_7d, 1)[0]) if len(vals_7d) >= 2 else 0.0

    # Consistency (inverse std deviation — lower std = more consistent)
    std_7d = float(np.std(vals_7d)) if vals_7d else 0.0

    # Days since goal was last met
    days_since_goal = 0
    for i in range(1, 31):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        if daily_totals.get(d, 0.0) >= goal_ml:
            break
        days_since_goal += 1

    # Day-of-week cyclical encoding
    dow = today.weekday()  # 0=Mon, 6=Sun
    dow_sin = float(np.sin(2 * np.pi * dow / 7))
    dow_cos = float(np.cos(2 * np.pi * dow / 7))

    # Peak drinking hour (from historical pattern)
    peak_hour = 12  # default noon
    if hourly_pattern:
        peak_hour = max(hourly_pattern, key=lambda h: hourly_pattern[h])

    peak_hour_sin = float(np.sin(2 * np.pi * peak_hour / 24))
    peak_hour_cos = float(np.cos(2 * np.pi * peak_hour / 24))

    return {
        "avg_3d":             round(avg_over(3), 1),
        "avg_7d":             round(avg_over(7), 1),
        "avg_14d":            round(avg_over(14), 1),
        "goal_rate_3d":       round(goal_rate(3), 3),
        "goal_rate_7d":       round(goal_rate(7), 3),
        "trend_slope_7d":     round(slope, 2),
        "std_7d":             round(std_7d, 1),
        "days_since_goal":    days_since_goal,
        "dow_sin":            round(dow_sin, 4),
        "dow_cos":            round(dow_cos, 4),
        "peak_hour_sin":      round(peak_hour_sin, 4),
        "peak_hour_cos":      round(peak_hour_cos, 4),
        "goal_ml":            goal_ml,
    }


def build_reminder_features(
    current_hour: int,
    intake_so_far_ml: float,
    goal_ml: float,
    last_log_minutes_ago: int,
    hourly_pattern: dict[int, float] | None = None,
) -> dict:
    """
    Features for predicting optimal reminder timing.
    Used by the smart scheduler to decide whether to fire a reminder now.
    """
    hours_remaining = max(0, 22 - current_hour)  # stops at 10 PM
    progress_pct = (intake_so_far_ml / goal_ml) * 100 if goal_ml else 0
    needed_per_hour = (
        (goal_ml - intake_so_far_ml) / hours_remaining
        if hours_remaining > 0 else 0
    )

    # How much user typically drinks at this hour
    typical_this_hour = 0.0
    if hourly_pattern:
        typical_this_hour = hourly_pattern.get(current_hour, 0.0)

    hour_sin = float(np.sin(2 * np.pi * current_hour / 24))
    hour_cos = float(np.cos(2 * np.pi * current_hour / 24))

    return {
        "current_hour":        current_hour,
        "hour_sin":            round(hour_sin, 4),
        "hour_cos":            round(hour_cos, 4),
        "intake_so_far_ml":    round(intake_so_far_ml, 1),
        "progress_pct":        round(progress_pct, 1),
        "needed_per_hour_ml":  round(needed_per_hour, 1),
        "last_log_minutes_ago": last_log_minutes_ago,
        "typical_this_hour_ml": round(typical_this_hour, 1),
        "hours_remaining":     hours_remaining,
        "goal_ml":             goal_ml,
    }


def features_to_vector(features: dict) -> np.ndarray:
    """Convert feature dict to ordered numpy array for sklearn."""
    FEATURE_ORDER = [
        "avg_3d", "avg_7d", "avg_14d",
        "goal_rate_3d", "goal_rate_7d",
        "trend_slope_7d", "std_7d",
        "days_since_goal",
        "dow_sin", "dow_cos",
        "peak_hour_sin", "peak_hour_cos",
        "goal_ml",
    ]
    return np.array([features.get(k, 0.0) for k in FEATURE_ORDER], dtype=np.float32)


def reminder_features_to_vector(features: dict) -> np.ndarray:
    FEATURE_ORDER = [
        "hour_sin", "hour_cos",
        "intake_so_far_ml", "progress_pct",
        "needed_per_hour_ml", "last_log_minutes_ago",
        "typical_this_hour_ml", "hours_remaining",
        "goal_ml",
    ]
    return np.array([features.get(k, 0.0) for k in FEATURE_ORDER], dtype=np.float32)
