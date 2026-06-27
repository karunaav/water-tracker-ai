"""
Analytics Module (v2)
Accepts pre-computed daily_totals dict (from DB) instead of raw logs.
"""

from datetime import date, timedelta
import numpy as np


def compute_analytics(
    daily_totals: dict[str, float],
    period: str = "week",
    goal_ml: float = 2500.0,
) -> dict:
    today = date.today()
    days = 7 if period == "week" else 30
    start_date = today - timedelta(days=days - 1)

    dates, amounts, goal_line = [], [], []
    for i in range(days):
        d = start_date + timedelta(days=i)
        d_str = d.strftime("%Y-%m-%d")
        dates.append(d_str)
        amounts.append(round(daily_totals.get(d_str, 0.0), 1))
        goal_line.append(goal_ml)

    total_intake = sum(amounts)
    active_days = sum(1 for a in amounts if a > 0)
    goal_met_days = sum(1 for a in amounts if a >= goal_ml)
    avg_daily = round(total_intake / days, 1)
    best_day_idx = amounts.index(max(amounts)) if amounts else 0

    # 7-day moving average
    moving_avg = []
    for i in range(len(amounts)):
        window = amounts[max(0, i - 6): i + 1]
        moving_avg.append(round(sum(window) / len(window), 1))

    # Trend: slope over the period
    if len(amounts) >= 2:
        x = np.arange(len(amounts))
        slope = float(np.polyfit(x, amounts, 1)[0])
    else:
        slope = 0.0

    return {
        "period": period,
        "start_date": str(start_date),
        "end_date": str(today),
        "daily_goal_ml": goal_ml,
        "chart": {
            "dates": dates,
            "amounts_ml": amounts,
            "goal_line": goal_line,
            "moving_avg_7d": moving_avg,
        },
        "summary": {
            "total_intake_ml": round(total_intake, 1),
            "average_daily_ml": avg_daily,
            "active_days": active_days,
            "goal_met_days": goal_met_days,
            "goal_streak": _compute_streak(amounts, goal_ml),
            "best_day": dates[best_day_idx] if dates else None,
            "best_day_ml": max(amounts) if amounts else 0,
            "completion_rate_pct": round((goal_met_days / days) * 100, 1),
            "trend_slope": round(slope, 2),
            "trend_direction": "up" if slope > 20 else "down" if slope < -20 else "stable",
        },
    }


def _compute_streak(amounts: list[float], goal: float) -> int:
    streak = 0
    for amount in reversed(amounts):
        if amount >= goal:
            streak += 1
        else:
            break
    return streak
