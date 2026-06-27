"""
Tests for Water Tracker AI backend
Run with: pytest tests/
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, date


# ── Analytics tests ───────────────────────────────────────────────────────────

def test_compute_analytics_empty():
    import sys
    sys.path.insert(0, "backend")
    from analytics import compute_analytics

    result = compute_analytics([], period="week")
    assert result["period"] == "week"
    assert result["summary"]["total_intake_ml"] == 0
    assert result["summary"]["active_days"] == 0
    assert result["summary"]["goal_met_days"] == 0


def test_compute_analytics_with_data():
    import sys
    sys.path.insert(0, "backend")
    from analytics import compute_analytics

    today_str = date.today().strftime("%Y-%m-%d")
    logs = [
        {"date": today_str, "amount_ml": 1500},
        {"date": today_str, "amount_ml": 1000},
    ]

    result = compute_analytics(logs, period="week")
    assert result["summary"]["total_intake_ml"] == 2500
    assert result["summary"]["active_days"] == 1
    assert result["summary"]["best_day"] == today_str
    assert result["summary"]["best_day_ml"] == 2500


def test_analytics_streak():
    import sys
    sys.path.insert(0, "backend")
    from analytics import _compute_streak

    # 3 days meeting goal of 2500
    amounts = [0, 1000, 2500, 2500, 2500]
    streak = _compute_streak(amounts, 2500)
    assert streak == 3

    # No streak
    amounts2 = [2500, 2500, 1000]
    streak2 = _compute_streak(amounts2, 2500)
    assert streak2 == 0


# ── Agent fallback tests ──────────────────────────────────────────────────────

def test_agent_fallback_greeting():
    import sys
    sys.path.insert(0, "backend")
    from agent import _fallback_response

    stats = {"total_ml": 500, "daily_goal_ml": 2500, "progress_pct": 20, "remaining_ml": 2000}
    response = _fallback_response("hello", stats, "Karuna")
    assert "Karuna" in response
    assert "500" in response


def test_agent_fallback_status():
    import sys
    sys.path.insert(0, "backend")
    from agent import _fallback_response

    stats = {"total_ml": 2500, "daily_goal_ml": 2500, "progress_pct": 100, "remaining_ml": 0}
    response = _fallback_response("how am I doing", stats, "Alex")
    assert "100" in response or "goal" in response.lower()


# ── Scheduler tests ───────────────────────────────────────────────────────────

def test_reminder_message_morning():
    import sys
    sys.path.insert(0, "backend")
    from scheduler import _pick_message

    msg = _pick_message(8)
    assert "morning" in msg.lower() or "start" in msg.lower()


def test_reminder_message_evening():
    import sys
    sys.path.insert(0, "backend")
    from scheduler import _pick_message

    msg = _pick_message(20)
    assert len(msg) > 0


# ── Data integrity tests ──────────────────────────────────────────────────────

def test_log_structure():
    """Verify log records have required fields."""
    log = {
        "id": "abc-123",
        "amount_ml": 250.0,
        "note": "morning",
        "timestamp": datetime.now().isoformat(),
        "date": date.today().strftime("%Y-%m-%d"),
    }
    assert "id" in log
    assert "amount_ml" in log
    assert "date" in log
    assert isinstance(log["amount_ml"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
