"""
Comprehensive tests — Water Tracker AI v2
Covers: DB CRUD, ML features, analytics, API endpoints, ML predictor.

Run: pytest tests/ -v --tb=short
"""

import sys
import json
import pytest
import numpy as np
from datetime import date, timedelta, datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ══════════════════════════════════════════════════════════════════════════════
# ML FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    def _make_totals(self, n_days=30, base=2000.0):
        today = date.today()
        return {
            (today - timedelta(days=i)).strftime("%Y-%m-%d"): base + (i % 3) * 100
            for i in range(n_days)
        }

    def test_build_daily_features_shape(self):
        from ml.features import build_daily_features, features_to_vector
        totals = self._make_totals()
        feats = build_daily_features(totals, date.today(), goal_ml=2500)
        vec = features_to_vector(feats)
        assert vec.shape == (13,), f"Expected 13 features, got {vec.shape[0]}"

    def test_feature_rolling_averages(self):
        from ml.features import build_daily_features
        totals = {
            (date.today() - timedelta(days=i)).strftime("%Y-%m-%d"): 1000.0
            for i in range(15)
        }
        feats = build_daily_features(totals, date.today(), goal_ml=2500)
        assert feats["avg_3d"] == pytest.approx(1000.0, abs=1)
        assert feats["avg_7d"] == pytest.approx(1000.0, abs=1)

    def test_goal_rate_all_met(self):
        from ml.features import build_daily_features
        totals = {
            (date.today() - timedelta(days=i)).strftime("%Y-%m-%d"): 3000.0
            for i in range(10)
        }
        feats = build_daily_features(totals, date.today(), goal_ml=2500)
        assert feats["goal_rate_7d"] == pytest.approx(1.0)
        assert feats["goal_rate_3d"] == pytest.approx(1.0)

    def test_goal_rate_none_met(self):
        from ml.features import build_daily_features
        totals = {
            (date.today() - timedelta(days=i)).strftime("%Y-%m-%d"): 500.0
            for i in range(10)
        }
        feats = build_daily_features(totals, date.today(), goal_ml=2500)
        assert feats["goal_rate_7d"] == pytest.approx(0.0)
        assert feats["days_since_goal"] == 7   # capped at 7 in window

    def test_dow_cyclical_encoding(self):
        from ml.features import build_daily_features
        feats = build_daily_features({}, date.today(), goal_ml=2500)
        # sin²+cos² must equal 1 (unit circle)
        sin_sq = feats["dow_sin"] ** 2
        cos_sq = feats["dow_cos"] ** 2
        assert (sin_sq + cos_sq) == pytest.approx(1.0, abs=1e-5)

    def test_reminder_features_shape(self):
        from ml.features import build_reminder_features, reminder_features_to_vector
        feats = build_reminder_features(
            current_hour=14,
            intake_so_far_ml=1200,
            goal_ml=2500,
            last_log_minutes_ago=90,
        )
        vec = reminder_features_to_vector(feats)
        assert vec.shape == (9,)

    def test_features_no_nan(self):
        from ml.features import build_daily_features, features_to_vector
        feats = build_daily_features({}, date.today(), goal_ml=2500)
        vec = features_to_vector(feats)
        assert not np.any(np.isnan(vec))


# ══════════════════════════════════════════════════════════════════════════════
# ML MODELS
# ══════════════════════════════════════════════════════════════════════════════

class TestMLModels:
    def _make_totals(self):
        today = date.today()
        return {
            (today - timedelta(days=i)).strftime("%Y-%m-%d"): 2000 + (i % 5) * 100
            for i in range(30)
        }

    def test_train_and_predict(self):
        from ml.predictor import train_models, HydrationPredictor
        # Force train (use small dataset for speed)
        metrics = train_models(force=True)
        assert "intake_predictor" in metrics
        assert metrics["intake_predictor"]["cv_mae_mean"] > 0

        result = HydrationPredictor.predict_daily_intake(
            self._make_totals(),
            goal_ml=2500.0,
        )
        assert "predicted_intake_ml" in result
        assert result["predicted_intake_ml"] >= 0
        assert 0 <= result["goal_met_probability"] <= 1
        assert result["model_version"] is not None

    def test_prediction_range(self):
        from ml.predictor import HydrationPredictor
        result = HydrationPredictor.predict_daily_intake(
            self._make_totals(),
            goal_ml=2500.0,
        )
        # Prediction should be physically reasonable
        assert 0 <= result["predicted_intake_ml"] <= 10000

    def test_reminder_scoring(self):
        from ml.predictor import HydrationPredictor
        result = HydrationPredictor.score_reminder_urgency(
            current_hour=14,
            intake_so_far_ml=500,
            goal_ml=2500,
            last_log_minutes_ago=120,
        )
        assert "urgency_score" in result
        assert 0 <= result["urgency_score"] <= 1
        assert isinstance(result["should_remind"], bool)

    def test_reminder_not_urgent_when_on_track(self):
        from ml.predictor import HydrationPredictor
        # Just logged, well on track
        result = HydrationPredictor.score_reminder_urgency(
            current_hour=12,
            intake_so_far_ml=2400,
            goal_ml=2500,
            last_log_minutes_ago=5,
        )
        # Should be low urgency when nearly done and just logged
        assert result["urgency_score"] < 0.9  # not maximum urgency


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS MODULE
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalytics:
    def test_empty_data(self):
        from analytics import compute_analytics
        result = compute_analytics({}, period="week", goal_ml=2500)
        assert result["summary"]["total_intake_ml"] == 0
        assert result["summary"]["active_days"] == 0
        assert len(result["chart"]["dates"]) == 7

    def test_week_has_7_days(self):
        from analytics import compute_analytics
        result = compute_analytics({}, period="week", goal_ml=2500)
        assert len(result["chart"]["dates"]) == 7
        assert len(result["chart"]["amounts_ml"]) == 7

    def test_month_has_30_days(self):
        from analytics import compute_analytics
        result = compute_analytics({}, period="month", goal_ml=2500)
        assert len(result["chart"]["dates"]) == 30

    def test_goal_met_count(self):
        from analytics import compute_analytics
        today = date.today()
        totals = {
            (today - timedelta(days=i)).strftime("%Y-%m-%d"): 3000.0
            for i in range(5)
        }
        result = compute_analytics(totals, period="week", goal_ml=2500)
        assert result["summary"]["goal_met_days"] == 5

    def test_streak_calculation(self):
        from analytics import _compute_streak
        # 3 consecutive days meeting goal
        assert _compute_streak([0, 500, 2500, 2500, 2500], 2500) == 3
        assert _compute_streak([2500, 2500, 0], 2500) == 0
        assert _compute_streak([], 2500) == 0

    def test_moving_average_length(self):
        from analytics import compute_analytics
        result = compute_analytics({}, period="week", goal_ml=2500)
        assert len(result["chart"]["moving_avg_7d"]) == 7

    def test_trend_direction(self):
        from analytics import compute_analytics
        today = date.today()
        # Increasing trend
        totals = {
            (today - timedelta(days=6 - i)).strftime("%Y-%m-%d"): 1000 + i * 300.0
            for i in range(7)
        }
        result = compute_analytics(totals, period="week", goal_ml=2500)
        assert result["summary"]["trend_direction"] == "up"

    def test_completion_rate(self):
        from analytics import compute_analytics
        today = date.today()
        # All 7 days met goal
        totals = {
            (today - timedelta(days=i)).strftime("%Y-%m-%d"): 3000.0
            for i in range(7)
        }
        result = compute_analytics(totals, period="week", goal_ml=2500)
        assert result["summary"]["completion_rate_pct"] == 100.0


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE CRUD
# ══════════════════════════════════════════════════════════════════════════════

class TestDatabase:
    @pytest.fixture
    def db_session(self):
        """Create an in-memory SQLite DB for each test."""
        import os
        os.environ["DATABASE_URL"] = "sqlite://"   # in-memory
        from db.session import init_db, SessionLocal, engine
        from db.models import Base
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        yield db
        db.close()
        Base.metadata.drop_all(bind=engine)

    def test_create_and_retrieve_log(self, db_session):
        from db import crud
        log = crud.create_log(db_session, amount_ml=250.0, note="test")
        assert log.id is not None
        assert log.amount_ml == 250.0
        assert log.date_str == date.today().strftime("%Y-%m-%d")

    def test_today_stats_empty(self, db_session):
        from db import crud
        crud.get_or_create_profile(db_session)
        stats = crud.today_stats(db_session)
        assert stats["total_ml"] == 0
        assert stats["progress_pct"] == 0

    def test_today_stats_with_logs(self, db_session):
        from db import crud
        crud.get_or_create_profile(db_session)
        crud.create_log(db_session, 500.0)
        crud.create_log(db_session, 750.0)
        stats = crud.today_stats(db_session)
        assert stats["total_ml"] == 1250.0
        assert stats["entries"] == 2

    def test_delete_log(self, db_session):
        from db import crud
        log = crud.create_log(db_session, 300.0)
        deleted = crud.delete_log(db_session, log.id)
        assert deleted is True
        remaining = crud.get_logs(db_session)
        assert len(remaining) == 0

    def test_delete_nonexistent_log(self, db_session):
        from db import crud
        result = crud.delete_log(db_session, "does-not-exist")
        assert result is False

    def test_profile_defaults(self, db_session):
        from db import crud
        profile = crud.get_or_create_profile(db_session)
        assert profile.daily_goal_ml == 2500.0
        assert profile.reminder_interval_min == 60

    def test_update_profile(self, db_session):
        from db import crud
        crud.get_or_create_profile(db_session)
        updated = crud.update_profile(db_session, "default", daily_goal_ml=3000.0, name="Karuna")
        assert updated.daily_goal_ml == 3000.0
        assert updated.name == "Karuna"

    def test_daily_totals_aggregation(self, db_session):
        from db import crud
        crud.get_or_create_profile(db_session)
        crud.create_log(db_session, 500.0)
        crud.create_log(db_session, 750.0)
        totals = crud.daily_totals(db_session, "default", days=7)
        today_str = date.today().strftime("%Y-%m-%d")
        assert totals.get(today_str) == pytest.approx(1250.0)

    def test_clear_day(self, db_session):
        from db import crud
        crud.get_or_create_profile(db_session)
        crud.create_log(db_session, 250.0)
        crud.create_log(db_session, 500.0)
        n = crud.clear_day(db_session, "default", date.today().strftime("%Y-%m-%d"))
        assert n == 2
        stats = crud.today_stats(db_session)
        assert stats["total_ml"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# API INTEGRATION (via TestClient)
# ══════════════════════════════════════════════════════════════════════════════

class TestAPI:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite://")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        from fastapi.testclient import TestClient
        import importlib, main as m
        importlib.reload(m)
        with TestClient(m.app) as c:
            yield c

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "uptime_seconds" in data
        assert "database" in data

    def test_log_and_today(self, client):
        r = client.post("/log", json={"amount_ml": 500.0, "note": "test"})
        assert r.status_code == 200
        assert r.json()["status"] == "logged"

        r2 = client.get("/today")
        assert r2.status_code == 200
        assert r2.json()["total_ml"] == 500.0

    def test_log_validation(self, client):
        r = client.post("/log", json={"amount_ml": -100})
        assert r.status_code == 422   # Pydantic validation error

    def test_delete_log(self, client):
        r = client.post("/log", json={"amount_ml": 250.0})
        log_id = r.json()["record"]["id"]
        r2 = client.delete(f"/log/{log_id}")
        assert r2.status_code == 200

    def test_delete_nonexistent(self, client):
        r = client.delete("/log/does-not-exist")
        assert r.status_code == 404

    def test_analytics_week(self, client):
        r = client.get("/analytics?period=week")
        assert r.status_code == 200
        data = r.json()
        assert data["period"] == "week"
        assert len(data["chart"]["dates"]) == 7

    def test_get_profile(self, client):
        r = client.get("/profile")
        assert r.status_code == 200
        assert "daily_goal_ml" in r.json()

    def test_update_profile(self, client):
        r = client.patch("/profile", json={"daily_goal_ml": 3000.0, "name": "Karuna"})
        assert r.status_code == 200
        assert r.json()["daily_goal_ml"] == 3000.0

    def test_trace_id_header(self, client):
        r = client.get("/today")
        assert "X-Trace-ID" in r.headers
        assert len(r.headers["X-Trace-ID"]) == 8

    def test_latency_header(self, client):
        r = client.get("/today")
        assert "X-Latency-MS" in r.headers
        assert float(r.headers["X-Latency-MS"]) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
