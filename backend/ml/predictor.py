"""
ML Models — Water Tracker AI
1. DailyIntakePredictor  — predicts tomorrow's intake in ml
2. GoalMet classifier    — P(user meets daily goal today)
3. SmartReminderScorer   — scores urgency of firing a reminder right now

All models use scikit-learn pipelines with StandardScaler preprocessing.
Models are persisted to disk with joblib and lazily reloaded on startup.
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Any

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from ml.features import (
    build_daily_features,
    features_to_vector,
    build_reminder_features,
    reminder_features_to_vector,
)

log = logging.getLogger(__name__)

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INTAKE_MODEL_PATH  = MODEL_DIR / "intake_predictor.joblib"
GOAL_MODEL_PATH    = MODEL_DIR / "goal_classifier.joblib"
REMINDER_MODEL_PATH = MODEL_DIR / "reminder_scorer.joblib"
METRICS_PATH       = MODEL_DIR / "model_metrics.json"

MODEL_VERSION = "v1.2"


# ── Synthetic training data generator ─────────────────────────────────────────

def _generate_synthetic_data(n_users: int = 20, days: int = 90):
    """
    Generate realistic synthetic hydration data for bootstrapping.
    Simulates users with different habits, goals, and consistency levels.
    Returns (X_intake, y_intake, X_goal, y_goal).
    """
    rng = np.random.default_rng(42)
    X_intake, y_intake = [], []
    X_goal, y_goal = [], []

    goal_options = [2000, 2500, 3000]

    for _ in range(n_users):
        goal = float(rng.choice(goal_options))
        base_intake = rng.uniform(1500, 3200)
        consistency = rng.uniform(0.3, 0.9)   # std factor
        
        # Simulate daily totals
        totals: dict[str, float] = {}
        for d_offset in range(days + 14):
            d = (date.today() - timedelta(days=days + 14 - d_offset)).strftime("%Y-%m-%d")
            noise = rng.normal(0, base_intake * (1 - consistency))
            day_total = max(0, base_intake + noise)
            totals[d] = round(day_total, 1)

        # Build features for each day
        hourly = {h: float(rng.uniform(0, 200)) for h in range(24)}
        for d_offset in range(1, days):
            target = date.today() - timedelta(days=days - d_offset)
            feats = build_daily_features(totals, target, goal_ml=goal, hourly_pattern=hourly)
            x_vec = features_to_vector(feats)
            actual = totals.get(target.strftime("%Y-%m-%d"), 0.0)
            X_intake.append(x_vec)
            y_intake.append(actual)
            X_goal.append(x_vec)
            y_goal.append(1 if actual >= goal else 0)

    return (
        np.array(X_intake, dtype=np.float32),
        np.array(y_intake, dtype=np.float32),
        np.array(X_goal, dtype=np.float32),
        np.array(y_goal, dtype=np.int32),
    )


def _generate_reminder_data(n_samples: int = 2000):
    rng = np.random.default_rng(99)
    X, y = [], []
    for _ in range(n_samples):
        hour = rng.integers(7, 22)
        goal = float(rng.choice([2000, 2500, 3000]))
        intake_so_far = rng.uniform(0, goal * 1.1)
        last_log_min = rng.integers(10, 200)
        hourly = {h: float(rng.uniform(0, 300)) for h in range(24)}
        feats = build_reminder_features(
            current_hour=int(hour),
            intake_so_far_ml=float(intake_so_far),
            goal_ml=goal,
            last_log_minutes_ago=int(last_log_min),
            hourly_pattern=hourly,
        )
        x = reminder_features_to_vector(feats)

        # Label: reminder is "useful" if behind pace and hasn't logged in >90 min
        progress = intake_so_far / goal
        hours_elapsed = hour - 7
        expected_progress = hours_elapsed / 15 if hours_elapsed > 0 else 0
        useful = 1 if (progress < expected_progress - 0.1 and last_log_min > 90) else 0
        X.append(x)
        y.append(useful)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── Training ──────────────────────────────────────────────────────────────────

def train_models(force: bool = False) -> dict:
    """
    Train all three models. Skips if models already exist unless force=True.
    Returns training metrics dict.
    """
    if not force and INTAKE_MODEL_PATH.exists() and GOAL_MODEL_PATH.exists():
        log.info("Models already exist — skipping training (use force=True to retrain)")
        return load_metrics()

    log.info("Training ML models on synthetic data…")
    X_int, y_int, X_goal, y_goal = _generate_synthetic_data(n_users=30, days=90)
    X_rem, y_rem = _generate_reminder_data(n_samples=3000)

    # 1. Intake predictor (regression)
    intake_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=5, random_state=42,
        )),
    ])
    cv_mae = -cross_val_score(intake_pipe, X_int, y_int, cv=5, scoring="neg_mean_absolute_error")
    intake_pipe.fit(X_int, y_int)
    joblib.dump(intake_pipe, INTAKE_MODEL_PATH)

    # 2. Goal-met classifier
    goal_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=42,
        )),
    ])
    cv_auc = cross_val_score(goal_pipe, X_goal, y_goal, cv=5, scoring="roc_auc")
    goal_pipe.fit(X_goal, y_goal)
    joblib.dump(goal_pipe, GOAL_MODEL_PATH)

    # 3. Reminder scorer
    reminder_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42,
        )),
    ])
    cv_rem_auc = cross_val_score(reminder_pipe, X_rem, y_rem, cv=5, scoring="roc_auc")
    reminder_pipe.fit(X_rem, y_rem)
    joblib.dump(reminder_pipe, REMINDER_MODEL_PATH)

    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.utcnow().isoformat(),
        "intake_predictor": {
            "cv_mae_mean": round(float(cv_mae.mean()), 1),
            "cv_mae_std":  round(float(cv_mae.std()), 1),
            "training_samples": len(X_int),
        },
        "goal_classifier": {
            "cv_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_auc_std":  round(float(cv_auc.std()), 4),
            "training_samples": len(X_goal),
        },
        "reminder_scorer": {
            "cv_auc_mean": round(float(cv_rem_auc.mean()), 4),
            "cv_auc_std":  round(float(cv_rem_auc.std()), 4),
            "training_samples": len(X_rem),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log.info("Training complete: %s", metrics)
    return metrics


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


# ── Inference ─────────────────────────────────────────────────────────────────

class HydrationPredictor:
    """Lazy-loaded wrapper around trained pipelines."""

    _intake_model: Any = None
    _goal_model: Any = None
    _reminder_model: Any = None

    @classmethod
    def _ensure_loaded(cls):
        if cls._intake_model is None:
            if not INTAKE_MODEL_PATH.exists():
                train_models()
            cls._intake_model  = joblib.load(INTAKE_MODEL_PATH)
            cls._goal_model    = joblib.load(GOAL_MODEL_PATH)
            cls._reminder_model = joblib.load(REMINDER_MODEL_PATH)

    @classmethod
    def predict_daily_intake(
        cls,
        daily_totals: dict[str, float],
        goal_ml: float = 2500.0,
        hourly_pattern: dict[int, float] | None = None,
        target_date: date | None = None,
    ) -> dict:
        cls._ensure_loaded()
        target = target_date or date.today()
        feats = build_daily_features(daily_totals, target, goal_ml=goal_ml, hourly_pattern=hourly_pattern)
        x = features_to_vector(feats).reshape(1, -1)
        predicted_ml = float(cls._intake_model.predict(x)[0])
        predicted_ml = max(0, predicted_ml)

        goal_prob = float(cls._goal_model.predict_proba(x)[0][1])

        return {
            "predicted_intake_ml": round(predicted_ml, 1),
            "goal_met_probability": round(goal_prob, 3),
            "goal_met_probability_pct": round(goal_prob * 100, 1),
            "features_used": feats,
            "model_version": MODEL_VERSION,
        }

    @classmethod
    def score_reminder_urgency(
        cls,
        current_hour: int,
        intake_so_far_ml: float,
        goal_ml: float,
        last_log_minutes_ago: int,
        hourly_pattern: dict[int, float] | None = None,
    ) -> dict:
        cls._ensure_loaded()
        feats = build_reminder_features(
            current_hour=current_hour,
            intake_so_far_ml=intake_so_far_ml,
            goal_ml=goal_ml,
            last_log_minutes_ago=last_log_minutes_ago,
            hourly_pattern=hourly_pattern,
        )
        x = reminder_features_to_vector(feats).reshape(1, -1)
        urgency_prob = float(cls._reminder_model.predict_proba(x)[0][1])
        should_remind = urgency_prob > 0.5

        return {
            "urgency_score": round(urgency_prob, 3),
            "urgency_pct": round(urgency_prob * 100, 1),
            "should_remind": should_remind,
            "features": feats,
        }
