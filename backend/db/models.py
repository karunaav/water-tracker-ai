"""
SQLAlchemy ORM models — Water Tracker AI
Supports PostgreSQL (prod) and SQLite (local dev) via DATABASE_URL.
"""

from sqlalchemy import (
    Column, String, Float, DateTime, Integer, Boolean, Text, Index, func
)
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
import uuid


class Base(DeclarativeBase):
    pass


class HydrationLog(Base):
    __tablename__ = "hydration_logs"

    id          = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    amount_ml   = Column(Float, nullable=False)
    note        = Column(Text, default="")
    timestamp   = Column(DateTime, default=datetime.utcnow, nullable=False)
    date_str    = Column(String(10), nullable=False, index=True)   # "YYYY-MM-DD"
    user_id     = Column(String(64), default="default", index=True)
    source      = Column(String(32), default="manual")            # manual | reminder | api
    created_at  = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_logs_user_date", "user_id", "date_str"),
    )

    def to_dict(self) -> dict:
        return {
            "id":        self.id,
            "amount_ml": self.amount_ml,
            "note":      self.note,
            "timestamp": self.timestamp.isoformat(),
            "date":      self.date_str,
            "user_id":   self.user_id,
            "source":    self.source,
        }


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id                      = Column(String(64), primary_key=True, default="default")
    name                    = Column(String(128), default="User")
    daily_goal_ml           = Column(Float, default=2500.0)
    reminder_interval_min   = Column(Integer, default=60)
    weight_kg               = Column(Float, nullable=True)
    activity_level          = Column(String(32), default="moderate")  # low|moderate|high
    climate                 = Column(String(32), default="temperate")  # cold|temperate|hot
    created_at              = Column(DateTime, default=datetime.utcnow)
    updated_at              = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id":                    self.id,
            "name":                  self.name,
            "daily_goal_ml":         self.daily_goal_ml,
            "reminder_interval_min": self.reminder_interval_min,
            "weight_kg":             self.weight_kg,
            "activity_level":        self.activity_level,
            "climate":               self.climate,
        }


class ReminderLog(Base):
    __tablename__ = "reminder_logs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    timestamp   = Column(DateTime, default=datetime.utcnow)
    message     = Column(Text)
    responded   = Column(Boolean, default=False)   # did user log within 30 min?
    response_ml = Column(Float, nullable=True)      # how much they logged after

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "timestamp":   self.timestamp.isoformat(),
            "message":     self.message,
            "responded":   self.responded,
            "response_ml": self.response_ml,
        }


class MLPrediction(Base):
    """Stores ML model predictions for audit trail and retraining."""
    __tablename__ = "ml_predictions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    prediction_type = Column(String(64))   # reminder_timing | daily_intake | goal_met
    input_features  = Column(Text)         # JSON of features used
    predicted_value = Column(Float)
    confidence      = Column(Float, nullable=True)
    actual_value    = Column(Float, nullable=True)   # filled in later for evaluation
    timestamp       = Column(DateTime, default=datetime.utcnow)
    model_version   = Column(String(32), default="v1.0")

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "prediction_type":  self.prediction_type,
            "predicted_value":  self.predicted_value,
            "confidence":       self.confidence,
            "actual_value":     self.actual_value,
            "timestamp":        self.timestamp.isoformat(),
            "model_version":    self.model_version,
        }
