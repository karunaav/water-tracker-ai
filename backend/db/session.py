"""
Database session factory — supports PostgreSQL + SQLite fallback.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv

from db.models import Base

load_dotenv()

# ── Connection URL ────────────────────────────────────────────────────────────

_RAW_URL = os.getenv("DATABASE_URL", "")

if _RAW_URL.startswith("postgresql"):
    DATABASE_URL = _RAW_URL
    _CONNECT_ARGS = {}
    _POOL_CLASS = None
else:
    # SQLite dev fallback
    Path("data").mkdir(parents=True, exist_ok=True)
    DATABASE_URL = "sqlite:///./data/water_tracker.db"
    _CONNECT_ARGS = {"check_same_thread": False}
    _POOL_CLASS = StaticPool

_engine_kwargs: dict = {
    "connect_args": _CONNECT_ARGS,
    "echo": os.getenv("APP_ENV", "development") == "development",
}
if _POOL_CLASS:
    _engine_kwargs["poolclass"] = _POOL_CLASS

engine = create_engine(DATABASE_URL, **_engine_kwargs)

# Enable WAL mode for SQLite (better concurrent read performance)
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables. Safe to call multiple times (no-op if tables exist)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and ensures it is closed."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
