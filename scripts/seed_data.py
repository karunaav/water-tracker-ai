#!/usr/bin/env python3
"""
Seed script — generates 30 days of realistic demo data.
Run: python scripts/seed_data.py
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from datetime import date, timedelta, datetime
from db.session import init_db, SessionLocal
from db import crud

def seed():
    init_db()
    db = SessionLocal()

    try:
        # Create default profile
        crud.update_profile(db, "default", name="Karuna", daily_goal_ml=2500.0,
                            activity_level="moderate", climate="temperate")

        today = date.today()
        total_seeded = 0

        for days_ago in range(30, 0, -1):
            d = today - timedelta(days=days_ago)

            # Simulate realistic variation: some good days, some bad
            quality = random.choices(["great", "good", "poor"], weights=[0.3, 0.5, 0.2])[0]
            if quality == "great":
                target = random.uniform(2500, 3200)
            elif quality == "good":
                target = random.uniform(1800, 2600)
            else:
                target = random.uniform(600, 1700)

            # Spread intake across the day in 3–7 logs
            n_logs = random.randint(3, 7)
            amounts = [target / n_logs + random.gauss(0, 50) for _ in range(n_logs)]
            amounts = [max(100, a) for a in amounts]

            for i, amt in enumerate(amounts):
                hour = random.randint(7, 21)
                minute = random.randint(0, 59)
                ts = datetime(d.year, d.month, d.day, hour, minute)
                crud.create_log(db, amount_ml=round(amt, 1),
                                note=random.choice(["", "morning", "with meal", "post-workout", "evening"]),
                                timestamp=ts, source="manual")
                total_seeded += 1

        print(f"✅ Seeded {total_seeded} log entries across 30 days")

    finally:
        db.close()


if __name__ == "__main__":
    seed()
