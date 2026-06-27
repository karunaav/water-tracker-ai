"""
Locust Load Test — Water Tracker AI
Simulates concurrent users logging water, checking progress, and chatting.

Run:
    locust -f load_tests/locustfile.py --host http://localhost:8000

Then open: http://localhost:8089
"""

import random
from locust import HttpUser, task, between, constant_pacing


class HydrationUser(HttpUser):
    """
    Simulates a typical app user:
    - Frequently checks today's summary
    - Logs water at intervals
    - Occasionally checks analytics
    """
    wait_time = between(0.5, 2.0)

    def on_start(self):
        """Verify the backend is reachable before running tasks."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure("Backend health check failed")

    @task(5)
    def check_today(self):
        """Most common action — checking today's progress."""
        with self.client.get("/today", catch_response=True) as r:
            if r.status_code == 200:
                data = r.json()
                if "total_ml" not in data:
                    r.failure("Missing total_ml in response")
            else:
                r.failure(f"HTTP {r.status_code}")

    @task(4)
    def log_water(self):
        """Second most common — logging water intake."""
        amount = random.choice([150, 200, 250, 300, 350, 500])
        note = random.choice(["morning", "lunch", "workout", "afternoon", ""])
        with self.client.post(
            "/log",
            json={"amount_ml": amount, "note": note, "source": "manual"},
            catch_response=True,
        ) as r:
            if r.status_code == 200:
                data = r.json()
                if data.get("status") != "logged":
                    r.failure("Expected status=logged")
            else:
                r.failure(f"HTTP {r.status_code}")

    @task(2)
    def check_analytics_week(self):
        with self.client.get("/analytics?period=week", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")

    @task(1)
    def check_analytics_month(self):
        with self.client.get("/analytics?period=month", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")

    @task(1)
    def get_logs(self):
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        with self.client.get(f"/logs?date_filter={today}", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")

    @task(1)
    def check_profile(self):
        with self.client.get("/profile", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")


class MLPowerUser(HttpUser):
    """
    Simulates users hitting the ML prediction endpoints.
    Lower weight — fewer users hit these but they're more expensive.
    """
    wait_time = between(2.0, 5.0)
    weight = 1   # 1 ML user for every ~5 regular users

    @task(3)
    def predict_intake(self):
        with self.client.get("/predict/intake", catch_response=True) as r:
            if r.status_code == 200:
                data = r.json()
                if "predicted_intake_ml" not in data:
                    r.failure("Missing prediction in response")
            else:
                r.failure(f"HTTP {r.status_code}")

    @task(2)
    def predict_reminder(self):
        with self.client.get("/predict/reminder", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")

    @task(1)
    def get_ml_metrics(self):
        with self.client.get("/ml/metrics", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")

    @task(1)
    def health_check(self):
        with self.client.get("/health", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
