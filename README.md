# 💧 Water Tracker

## 🏗️ Project Structure
<img width="1907" height="942" alt="Screenshot 2026-06-27 181141" src="https://github.com/user-attachments/assets/fad367c2-ec0c-4733-bf27-5ec7004acd8a" />


```
water-tracker-ai/
├── backend/
│   ├── main.py               # FastAPI v2 — 15+ endpoints, lifespan hooks
│   ├── agent.py              # LangChain + ML context injection
│   ├── analytics.py          # DB-backed analytics with 7d moving avg + trend
│   ├── scheduler.py          # APScheduler + ML-based smart reminder suppression
│   ├── db/
│   │   ├── models.py         # SQLAlchemy ORM (4 tables)
│   │   ├── session.py        # PG + SQLite connection factory
│   │   ├── crud.py           # Repository layer (15+ operations)
│   │   └── __init__.py
│   ├── ml/
│   │   ├── features.py       # Feature engineering (cyclical encoding, rolling stats)
│   │   ├── predictor.py      # GBM + RF pipelines, training, inference
│   │   └── __init__.py
│   └── middleware/
│       ├── logging.py        # structlog + RequestTracingMiddleware
│       ├── metrics.py        # Prometheus counters, histograms, gauges
│       └── __init__.py
├── frontend/
│   └── app.py                # Streamlit (7 tabs including ML Insights + System Health)
├── tests/
│   ├── test_v2.py            # 30+ pytest tests (features, CRUD, ML, API)
│   └── test_backend.py       # Original v1 tests (kept for regression)
├── load_tests/
│   └── locustfile.py         # Locust with HydrationUser + MLPowerUser personas
├── monitoring/
│   ├── prometheus.yml        # Scrape config
│   └── grafana_datasource.yml
├── scripts/
│   └── seed_data.py          # 30-day synthetic data seeder
├── docker-compose.yml        # Full stack: API + DB + Prometheus + Grafana + Frontend
├── Dockerfile                # FastAPI backend image
├── Dockerfile.frontend       # Streamlit image
├── requirements.txt
├── .env.example
└── start.sh
```

---

## 🚀 Quick Start
<img width="1911" height="947" alt="Screenshot 2026-06-27 181202" src="https://github.com/user-attachments/assets/6b5c4010-6b50-4111-85e3-8d2ab40dd375" />


### Local dev (SQLite, no Docker needed)

```bash
cd water-tracker-ai
pip install -r requirements.txt
cp .env.example .env         # Add ANTHROPIC_API_KEY (optional)

# Seed 30 days of demo data
python scripts/seed_data.py

# Terminal 1 — Backend
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend && streamlit run app.py
```

### Full stack with Docker + PostgreSQL + Prometheus + Grafana
<img width="1913" height="952" alt="Screenshot 2026-06-27 181229" src="https://github.com/user-attachments/assets/9a5b698c-1ec3-47b6-a7d9-fd496d61fdfa" />

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Swagger | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

---
<img width="1915" height="955" alt="Screenshot 2026-06-27 181241" src="https://github.com/user-attachments/assets/c1be8547-cf66-431d-9368-b3fb42cfc94e" />

## 🤖 ML Models
<img width="1908" height="936" alt="Screenshot 2026-06-27 181252" src="https://github.com/user-attachments/assets/f57463fb-a2b1-4569-a47b-a4b02be16d46" />

Three scikit-learn pipelines with StandardScaler preprocessing:

### 1. Daily Intake Predictor (GBM Regression)
- **Features:** 3d/7d/14d rolling averages, goal attainment rate, day-of-week (sin/cos), trend slope, consistency score, peak-hour signal
- **Model:** `GradientBoostingRegressor(n_estimators=200, max_depth=4)`
- **Metric:** CV MAE ~280 ml
- **Endpoint:** `GET /predict/intake`

### 2. Goal-Met Classifier (Random Forest)
- **Same features as above**
- **Model:** `RandomForestClassifier(n_estimators=150, class_weight="balanced")`
- **Metric:** CV AUC ~0.79
- **Output:** P(goal met today)

### 3. Smart Reminder Scorer (Random Forest)
- **Features:** hour-of-day (sin/cos), intake progress, pace deficit, last-log recency, typical intake at this hour
- **Usage:** Suppresses reminders when user is on track — reduces notification fatigue
- **Endpoint:** `GET /predict/reminder`

All models persist to `data/models/` via joblib and retrain via `POST /ml/retrain`.

---

## 📡 Observability
<img width="1911" height="943" alt="Screenshot 2026-06-27 181304" src="https://github.com/user-attachments/assets/4994fcb9-cb9b-46fe-95cb-940e834474d8" />

### Prometheus metrics
```
water_logged_ml_total{source}       # cumulative ml by source
api_request_latency_seconds{endpoint} # p50/p95/p99 per route
ml_inference_latency_seconds{model}   # model inference timing
goal_achievement_ratio              # today's progress (gauge)
reminder_fired_total               # reminders sent
```

### Structured logging
```json
{"level": "info", "timestamp": "2026-06-27T10:23:11Z", "event": "request_completed",
 "trace_id": "a3f9b2c1", "method": "POST", "path": "/log",
 "status_code": 200, "latency_ms": 4.7}
```

Every response includes `X-Trace-ID` and `X-Latency-MS` headers.

---

## 🧪 Tests

```bash
# All tests
pytest tests/ -v --tb=short

# Just ML tests
pytest tests/test_v2.py::TestMLModels -v

# Just API tests
pytest tests/test_v2.py::TestAPI -v
```

30+ test cases covering: feature engineering, ML training/inference, DB CRUD, API endpoints, Pydantic validation, analytics, header propagation.

---

## ⚡ Load Testing

```bash
pip install locust
locust -f load_tests/locustfile.py --host http://localhost:8000
# Open http://localhost:8089 → set users and ramp rate
```

Two personas:
- **HydrationUser** — log water, check today, get analytics (weight=5)
- **MLPowerUser** — hit prediction endpoints under load (weight=1)

---
<img width="1915" height="970" alt="Screenshot 2026-06-27 181312" src="https://github.com/user-attachments/assets/b8a7ace4-fefc-476a-9ab1-45f4cfddb65c" />


## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Detailed health (DB, ML, uptime) |
| GET | `/metrics` | Prometheus scrape |
| POST | `/log` | Log intake (validated, traced) |
| GET | `/today` | Today's summary |
| GET | `/logs` | History with date filter |
| GET | `/analytics` | Week/month with moving avg + trend |
| POST | `/chat` | AI coach (ML-enriched context) |
| GET | `/predict/intake` | GBM intake forecast for tomorrow |
| GET | `/predict/reminder` | RF reminder urgency score |
| GET | `/ml/metrics` | Model CV scores and metadata |
| POST | `/ml/retrain` | Force model retraining |
| GET | `/profile` | User profile |
| PATCH | `/profile` | Update goal, activity, climate |
| DELETE | `/log/{id}` | Delete a log entry |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.111 + Uvicorn |
| ORM | SQLAlchemy 2.0 |
| Database | PostgreSQL 16 / SQLite (dev) |
| AI | LangChain + Claude Sonnet 4.6 |
| ML | scikit-learn (GBM + RF pipelines) |
| Scheduling | APScheduler + ML urgency scoring |
| Observability | Prometheus + structlog |
| Frontend | Streamlit + Plotly |
| Load testing | Locust |
| Testing | pytest (30+ tests) |
| Deployment | Docker Compose |

---

