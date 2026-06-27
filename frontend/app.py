"""
Water Tracker AI — Streamlit Frontend (v2)
New tabs: ML Predictions, System Health. Enhanced charts with moving avg + trend.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="💧 Water Tracker AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{font-size:2.1rem;font-weight:700;color:#1565c0;text-align:center;margin-bottom:0.1rem}
.sub-header{text-align:center;color:#546e7a;font-size:0.95rem;margin-bottom:1.2rem}
.chat-user{background:#1565c0;color:#ffffff;border-radius:12px 12px 2px 12px;padding:.6rem 1rem;margin:.3rem 0;text-align:right}
.chat-ai{background:#1b5e20;color:#ffffff;border-radius:12px 12px 12px 2px;padding:.6rem 1rem;margin:.3rem 0}
.metric-pill{background:#e8f5e9;border-radius:20px;padding:.2rem .8rem;font-size:.85rem;font-weight:600;color:#2e7d32;display:inline-block}
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────
def api(method: str, path: str, **kwargs):
    try:
        r = getattr(requests, method)(f"{API_URL}{path}", timeout=8, **kwargs)
        return r.json() if r.ok else None
    except Exception:
        return None

def get(path): return api("get", path)
def post(path, data): return api("post", path, json=data)
def delete(path): return api("delete", path)
def patch(path, data): return api("patch", path, json=data)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    user_name = st.text_input("Your name", value="Karuna")

    st.divider()
    profile = get("/profile") or {}
    st.markdown("### 🎯 Profile")
    new_goal = st.number_input("Daily goal (ml)", 500, 6000,
                               int(profile.get("daily_goal_ml", 2500)), 250)
    activity = st.selectbox("Activity level",
                            ["low", "moderate", "high"],
                            index=["low","moderate","high"].index(profile.get("activity_level","moderate")))
    climate = st.selectbox("Climate",
                           ["cold", "temperate", "hot"],
                           index=["cold","temperate","hot"].index(profile.get("climate","temperate")))
    if st.button("💾 Save Profile", use_container_width=True):
        patch("/profile", {"daily_goal_ml": new_goal, "activity_level": activity,
                           "climate": climate, "name": user_name})
        st.success("Profile saved!")
        st.rerun()

    st.divider()
    st.markdown("### ⚡ Quick Log")
    for amt in [150, 250, 350, 500]:
        if st.button(f"💧 {amt} ml", use_container_width=True, key=f"q{amt}"):
            post("/log", {"amount_ml": amt, "note": "Quick log"})
            st.toast(f"Logged {amt} ml 💧")
            st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">💧 Water Tracker AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered hydration · ML predictions · Real-time observability</div>',
            unsafe_allow_html=True)

tabs = st.tabs(["🏠 Dashboard", "➕ Log", "🤖 AI Coach", "📊 Analytics", "🔮 ML Insights", "🏥 System", "📋 History"])
tab_home, tab_log, tab_chat, tab_analytics, tab_ml, tab_health, tab_history = tabs


# ═══════════════════ DASHBOARD ════════════════════════════════════════════════
with tab_home:
    today = get("/today") or {}
    if today:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💧 Intake Today", f"{today['total_ml']} ml")
        c2.metric("🥤 Glasses", str(today['total_glasses']))
        c3.metric("🎯 Progress", f"{today['progress_pct']}%")
        c4.metric("⬆️ Remaining", f"{today['remaining_ml']} ml")

        # Progress bar
        pct = min(today["progress_pct"], 100)
        color = "#43a047" if pct >= 100 else "#1e88e5" if pct >= 50 else "#fb8c00"
        st.markdown(f"""
        <div style="margin:1rem 0">
          <p style="font-weight:600;margin-bottom:.3rem">Progress — {today['progress_pct']}% of {today['daily_goal_ml']} ml</p>
          <div style="background:#e0e0e0;border-radius:10px;height:30px">
            <div style="background:{color};width:{pct}%;height:30px;border-radius:10px;
                        display:flex;align-items:center;padding-left:10px;color:white;font-weight:700">
              {'🎉 Goal hit!' if pct >= 100 else f'{today["total_ml"]} ml'}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        if today['progress_pct'] >= 100:
            st.success(f"🎉 Amazing, {user_name}! You've hit your hydration goal!")
        elif today['progress_pct'] >= 70:
            st.info(f"💪 Great pace, {user_name}! {today['remaining_ml']} ml to go.")
        elif today['progress_pct'] >= 40:
            st.warning(f"⏰ Behind pace, {user_name}. Try drinking {today['remaining_ml']} ml before tonight.")
        else:
            st.error(f"🚨 {user_name}, you're significantly behind. Drink a glass now!")

        # Glass visualisation
        st.markdown("---")
        glasses_done = int(today['total_glasses'])
        glasses_left = max(0, int(today['daily_goal_ml'] / 250) - glasses_done)
        st.markdown(
            f"<div style='font-size:1.5rem;letter-spacing:3px'>{'🥤'*glasses_done}{'⬜'*min(glasses_left,10)}</div>",
            unsafe_allow_html=True)
        st.caption(f"{glasses_done} glasses done · {glasses_left} to go")
    else:
        st.error("⚠️ Cannot reach backend. Run: `cd backend && uvicorn main:app --reload`")


# ═══════════════════ LOG WATER ════════════════════════════════════════════════
with tab_log:
    st.subheader("➕ Log Water Intake")
    ca, cb = st.columns(2)
    with ca:
        amount = st.number_input("Amount (ml)", 50, 2000, 250, 50)
        note = st.text_input("Note", placeholder="e.g. pre-workout, with lunch…")
        source = st.selectbox("Source", ["manual", "reminder", "api"])
        use_now = st.checkbox("Current time", True)
        if not use_now:
            ctime = st.time_input("Custom time")
        if st.button("💧 Log Water", type="primary", use_container_width=True):
            payload = {"amount_ml": amount, "note": note, "source": source}
            if not use_now:
                from datetime import datetime as dt2
                payload["timestamp"] = dt2.combine(date.today(), ctime).isoformat()
            result = post("/log", payload)
            if result:
                st.success(f"✅ Logged {amount} ml!")
                st.rerun()
            else:
                st.error("Log failed — is the backend running?")
    with cb:
        st.markdown("#### 💡 Presets")
        presets = {"Small glass (150 ml)": 150, "Standard glass (250 ml)": 250,
                   "Large glass (350 ml)": 350, "Bottle (500 ml)": 500,
                   "Large bottle (750 ml)": 750, "1 Litre": 1000}
        for label, ml in presets.items():
            if st.button(f"💧 {label}", use_container_width=True, key=f"p{ml}"):
                post("/log", {"amount_ml": ml, "note": label})
                st.success(f"✅ {ml} ml logged!")
                st.rerun()


# ═══════════════════ AI COACH ════════════════════════════════════════════════
with tab_chat:
    st.subheader("🤖 HydroCoach — AI Hydration Assistant")
    if "history" not in st.session_state:
        st.session_state.history = []

    for m in st.session_state.history:
        cls = "chat-user" if m["role"] == "user" else "chat-ai"
        icon = "🙋" if m["role"] == "user" else "🤖"
        label = "You" if m["role"] == "user" else "HydroCoach"
        st.markdown(f'<div class="{cls}">{icon} <strong>{label}:</strong> {m["content"]}</div>',
                    unsafe_allow_html=True)

    st.divider()
    if not st.session_state.history:
        st.markdown("##### 💬 Try asking:")
        suggestions = ["How am I doing today?", "What's my predicted intake tomorrow?",
                       "Why is hydration important?", "Give me hydration tips"]
        cs = st.columns(2)
        for i, s in enumerate(suggestions):
            with cs[i % 2]:
                if st.button(s, use_container_width=True, key=f"sg{i}"):
                    st.session_state._pending = s
                    st.rerun()

    user_input = st.text_input("Ask HydroCoach…",
                               value=st.session_state.pop("_pending", ""),
                               placeholder="e.g. Am I on track today?")
    col_s, col_c = st.columns([3, 1])
    with col_s:
        send = st.button("Send 💬", type="primary", use_container_width=True)
    with col_c:
        if st.button("Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    if send and user_input.strip():
        st.session_state.history.append({"role": "user", "content": user_input})
        with st.spinner("HydroCoach is thinking…"):
            result = post("/chat", {"message": user_input, "user_name": user_name})
        reply = result["reply"] if result else "⚠️ Could not reach AI coach."
        st.session_state.history.append({"role": "assistant", "content": reply})
        st.rerun()


# ═══════════════════ ANALYTICS ════════════════════════════════════════════════
with tab_analytics:
    st.subheader("📊 Hydration Analytics")
    period = st.radio("Period", ["week", "month"], horizontal=True)
    data = get(f"/analytics?period={period}")

    if data:
        s = data["summary"]
        chart = data["chart"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📅 Active Days", s["active_days"])
        c2.metric("✅ Goal Met", s["goal_met_days"])
        c3.metric("📈 Avg Daily", f"{s['average_daily_ml']} ml")
        c4.metric("🔥 Streak", f"{s['goal_streak']}d")
        trend_arrow = "📈" if s["trend_direction"] == "up" else "📉" if s["trend_direction"] == "down" else "➡️"
        c5.metric("Trend", f"{trend_arrow} {s['trend_direction'].title()}")

        st.divider()

        # Main chart with moving average
        fig = go.Figure()
        bar_colors = ["#43a047" if v >= data["daily_goal_ml"] else "#1e88e5"
                      for v in chart["amounts_ml"]]
        fig.add_trace(go.Bar(x=chart["dates"], y=chart["amounts_ml"],
                             name="Intake", marker_color=bar_colors,
                             text=chart["amounts_ml"], textposition="outside"))
        fig.add_trace(go.Scatter(x=chart["dates"], y=chart["goal_line"],
                                 name="Goal", mode="lines",
                                 line=dict(color="#e53935", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=chart["dates"], y=chart["moving_avg_7d"],
                                 name="7-day avg", mode="lines",
                                 line=dict(color="#7b1fa2", width=2)))
        fig.update_layout(title="Daily Water Intake with 7-Day Moving Average",
                          xaxis_title="Date", yaxis_title="ml",
                          plot_bgcolor="white", paper_bgcolor="white", height=380,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"))
        st.plotly_chart(fig, use_container_width=True)

        # Donut
        cd1, cd2 = st.columns(2)
        with cd1:
            total_days = 7 if period == "week" else 30
            fig2 = go.Figure(go.Pie(
                labels=["Goal Met", "Missed"],
                values=[s["goal_met_days"], max(0, total_days - s["goal_met_days"])],
                hole=0.55, marker_colors=["#43a047", "#e0e0e0"]))
            fig2.update_layout(title=f"Goal Completion — {s['completion_rate_pct']}%", height=280)
            fig2.add_annotation(text=f"{s['completion_rate_pct']}%",
                                x=0.5, y=0.5, font_size=22, showarrow=False)
            st.plotly_chart(fig2, use_container_width=True)
        with cd2:
            fig3 = go.Figure(go.Bar(
                x=chart["dates"], y=chart["amounts_ml"],
                marker_color=["#43a047" if v >= data["daily_goal_ml"] else "#ef5350"
                               for v in chart["amounts_ml"]]))
            fig3.add_hline(y=data["daily_goal_ml"], line_dash="dash",
                           line_color="navy", annotation_text="Goal")
            fig3.update_layout(title="Goal vs Actual (Green = Met)",
                               plot_bgcolor="white", paper_bgcolor="white", height=280)
            st.plotly_chart(fig3, use_container_width=True)

        if s["best_day"]:
            st.info(f"🏆 Best day: **{s['best_day']}** with **{s['best_day_ml']} ml** logged")
    else:
        st.warning("No data yet. Start logging water to see analytics!")


# ═══════════════════ ML INSIGHTS ══════════════════════════════════════════════
with tab_ml:
    st.subheader("🔮 ML Predictions & Model Insights")

    col_p, col_r = st.columns(2)
    with col_p:
        st.markdown("#### 📅 Tomorrow's Intake Forecast")
        with st.spinner("Running intake predictor…"):
            pred = get("/predict/intake")
        if pred:
            st.metric("Predicted Intake", f"{pred['predicted_intake_ml']} ml")
            prob = pred["goal_met_probability_pct"]
            prob_color = "#43a047" if prob >= 70 else "#fb8c00" if prob >= 40 else "#e53935"
            st.markdown(f"""
            <div style="margin:.5rem 0">
              <p style="font-weight:600;margin-bottom:.3rem">Goal probability: {prob}%</p>
              <div style="background:#e0e0e0;border-radius:8px;height:22px">
                <div style="background:{prob_color};width:{prob}%;height:22px;border-radius:8px;
                            display:flex;align-items:center;padding-left:8px;color:white;font-size:.8rem;font-weight:700">
                  {prob}%
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            st.caption(f"Model: {pred.get('model_version','v1')} · Inference: {pred.get('inference_latency_ms','?')} ms")

            with st.expander("📐 Feature values used"):
                feats = pred.get("features_used", {})
                feat_df = pd.DataFrame([
                    {"Feature": k, "Value": v} for k, v in feats.items()
                ])
                st.dataframe(feat_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Prediction unavailable — check backend logs.")

    with col_r:
        st.markdown("#### ⏰ Reminder Urgency Scorer")
        with st.spinner("Scoring reminder urgency…"):
            rem = get("/predict/reminder")
        if rem:
            urgency = rem["urgency_pct"]
            urg_color = "#e53935" if urgency >= 70 else "#fb8c00" if urgency >= 40 else "#43a047"
            st.metric("Urgency Score", f"{urgency}%",
                      help="How urgently the ML model thinks you need a reminder right now")
            st.markdown(f"""
            <div style="margin:.5rem 0">
              <div style="background:#e0e0e0;border-radius:8px;height:22px">
                <div style="background:{urg_color};width:{urgency}%;height:22px;border-radius:8px;
                            display:flex;align-items:center;padding-left:8px;color:white;font-size:.8rem;font-weight:700">
                  {urgency}%
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            should = rem.get("should_remind", False)
            if should:
                st.warning("🔔 Model recommends firing a reminder now!")
            else:
                st.success("✅ No reminder needed right now.")
        else:
            st.warning("Reminder score unavailable.")

    st.divider()
    st.markdown("#### 📊 Model Training Metrics")
    metrics = get("/ml/metrics") or {}
    if metrics:
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown("**Intake Predictor (GBM)**")
            im = metrics.get("intake_predictor", {})
            st.metric("CV MAE", f"{im.get('cv_mae_mean','?')} ml",
                      help="Mean Absolute Error across 5 cross-validation folds")
            st.caption(f"Trained on {im.get('training_samples','?')} samples")
        with mc2:
            st.markdown("**Goal Classifier (RF)**")
            gm = metrics.get("goal_classifier", {})
            st.metric("CV AUC", str(gm.get("cv_auc_mean","?")),
                      help="ROC-AUC across 5 cross-validation folds")
            st.caption(f"Trained on {gm.get('training_samples','?')} samples")
        with mc3:
            st.markdown("**Reminder Scorer (RF)**")
            rm = metrics.get("reminder_scorer", {})
            st.metric("CV AUC", str(rm.get("cv_auc_mean","?")))
            st.caption(f"Trained on {rm.get('training_samples','?')} samples")

        st.caption(f"Model version: **{metrics.get('model_version','?')}** · "
                   f"Trained: {metrics.get('trained_at','?')[:10]}")

    if st.button("🔄 Retrain Models", help="Force retrain all ML models with latest data"):
        with st.spinner("Retraining… this may take 30–60 seconds"):
            result = post("/ml/retrain", {})
        if result:
            st.success("Models retrained successfully!")
            st.rerun()


# ═══════════════════ SYSTEM HEALTH ════════════════════════════════════════════
with tab_health:
    st.subheader("🏥 System Health & Observability")

    health = get("/health") or {}
    rem_status = get("/reminder/status") or {}

    if health:
        h1, h2, h3 = st.columns(3)
        db_ok = health.get("database") == "healthy"
        ml_ok = health.get("ml_models_loaded", False)
        h1.metric("⚡ Uptime", f"{health.get('uptime_seconds', 0)} s")
        h2.metric("🗄️ Database", "✅ Healthy" if db_ok else "❌ Down")
        h3.metric("🤖 ML Models", "✅ Loaded" if ml_ok else "⚠️ Not loaded")

        st.divider()
        st.markdown("#### 🔔 Reminder Effectiveness")
        rr = rem_status.get("response_rate_pct", 0)
        c1r, c2r = st.columns(2)
        c1r.metric("Response Rate", f"{rr}%",
                   help="% of reminders that led to a log within 30 min (last 14 days)")
        c1r.caption(f"Interval: every {rem_status.get('interval_minutes','?')} min")
        c2r.progress(min(rr / 100, 1.0))

        st.divider()
        st.markdown("#### 📡 Prometheus Metrics")
        st.code(f"Scrape endpoint: {API_URL}/metrics", language="bash")
        st.markdown("""
**Key metrics exposed:**
- `water_logged_ml_total` — cumulative ml logged by source
- `log_requests_total` — API call counts
- `api_request_latency_seconds` — per-endpoint latency histogram
- `ml_inference_latency_seconds` — model inference time
- `ml_predictions_total` — prediction call counts
- `reminder_fired_total` — reminders sent
- `goal_achievement_ratio` — today's progress gauge
        """)

        with st.expander("📄 Raw health response"):
            st.json(health)
    else:
        st.error("Backend unreachable.")


# ═══════════════════ HISTORY ══════════════════════════════════════════════════
with tab_history:
    st.subheader("📋 Log History")
    c1h, c2h = st.columns([2, 1])
    with c1h:
        date_filter = st.date_input("Date", date.today())
    with c2h:
        show_all = st.checkbox("Show all")

    date_str = "" if show_all else date_filter.strftime("%Y-%m-%d")
    ep = "/logs" + (f"?date_filter={date_str}" if date_str else "?limit=200")
    result = get(ep)

    if result and result["logs"]:
        logs = result["logs"]
        st.markdown(f"**{result['count']} entries**")
        daily_total = sum(l["amount_ml"] for l in logs)
        st.markdown(f"**Total: {daily_total} ml ({round(daily_total/250,1)} glasses)**")
        st.divider()

        for l in logs:
            ts = datetime.fromisoformat(l["timestamp"]).strftime("%b %d %I:%M %p")
            src_badge = f"<span style='background:#e3f2fd;padding:1px 6px;border-radius:8px;font-size:.75rem'>{l.get('source','manual')}</span>"
            ci, cd = st.columns([5, 1])
            with ci:
                st.markdown(
                    f"💧 **{l['amount_ml']} ml** &nbsp; "
                    f"<span style='color:#777'>{ts}</span> {src_badge}"
                    + (f"<br><em style='color:#888;font-size:.85rem'>{l['note']}</em>" if l.get("note") else ""),
                    unsafe_allow_html=True)
            with cd:
                if st.button("🗑️", key=f"d{l['id']}"):
                    delete(f"/log/{l['id']}")
                    st.rerun()

        if st.button("🗑️ Clear today's logs", type="secondary"):
            delete("/logs/clear-today")
            st.rerun()
    else:
        st.info("No logs found for this date. Start hydrating! 💧")
