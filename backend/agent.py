"""
LangChain AI Agent — HydroCoach (v2)
Enhanced with ML prediction context and richer system prompt.
"""

import os
from typing import Optional
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are HydroCoach, a science-backed AI hydration assistant.
You have full context of the user's hydration data for today:
- Total intake: {total_ml} ml ({total_glasses} glasses)
- Daily goal: {daily_goal_ml} ml
- Progress: {progress_pct}%
- Remaining: {remaining_ml} ml
- Log entries today: {entries}

ML Model Insights:
- Predicted intake for tomorrow: {predicted_tomorrow_ml} ml
- Probability of meeting goal today: {goal_prob_pct}%
- Trend: {trend_direction}

Guidelines:
- Be encouraging, specific, and data-driven
- Reference the user's actual numbers when giving advice
- Mention the ML prediction when relevant to motivate
- Keep responses concise (2-4 sentences) unless depth is requested
- Use 💧 occasionally, but don't overdo it
- Address the user by name
- If the user is behind pace, be constructively urgent
"""


def get_water_ai_response(
    user_message: str,
    today_stats: dict,
    user_name: str = "User",
    ml_context: Optional[dict] = None,
) -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        return _fallback_response(user_message, today_stats, user_name, ml_context)

    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=api_key,
            max_tokens=512,
            temperature=0.7,
        )

        ml_predicted = ml_context.get("predicted_intake_ml", "N/A") if ml_context else "N/A"
        ml_prob = ml_context.get("goal_met_probability_pct", "N/A") if ml_context else "N/A"
        trend = today_stats.get("trend_direction", "stable")

        system_content = SYSTEM_PROMPT.format(
            total_ml=today_stats.get("total_ml", 0),
            total_glasses=today_stats.get("total_glasses", 0),
            daily_goal_ml=today_stats.get("daily_goal_ml", 2500),
            progress_pct=today_stats.get("progress_pct", 0),
            remaining_ml=today_stats.get("remaining_ml", 2500),
            entries=today_stats.get("entries", 0),
            predicted_tomorrow_ml=ml_predicted,
            goal_prob_pct=ml_prob,
            trend_direction=trend,
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"{user_name} says: {user_message}"),
        ]
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return _fallback_response(user_message, today_stats, user_name, ml_context)


def _fallback_response(
    user_message: str,
    stats: dict,
    user_name: str,
    ml_context: Optional[dict] = None,
) -> str:
    total = stats.get("total_ml", 0)
    goal = stats.get("daily_goal_ml", 2500)
    pct = stats.get("progress_pct", 0)
    remaining = stats.get("remaining_ml", goal)
    msg_lower = user_message.lower()

    ml_snippet = ""
    if ml_context:
        prob = ml_context.get("goal_met_probability_pct", 0)
        predicted = ml_context.get("predicted_intake_ml", 0)
        ml_snippet = (
            f" Our model gives you a {prob}% chance of hitting your goal today "
            f"and predicts ~{predicted} ml tomorrow."
        )

    if any(w in msg_lower for w in ["hello", "hi", "hey"]):
        return (
            f"👋 Hey {user_name}! You've had {total} ml — {pct}% of your {goal} ml goal.{ml_snippet} 💧"
        )
    elif any(w in msg_lower for w in ["predict", "tomorrow", "forecast"]):
        if ml_context:
            return (
                f"💧 Based on your recent patterns, you're predicted to drink "
                f"{ml_context.get('predicted_intake_ml', '?')} ml tomorrow, "
                f"with a {ml_context.get('goal_met_probability_pct', '?')}% chance of hitting your goal."
            )
        return "I don't have enough data yet to make a prediction — keep logging!"
    elif any(w in msg_lower for w in ["tip", "advice", "help"]):
        return (
            "💡 Try pairing water with habits — a glass when you wake up, before each meal, "
            f"and after any screen time. You need {remaining} ml more today, {user_name}!"
        )
    elif pct >= 100:
        return f"🎉 Goal crushed, {user_name}! {total} ml logged — you're {round(total - goal)} ml over target!"
    else:
        return (
            f"💧 {user_name}: {total} ml down, {remaining} ml to go ({pct}% done).{ml_snippet}"
        )
