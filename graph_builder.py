from langgraph.graph import StateGraph, END
from tools.emotion_detector import detect_emotion
from tools.prompt_tailor import tailor_prompt
from tools.self_care_recommender import suggest_care
from tools.mood_forecaster import forecast_mood
from tools.memory_store import store_mood
from tools.router import emotion_router

# Define the graph

graph = StateGraph()

graph.add_node("detect_emotion", detect_emotion)
graph.add_node("tailor_prompt", tailor_prompt)
graph.add_node("suggest_care", suggest_care)
graph.add_node("forecast_mood", forecast_mood)
graph.add_node("store_mood", store_mood)
graph.add_node("router", emotion_router)

# Flows

graph.add_flow("support_flow", ["tailor_prompt", "suggest_care", "forecast_mood", "store_mood", END])
graph.add_flow("positive_flow", ["tailor_prompt", "forecast_mood", "store_mood", END])
graph.add_flow("default_flow", ["tailor_prompt", "forecast_mood", END])

graph.set_entrypoint(["detect_emotion", "router"])
graph.set_router("router")

# Export for use in app.py
analyze_graph = graph.compile()
