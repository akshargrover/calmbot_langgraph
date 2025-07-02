from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from tools.emotion_detector import detect_emotion
from tools.prompt_tailor import tailor_prompt
from tools.self_care_recommender import suggest_care
from tools.mood_forecaster import forecast_mood
from tools.memory_store import store_mood
from tools.router import emotion_router

class GraphState(TypedDict, total=False):
    user_id: str
    text: str
    emotions: str
    confidence: float
    details: str
    suggestion: str
    tailored_prompt: str
    forecast: str
    similar_past_moods: List[str]
    route: str

# Wrapper nodes to control flow

def detect_emotion_node(state: GraphState):
    new_state = detect_emotion(state)
    return "router", new_state

def router_node(state: GraphState):
    route = emotion_router(state)
    state["route"] = route
    return "tailor_prompt", state

def tailor_prompt_node(state: GraphState):
    new_state = tailor_prompt(state)
    if state.get("route") == "support_flow":
        return "suggest_care", new_state
    else:
        return "forecast_mood", new_state

def suggest_care_node(state: GraphState):
    new_state = suggest_care(state)
    return "forecast_mood", new_state

def forecast_mood_node(state: GraphState):
    new_state = forecast_mood(state)
    if state.get("route") in ["support_flow", "positive_flow"]:
        return "store_mood", new_state
    else:
        return END, new_state

def store_mood_node(state: GraphState):
    new_state = store_mood(state)
    return END, new_state

# Build the graph

graph = StateGraph(GraphState)

graph.add_node("detect_emotion", detect_emotion_node)
graph.add_node("router", router_node)
graph.add_node("tailor_prompt", tailor_prompt_node)
graph.add_node("suggest_care", suggest_care_node)
graph.add_node("forecast_mood", forecast_mood_node)
graph.add_node("store_mood", store_mood_node)

graph.add_edge("detect_emotion", "router")
graph.add_edge("router", "tailor_prompt")
graph.add_edge("tailor_prompt", "suggest_care")
graph.add_edge("tailor_prompt", "forecast_mood")
graph.add_edge("suggest_care", "forecast_mood")
graph.add_edge("forecast_mood", "store_mood")
graph.add_edge("forecast_mood", END)
graph.add_edge("store_mood", END)

graph.set_entry_point("detect_emotion")

analyze_graph = graph.compile()
