from langgraph.graph import StateGraph, END
from langgraph.router import Router
from emotion_detector import detect_emotion
from prompt_tailor import generate_prompt
from self_care_recommender import suggest_care
from mood_forecaster import forecast_mood
from memory_store import fetch_user_history

def emotion_router(state):
    emotions = state["emotions"]
    if "anxiety" in emotions.lower():
        return "anxiety_flow"
    elif "joy" in emotions.lower():
        return "joy_flow"
    else:
        return "generic_flow"

def build_graph():
    graph = StateGraph()
    graph.add_node("DetectEmotion", detect_emotion)

    # Define router
    router = Router(emotion_router)
    router.add_route("anxiety_flow", ["FetchMemory", "ForecastMood", "TailorPrompt", "SuggestCare"])
    router.add_route("joy_flow", ["TailorPrompt", "SuggestCare"])
    router.add_route("generic_flow", ["TailorPrompt", "SuggestCare"])

    # Register all nodes used in routing
    graph.add_node("FetchMemory", fetch_user_history)
    graph.add_node("ForecastMood", forecast_mood)
    graph.add_node("TailorPrompt", generate_prompt)
    graph.add_node("SuggestCare", suggest_care)

    # Plug router in
    graph.add_node("EmotionRouter", router)

    graph.set_entry_point("DetectEmotion")
    graph.add_edge("DetectEmotion", "EmotionRouter")
    graph.add_edge("EmotionRouter", END)

    return graph.compile()
