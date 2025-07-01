from langgraph.graph import StateGraph, END
from langgraph.router import Router
from tools.emotion_detector import detect_emotion
from tools.prompt_tailor import generate_prompt
from tools.self_care_recommender import suggest_care
from tools.mood_forecaster import forecast_mood
from memory_store import fetch_user_history

def emotion_router(state):
    emotion = state.get("emotions", "other").lower()
    if emotion in ["anxiety", "fear", "shame"]:
        return "support_flow"
    elif emotion in ["joy", "gratitude", "surprise"]:
        return "positive_flow"
    else:
        return "default_flow"

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
