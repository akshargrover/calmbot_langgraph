from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
# Import tools
from tools.emotion_detector import detect_emotion
from tools.emotion_context_search import search_emotion_context
from tools.memory_store import fetch_user_history
# from tools.mood_forecaster import forecast_mood
from tools.self_care_websearch import search_self_care_methods
from tools.selfcare_rag_suggester import rag_selfcare_suggestion
from tools.therapist_match_rag import rag_match_therapist
from tools.prompt_tailor import tailor_prompt
from tools.self_care_recommender import suggest_care
from tools.appointment_tool import offer_appointment, book_appointment
from graphviz import Digraph
from tools.agent_router import agent_router_node
from tools.crisis_responder import crisis_responder

# -----------------------------
# ✨ ROUTER FUNCTION 1: Emotion complexity
def emotion_risk_router(state):
    emotion = state.get("emotions", "").lower()
    high_risk_emotions = ["anxiety", "depression", "grief", "loneliness", "stress"]
    return "complex" if any(e in emotion for e in high_risk_emotions) else "simple"

# ✨ ROUTER FUNCTION 2: Should we use RAG or not
def use_rag_or_not(state):
    emotion = state.get("emotions", "").lower()
    positive_emotions = ["gratitude", "confidence", "joy", "hope"]
    return "skip_rag" if any(e in emotion for e in positive_emotions) else "use_rag"

def crisis_router(state):
    emotion = state.get("emotions", "").lower()
    suicidal_keywords = ["suicidal", "want to die", "end my life", "no will to live", "kill myself"]
    if any(keyword in emotion for keyword in suicidal_keywords):
        return "crisis"
    return "normal"

# -----------------------------

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
       appointment_offer: Optional[str]
       appointment_status: Optional[str]
       matched_therapist_rag: Optional[str]
       agent_router_output: Optional[str]
       router_trace: List[str]
       crisis_response: Optional[str]
       memory: List[str]
       emotion_context_links: List[str]
       self_care_articles: List[str]
       rag_self_care: Optional[str]
       matched_therapist_rag: Optional[str]
       agent_router_output: Optional[str]
       router_trace: List[str]

# -----------------------------





# -----------------------------
def build_graph():
    graph = StateGraph(GraphState)

    # Core node definitions
    graph.add_node("DetectEmotion", detect_emotion)
    graph.add_node("AgentRouter", agent_router_node)
    graph.add_node("FetchMemory", fetch_user_history)
    graph.add_node("SearchSelfCare", search_self_care_methods)
    graph.add_node("RAGSelfCare", rag_selfcare_suggestion)
    graph.add_node("RAGTherapistMatch", rag_match_therapist)
    graph.add_node("TailorPrompt", tailor_prompt)
    graph.add_node("SuggestCare", suggest_care)
    graph.add_node("OfferAppointment", offer_appointment)
    graph.add_node("BookAppointment", book_appointment)
    graph.add_node("CrisisResponder", crisis_responder)

    # Set entry point
    graph.set_entry_point("DetectEmotion")
    graph.add_edge("DetectEmotion", "AgentRouter")
    graph.add_edge("AgentRouter", "TailorPrompt")
    graph.add_edge("TailorPrompt", "FetchMemory")
    graph.add_edge("TailorPrompt", "SearchSelfCare")
    graph.add_edge("AgentRouter", "OfferAppointment")
    graph.add_edge("OfferAppointment", "RAGTherapistMatch")
    graph.add_edge("RAGTherapistMatch", "BookAppointment")
    graph.add_edge("FetchMemory", "RAGSelfCare")
    graph.add_edge("RAGSelfCare", "SuggestCare")
    graph.add_edge("AgentRouter", "CrisisResponder")
    graph.add_edge("BookAppointment", END)
    graph.add_edge("CrisisResponder", END)
    graph.add_edge("RAGSelfCare", END)
    graph.add_edge("SearchSelfCare", END)
    
    return graph.compile()


def export_graph_visual(graph_obj, output_path="graph.png"):
    
    png_graph = graph_obj.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_graph)

graph=build_graph()

if __name__ == "__main__":
    graph = build_graph()
    export_graph_visual(graph)


