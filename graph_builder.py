from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Annotated
from tools.emotion_detector import detect_emotion
from tools.memory_store import fetch_user_history, store_user_turn
from tools.selfcare_rag_suggester import suggest_care
from tools.crisis_responder import crisis_responder 
from tools.appointment_tool import appointment_booking_node
from tools.agent_router import (
    smart_unified_router, 
    route_state,
)

class GraphState(TypedDict, total=False):
    user_id: Annotated[str, ...]
    text: Annotated[list, ...]
    current_input: Annotated[str, ...]
    emotions: Annotated[str, ...]
    confidence: float
    details: str
    next_action: str
    expected_input: str
    current_stage: str
    appointment_stage: str
    appointment_offer: Optional[str]
    appointment_status: Optional[str]
    appointment_response: Optional[str]
    matched_therapist_rag: Optional[str]
    booked_therapist: Optional[str]
    booked_slot: Optional[str]
    preferred_time: Optional[str]
    preferred_therapist: Optional[str]
    location: Optional[str]
    suggestion: str
    tailored_prompt: str
    rag_self_care: Optional[str]
    crisis_response: Optional[str]
    route: str
    memory: List[str]
    emotion_context_links: List[str]
    agent_router_output: Optional[str]
    router_trace: List[str]
    rag_error: Optional[str]
    web_search_error: Optional[str]
    emotion_clarification: Optional[str]
    clarification_count: int
    route_decision: Optional[str]

# New: Input handler node (entry point)
def input_handler(state):
    # Just pass through, or could do preprocessing if needed
    return state

# New: SelfCareNode combines memory fetch, suggestion, and memory store
def self_care_node(state):
    state = fetch_user_history(state)
    state = suggest_care(state)
    state = store_user_turn(state)
    return state

# New: CrisisResponder node with memory store
def crisis_responder_node(state):
    state = crisis_responder(state)
    state = store_user_turn(state)
    return state

# New: AppointmentBooking node with memory store
def appointment_booking_node_with_memory(state):
    state = appointment_booking_node(state)
    state = store_user_turn(state)
    return state

# New: Router node that handles clarifications and routing
def router_node(state):
    result = smart_unified_router(state)
    # If clarification is needed, store the turn and return
    if result.get("next_action") == "wait_for_input":
        result = store_user_turn(result)
    return result

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("InputHandler", input_handler)
    graph.add_node("EmotionDetector", detect_emotion)
    graph.add_node("Router", router_node)
    graph.add_node("CrisisResponder", crisis_responder_node)
    graph.add_node("AppointmentBooking", appointment_booking_node_with_memory)
    graph.add_node("SelfCareNode", self_care_node)

    graph.set_entry_point("InputHandler")
    graph.add_edge("InputHandler", "EmotionDetector")
    graph.add_edge("EmotionDetector", "Router")
    graph.add_conditional_edges(
        "Router",
        route_state,
        {
            "crisis": "CrisisResponder",
            "appointment": "AppointmentBooking",
            "self_care": "SelfCareNode",
            "wait_for_input": END,
            "end_conversation": END,
        }
    )
    graph.add_edge("CrisisResponder", END)
    graph.add_edge("AppointmentBooking", END)
    graph.add_edge("SelfCareNode", END)
    return graph.compile()

def export_graph_visual(graph_obj, output_path="graph.png"):
    png_graph = graph_obj.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_graph)

graph = build_graph()

if __name__ == "__main__":
    graph = build_graph()
    export_graph_visual(graph)
