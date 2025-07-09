from langgraph.graph import StateGraph, END, add_messages
from typing import TypedDict, List, Optional, Annotated
from tools.emotion_detector import detect_emotion
from tools.memory_store import fetch_user_history, store_user_turn
from tools.selfcare_rag_suggester import suggest_care
from tools.crisis_responder import crisis_responder 
from tools.appointment_tool import appointment_booking_node


# Import the unified router - now simplified
from tools.agent_router import (
    smart_unified_router, 
    handle_user_input,
    route_state,  # This is now the single routing function
    crisis_checker_node,
)

class GraphState(TypedDict, total=False):
    # User input and identification
    user_id: Optional[str]
    text: Annotated[list, add_messages]
    current_input: str
    
    # Emotion analysis
    emotions: str
    confidence: float
    details: str
    
    # Agent control flow
    next_action: str  # "continue", "wait_for_input", "end"
    expected_input: str  # What type of input we're waiting for
    current_stage: str
    
    # Appointment booking
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
    
    # Self-care and recommendations
    suggestion: str
    tailored_prompt: str
    rag_self_care: Optional[str]
    
    # Crisis and routing
    crisis_response: Optional[str]
    route: str
    
    # Memory and context
    memory: List[str]
    emotion_context_links: List[str]
    
    # Agent decisions
    agent_router_output: Optional[str]
    router_trace: List[str]
    rag_error: Optional[str]
    web_search_error: Optional[str]
    emotion_clarification: Optional[str]
    clarification_count: int
    route_decision: Optional[str]  # Added to track routing decisions

def user_input_handler(state, new_input=None):
    """Handle when we need to wait for user input. Accepts new_input as argument for API/streamlit, or falls back to CLI input() for testing."""
    if new_input is None:
        # For CLI/testing: prompt for input
        new_input = input("Waiting for user input: ")
    
    # Update state with the new input
    updated_state = {**state, "current_input": new_input}
    
    # Process the input using the unified router's input handler
    processed_state = handle_user_input(updated_state)
    
    # If this is a follow-up response (like "Yes can you help me with that"), 
    # we need to continue the conversation flow
    if processed_state.get("next_action") == "continue":
        # Re-run emotion detection and routing for the new input
        return processed_state
    
    return processed_state

def build_graph():
    """Build the simplified graph with unified router"""
    graph = StateGraph(GraphState)

    # Crisis pre-check node
    graph.add_node("CrisisChecker", crisis_checker_node)
    # Core processing nodes
    graph.add_node("DetectEmotion", detect_emotion)
    graph.add_node("Router", smart_unified_router)
    graph.add_node("UserInputHandler", user_input_handler)
    graph.add_node("CrisisResponder", crisis_responder)
    
    # Self-care path
    graph.add_node("FetchMemory", fetch_user_history)
    graph.add_node("SuggestCare", suggest_care)
    graph.add_node("StoreMemory", store_user_turn)
    # Appointment booking
    graph.add_node("AppointmentBooking", appointment_booking_node)

    # Entry point is now CrisisChecker
    graph.set_entry_point("CrisisChecker")

    # CrisisChecker: if crisis, go to CrisisResponder, else DetectEmotion
    graph.add_conditional_edges(
        "CrisisChecker",
        lambda state: "crisis" if state.get("next_action") == "crisis" else "no_crisis",
        {
            "crisis": "CrisisResponder",
            "no_crisis": "DetectEmotion",
        }
    )

    # Main flow: Emotion Detection -> Unified Router
    graph.add_edge("DetectEmotion", "Router")

    # Centralized routing using the unified route_state function
    graph.add_conditional_edges(
        "Router",
        route_state,  # Now uses the unified routing function
        {
            "crisis": "CrisisResponder",
            "appointment": "AppointmentBooking",
            "self_care": "FetchMemory",
            "wait_for_input": "UserInputHandler",
            "end_conversation": END,
        }
    )

    # User input handling (loop back to router after input)
    graph.add_conditional_edges(
        "UserInputHandler",
        route_state,
        {
            "wait_for_input": "DetectEmotion",
            "crisis": "CrisisResponder",
            "appointment": "AppointmentBooking",
            "self_care": "FetchMemory",
            
            "end_conversation": END,
        }
    )

    # Appointment booking flow (loop back to router after booking step)
    graph.add_conditional_edges(
        "AppointmentBooking",
        route_state,
        {
            "wait_for_input": "UserInputHandler",
            "appointment": "AppointmentBooking",  # <-- Add this line
            "end_conversation": END,
        }
    )

    # Self-care flow
    graph.add_edge("FetchMemory", "SuggestCare")
    graph.add_edge("SuggestCare", END)
    graph.add_edge("CrisisResponder", END)

    # After user input:
    graph.add_edge("UserInputHandler", "FetchMemory")
    graph.add_edge("FetchMemory", "Router")

    # After each agent (CrisisResponder, AppointmentBooking, SuggestCare, etc.):
    graph.add_edge("CrisisResponder", "StoreMemory")
    graph.add_edge("AppointmentBooking", "StoreMemory")
    graph.add_edge("SuggestCare", "StoreMemory")
    graph.add_edge("StoreMemory", END)  # or loop back as needed

    return graph.compile()

def export_graph_visual(graph_obj, output_path="graph.png"):
    """Export graph visualization"""
    png_graph = graph_obj.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_graph)

# Create the graph
graph = build_graph()

if __name__ == "__main__":
    graph = build_graph()
    export_graph_visual(graph)
