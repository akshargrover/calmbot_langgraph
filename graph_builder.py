from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Annotated
from tools.emotion_detector import detect_emotion
from tools.memory_store import fetch_user_history, store_user_turn
from tools.selfcare_rag_suggester import suggest_care
from tools.crisis_responder import crisis_responder 
from tools.appointment_tool import appointment_booking_node, get_appointment_input_prompt
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
    crisis_response: Optional[str]
    route: str
    memory: List[str]
    emotion_context_links: List[str]
    agent_router_output: Optional[str]
    router_trace: List[str]
    rag_error: Optional[str]
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

# UserInput node for appointment loop
def user_input_node(state):
    """
    Enhanced user input node that properly handles user responses
    during appointment booking flow
    """
    current_input = state.get("current_input", "")
    expected_input = state.get("expected_input", "")
    if not current_input:
        # Use the new appointment input prompt function
        prompt = get_appointment_input_prompt(state)
        return {
            **state,
            "next_action": "wait_for_input",
            "agent_output": prompt
        }
    # Use the input and continue
    return {
        **state,
        "user_input": current_input,  # Pass new input to the next node
        "current_input": "",          # Clear after use
        "next_action": "continue_appointment"
    }
    
def appointment_flow_condition(state):
    """
    Improved conditional routing for appointment booking flow
    """
    appointment_stage = state.get("appointment_stage", "")
    next_action = state.get("next_action", "")
    
    # If we're waiting for input, go to UserInput node
    if next_action == "wait_for_input":
        return "user_input"
    
    # If appointment is complete, end the flow
    if appointment_stage in ["complete", "declined", "booking_confirmed"]:
        return "complete"
    
    # If we have input to process, continue with appointment booking
    if next_action == "continue_appointment":
        return "appointment"
    
    # Default to ending
    return "complete"

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("InputHandler", input_handler)
    graph.add_node("EmotionDetector", detect_emotion)
    graph.add_node("Router", router_node)
    graph.add_node("CrisisResponder", crisis_responder_node)
    graph.add_node("AppointmentBooking", appointment_booking_node_with_memory)
    graph.add_node("SelfCareNode", self_care_node)
    graph.add_node("UserInput", user_input_node)

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
    
    # Improved appointment booking flow
    # After AppointmentBooking, always go to UserInput
    graph.add_edge("AppointmentBooking", "UserInput")
    
    # UserInput node routing
    graph.add_conditional_edges(
        "UserInput",
        appointment_flow_condition,
        {
            "appointment": "AppointmentBooking",
            "complete": END
        }
    )
    
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

def update_appointment_booking_node(state):
    """
    Updated appointment booking node that properly handles the user input
    """
    # Get user input from the state
    user_input = state.get("user_input", "")
    appointment_stage = state.get("appointment_stage", "initial")
    
    # Create a new state with the user input
    updated_state = {**state}
    if user_input:
        updated_state["current_input"] = user_input
    
    # Call the original appointment booking function
    result = appointment_booking_node(updated_state)
    
    # Make sure we clear the user_input after processing
    result["user_input"] = ""
    
    return result