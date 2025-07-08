from typing import Dict, Tuple, Optional
from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
from pydantic import BaseModel


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file or directly.")


class UnifiedRouter:
    """
    Single, robust router that handles all routing decisions for the AI agent
    """
    def __init__(self):
        self.crisis_keywords = [
            "suicidal", "want to die", "end my life", "no will to live", 
            "kill myself", "hurt myself", "can't go on", "no point living",
            "end it all", "suicide", "self harm"
        ]
        
        self.therapy_emotions = [
            "anxiety", "depression", "grief", "loneliness", "stress", 
            "trauma", "panic", "overwhelmed", "hopeless", "despair"
        ]
        
        self.vague_responses = [
            "fine", "okay", "nothing", "idk", "dunno", "whatever", 
            "sure", "yes", "no", "neutral", "unclear", "mixed", "confused"
        ]

    @tool
    def extract_text_from_state(self, state: Dict) -> str:
        """Extract and normalize text from state"""
        
        text = state.get("text", [])
        if isinstance(text, list):
            text = " ".join(x.content if hasattr(x, "content") else str(x) for x in text)
        return text.strip().lower()
    
    @tool
    def validate_input(self, state: Dict) -> Tuple[bool, str]:
        """
        Validate if we have sufficient information to proceed
        Returns: (is_valid, clarification_message)
        """
        text = self.extract_text_from_state(state)
        emotions = state.get("emotions", "").strip().lower()
        
        # Check for minimal input
        if len(text) < 3:
            return False, "I'd like to understand better. Can you share more about what's on your mind?"
        
        # Check for vague responses
        if text in self.vague_responses:
            return False, "I want to help you effectively. Could you tell me more about what's been happening or how you've been feeling?"
        
        # Check for missing or unclear emotions
        if not emotions or emotions in self.vague_responses:
            return False, "I'm having trouble understanding how you're feeling. Could you describe your emotions or what's troubling you?"
        
        return True, ""
    @tool
    def check_crisis(self, state: Dict) -> bool:
        """Check if user is in crisis situation"""
        text = self.extract_text_from_state(state)
        emotions = state.get("emotions", "").lower()
        
        combined_text = f"{text} {emotions}"
        return any(keyword in combined_text for keyword in self.crisis_keywords)
    @tool
    def check_needs_therapy(self, state: Dict) -> bool:
        """Check if user might benefit from therapy"""
        text = self.extract_text_from_state(state)
        emotions = state.get("emotions", "").lower()
        
        # Check for explicit therapy mentions
        therapy_mentions = ["therapy", "therapist", "appointment", "counseling", "counselor"]
        if any(mention in text for mention in therapy_mentions):
            return True
        
        # Check for high-risk emotions
        return any(emotion in emotions for emotion in self.therapy_emotions)
    @tool
    def determine_route(self, state: Dict) -> str:
        """
        Main routing logic - determines the next action
        Returns: "crisis", "appointment", "self_care", "wait_for_input", or "end"
        """
        # First, validate input
        is_valid, clarification_msg = self.validate_input(state)
        if not is_valid:
            return "wait_for_input"
        
        # Check for crisis first (highest priority)
        if self.check_crisis(state):
            return "crisis"
        
        # Check if user needs therapy
        if self.check_needs_therapy(state):
            return "appointment"
        
        # Default to self-care
        return "self_care"
    
    
    @tool
    def generate_clarification_message(self, state: Dict) -> dict:
        """Generate appropriate clarification message based on context and signal if agent should wait for input"""
        text = self.extract_text_from_state(state)
        emotions = state.get("emotions", "").lower()
        clarification_count = state.get("clarification_count", 0)
        
        # If we've asked for clarification too many times, provide fallback and signal to wait for input
        if clarification_count >= 2:
            return {
                "message": "I'm here to support you. Would you like to explore some self-care resources, or would you prefer to speak with a professional?",
                "wait_for_input": True
            }
        # Context-specific clarifications
        if not emotions or emotions in self.vague_responses:
            return {
                "message": "I want to understand your feelings better. Could you describe what emotions you're experiencing right now?",
                "wait_for_input": False
            }
        if len(text) < 3:
            return {
                "message": "I'd like to help you more effectively. Can you tell me more about what's been on your mind?",
                "wait_for_input": False
            }
        # Default clarification
        return {
            "message": "I want to make sure I give you the best support. Could you share more details about your situation?",
            "wait_for_input": False
        }
    
    @tool
    def route(self, state: Dict) -> Dict:
        """
        Main routing function that updates state based on routing decision
        """
        route_decision = self.determine_route(state)
        clarification_count = state.get("clarification_count", 0)
        response = {
            **state,
            "route": route_decision,
            "clarification_count": clarification_count
        }
        if route_decision == "wait_for_input":
            clarification_result = self.generate_clarification_message(state)
            if clarification_result.get("wait_for_input"):
                response.update({
                    "agent_router_output": clarification_result["message"] + " Please provide more information to continue.",
                    "next_action": "end",
                    "expected_input": "clarification",
                    "clarification_count": clarification_count + 1
                })
            else:
                response.update({
                    "agent_router_output": clarification_result["message"],
                    "next_action": "wait_for_input",
                    "expected_input": "clarification",
                    "clarification_count": clarification_count + 1
                })
        elif route_decision == "crisis":
            response.update({
                "agent_router_output": "I can see you're going through a really difficult time. Let me connect you with immediate support.",
                "next_action": "CrisiResponder"
            })
        elif route_decision == "appointment":
            emotions = state.get("emotions", "")
            response.update({
                "agent_router_output": f"I understand you're feeling {emotions}. Let me help you find professional support that could be really beneficial.",
                "next_action": "AppointmentBooking"
            })
        elif route_decision == "self_care":
            emotions = state.get("emotions", "")
            response.update({
                "agent_router_output": f"I hear that you're feeling {emotions}. Let me suggest some self-care strategies that might help.",
                "next_action": "SuggestCare"
            })
        return {
            **response,
            "route_decision": route_decision
        }

# Create singleton instance
unified_router = UnifiedRouter()

class StateInput(BaseModel):
    state: dict

# Main router function to use in your graph
@tool()
def smart_unified_router(state: Dict) -> Dict:
    """
    Main router function to be used in the StateGraph
    """
    return unified_router.route(state)

# Helper function to handle user input responses
@tool
def handle_user_input(state: Dict) -> Dict:
    """
    Handle user input and update state accordingly
    """
    current_input = state.get("current_input", "")
    expected_input = state.get("expected_input", "")
    
    if not current_input:
        # Still waiting for input
        return {**state, "next_action": "wait_for_input"}
    
    # Process different types of expected input
    if expected_input == "clarification":
        # Add new input to text history and re-route
        text_history = state.get("text", [])
        if not isinstance(text_history, list):
            text_history = [text_history]
        text_history.append(current_input)
        
        return {
            **state,
            "text": text_history,
            "current_input": "",
            "next_action": "continue",
            "expected_input": ""
        }
    
    elif expected_input == "appointment_response":
        # Handle appointment booking responses
        return handle_appointment_response(state, current_input)
    
    elif expected_input == "booking_details":
        # Handle booking detail responses
        return handle_booking_details(state, current_input)
    
    # Default: continue with current input
    return {
        **state,
        "current_input": "",
        "next_action": "continue",
        "expected_input": ""
    }


@tool
def handle_appointment_response(state: Dict, user_input: str) -> Dict:
    """Handle appointment booking responses"""
    user_input_lower = user_input.lower()
    
    if any(word in user_input_lower for word in ["yes", "sure", "okay", "book", "schedule"]):
        return {
            **state,
            "appointment_response": "yes",
            "current_input": "",
            "next_action": "continue",
            "expected_input": ""
        }
    else:
        return {
            **state,
            "appointment_response": "no",
            "current_input": "",
            "next_action": "continue",
            "expected_input": ""
        }


@tool
def handle_booking_details(state: Dict, user_input: str) -> Dict:
    """Handle booking detail responses"""
    user_input_lower = user_input.lower()
    
    # Parse time preferences
    if "morning" in user_input_lower:
        state["preferred_time"] = "morning"
    elif "afternoon" in user_input_lower:
        state["preferred_time"] = "afternoon"
    elif "evening" in user_input_lower:
        state["preferred_time"] = "evening"
    
    # Parse location preferences
    if "online" in user_input_lower or "virtual" in user_input_lower:
        state["location"] = "online"
    elif "in-person" in user_input_lower or "office" in user_input_lower:
        state["location"] = "in-person"
    
    return {
        **state,
        "current_input": "",
        "next_action": "continue",
        "expected_input": ""
    }

# Conditional routing functions for StateGraph
def input_flow_condition(state: Dict) -> str:
    """Conditional routing for input flow"""
    next_action = state.get("next_action", "continue")
    if next_action == "wait_for_input":
        return "wait_for_input"
    elif next_action == "continue":
        return "continue"
    else:
        return "end"