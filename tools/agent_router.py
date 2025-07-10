from typing import Dict, Tuple, Optional, List
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in a .env file or directly.")


# Abstract base class for agents
class BaseAgent(ABC):
    """Base class for all specialized agents"""
    
    @abstractmethod
    def process(self, state: Dict) -> Dict:
        """Process the state and return updated state with agent output"""
        pass
    
    @abstractmethod
    def get_tools(self) -> List[str]:
        """Return list of tools this agent can use"""
        pass


# Crisis Support Agent
class CrisisAgent(BaseAgent):
    """Handles crisis situations with appropriate resources and tools"""
    
    def get_tools(self) -> List[str]:
        return ["crisis_hotline_lookup", "emergency_contacts", "crisis_resources"]
    
    def process(self, state: Dict) -> Dict:
        """Process crisis situation with specialized tools"""
        try:
            # Get crisis resources based on user location/context
            crisis_resources = [
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741",
                "International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"
            ]
            
            # Get emergency contacts based on location
            emergency_contacts = [
                "Emergency Services: 911 (US) / 112 (EU) / 100 (India)",
                "Local Crisis Centers: Check your local directory"
            ]
            
            # Generate personalized crisis response
            user_context = state.get("text", "")
            emotions = state.get("emotions", "")
            
            response = f"I'm really concerned about you and want to help. Based on what you've shared, here are immediate resources:\n\n"
            
            for resource in crisis_resources:
                response += f"• {resource}\n"
            
            response += f"\nEmergency contacts:\n"
            for contact in emergency_contacts:
                response += f"• {contact}\n"
            
            response += f"\nYou matter, and there are people who want to help you through this difficult time."
            
            return {
                **state,
                "agent_output": response,
                "agent_used": "crisis",
                "tools_used": self.get_tools(),
                "next_action": "crisis_handled"
            }
        except Exception as e:
            # Fallback to basic crisis response
            return {
                **state,
                "agent_output": "I'm really sorry you're feeling this way. You're not alone — there are people who care about you and want to help.\n\n💙 Please reach out to someone you trust or contact a mental health professional.\n\n**If you're in immediate danger**, please call emergency services or reach out to a suicide prevention hotline:\n - 🇺🇸 USA: 988\n - 🇮🇳 India: 9152987821 (AASRA)\n\nYou're valued and your life matters. Talking to someone can make a big difference.",
                "agent_used": "crisis",
                "tools_used": [],
                "error": str(e)
            }


# Appointment Agent
class AppointmentAgent(BaseAgent):
    """Handles therapy appointment booking with specialized tools"""
    
    def get_tools(self) -> List[str]:
        return ["therapist_finder", "appointment_scheduler", "insurance_checker", "availability_checker"]
    
    def process(self, state: Dict) -> Dict:
        """Delegate to the actual appointment booking tool and ensure agent_output is set."""
        from tools.appointment_tool import appointment_booking_node
        result = appointment_booking_node(state)
        # If appointment_status is present but agent_output is not, map it
        if "agent_output" not in result and "appointment_status" in result:
            result["agent_output"] = result["appointment_status"]
        return result


# Self-Care Agent
class SelfCareAgent(BaseAgent):
    """Handles self-care recommendations with specialized tools"""
    
    def get_tools(self) -> List[str]:
        return ["rag_search", "emotion_analyzer", "personalized_recommender", "wellness_tracker"]
    
    def process(self, state: Dict) -> Dict:
        """Delegate to the actual self-care suggestion tool and ensure agent_output is set."""
        from tools.selfcare_rag_suggester import suggest_care
        result = suggest_care(state)
        # If agent_output is not set but suggestion is, map it
        if "agent_output" not in result and "suggestion" in result:
            result["agent_output"] = result["suggestion"]
        return result


class UnifiedRouter:
    """
    Enhanced unified router that actually uses specialized agents and tools
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
        
        # Initialize specialized agents
        self.crisis_agent = CrisisAgent()
        self.appointment_agent = AppointmentAgent()
        self.self_care_agent = SelfCareAgent()

    def extract_text_from_state(self, state: Dict) -> str:
        """Extract and normalize text from state"""
        text = state.get("text", [])
        if isinstance(text, list):
            text = " ".join(x.content if hasattr(x, "content") else str(x) for x in text)
        return text.strip().lower()
    
    def validate_input(self, state: Dict) -> Tuple[bool, str]:
        """Validate if we have sufficient information to proceed"""
        text = self.extract_text_from_state(state)
        emotions = state.get("emotions") or ""
        if isinstance(emotions, list):
            emotions = " ".join(str(e) for e in emotions)
        emotions = emotions.strip().lower()
        
        if len(text) < 3:
            return False, "I'd like to understand better. Can you share more about what's on your mind?"
        
        if text in self.vague_responses:
            return False, "I want to help you effectively. Could you tell me more about what's been happening or how you've been feeling?"
        
        if not emotions or emotions in self.vague_responses:
            return False, "I'm having trouble understanding how you're feeling. Could you describe your emotions or what's troubling you?"
        
        return True, ""
    
    def check_crisis(self, state: Dict) -> bool:
        """Check if user is in crisis situation"""
        text = self.extract_text_from_state(state)
        emotions = state.get("emotions") or ""
        if isinstance(emotions, list):
            emotions = " ".join(str(e) for e in emotions)
        emotions = emotions.strip().lower()
        
        combined_text = f"{text} {emotions}"
        return any(keyword in combined_text for keyword in self.crisis_keywords)
    
    def check_needs_therapy(self, state: Dict) -> bool:
        """Check if user might benefit from therapy"""
        text = self.extract_text_from_state(state)
        emotions = state.get("emotions") or ""
        if isinstance(emotions, list):
            emotions = " ".join(str(e) for e in emotions)
        emotions = emotions.strip().lower()        
        therapy_mentions = ["therapy", "therapist", "appointment", "counseling", "counselor"]
        if any(mention in text for mention in therapy_mentions):
            return True
        
        return any(emotion in emotions for emotion in self.therapy_emotions)
    
    def determine_route(self, state: Dict) -> str:
        """Determine which agent should handle the request"""
        # Only require emotion detection for self-care or crisis if emotion is missing
        text = self.extract_text_from_state(state)
        emotions = state.get("emotions") or ""
        if isinstance(emotions, list):
            emotions = " ".join(str(e) for e in emotions)
        emotions = emotions.strip().lower()
        is_valid, _ = self.validate_input(state)
        if not is_valid:
            return "wait_for_input"

        # If appointment is directly requested, route to appointment (even if emotion is missing)
        if self.check_needs_therapy(state):
            return "appointment"

        # If self-care or crisis is needed but emotion is missing, trigger emotion detection
        if (not emotions or emotions in self.vague_responses):
            return "detect_emotion"

        if self.check_crisis(state):
            return "crisis"

        return "self_care"
    
    def route(self, state: Dict) -> Dict:
        """Main routing function that delegates to appropriate agents"""
        route_decision = self.determine_route(state)
        state.setdefault("router_trace", []).append(f"Routing decision: {route_decision}")

        # Handle input validation
        if route_decision == "wait_for_input":
            return self._handle_input_validation(state)
        
        # Delegate to appropriate agent
        if route_decision == "crisis":
            return self.crisis_agent.process(state)
        elif route_decision == "appointment":
            return self.appointment_agent.process(state)
        elif route_decision == "self_care":
            return self.self_care_agent.process(state)
        
        # Fallback
        return {
            **state,
            "agent_output": "I'm not sure how to help with that. Could you tell me more about what you're going through?",
            "agent_used": "router",
            "tools_used": [],
            "next_action": "wait_for_input"
        }
    
    def _handle_input_validation(self, state: Dict) -> Dict:
        """Handle input validation and clarification"""
        clarification_count = state.get("clarification_count", 0)
        
        if clarification_count >= 2:
            return {
                **state,
                "agent_output": "I'm here to support you. Would you like to explore some self-care resources, or would you prefer to speak with a professional? If you need more help, please start a new conversation.",
                "agent_used": "router",
                "tools_used": [],
                "next_action": "end",
                "clarification_count": clarification_count + 1
            }
        
        _, clarification_msg = self.validate_input(state)
        
        return {
            **state,
            "agent_output": clarification_msg,
            "agent_used": "router",
            "tools_used": [],
            "next_action": "wait_for_input",
            "expected_input": "clarification",
            "clarification_count": clarification_count + 1
        }

# Create singleton instance
unified_router = UnifiedRouter()

# Main router function to use in your graph
def smart_unified_router(state: Dict) -> Dict:
    """Main router function that uses agents and tools"""
    return unified_router.route(state)

# Keep your existing helper functions for compatibility
def route_state(state: Dict) -> str:
    """Single routing function that handles all conditional edge routing"""
    next_action = state.get("next_action", "")
    
    route_mapping = {
        "crisis_handled": "crisis",
        "appointment_processed": "appointment",
        "self_care_provided": "self_care",
        "wait_for_input": "wait_for_input",
        "end": "end_conversation"
    }
    
    return route_mapping.get(next_action, "end_conversation")

def handle_user_input(state: Dict) -> Dict:
    """Handle user input and update state accordingly"""
    current_input = state.get("current_input", "")
    expected_input = state.get("expected_input", "")
    
    if not current_input:
        return {**state, "next_action": "wait_for_input"}
    
    # Add current input to conversation history
    text_history = state.get("text", [])
    if not isinstance(text_history, list):
        text_history = []
    
    text_history.append(HumanMessage(current_input))
    
    if expected_input == "clarification":
        return {
            **state,
            "text": text_history,
            "current_input": current_input,
            "next_action": "continue",
            "expected_input": ""
        }
    
    elif expected_input == "appointment_response":
        # Fix: set appointment_stage to user_responded so the booking node processes the reply
        return {
            **state,
            "text": text_history,
            "appointment_response": current_input,
            "appointment_stage": "user_responded",
            "current_input": "",
            "next_action": "continue",
            "expected_input": ""
        }
    
    elif expected_input == "booking_details":
        return handle_booking_details(state, current_input)
    
    return {
        **state,
        "text": text_history,
        "current_input": current_input,
        "next_action": "continue",
        "expected_input": ""
    }

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

def handle_booking_details(state: Dict, user_input: str) -> Dict:
    """Handle booking detail responses"""
    user_input_lower = user_input.lower()
    
    if "morning" in user_input_lower:
        state["preferred_time"] = "morning"
    elif "afternoon" in user_input_lower:
        state["preferred_time"] = "afternoon"
    elif "evening" in user_input_lower:
        state["preferred_time"] = "evening"
    
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

def input_flow_condition(state: Dict) -> str:
    """Conditional routing for input flow"""
    next_action = state.get("next_action", "continue")
    if next_action == "wait_for_input":
        return "wait_for_input"
    elif next_action == "continue":
        return "continue"
    else:
        return "end"

# Standalone crisis checker node for pre-emotion detection use in the graph

def crisis_checker_node(state: Dict) -> Dict:
    """
    Checks for crisis keywords in the raw user input (before emotion detection).
    If a crisis is detected, sets next_action='crisis', else next_action='continue'.
    """
    # Use the same crisis keywords as UnifiedRouter
    crisis_keywords = [
        "suicidal", "want to die", "end my life", "no will to live","want to give up",
        "kill myself", "hurt myself", "can't go on", "no point living",
        "end it all", "suicide", "self harm"
    ]
    # Extract raw user input (assume it's in state['text'] as a list or string)
    text = state.get("text", "")
    if isinstance(text, list):
        text = " ".join(str(x) for x in text)
    text = text.strip().lower()
    # Check for any crisis keyword
    if any(keyword in text for keyword in crisis_keywords):
        return {**state, "next_action": "crisis"}
    else:
        return {**state, "next_action": "continue"}