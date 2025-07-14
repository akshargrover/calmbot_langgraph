import sqlite3
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handles database operations with proper error handling"""
    
    def __init__(self, db_path="data/therapist.db"):
        self.db_path = db_path
    
    def get_connection(self):
        """Get database connection with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            return conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def find_therapists(self, specialty=None, location=None, online_preferred=False):
        """Find therapists based on criteria"""
        conn = self.get_connection()
        try:
            query = "SELECT id, name, specialty, location, online_available, rating FROM therapists WHERE 1=1"
            params = []
            
            if specialty:
                query += " AND (specialty LIKE ? OR specialty IS NULL)"
                params.append(f"%{specialty}%")
            
            if location and not online_preferred:
                query += " AND location LIKE ?"
                params.append(f"%{location}%")
            
            if online_preferred:
                query += " AND online_available = 1"
            
            query += " ORDER BY rating DESC"
            
            cur = conn.cursor()
            cur.execute(query, params)
            return cur.fetchall()
            
        except sqlite3.Error as e:
            logger.error(f"Error finding therapists: {e}")
            return []
        finally:
            conn.close()
    
    def get_available_slots(self, therapist_id, preferred_time=None):
        """Get available slots for a therapist"""
        conn = self.get_connection()
        try:
            query = """
                SELECT slot FROM availability 
                WHERE therapist_id = ? AND is_available = 1 
                AND slot NOT IN (
                    SELECT slot FROM appointments 
                    WHERE therapist_id = ? AND status = 'booked'
                )
                ORDER BY slot ASC
            """
            
            cur = conn.cursor()
            cur.execute(query, (therapist_id, therapist_id))
            slots = cur.fetchall()
            
            # Filter by preferred time if specified
            if preferred_time and slots:
                filtered_slots = []
                for slot in slots:
                    slot_time = slot['slot']
                    if self._matches_preferred_time(slot_time, preferred_time):
                        filtered_slots.append(slot)
                return filtered_slots
            
            return slots
            
        except sqlite3.Error as e:
            logger.error(f"Error getting available slots: {e}")
            return []
        finally:
            conn.close()
    
    def _matches_preferred_time(self, slot_time, preferred_time):
        """Check if slot matches preferred time"""
        try:
            dt = datetime.strptime(slot_time, "%Y-%m-%d %H:%M")
            hour = dt.hour
            
            preferred_time = preferred_time.lower()
            if "morning" in preferred_time and 6 <= hour < 12:
                return True
            elif "afternoon" in preferred_time and 12 <= hour < 18:
                return True
            elif "evening" in preferred_time and 18 <= hour < 22:
                return True
            
            return False
        except ValueError:
            return False
    
    def book_appointment(self, therapist_id, slot, user_id, notes=""):
        """Book an appointment"""
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            
            # Check if slot is still available
            cur.execute("""
                SELECT 1 FROM availability 
                WHERE therapist_id = ? AND slot = ? AND is_available = 1
            """, (therapist_id, slot))
            
            if not cur.fetchone():
                return {"success": False, "message": "Slot is no longer available"}
            
            # Check if appointment already exists
            cur.execute("""
                SELECT 1 FROM appointments 
                WHERE therapist_id = ? AND slot = ? AND status = 'booked'
            """, (therapist_id, slot))
            
            if cur.fetchone():
                return {"success": False, "message": "Appointment already booked"}
            
            # Book the appointment
            cur.execute("""
                INSERT INTO appointments (therapist_id, user_id, slot, notes)
                VALUES (?, ?, ?, ?)
            """, (therapist_id, user_id, slot, notes))
            
            conn.commit()
            return {"success": True, "appointment_id": cur.lastrowid}
            
        except sqlite3.Error as e:
            logger.error(f"Error booking appointment: {e}")
            return {"success": False, "message": f"Database error: {e}"}
        finally:
            conn.close()

# New: Function to generate context-aware input prompt for appointment phase

def get_appointment_input_prompt(state):
    """
    Returns a context-aware prompt string for the user during the appointment phase.
    """
    stage = state.get("appointment_stage", "")
    expected_input = state.get("expected_input", "")
    available_therapists = state.get("available_therapists", [])
    available_slots = state.get("available_slots", [])

    # Custom prompts for each stage/expected input
    if expected_input == "appointment_response":
        return "Would you like to book an appointment with a therapist? (yes/no)"
    elif expected_input == "booking_details":
        return "Please provide your preferences: preferred time (morning/afternoon/evening) and whether you prefer online or in-person sessions."
    elif expected_input == "therapist_selection":
        if available_therapists:
            options = "\n".join([
                f"{i+1}. {t['name']} - {t['specialty']} (Rating: {t['rating']:.1f})"
                for i, t in enumerate(available_therapists[:3])
            ])
            return f"Which therapist would you like to choose? Please reply with the number or name.\n{options}"
        else:
            return "Please select a therapist by number or name."
    elif expected_input == "slot_selection":
        if available_slots:
            slots = "\n".join([
                f"{i+1}. {slot['slot']}" for i, slot in enumerate(available_slots[:5])
            ])
            return f"Which time slot would you prefer? Please reply with the number.\n{slots}"
        else:
            return "Please specify your preferred time slot."
    elif expected_input == "final_booking_confirmation":
        return "Do you confirm this booking? (yes/no)"
    # Fallbacks for stages
    elif stage == "collecting_info":
        return "Please provide the requested information to help book your appointment."
    elif stage == "therapist_selection":
        return "Please select a therapist by number or name."
    elif stage == "slot_selection":
        return "Please select a time slot by number."
    # Default fallback
    return "Please provide the requested information."


def appointment_booking_node(state):
    """
    Enhanced appointment booking node with proper input handling
    """
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # Get current stage and user input
    current_stage = state.get("appointment_stage", "initial")
    user_input = state.get("user_input", "")
    
    # Log for debugging
    logger.info(f"Appointment booking - Stage: {current_stage}, Input: {user_input}")
    
    try:
        if current_stage == "initial":
            return _offer_appointment(state)
        
        elif current_stage == "user_responded":
            return _process_user_response(state, db_manager)
        
        elif current_stage == "collecting_info":
            return _collect_additional_info(state, db_manager)
        
        elif current_stage == "therapist_selected":
            return _handle_therapist_selection(state, db_manager)
        
        elif current_stage == "confirm_booking":
            return _confirm_booking(state, db_manager)
        
        elif current_stage == "awaiting_final_confirmation":
            return _handle_final_confirmation(state, db_manager)
        
        else:
            # Unknown stage, provide helpful message
            return {
                **state,
                "appointment_status": "I'm not sure where we are in the booking process. Would you like to start over?",
                "agent_output": "I'm not sure where we are in the booking process. Would you like to start over?",
                "appointment_stage": "initial",
                "next_action": "wait_for_input",
                "expected_input": "appointment_response"
            }
            
    except Exception as e:
        logger.error(f"Error in appointment booking node: {e}")
        return {
            **state,
            "appointment_status": "I'm sorry, there was an error. Let me help you start the appointment booking process again.",
            "agent_output": "I'm sorry, there was an error. Let me help you start the appointment booking process again.",
            "appointment_stage": "initial",
            "next_action": "continue"
        }


def _offer_appointment(state):
    """Step 1: Check if user needs appointment and offer it"""
    emotions = state.get("emotions", "")
    if isinstance(emotions, list):
        emotions = " ".join(str(e) for e in emotions)
    emotions = emotions.lower()
    
    trigger_emotions = [
        "anxiety", "depression", "grief", "loneliness", "stress", "trauma",
        "sadness", "hopelessness", "overwhelm", "panic", "despair", "worry", 
        "fear", "loss", "isolation", "burnout", "confused", "angry"
    ]
    
    # Build conversation context
    context = _build_conversation_context(state)
    
    if any(e in emotions for e in trigger_emotions):
        offer = (
            "Based on what you're experiencing, I think speaking with a therapist could be helpful. "
            "Would you like me to help you book an appointment?"
        )
        
        full_response = f"{offer}{context}"
        
        return {
            **state,
            "appointment_stage": "waiting_for_response",
            "appointment_offer": offer,
            "agent_output": full_response,
            "next_action": "wait_for_input",
            "expected_input": "appointment_response"
        }
    
    return {
        **state, 
        "appointment_offer": None,
        "next_action": "continue"
    }


def _process_user_response(state, db_manager):
    """Enhanced user response processing"""
    user_input = state.get("user_input", "").lower().strip()
    
    if not user_input:
        return {
            **state,
            "appointment_status": "I didn't receive your response. Would you like to book an appointment? Please say yes or no.",
            "agent_output": "I didn't receive your response. Would you like to book an appointment? Please say yes or no.",
            "appointment_stage": "waiting_for_response",
            "next_action": "wait_for_input",
            "expected_input": "appointment_response"
        }
    
    # Positive responses
    if any(word in user_input for word in ["yes", "sure", "ok", "yeah", "please", "book", "schedule"]):
        # Check if we have required information
        required_info = _validate_booking_info(state)
        
        if required_info["missing"]:
            missing_items = ", ".join(required_info["missing"])
            msg = f"Great! I'd like to help you find the right therapist. I need some information: {missing_items}. Let's start - what would you prefer?"
            
            return {
                **state,
                "appointment_stage": "collecting_info",
                "appointment_status": msg,
                "agent_output": msg,
                "next_action": "wait_for_input",
                "expected_input": "booking_details"
            }
        else:
            return _find_and_present_options(state, db_manager)
    
    # Negative responses
    elif any(word in user_input for word in ["no", "not now", "maybe later", "not interested"]):
        context = _build_conversation_context(state)
        msg = f"That's completely okay. I'm here whenever you're ready or need other support.{context}"
        
        return {
            **state,
            "appointment_stage": "declined",
            "appointment_status": msg,
            "agent_output": msg,
            "next_action": "continue"
        }
    
    else:
        # Unclear response
        msg = "I'm not sure if you'd like to book an appointment. Could you please let me know - yes or no?"
        return {
            **state,
            "appointment_stage": "waiting_for_response",
            "appointment_status": msg,
            "agent_output": msg,
            "next_action": "wait_for_input",
            "expected_input": "appointment_response"
        }

def _handle_therapist_selection(state, db_manager):
    """Handle therapist selection from user"""
    user_input = state.get("user_input", "").lower().strip()
    available_therapists = state.get("available_therapists", [])
    
    if not available_therapists:
        return {
            **state,
            "appointment_status": "I don't have the therapist options available. Let me search again.",
            "agent_output": "I don't have the therapist options available. Let me search again.",
            "appointment_stage": "collecting_info",
            "next_action": "continue"
        }
    
    # Try to parse therapist selection
    selected_therapist = None
    
    # Check for number selection (1, 2, 3, etc.)
    if user_input.isdigit():
        try:
            index = int(user_input) - 1
            if 0 <= index < len(available_therapists):
                selected_therapist = available_therapists[index]
        except ValueError:
            pass
    
    # Check for name selection
    if not selected_therapist:
        for therapist in available_therapists:
            if therapist['name'].lower() in user_input:
                selected_therapist = therapist
                break
    
    if selected_therapist:
        # Get available slots
        slots = db_manager.get_available_slots(
            selected_therapist['id'], 
            state.get("preferred_time")
        )
        
        if slots:
            slots_text = "Available slots:\n"
            for i, slot in enumerate(slots[:5], 1):  # Show first 5 slots
                slots_text += f"{i}. {slot['slot']}\n"
            
            msg = f"Great choice! {selected_therapist['name']} is available at these times:\n{slots_text}\nWhich slot would you prefer?"
            
            return {
                **state,
                "selected_therapist": selected_therapist,
                "available_slots": slots,
                "appointment_stage": "slot_selection",
                "appointment_status": msg,
                "agent_output": msg,
                "next_action": "wait_for_input",
                "expected_input": "slot_selection"
            }
        else:
            msg = f"I'm sorry, {selected_therapist['name']} doesn't have available slots matching your preferences. Would you like to see other therapists?"
            return {
                **state,
                "appointment_stage": "no_slots_available",
                "appointment_status": msg,
                "agent_output": msg,
                "next_action": "wait_for_input",
                "expected_input": "therapist_selection"
            }
    else:
        # Invalid selection
        msg = "I couldn't understand your selection. Please choose a therapist by number (1, 2, 3) or by name."
        return {
            **state,
            "appointment_stage": "therapist_selection",
            "appointment_status": msg,
            "agent_output": msg,
            "next_action": "wait_for_input",
            "expected_input": "therapist_selection"
        }

# Add debugging function
def debug_appointment_state(state):
    """Debug function to log appointment state"""
    logger.info(f"Appointment Debug:")
    logger.info(f"  Stage: {state.get('appointment_stage')}")
    logger.info(f"  Next Action: {state.get('next_action')}")
    logger.info(f"  Expected Input: {state.get('expected_input')}")
    logger.info(f"  User Input: {state.get('user_input')}")
    logger.info(f"  Current Input: {state.get('current_input')}")
    return state


def _collect_additional_info(state, db_manager):
    """Collect missing information for booking"""
    user_input = state.get("user_input", "")
    
    # Update state with new information
    updated_state = _extract_booking_info(state, user_input)
    
    # Check if we have enough information now
    required_info = _validate_booking_info(updated_state)
    
    if required_info["missing"]:
        missing_items = ", ".join(required_info["missing"])
        msg = f"Thanks! I still need: {missing_items}. Could you provide this information?"
        
        return {
            **updated_state,
            "appointment_stage": "collecting_info",
            "next_action": "wait_for_input",
            "appointment_status": msg,
            "agent_output": msg,
            "expected_input": "booking_details"
        }
    else:
        return _find_and_present_options(updated_state, db_manager)


def _find_and_present_options(state, db_manager):
    """Find therapists and present options to user"""
    emotions = state.get("emotions", "")
    if isinstance(emotions, list):
        emotions = " ".join(str(e) for e in emotions)
    
    location = state.get("location", "")
    online_preferred = "online" in location.lower() if location else False
    
    # Find therapists
    therapists = db_manager.find_therapists(
        specialty=emotions,
        location=location if not online_preferred else None,
        online_preferred=online_preferred
    )
    
    if not therapists:
        msg = "I couldn't find therapists matching your specific criteria. Would you like me to show you all available therapists?"
        return {
            **state,
            "appointment_stage": "no_match",
            "appointment_status": msg,
            "agent_output": msg,
            "next_action": "wait_for_input",
            "expected_input": "show_all_therapists"
        }
    
    # Present options
    options_text = "Here are some therapists I found for you:\n"
    for i, therapist in enumerate(therapists[:3], 1):  # Show top 3
        options_text += f"{i}. {therapist['name']} - {therapist['specialty']} (Rating: {therapist['rating']:.1f})\n"
    
    context = _build_conversation_context(state)
    msg = f"{options_text}\nWhich therapist would you prefer, or would you like more information about any of them?{context}"
    
    return {
        **state,
        "appointment_stage": "therapist_selection",
        "available_therapists": therapists,
        "appointment_status": msg,
        "agent_output": msg,
        "next_action": "wait_for_input",
        "expected_input": "therapist_selection"
    }


def _build_conversation_context(state):
    """Build conversation context for continuity"""
    memory = state.get("memory", [])
    if not memory:
        return ""
    
    # Get recent relevant exchanges
    recent_exchanges = []
    for turn in memory[-3:]:  # Last 3 exchanges
        if turn.get("user_input") and turn.get("agent_output"):
            recent_exchanges.append(f"You: {turn['user_input'][:50]}...")
    
    if recent_exchanges:
        return f"\n\nBased on our conversation: {' | '.join(recent_exchanges)}"
    return ""


def _extract_booking_info(state, user_input):
    """Extract booking information from user input"""
    updated_state = state.copy()
    user_input_lower = user_input.lower()
    
    # Extract time preferences
    if any(time in user_input_lower for time in ["morning", "afternoon", "evening"]):
        for time_pref in ["morning", "afternoon", "evening"]:
            if time_pref in user_input_lower:
                updated_state["preferred_time"] = time_pref
                break
    
    # Extract location preferences
    if any(loc in user_input_lower for loc in ["online", "virtual", "remote"]):
        updated_state["location"] = "online"
    elif any(loc in user_input_lower for loc in ["in-person", "office", "clinic"]):
        updated_state["location"] = "in-person"
    
    return updated_state


def _validate_booking_info(state):
    """Check what information we have and what we need"""
    required_fields = {
        "emotions": "your current concerns or what you'd like to work on",
        "preferred_time": "your preferred time (morning/afternoon/evening)",
        "location": "whether you prefer online or in-person sessions"
    }
    
    missing = []
    available = {}
    
    for field, description in required_fields.items():
        if field not in state or not state[field]:
            missing.append(description)
        else:
            available[field] = state[field]
    
    return {
        "missing": missing,
        "available": available,
        "is_complete": len(missing) == 0
    }


def _confirm_booking(state, db_manager):
    """Confirm booking details with user"""
    # This would be implemented based on your specific flow
    pass


def _complete_booking(state, db_manager):
    """Complete the actual booking"""
    # This would be implemented based on your specific flow
    pass


def _handle_final_confirmation(state, db_manager):
    """Handle final booking confirmation"""
    # This would be implemented based on your specific flow
    pass