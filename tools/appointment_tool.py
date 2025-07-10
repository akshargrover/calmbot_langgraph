import sqlite3
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os


def appointment_booking_node(state):
    """
    Unified appointment booking node that handles:
    1. Offering appointments based on emotions
    2. Actual booking process 
    
    Returns state with next_action to control flow
    """
    
    # Check what stage we're at
    current_stage = state.get("appointment_stage", "initial")
    
    if current_stage == "initial":
        return _offer_appointment(state)
    elif current_stage == "user_responded":
        return _process_user_response(state)
    elif current_stage == "booking_confirmed":
        return _complete_booking(state)
    else:
        return {**state, "appointment_status": "Unknown stage", "next_action": "continue"}


def _offer_appointment(state):
    """Step 1: Check if user needs appointment and offer it"""
    emotions = state.get("emotions", "")
    if isinstance(emotions, list):
        emotions = " ".join(str(e) for e in emotions)
    emotions = emotions.lower()
    trigger_emotions = [
        "anxiety", "depression", "grief", "loneliness", "stress", "trauma",
        "sadness", "hopelessness", "overwhelm", "panic", "despair", "worry", "fear", "loss", "isolation"
    ]
    # --- Memory context ---
    memory = state.get("memory", [])
    memory_text = "\n".join(
        f"User: {turn.get('user_input','')}, Agent: {turn.get('agent_output','')}" for turn in memory if turn.get("user_input") and turn.get("agent_output")
    )
    memory_summary = f"\n\nRecent conversation:\n{memory_text}" if memory_text else ""
    
    if any(e in emotions for e in trigger_emotions):
        offer = "Based on what you're experiencing, I think speaking with a therapist could be helpful. Would you like me to book an appointment for you?"
        offer_with_memory = offer + memory_summary
        return {
            **state,
            "appointment_stage": "waiting_for_response",
            "appointment_offer": offer_with_memory,
            "agent_output": offer_with_memory,
            "next_action": "wait_for_input",  # This tells the agent to stop and wait
            "expected_input": "appointment_response"
        }
    
    return {
        **state, 
        "appointment_offer": None,
        "next_action": "continue"
    }


def _process_user_response(state):
    """Step 2: Process user's response to appointment offer"""
    appointment_response = state.get("appointment_response", "").lower()
    emotions = state.get("emotions", "")
    if isinstance(emotions, list):
        emotions = " ".join(str(e) for e in emotions)
    emotions = emotions.lower()
    
    if "yes" in appointment_response or "sure" in appointment_response or "ok" in appointment_response:
        # Check if we have all required information
        required_info = _validate_booking_info(state)
        
        if required_info["missing"]:
            msg = f"Great! I need some additional information: {', '.join(required_info['missing'])}"
            return {
                **state,
                "appointment_stage": "collecting_info",
                "next_action": "wait_for_input",
                "appointment_status": msg,
                "agent_output": msg,
                "expected_input": "booking_details"
            }
        else:
            # Instead of booking immediately, ask for confirmation
            msg = "I have all the information I need. Would you like to confirm and book the appointment now?"
            return {
                **state,
                "appointment_stage": "confirm_booking",
                "next_action": "wait_for_input",
                "appointment_status": msg,
                "agent_output": msg,
                "expected_input": "booking_confirmation"
            }
    
    elif "no" in appointment_response:
        msg = "That's okay. I'm here if you change your mind or need other support."
        return {
            **state,
            "appointment_stage": "declined",
            "appointment_status": msg,
            "agent_output": msg,
            "next_action": "continue"
        }
    
    else:
        msg = "I'm not sure if you'd like to book an appointment. Could you please say 'yes' or 'no'?"
        return {
            **state,
            "appointment_stage": "waiting_for_response",
            "appointment_status": msg,
            "agent_output": msg,
            "next_action": "wait_for_input",
            "expected_input": "appointment_response"
        }


def _complete_booking(state):
    """Step 3: Actually book the appointment"""
    conn = sqlite3.connect("data/therapist.db")
    cur = conn.cursor()

    # Get user preferences or use defaults
    emotion = state["emotions"]
    if isinstance(emotion, list):
        emotion = " ".join(str(e) for e in emotion)
    emotion = emotion.strip().lower()
    preferred_time = state.get("preferred_time", None)
    preferred_therapist = state.get("preferred_therapist", None)
    
    # Match therapists based on criteria
    if preferred_therapist:
        cur.execute("SELECT id, name FROM therapists WHERE name LIKE ?", (f"%{preferred_therapist}%",))
    else:
        cur.execute("SELECT id, name FROM therapists WHERE specialty LIKE ?", (f"%{emotion}%",))
    
    candidates = cur.fetchall()

    if not candidates:
        conn.close()
        msg = "I couldn't find a therapist matching your needs. Would you like me to show you all available therapists?"
        return {
            **state,
            "appointment_stage": "no_match",
            "appointment_status": msg,
            "agent_output": msg,
            "next_action": "wait_for_input",
            "expected_input": "show_all_therapists"
        }

    # Find best available slot
    best_match = None
    best_slot = None
    earliest_time = datetime.max

    for therapist_id, name in candidates:
        cur.execute("SELECT slot FROM availability WHERE therapist_id = ? ORDER BY slot ASC", (therapist_id,))
        slots = cur.fetchall()
        
        for (slot,) in slots:
            dt_slot = datetime.strptime(slot, "%Y-%m-%d %H:%M")
            
            # If user has preferred time, try to match it
            if preferred_time:
                # Simple time matching logic - you can make this more sophisticated
                if preferred_time.lower() in slot.lower():
                    best_match = (therapist_id, name)
                    best_slot = slot
                    break
            elif dt_slot < earliest_time:
                earliest_time = dt_slot
                best_match = (therapist_id, name)
                best_slot = slot

    if not best_match:
        conn.close()
        msg = "No available slots match your preferences. Would you like to see other available times?"
        return {
            **state,
            "appointment_stage": "no_slots",
            "appointment_status": msg,
            "agent_output": msg,
            "next_action": "wait_for_input",
            "expected_input": "alternative_times"
        }

    # --- Memory context ---
    memory = state.get("memory", [])
    memory_text = "\n".join(
        f"User: {turn.get('user_input','')}, Agent: {turn.get('agent_output','')}" for turn in memory if turn.get("user_input") and turn.get("agent_output")
    )
    memory_summary = f"\n\nRecent conversation:\n{memory_text}" if memory_text else ""
    # Instead of booking, ask for confirmation
    msg = f"I found a slot with {best_match[1]} on {best_slot}. Would you like to confirm this booking?{memory_summary}"
    return {
        **state,
        "appointment_stage": "awaiting_final_confirmation",
        "appointment_status": msg,
        "agent_output": msg,
        "booked_therapist": best_match[1],
        "booked_slot": best_slot,
        "next_action": "wait_for_input",
        "expected_input": "final_booking_confirmation"
    }


def _validate_booking_info(state):
    """Check what information we have and what we need"""
    required_fields = {
        "emotions": "your current emotional state",
        "preferred_time": "your preferred appointment time (morning/afternoon/evening)",
        "location": "your location or if you prefer online sessions"
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
