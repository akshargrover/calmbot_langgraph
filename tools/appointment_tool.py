import sqlite3
from datetime import datetime

def offer_appointment(state):
    emotions = state.get("emotions", "").lower()
    trigger_emotions = ["anxiety", "depression", "grief", "loneliness"]
    if any(e in emotions for e in trigger_emotions):
        return {**state, "appointment_offer": "Would you like to book a session with a therapist?"}
    return {**state, "appointment_offer": None}

def book_appointment(state):
    if "yes" not in state.get("appointment_response", "").lower():
        return {**state, "appointment_status": "Declined"}

    conn = sqlite3.connect("data/therapist.db")
    cur = conn.cursor()

    # Match therapists based on dominant emotion
    emotion = state["emotions"].split(",")[0].strip().lower()
    cur.execute("SELECT id, name FROM therapists WHERE specialty LIKE ?", (f"%{emotion}%",))
    candidates = cur.fetchall()

    best_match = None
    best_slot = None
    earliest_time = datetime.max

    for therapist_id, name in candidates:
        cur.execute("SELECT slot FROM availability WHERE therapist_id = ? ORDER BY slot ASC", (therapist_id,))
        slots = cur.fetchall()
        for (slot,) in slots:
            dt_slot = datetime.strptime(slot, "%Y-%m-%d %H:%M")
            if dt_slot < earliest_time:
                earliest_time = dt_slot
                best_match = (therapist_id, name)
                best_slot = slot

    if not best_match:
        conn.close()
        return {**state, "appointment_status": "No therapist available right now."}

    # Book it (remove from availability, add to user_logs)
    cur.execute("DELETE FROM availability WHERE therapist_id = ? AND slot = ?", (best_match[0], best_slot))

    cur.execute("""
        INSERT INTO user_logs (user_input, emotion, therapist_name, slot, booked_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        state["input"],
        emotion,
        best_match[1],
        best_slot,
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))

    conn.commit()
    conn.close()

    return {
        **state,
        "appointment_status": f"Appointment confirmed with {best_match[1]} at {best_slot}"
    }
