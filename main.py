from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph_builder import graph
from tools.memory_store import fetch_user_history, clear_user_memory  # Import your memory fetcher

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    user_input: str

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    user_id = "demo_user"
    # 1. Fetch last state from memory
    last_state = fetch_user_history({"user_id": user_id})
    expected_input = last_state.get("expected_input")
    
    # 2. Prepare new state
    input_state = last_state.copy() if last_state else {}
    input_state["user_id"] = user_id
    input_state["current_input"] = request.user_input
    input_state["text"] = [request.user_input]
    
    # 3. If expecting a specific field, fill it
    if expected_input:
        input_state[expected_input] = request.user_input
        # Optionally clear expected_input so the node knows it was filled
        input_state["expected_input"] = None

    # 4. Always reset clarification count if needed
    input_state.setdefault("clarification_count", 0)
    input_state.setdefault("memory", [])
    input_state.setdefault("emotion_context_links", [])

    final_state = graph.invoke(input_state)
    # Remove 'text' from the final state to avoid post-chain updates
    if 'text' in final_state:
        del final_state['text']
    # Prefer agent_output, then care_suggestion, then suggestion, then appointment_offer, then appointment_status, then agent_router_output
    agent_output = (
        final_state.get("agent_output")
        or final_state.get("care_suggestion")
        or final_state.get("suggestion")
        or final_state.get("appointment_offer")
        or final_state.get("appointment_status")
        or final_state.get("agent_router_output", "")
    )
    clarification = final_state.get("clarification_question")
    needs_clarification = clarification is not None and clarification.strip() != ""

    # If the agent is asking for clarification, return immediately
    if needs_clarification:
        return {
            "agent_message": clarification,
            "needs_clarification": True,
            "emotion": final_state.get("emotions"),
            "forecast": final_state.get("forecast"),
            "rag_self_care": final_state.get("rag_self_care"),
            "prompt": final_state.get("tailored_prompt"),
            "care_suggestion": final_state.get("care_suggestion"),
            "appointment_offer": final_state.get("appointment_offer"),
            "appointment_status": final_state.get("appointment_status"),
            "therapist_match": final_state.get("matched_therapist_rag"),
            "agent_router_output": final_state.get("agent_router_output"),
            "router_trace": final_state.get("router_trace"),
            "crisis_response": final_state.get("crisis_response"),
        }

    # Otherwise, proceed as normal
    return {
        "agent_message": agent_output,
        "needs_clarification": False,
        "emotion": final_state.get("emotions"),
        "forecast": final_state.get("forecast"),
        "rag_self_care": final_state.get("rag_self_care"),
        "prompt": final_state.get("tailored_prompt"),
        "care_suggestion": final_state.get("care_suggestion"),
        "appointment_offer": final_state.get("appointment_offer"),
        "appointment_status": final_state.get("appointment_status"),
        "therapist_match": final_state.get("matched_therapist_rag"),
        "agent_router_output": final_state.get("agent_router_output"),
        "router_trace": final_state.get("router_trace"),
        "crisis_response": final_state.get("crisis_response"),
    }

@app.post("/clear_memory")
async def clear_memory(user_id: str = "demo_user"):
    success = clear_user_memory(user_id)
    return {"success": success}

# To run: uvicorn main:app --reload