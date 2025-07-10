from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph_builder import graph
from tools.memory_store import fetch_user_history, clear_user_memory

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

class ClearMemoryRequest(BaseModel):
    user_id: str

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    user_id = "demo_user"
    
    # 1. Fetch last state from memory
    last_state = fetch_user_history({"user_id": user_id})
    expected_input = last_state.get("expected_input") if last_state else None
    
    # 2. Prepare new state
    input_state = last_state.copy() if last_state else {}
    input_state["user_id"] = user_id
    input_state["current_input"] = request.user_input
    input_state["text"] = [request.user_input]
    
    # 3. Handle expected input types
    if expected_input:
        if expected_input == "appointment_response":
            input_state["appointment_response"] = request.user_input
        elif expected_input == "booking_details":
            input_state["booking_details"] = request.user_input
        elif expected_input == "final_booking_confirmation":
            input_state["final_booking_confirmation"] = request.user_input
        else:
            input_state[expected_input] = request.user_input
        
        # Clear expected_input so the node knows it was filled
        input_state["expected_input"] = None

    # 4. Initialize default values
    input_state.setdefault("clarification_count", 0)
    input_state.setdefault("memory", [])
    input_state.setdefault("emotion_context_links", [])

    # 5. Invoke the graph
    final_state = graph.invoke(input_state)
    
    # Remove 'text' from the final state to avoid post-chain updates
    if 'text' in final_state:
        del final_state['text']
    
    # 6. Determine the response message
    agent_output = (
        final_state.get("agent_output")
        or final_state.get("care_suggestion")
        or final_state.get("suggestion")
        or final_state.get("appointment_offer")
        or final_state.get("appointment_status")
        or final_state.get("agent_router_output")
        or final_state.get("crisis_response")
        or "(No response)"
    )
    
    # 7. Check for clarification needs
    clarification = final_state.get("clarification_question")
    needs_clarification = clarification is not None and clarification.strip() != ""
    
    # 8. Check if we're waiting for specific input
    waiting_for_input = final_state.get("expected_input") is not None
    
    # 9. Return response
    return {
        "agent_message": agent_output,
        "needs_clarification": needs_clarification or waiting_for_input,
        "waiting_for_input": waiting_for_input,
        "expected_input": final_state.get("expected_input"),
        "appointment_stage": final_state.get("appointment_stage"),
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
        "next_action": final_state.get("next_action"),
        "debug_info": {
            "appointment_stage": final_state.get("appointment_stage"),
            "expected_input": final_state.get("expected_input"),
            "current_input": final_state.get("current_input"),
            "user_response": final_state.get("appointment_response")
        }
    }

@app.post("/clear_memory")
async def clear_memory(request: ClearMemoryRequest):
    success = clear_user_memory(request.user_id)
    return {"success": success}

# To run: uvicorn main:app --reload