from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph_builder import graph

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
    input_state = {"text": request.user_input}
    final_state = graph.invoke(input_state)
    agent_output = final_state.get("agent_router_output", "")
    needs_clarification = agent_output.strip().endswith("?") or "clarify" in agent_output.lower()

    return {
        "agent_message": agent_output,
        "needs_clarification": needs_clarification,
        "emotion": final_state.get("emotions"),
        "forecast": final_state.get("forecast"),
        "emotion_context_links": final_state.get("emotion_context_links"),
        "self_care_articles": final_state.get("self_care_articles"),
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

# To run: uvicorn main:app --reload