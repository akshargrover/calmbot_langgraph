from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from utils.embedding import get_text_embedding
from utils.faiss_utils import load_faiss_index, save_faiss_index, add_embedding
import numpy as np
import os
import json
from datetime import datetime

def fetch_user_history(state, n_turns=5):
    user_id = state.get("user_id", "default_user")
    log_path = os.path.join("data/user_logs", f"{user_id}.jsonl")
    memory = []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-n_turns:]
            memory = [json.loads(line) for line in lines]
    return {**state, "memory": memory}

def store_mood(state):
    user_text = state["text"]
    if isinstance(user_text, list):
        user_text = " ".join(x.content if hasattr(x, "content") else str(x) for x in user_text)
    embedding = np.array(get_text_embedding(user_text))
    index = load_faiss_index()
    add_embedding(index, embedding)
    save_faiss_index(index)
    return state

def store_user_turn(state):
    user_id = state.get("user_id", "default_user")
    if not isinstance(user_id, str):
        # If it's a list, take the first string element, or fallback to 'default_user'
        if isinstance(user_id, list) and user_id:
            user_id = str(user_id[0])
        else:
            user_id = "default_user"
    user_id = user_id.replace("/", "_").replace("\\", "_")  # sanitize for filesystem
    log_dir = "data/user_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{user_id}.jsonl")

    def safe_str(val):
        # Convert HumanMessage or other objects to string
        if hasattr(val, "content"):
            return str(val.content)
        if isinstance(val, list):
            return [safe_str(v) for v in val]
        return str(val) if val is not None else ""

    turn = {
        "timestamp": datetime.now().isoformat(),
        "user_input": safe_str(state.get("current_input")),
        "agent_output": safe_str(state.get("agent_output")),
        "emotions": safe_str(state.get("emotions")),
        "details": safe_str(state.get("details")),
        "suggestion": safe_str(state.get("suggestion"))
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(turn) + "\n")
    return state

