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
    log_dir = "data/user_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{user_id}.jsonl")

    turn = {
        "timestamp": datetime.now().isoformat(),
        "user_input": state.get("current_input"),
        "agent_output": state.get("agent_output"),
        "emotions": state.get("emotions"),
        "details": state.get("details"),
        "next_action": state.get("next_action"),
        "suggestion" : state.get("suggestion")
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(turn) + "\n")
    return state

