import pandas as pd
import datetime
from utils.embedding import get_text_embedding
from utils.faiss_utils import load_faiss_index, query_similar
import numpy as np

def forecast_mood(state):
    user_text = state["text"]
    embedding = np.array(get_text_embedding(user_text))
    index = load_faiss_index()
    idxs, dists = query_similar(index, embedding, top_k=3)
    # For demo, just use dummy past moods (in real app, fetch metadata)
    similar_past_moods = [f"Past mood #{i+1}" for i in idxs]
    # Simple rule-based forecast
    emotion = state.get("emotions", "other").lower()
    forecast_map = {
        "anxiety": "You may feel calmer if you practice relaxation techniques.",
        "joy": "Your positive mood is likely to continue!",
        "gratitude": "Expressing gratitude can boost your well-being.",
        "shame": "Self-compassion may help you feel better soon.",
        "sadness": "Connecting with others may lift your mood.",
        "anger": "Taking time to cool off can help you regain balance.",
        "fear": "Facing your fears gradually can reduce anxiety.",
        "surprise": "Processing surprises can help you adapt.",
        "other": "Your mood may shift—check in with yourself later."
    }
    forecast = forecast_map.get(emotion, forecast_map["other"])
    state["forecast"] = forecast
    state["similar_past_moods"] = similar_past_moods
    return state
