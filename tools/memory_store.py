from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from utils.embedding import get_text_embedding
from utils.faiss_utils import load_faiss_index, save_faiss_index, add_embedding
import numpy as np
import os
def fetch_user_history(state):
    emotion = state["emotions"]
    
    index_path = "data/faiss_index"
    vectorstore = FAISS.load_local(
        "data/therapist_rag",
        GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")),
        allow_dangerous_deserialization=True
    )
    # Using MMR (Maximum Marginal Relevance) instead of similarity search
    mmr_moods = vectorstore.max_marginal_relevance_search(emotion, k=5)
    
    return {**state, "memory": mmr_moods}

def store_mood(state):
    user_text = state["text"]
    if isinstance(user_text, list):
        user_text = " ".join(x.content if hasattr(x, "content") else str(x) for x in user_text)
    embedding = np.array(get_text_embedding(user_text))
    index = load_faiss_index()
    add_embedding(index, embedding)
    save_faiss_index(index)
    return state

