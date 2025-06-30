import faiss
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings  # Or OpenAI

def fetch_user_history(state):
    emotion = state["emotions"]
    
    index_path = "data/faiss_index"
    vectorstore = FAISS.load_local(index_path, GoogleGenerativeAIEmbeddings())
    
    similar_moods = vectorstore.similarity_search(emotion, k=5)
    
    return {**state, "memory": similar_moods}
