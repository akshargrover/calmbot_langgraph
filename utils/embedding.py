from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import get_gemini_api_key

# Optionally, fallback to OpenAI or other embedding models if needed

def get_text_embedding(text: str):
    """Get embedding for a given text using Gemini 1.5 Flash."""
    embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=get_gemini_api_key())
    return embedder.embed_query(text)
