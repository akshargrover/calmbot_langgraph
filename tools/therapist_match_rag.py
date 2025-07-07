from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
def rag_match_therapist(state):
    user_input = state["input"]
    
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local(
        "data/therapist_rag",
        embed_model,
        allow_dangerous_deserialization=True
    )
    docs = vectorstore.similarity_search(user_input, k=2)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
    suggestion = model.invoke(f"Match the user to the best therapist from the following:\n{docs}\nUser text: {user_input}")
    
    return {**state, "matched_therapist_rag": suggestion}

