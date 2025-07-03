from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from vertexai.language_models import ChatModel

def rag_match_therapist(state):
    user_input = state["input"]
    
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("data/therapist_rag", embed_model)
    docs = vectorstore.similarity_search(user_input, k=2)

    model = ChatModel.from_pretrained("gemini-1.5-flash")
    suggestion = model.predict(f"Match the user to the best therapist from the following:\n{docs}\nUser text: {user_input}")
    
    return {**state, "matched_therapist_rag": suggestion.text}

