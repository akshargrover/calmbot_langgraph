from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def rag_selfcare_suggestion(state):
    emotion = state["emotions"]
    vectorstore = FAISS.load_local("data/selfcare_rag", GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")))
    docs = vectorstore.similarity_search(emotion, k=3)
    content = "\n".join([doc.page_content for doc in docs[:3]])

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
    prompt = (
        f"I have the following self-care content:\n{content}\n"
        f"Suggest personalized self-care steps for someone feeling {emotion}."
    )
    response = model.predict(prompt)
    return {**state, "rag_self_care": response.text}

