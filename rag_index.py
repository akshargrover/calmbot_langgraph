from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
import json

def create_therapist_rag_index():
    with open("data/therapist_profiles.json") as f:
        profiles = json.load(f)

    docs = [Document(page_content=f"{p['name']}: {p['bio']} | Specialties: {p['specialty']}, Approach: {p['approach']}")
            for p in profiles]

    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embed_model)
    vectorstore.save_local("data/therapist_rag")
