import json
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
import os

def build_therapist_rag_index(json_path="data/therapist_profiles.json", index_path="data/therapist_rag"):
    # Load therapist profiles
    with open(json_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    # Convert each profile to a LangChain Document
    documents = []
    for profile in profiles:
        content = (
            f"Name: {profile['name']}\n"
            f"Specialty: {profile['specialty']}\n"
            f"Approach: {profile['approach']}\n"
            f"Bio: {profile['bio']}"
        )
        documents.append(Document(page_content=content))

    # Load Gemini embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save the index locally
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    print(f"✅ Therapist RAG index built and saved at '{index_path}'")

if __name__ == "__main__":
    build_therapist_rag_index()
