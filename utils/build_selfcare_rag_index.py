import os, glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader

def build_selfcare_rag_index(pdf_folder="data/selfcare_pdfs", index_path="data/selfcare_rag"):
    os.makedirs(index_path, exist_ok=True)
    docs = []

    for pdf in glob.glob(f"{pdf_folder}/*.pdf"):
        loader = PyPDFLoader(pdf)
        for page in loader.load_and_split():
            docs.append(Document(page_content=page.page_content, metadata={"source": pdf}))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, overlap=200)
    chunks = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)

    print(f"✅ Built self-care RAG index at {index_path}")

if __name__ == "__main__":
    build_selfcare_rag_index()
