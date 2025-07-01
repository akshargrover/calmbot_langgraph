import os
import faiss
import numpy as np
from config.settings import FAISS_INDEX_PATH, EMBEDDING_DIM

# Each entry: (embedding, metadata dict)

def create_faiss_index():
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    return index

def save_faiss_index(index, path=FAISS_INDEX_PATH):
    faiss.write_index(index, os.path.join(path, 'faiss.index'))

def load_faiss_index(path=FAISS_INDEX_PATH):
    index_path = os.path.join(path, 'faiss.index')
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        return create_faiss_index()

def add_embedding(index, embedding: np.ndarray):
    index.add(np.array([embedding]).astype('float32'))

def query_similar(index, embedding: np.ndarray, top_k=3):
    D, I = index.search(np.array([embedding]).astype('float32'), top_k)
    return I[0], D[0]
