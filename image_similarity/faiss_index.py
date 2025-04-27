import faiss
import numpy as np

def initialize_faiss():
    dimension = 768  # DINO feature dimension
    index = faiss.IndexFlatIP(dimension)  # Use IndexFlatIP for cosine similarity
    faiss.write_index(index, "image_index.faiss")
    return index

def add_to_faiss(index, features, faiss_id):
    index = faiss.read_index("image_index.faiss")
    features = np.array([features], dtype=np.float32)
    index.add(features)
    faiss.write_index(index, "image_index.faiss")

def search_faiss(index, query_features, k=5):
    index = faiss.read_index("image_index.faiss")
    query_features = np.array([query_features], dtype=np.float32)
    # Since vectors are L2-normalized, inner product is the cosine similarity
    similarities, indices = index.search(query_features, k)
    # Convert similarities to cosine distances (1 - similarity)
    distances = 1.0 - similarities[0]
    return distances, indices[0]