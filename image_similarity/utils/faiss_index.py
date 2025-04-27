import faiss
import os

INDEX_DIR = "faiss_indices"
os.makedirs(INDEX_DIR, exist_ok=True)

def initialize_faiss():
    return faiss.IndexFlatL2(512)  # or use IndexFlatIP for cosine similarity

def load_faiss_index(collection):
    path = os.path.join(INDEX_DIR, f"{collection}.faiss")
    if os.path.exists(path):
        return faiss.read_index(path)
    return initialize_faiss()

def save_faiss_index(collection, index):
    path = os.path.join(INDEX_DIR, f"{collection}.faiss")
    faiss.write_index(index, path)

def add_to_faiss(index, vector, faiss_id):
    index.add_with_ids(vector, np.array([faiss_id], dtype='int64'))

def search_faiss(index, vector, k=5):
    distances, indices = index.search(vector, k)
    return distances[0], indices[0]

def get_index_and_collection(collection):
    return
    # from image_similarity.utils.mongo_utils import initialize_mongo
    # mongo_client, mongo_db = initialize_mongo()
    # faiss_index = load_faiss_index(collection)
    # mongo_collection = mongo_db[collection]
    # return faiss_index, mongo_collection
