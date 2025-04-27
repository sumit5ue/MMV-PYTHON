from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from image_similarity.models import initialize_mongo, get_next_id, store_image_metadata, find_image_metadata
from image_similarity.dino import extract_dino_features
from image_similarity.faiss_index import initialize_faiss, add_to_faiss, search_faiss
from bson import ObjectId
import os
import faiss

app = FastAPI()

# Load or initialize FAISS index from disk
def load_or_initialize_faiss():
    if os.path.exists("image_index.faiss"):
        return faiss.read_index("image_index.faiss")
    else:
        return initialize_faiss()

# Initialize FAISS
faiss_index = load_or_initialize_faiss()
mongo_client, mongo_collection = None, None

@app.on_event("startup")
async def startup_event():
    global mongo_client, mongo_collection
    mongo_client, mongo_collection = initialize_mongo()

@app.post("/embed")
async def create_embedding(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    try:
        features = extract_dino_features(temp_path)
        print(features)
        faiss_id = get_next_id(mongo_collection)
        add_to_faiss(faiss_index, features, faiss_id)
        metadata = {
            "filename": file.filename,
            "faiss_id": faiss_id
        }
        object_id = store_image_metadata(mongo_collection, metadata)
        return JSONResponse(content={"object_id": str(object_id), "faiss_id": faiss_id})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/similar")
async def find_similar(file: UploadFile = File(...), k: int = 5):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    try:
        query_features = extract_dino_features(temp_path)
        distances, ids = search_faiss(faiss_index, query_features, k)
        # Batch query MongoDB for all faiss_ids
        metadata_list = list(mongo_collection.find({"faiss_id": {"$in": ids.tolist()}}))
        # Create a dictionary for quick lookup
        metadata_dict = {item["faiss_id"]: item for item in metadata_list}
        results = []
        for faiss_id, distance in zip(ids, distances):
            metadata = metadata_dict.get(int(faiss_id))
            if metadata:
                results.append({
                    "object_id": str(metadata["_id"]),
                    "filename": metadata["filename"],
                    "faiss_id": int(faiss_id),
                    "distance": float(distance)  # Now a cosine distance (0 to 2)
                })
        return JSONResponse(content={"similar_images": results})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)