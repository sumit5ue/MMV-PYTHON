from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from image_similarity.utils.face_utils import detect_faces, extract_face_embedding, filter_faces_by_quality
from image_similarity.utils.faiss_index import get_index_and_collection, search_faiss
from image_similarity.recognizer.face_recognizer import index_known_faces, recognize_faces_in_image

import os

router = APIRouter(prefix="/search-face", tags=["Face Search"])
known_faces = index_known_faces()

def cleanup_temp(path):
    if os.path.exists(path):
        os.remove(path)

@router.post("/recognize")
async def recognize_faces_api(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        results = recognize_faces_in_image(temp_path, known_faces)
        return JSONResponse(content={"results": results})
    finally:
        os.remove(temp_path)



@router.post("")
async def search_face_api(
    file: UploadFile = File(...),
    collection: str = Query(...),
    k: int = 5
):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        faces = detect_faces(temp_path)
        faces = filter_faces_by_quality(faces)
        if not faces:
            raise HTTPException(status_code=400, detail="No valid faces for search")

        faiss_index, mongo_collection = get_index_and_collection(collection)
        results = []

        for face in faces:
            embedding = extract_face_embedding(face)
            distances, ids = search_faiss(faiss_index, embedding, k)
            metadata_list = list(mongo_collection.find({"faiss_id": {"$in": ids.tolist()}}))
            metadata_dict = {item["faiss_id"]: item for item in metadata_list}
            matches = []
            for faiss_id, distance in zip(ids, distances):
                if int(faiss_id) in metadata_dict:
                    matches.append({
                        "faiss_id": int(faiss_id),
                        "distance": float(distance),
                        "filename": metadata_dict[int(faiss_id)]["filename"],
                        "object_id": str(metadata_dict[int(faiss_id)]["_id"])
                    })
            results.append({"bbox": face['bbox'], "matches": matches})

        return JSONResponse(content={"search_results": results})
    finally:
        cleanup_temp(temp_path)
