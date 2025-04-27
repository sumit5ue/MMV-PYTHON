from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from image_similarity.utils.face_utils import detect_faces, extract_face_embedding, filter_faces_by_quality
from image_similarity.utils.faiss_index import get_index_and_collection, add_to_faiss, save_faiss_index
# from image_similarity.utils.mongo_utils import get_next_id, store_image_metadata
import os

router = APIRouter(prefix="/index-face", tags=["Face Indexing"])

def cleanup_temp(path):
    if os.path.exists(path):
        os.remove(path)

@router.post("")
async def index_face_api(
    file: UploadFile = File(...),
    collection: str = Query(..., description="Name of the index/collection")
):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        faces = detect_faces(temp_path)
        faces = filter_faces_by_quality(faces)
        if not faces:
            raise HTTPException(status_code=400, detail="No high-quality faces found")

        results = []
        faiss_index, mongo_collection = get_index_and_collection(collection)

        for face in faces:
            embedding = extract_face_embedding(face)
            # faiss_id = get_next_id(mongo_collection)
        #     add_to_faiss(faiss_index, embedding, faiss_id)
        #     metadata = {
        #         "filename": file.filename,
        #         "faiss_id": faiss_id,
        #         "bbox": face['bbox'],
        #         "quality": face['quality']
        #     }
        #     object_id = store_image_metadata(mongo_collection, metadata)
        #     results.append({
        #         "object_id": str(object_id),
        #         "faiss_id": faiss_id,
        #         "bbox": face['bbox']
        #     })
        # save_faiss_index(collection, faiss_index)
        return JSONResponse(content={"indexed_faces": results})
    finally:
        cleanup_temp(temp_path)
