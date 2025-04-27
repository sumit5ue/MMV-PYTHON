from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from image_similarity.utils.face_utils import detect_faces, extract_face_embedding, filter_faces_by_quality
from image_similarity.utils.faiss_index import get_index_and_collection, add_to_faiss, search_faiss, save_faiss_index
# from image_similarity.utils.mongo_utils import get_next_id, store_image_metadata, find_image_metadata
import os

router = APIRouter(prefix="/process-photo", tags=["Face Pipeline"])

def cleanup_temp(path):
    if os.path.exists(path):
        os.remove(path)

@router.post("")
async def full_pipeline(
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
            raise HTTPException(status_code=400, detail="No usable faces")

        faiss_index, mongo_collection = get_index_and_collection(collection)
        results = []

        for face in faces:
            embedding = extract_face_embedding(face)
            distances, ids = search_faiss(faiss_index, embedding, k)
            is_duplicate = any(d < 0.1 for d in distances)  # customizable threshold

            # if not is_duplicate:
                # faiss_id = get_next_id(mongo_collection)
                # add_to_faiss(faiss_index, embedding, faiss_id)
                # object_id = store_image_metadata(mongo_collection, {
                #     "filename": file.filename,
                #     "faiss_id": faiss_id,
                #     "bbox": face['bbox'],
                #     "quality": face['quality']
                # })
                # save_faiss_index(collection, faiss_index)
            # else:
            #     faiss_id = int(ids[0])
            #     object_id = find_image_metadata(mongo_collection, faiss_id)

            # results.append({
            #     "faceId": faiss_id,
            #     "matched": not is_duplicate,
            #     "bbox": face['bbox'],
            #     "object_id": str(object_id)
            # })

        return JSONResponse(content={"processed_faces": results})
    finally:
        cleanup_temp(temp_path)
