# main.py

from fastapi import FastAPI,HTTPException
from services.clip_service import process_clip_folder  # <- NOT embed_clip_for_photo
from services.dino_service import process_dino_folder
from services.insightface_service import process_faces_folder,process_faces_folder_fast  # <- NOT embed_clip_for_photo
from services.clustering_service import cluster_faces_insightface,cluster_embeddings_and_build_response
from fastapi.middleware.cors import CORSMiddleware
from services.clustering_service import cluster_embeddings_and_build_response
from starlette.middleware.cors import CORSMiddleware as StaticCORSMiddleware  # needed separately
from fastapi import HTTPException
from pathlib import Path
from schemas.processing import PartnerRequest, PartnerModelRequest, ClusterRequest
from fastapi.responses import FileResponse
# from routes.photos import router as photos_router
# from routes.faces_clusters import router as faces_cluster_router
from routes.photo_api import router as photo_router
from routes.cluster_faces import router as cluster_faces_router
from routes.cluster_photo import router as cluster_photo_router



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← you can replace * with ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# app.include_router(photos_router)
# app.include_router(faces_cluster_router)
app.include_router(photo_router)
app.include_router(cluster_faces_router)
app.include_router(cluster_photo_router)

BASE_DIR = "/Users/sumit/Documents/ai_analysis"

# Dynamic endpoint for /faces_rep
# Dynamic endpoint for /<partner>/images
@app.get("/{partner}/photos/{filename}")
async def serve_images(partner: str, filename: str):
    # Validate partner to prevent path traversal
    if not partner or not all(c.isalnum() or c in '-_' for c in partner):
        raise HTTPException(status_code=400, detail="Invalid partner identifier")
    images_dir = Path(BASE_DIR) / partner / "photos"
    file_path = images_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/{partner}/face_reps/{filename}")
async def serve_faces_rep(partner: str, filename: str):
    # Validate partner to prevent path traversal
    if not partner or not all(c.isalnum() or c in '-_' for c in partner):
        raise HTTPException(status_code=400, detail="Invalid partner identifier")
    faces_rep_dir = Path(BASE_DIR) / partner / "face_reps"
    file_path = faces_rep_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.post("/clip/process-folder/")
async def clip_process_folder(req: PartnerRequest):
    partner = req.partner
    result = process_clip_folder(partner)
    return {
        "status": "ok",
        "partner": partner,
        "model": "clip",
        "processed": result['processed'],
        "skipped": result['skipped'],
        "total": result['total']
    }

@app.post("/dino/process-folder/")
async def dino_process_folder(req: PartnerRequest):
    partner = req.partner
    result = process_dino_folder(partner)
    return {
        "status": "ok",
        "partner": partner,
        "model": "dino",
        "processed": result['processed'],
        "skipped": result['skipped'],
        "total": result['total']
    }

@app.post("/faces/process-folder/")
async def faces_process_folder(req: PartnerRequest):
    partner = req.partner
    result = process_faces_folder_fast(partner)
    return {
        "status": "ok",
        "partner": partner,
        "model": "insightface",
        "processed": result['processed'],
        "skipped": result['skipped'],
        "total": result['total']
    }


@app.post("/cluster/")
async def cluster_embeddings(req: ClusterRequest):
    print("----IN CLUSTER API")
    return await cluster_embeddings_and_build_response(req)

@app.post("/cluster/faces")
async def cluster_embeddings(req: ClusterRequest):
    print("----IN CLUSTER FACES API")
    try:
        result = await cluster_faces_insightface(req)
        return result
    except Exception as e:
        print(f"Error in cluster_faces_insightface: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
# @app.post("/cluster/dbscan/")
# async def cluster_faces_dbscan_endpoint(req: PartnerModelRequest):
#     print("----IN dbscan API")
#     return cluster_faces_dbscan(req.partner, req.model)

# @app.post("/cluster/hdbscan/")
# async def cluster_faces_hdbscan_endpoint(req: PartnerModelRequest):
#     return cluster_faces_hdbscan(req.partner, req.model)
