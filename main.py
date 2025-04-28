# main.py

from fastapi import FastAPI
from services.clip_service import process_clip_folder  # <- NOT embed_clip_for_photo
from services.dino_service import process_dino_folder
from services.insightface_service import process_faces_folder  # <- NOT embed_clip_for_photo
from services.clustering_service import cluster_faces_dbscan, cluster_faces_hdbscan
from schemas.processing import PartnerModelRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from schemas.processing import ClusterRequest
from services.clustering_service import cluster_embeddings_and_build_response

from schemas.processing import PartnerRequest

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← you can replace * with ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/images", 
    StaticFiles(directory="/Users/sumit/Documents/ai_analysis/67c507b9fb7ebb148255e4af/photos"), 
    name="images"
)

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
    result = process_faces_folder(partner)
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

@app.post("/cluster/dbscan/")
async def cluster_faces_dbscan_endpoint(req: PartnerModelRequest):
    print("----IN dbscan API")
    return cluster_faces_dbscan(req.partner, req.model)

@app.post("/cluster/hdbscan/")
async def cluster_faces_hdbscan_endpoint(req: PartnerModelRequest):
    return cluster_faces_hdbscan(req.partner, req.model)
