from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from db.session import SessionLocal
from models.photo import Photo
from schemas.photo import PhotoCreate, PhotoSchema, PhotoUpdate,PhotoDownloadRequest,ClipEmbedRequest
import uuid
from typing import List, Optional, Dict, Any
from services.photo_processing_service import process_photos_async
from services.facial_rekognition_service import process_faces_for_partner
from pydantic import BaseModel
from services.clip_service import clip_photos_from_db,write_clip_cluster_to_mongo
from services.facial_rekognition_service import process_faces_for_partner

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/photos", response_model=PhotoSchema)
def create_photo(photo: PhotoCreate, db: Session = Depends(get_db)):
    new_photo = Photo(**photo.model_dump())
    db.add(new_photo)
    db.commit()
    db.refresh(new_photo)
    return new_photo

@router.get("/photos/{id}", response_model=PhotoSchema)
def get_photo(id: str, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    return photo


@router.patch("/photos/{id}", response_model=PhotoSchema)
def update_photo(id: str, updates: PhotoUpdate, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")

    update_data = updates.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(photo, key, value)

    db.commit()
    db.refresh(photo)
    return photo


@router.delete("/photos/{id}")
def delete_photo(id: str, db: Session = Depends(get_db)):
    photo = db.query(Photo).filter(Photo.id == id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    db.delete(photo)
    db.commit()
    return {"message": f"Photo {id} deleted"}

@router.get("/photos", response_model=List[PhotoSchema])
def list_photos(db: Session = Depends(get_db)):
    return db.query(Photo).limit(100).all()

# @router.post("/photos/search", response_model=List[PhotoSchema])
# def search_photos(filter: PhotoFilter, db: Session = Depends(get_db)):
#     query = db.query(Photo)

#     if filter.partner:
#         query = query.filter(Photo.partner == filter.partner)
#     if filter.isVideo is not None:
#         query = query.filter(Photo.is_video == filter.isVideo)
#     if filter.rosterId:
#         query = query.filter(Photo.roster_id == filter.rosterId)
#     if filter.start_date:
#         query = query.filter(Photo.photo_creation_date >= filter.start_date)
#     if filter.end_date:
#         query = query.filter(Photo.photo_creation_date <= filter.end_date)

#     return query.order_by(Photo.created_at.desc()).limit(100).all()


@router.post("/photos/download-and-process-faces")
async def download_and_process_faces_endpoint(
    body: PhotoDownloadRequest,
    background_tasks: BackgroundTasks
):
    return await process_photos_async(
        partner=body.partner,
        concurrency=body.concurrency,
        limit=body.limit,
        update_batch_size=body.update_batch_size,
    )


@router.post("/photos/embedding/clip")
async def clip_embed_photos_not_clipped(
    body: ClipEmbedRequest,
    background_tasks: BackgroundTasks   
):
    return await clip_photos_from_db(
        partner=body.partner,
        batch_size=body.batch_size,
    )


@router.post("/photos/aws-rekognition/search-faces")
async def aws_facial_rekognition(
    body: ClipEmbedRequest,
    background_tasks: BackgroundTasks
):
    return await process_faces_for_partner(
        partner=body.partner,
    )


@router.post("/photos/transfer-to-mongodb")
async def aws_facial_rekognition(
    body: ClipEmbedRequest,
    background_tasks: BackgroundTasks
):
    return await write_clip_cluster_to_mongo(
        partner=body.partner,
    )
