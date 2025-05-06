from pydantic import BaseModel
from typing import Optional, List, Dict
from uuid import UUID
from datetime import datetime

class ClusterPhotoRepBase(BaseModel):
    cluster_id: UUID
    run_id: UUID
    cluster_label: Optional[str] = None
    partner: Optional[str] = None
    rep_photo_id: Optional[str] = None
    centroid: Optional[List[float]] = None
    data: Optional[Dict] = {}
    chapter_id: Optional[str] = None
    chapter: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True  # This allows Pydantic to work with SQLAlchemy models directly
        

class ClusterPhotoRepCreate(ClusterPhotoRepBase):
    # For creating a new entry, run_id, cluster_label, and other fields can be passed
    pass


class ClusterPhotoRepResponse(ClusterPhotoRepBase):
    cluster_id: UUID

    class Config:
        orm_mode = True
