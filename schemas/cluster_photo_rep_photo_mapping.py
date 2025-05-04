from pydantic import BaseModel
from typing import Optional, Dict
from uuid import UUID
from datetime import datetime

class ClusterPhotoRepPhotoMappingBase(BaseModel):
    photo_id: UUID
    label: str  # Cluster's label
    run_id: UUID  # Clustering run identifier
    partner: Optional[str] = None
    confidence: Optional[float] = None
    chapter_id: Optional[str] = None
    data: Optional[Dict] = {}  # This will be a dictionary if provided
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True  # This allows Pydantic to work with SQLAlchemy models directly


class ClusterPhotoRepPhotoMappingCreate(ClusterPhotoRepPhotoMappingBase):
    # This model will be used for creating a new ClusterPhotoRepPhotoMapping record
    pass


class ClusterPhotoRepPhotoMappingResponse(ClusterPhotoRepPhotoMappingBase):
    mapping_id: UUID  # Include the primary key for responses

    class Config:
        orm_mode = True
