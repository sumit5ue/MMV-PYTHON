from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

# ✅ Base schema shared by all others
class FaceBase(BaseModel):
    face_id: UUID  # UUID of the face
    photo_id: UUID  # Foreign key to the `photos` table
    partner: Optional[str] = None  # Partner associated with the face
    bbox_x: Optional[float] = None  # Detection score
    bbox_y: Optional[float] = None  # Detection score
    bbox_width: Optional[float] = None  # Detection score
    bbox_height: Optional[float] = None  # Detection score
    embedding: Optional[List[float]] = None  # Embedding as an array of floats
    det_score: Optional[float] = None  # Detection score
    pose: Optional[Dict[str, Any]] = None  # Pose as a JSONB (a dictionary)
    landmarks: Optional[Dict[str, Any]] = None  # Landmarks as a JSONB (a dictionary)
    member_id: Optional[str] = None  # Member ID associated with the face
    roster_id: Optional[str] = None  # Roster ID associated with the face
    aws_face_id=Optional[str] = None 
    external_image_id=Optional[str] = None 
    similarity: Optional[float] = None 
    eye_direction: Optional[Dict[str, Any]] = None
    is_aws_error: Optional[bool] = False
    aws_error: Optional[Dict[str, Any]] = None
    
    
    data: Optional[Dict[str, Any]] = None  # Metadata as a JSONB (a dictionary)

    created_at: Optional[datetime] = None  # Timestamp when the face record was created
    updated_at: Optional[datetime] = None  # Timestamp when the face record was last updated

# 🟢 POST /faces — Create
class FaceCreate(FaceBase):
    id: Optional[int]  # Optional because it will be auto-generated by SQLAlchemy

# 🟡 PATCH /faces/{id} — Update
class FaceUpdate(BaseModel):
    partner: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None
    det_score: Optional[float] = None
    pose: Optional[Dict[str, Any]] = None
    landmarks: Optional[List[float]] = None
    member_id: Optional[str] = None
    roster_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    # Add only the fields that are allowed to be updated

# 🔵 GET /faces/{id} — Read (or list)
class FaceSchema(FaceCreate):
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True  # To read data from SQLAlchemy model instances
   