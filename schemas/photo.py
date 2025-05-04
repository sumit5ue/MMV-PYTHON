from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

# ✅ Base schema shared by all others
class PhotoBase(BaseModel):
    partner: str
    url: str
    shared_with: List[str] = []
    app_roster_id: Optional[str] = None
    caption: Optional[str] = None
    is_video: Optional[bool] = False
    saliency: Optional[Dict[str, Any]] = None  # Corrected to JSON as a Dict
    aesthetic_score: Optional[float] = None
    weighted_score: Optional[float] = None
    photo_creation_date: Optional[datetime] = None
    data: Optional[Dict[str, Any]] = None
    is_downloaded: Optional[bool] = False
    is_profile: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

# 🟢 POST /photos — Create
class PhotoCreate(PhotoBase):
    photo_id: str  # Same as the `photo_id` in the SQLAlchemy model

# 🟡 PATCH /photos/{id} — Update
class PhotoUpdate(BaseModel):
    caption: Optional[str] = None
    aesthetic_score: Optional[float] = None
    weighted_score: Optional[float] = None
    saliency: Optional[Dict[str, Any]] = None  # Corrected to JSON as a Dict
    data: Optional[Dict[str, Any]] = None
    is_downloaded: Optional[bool] = None
    data: Optional[Dict[str, Any]] = None
    is_downloaded: Optional[bool] = False
    # Add only the fields that are allowed to be updated

# 🔵 GET /photos/{id} — Read (or list)
class PhotoSchema(PhotoCreate):
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class PhotoDownloadRequest(BaseModel):
    partner: str
    concurrency: int = 20
    limit: int = 10000
    update_batch_size: int = 100

class ClipEmbedRequest(BaseModel):
    partner: str   
    batch_size: int = 4