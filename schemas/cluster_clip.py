from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic.networks import HttpUrl

class ClusterClipBase(BaseModel):
    photo_id: str
    label: str
    run_id: UUID
    partner: Optional[str] = None
    confidence: Optional[float] = None
    category: Optional[str] = None
    chapter: Optional[str] = None
    data: Optional[dict] = {}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class ClusterClipCreate(ClusterClipBase):
    pass

class ClusterClipUpdate(ClusterClipBase):
    pass
