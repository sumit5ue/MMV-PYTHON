from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime


class ClusterPhotoBase(BaseModel):
    partner: str
    cluster_ver: str
    cluster_id: str
    rep_id: str
    cluster_item_id: str
    confidence:float
    chapter:str
    data: Optional[dict] = None


class ClusterPhotoCreate(ClusterPhotoBase):
    pass


class ClusterPhotoRead(ClusterPhotoBase):
    id: int
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True