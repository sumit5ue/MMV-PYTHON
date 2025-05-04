from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime


class ClusterFaceBase(BaseModel):
    partner: str
    cluster_ver: str
    cluster_id: int
    rep_id: int
    cluster_item_id: str
    confidence:float
    member_id:str
    roster_id:str
    aws_face_id:str
    data: Optional[dict] = None


class ClusterFaceCreate(ClusterFaceBase):
    pass


class ClusterFaceRead(ClusterFaceBase):
    id: int
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True