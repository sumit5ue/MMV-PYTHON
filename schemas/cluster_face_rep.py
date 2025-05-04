from pydantic import BaseModel
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic.networks import HttpUrl


class ClusterFaceBase(BaseModel):
    run_id: UUID
    cluster_label:str
    partner: Optional[str] = None
    rep_face_id: Optional[UUID] = None
    centroid: Optional[list[float]] = None
    data: Optional[Dict[str, Any]] = {}
    aws_face_id: Optional[str] = None
    aws_external_image_id: Optional[str] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ClusterFaceCreate(ClusterFaceBase):
    pass

class ClusterFaceUpdate(ClusterFaceBase):
    pass

class ClusterFace(ClusterFaceBase):
    cluster_id: UUID

    class Config:
        orm_mode = True
