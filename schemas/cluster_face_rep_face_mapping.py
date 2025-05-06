from pydantic import BaseModel
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime

class ClusterFaceRepFaceMappingBase(BaseModel):
    face_id: str
    cluster_id:UUID
    run_id: UUID
    label: str
    partner: Optional[str] = None
    confidence: Optional[float] = None
    aws_face_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = {}

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ClusterFaceRepFaceMappingCreate(ClusterFaceRepFaceMappingBase):
    pass

class ClusterFaceRepFaceMappingUpdate(ClusterFaceRepFaceMappingBase):
    pass

class ClusterFaceRepFaceMapping(ClusterFaceRepFaceMappingBase):
    mapping_id: UUID

    class Config:
        orm_mode = True
