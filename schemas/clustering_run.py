from pydantic import BaseModel
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime

class ClusteringRunBase(BaseModel):
    partner: Optional[str] = None
    cluster_item_type: str
    algorithm: str
    parameters: Optional[Dict[str, Any]] = {}


class ClusteringRunCreate(ClusteringRunBase):
    pass

class ClusteringRunUpdate(ClusteringRunBase):
    pass

class ClusteringRun(ClusteringRunBase):
    run_id: UUID

    class Config:
        orm_mode = True
