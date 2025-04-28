# schemas/processing.py

from pydantic import BaseModel
from typing import Optional

class PartnerRequest(BaseModel):
    partner: str

class FolderProcessingRequest(BaseModel):
    partner: str
    modelType: str  # clip, dino, insightface

class PartnerModelRequest(BaseModel):
    partner: str
    model: str  # clip, dino, or insightface

class ClusterRequest(BaseModel):
    partner: str
    model: str  # "clip", "dino", "insightface"
    method: str = "hdbscan"  # "hdbscan" or "dbscan"
    distance: Optional[str] = None  # "cosine" or "euclidean"
    minClusterSize: Optional[int] = 2
    eps: Optional[float] = 0.5  # only for dbscan