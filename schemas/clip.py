from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime


class ClipBase(BaseModel):
    photoId: str
    embedding: List[float]
    data: Optional[dict] = None


class ClipCreate(ClipBase):
    pass


class ClipRead(ClipBase):
    id: int
    created_at: Optional[datetime]

    class Config:
        orm_mode = True
