from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import ARRAY  # ✅ correct import
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db.base import Base
import uuid

class Clip(Base):
    __tablename__ = "clip"

    id = Column(Integer, primary_key=True, autoincrement=True)  # ✅ Integer ID for Faiss

    embedding = Column(ARRAY(Float), nullable=False)  # ✅ Stored as float[]    
    photoId = Column(String, ForeignKey("photos.photo_id", ondelete="CASCADE"), nullable=False)
    data = Column(JSONB, nullable=True, default=dict)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
