from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
import uuid
from sqlalchemy.orm import relationship

# from models.photo import Photo


from db.base import Base

class Face(Base):
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    face_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    photo_id = Column(UUID(as_uuid=True), ForeignKey("photos.photo_id", ondelete="CASCADE"), nullable=False)
    partner = Column(String, nullable=True, index=True)
    bbox = Column(JSONB, nullable=True, default=dict)   
    embedding = Column(ARRAY(Float), nullable=True)
    det_score = Column(Float, nullable=True)
    pose = Column(JSONB, nullable=True, default=dict)   
    landmarks = Column(JSONB, nullable=True, default=dict) 
    member_id = Column(String, nullable=True)
    roster_id = Column(String, nullable=True)
    data = Column(JSONB, nullable=True, default=dict)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationship to the Photo model
    photo = relationship("Photo", back_populates="faces")

