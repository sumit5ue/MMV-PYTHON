from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON,Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY  # ✅ correct import


from db.base import Base
import uuid

# from models.face import Face

class Photo(Base):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    photo_id = Column(String, unique=True)  # Add unique constraint here
    partner = Column(String)
    url = Column(String)
    shared_with = Column(JSON)  # store list as JSON
    app_roster_id = Column(String, nullable=True)
    caption = Column(String, nullable=True)
    is_video = Column(Boolean, default=False)
    saliency = Column(JSON, nullable=True)  # if it's a nested dict/box
    aesthetic_score = Column(Float, nullable=True)
    weighted_score = Column(Float, nullable=True)
    photo_creation_date = Column(DateTime, nullable=True)
    data = Column(JSONB, nullable=True)  # <-- arbitrary key-value store
    is_downloaded = Column(Boolean, default=False)
    is_profile = Column(Boolean, nullable=False, default=False, index=True)
    is_clip_created = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime, server_default=func.now())  # Automatically set on insert
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())  # Automatically set on update
   
    # Define the relationship with the Face model (one-to-many)
    faces = relationship("Face", back_populates="photo", cascade="all, delete-orphan")


class Face(Base):
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    face_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    photo_id = Column(String, ForeignKey("photos.photo_id", ondelete="CASCADE"), nullable=False)
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


class ClusterPhotoRep(Base):
    __tablename__ = "cluster_rep_photo"

    cluster_id = Column(UUID(as_uuid=True), primary_key=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("clustering_runs.run_id", ondelete="CASCADE"), nullable=False)
    cluster_label = Column(String, nullable=True, index=True)
    partner = Column(String, nullable=True, index=True)
    rep_photo_id = Column(String, ForeignKey("photos.photo_id", ondelete="SET NULL"), nullable=True)
    centroid = Column(ARRAY(Float), nullable=True)
    data = Column(JSONB, nullable=True, default=dict)
    chapter_id=Column(String, nullable=True)
    chapter=Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ClusterPhotoRepPhotoMapping(Base):
    __tablename__ = "cluster_photo_rep_photo_mapping"

    mapping_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    photo_id = Column(String, ForeignKey("photos.photo_id", ondelete="CASCADE"), nullable=False)
    
    # Replacing `cluster_id` with `label` and `run_id`
    label = Column(String, nullable=False)  # Cluster's label
    run_id = Column(UUID(as_uuid=True), nullable=False)  # Clustering run identifier
    
    partner = Column(String, nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    chapter_id = Column(String, nullable=True)
    data = Column(JSONB, nullable=True, default=dict)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Optional index to ensure uniqueness of face_id and label combination
    # __table_args__ = (
    #     sqlalchemy.Index("ix_cluster_face_mapping_face_label", "face_id", "label", unique=True),
    # )
