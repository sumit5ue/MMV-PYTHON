from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from db.base import Base
import uuid

class ClusteringRun(Base):
    __tablename__ = "clustering_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    partner = Column(String, nullable=True, index=True)
    cluster_item_type = Column(String, nullable=False)
    algorithm = Column(String, nullable=False)
    parameters = Column(JSONB, nullable=True, default=dict)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class Photo(Base):
    __tablename__ = "photos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    photo_id = Column(String, unique=True, default=lambda: str(uuid.uuid4()))  # Kept as String
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
    face_count = Column(Integer, default=0)
    data = Column(JSONB, nullable=True)  # arbitrary key-value store
    is_downloaded = Column(Boolean, default=False)
    is_profile = Column(Boolean, nullable=False, default=False, index=True)
    is_clip_created = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    faces = relationship("Face", back_populates="photo", cascade="all, delete-orphan")

class Face(Base):
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    face_id = Column(String, unique=True, default=lambda: str(uuid.uuid4()))  # Kept as String
    photo_id = Column(String, ForeignKey("photos.photo_id", ondelete="CASCADE"), nullable=False)
    partner = Column(String, nullable=True, index=True)
    bbox_x = Column(Float, nullable=True)
    bbox_y = Column(Float, nullable=True)
    bbox_width = Column(Float, nullable=True)
    bbox_height = Column(Float, nullable=True)
    embedding = Column(ARRAY(Float), nullable=True)
    det_score = Column(Float, nullable=True)
    pose = Column(JSONB, nullable=True, default=dict)   
    member_id = Column(String, nullable=True)
    roster_id = Column(String, nullable=True)
    aws_face_id=Column(String, nullable=True)
    similarity = Column(Float, nullable=True)
    eye_direction = Column(JSONB, nullable=True, default=dict)
    landmarks = Column(JSONB, nullable=True, default=dict) 
    is_aws_error = Column(Boolean, nullable=True,default = False)
    external_image_id=Column(String, nullable=True)
    aws_error = Column(JSONB, nullable=True, default=dict) 
    
    data = Column(JSONB, nullable=True, default=dict)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    photo = relationship("Photo", back_populates="faces")

class ClusterPhotoRep(Base):
    __tablename__ = "cluster_rep_photo"

    cluster_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("clustering_runs.run_id", ondelete="CASCADE"), nullable=False)
    cluster_label = Column(String, nullable=True, index=True)
    partner = Column(String, nullable=True, index=True)
    rep_photo_id = Column(String, ForeignKey("photos.photo_id", ondelete="SET NULL"), nullable=True)
    centroid = Column(ARRAY(Float), nullable=True)
    data = Column(JSONB, nullable=True, default=dict)
    chapter_id = Column(String, nullable=True)
    chapter = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class ClusterPhotoRepPhotoMapping(Base):
    __tablename__ = "cluster_photo_rep_photo_mapping"

    mapping_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("cluster_rep_photo.cluster_id", ondelete="CASCADE"), nullable=False)
    photo_id = Column(String, ForeignKey("photos.photo_id", ondelete="CASCADE"), nullable=False)
    label = Column(String, nullable=False)  # Cluster's label
    run_id = Column(UUID(as_uuid=True), ForeignKey("clustering_runs.run_id", ondelete="CASCADE"), nullable=False)
    partner = Column(String, nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    chapter_id = Column(String, nullable=True)
    data = Column(JSONB, nullable=True, default=dict)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class ClusterFaceRep(Base):
    __tablename__ = "cluster_rep_face"

    cluster_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("clustering_runs.run_id", ondelete="CASCADE"), nullable=False)
    cluster_label = Column(String, nullable=True, index=True)
    partner = Column(String, nullable=True, index=True)
    rep_face_id = Column(String, ForeignKey("faces.face_id", ondelete="SET NULL"), nullable=True)
    centroid = Column(ARRAY(Float), nullable=True)
    data = Column(JSONB, nullable=True, default=dict)
    aws_face_id = Column(String, nullable=True)
    aws_external_image_id = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

class ClusterFaceRepFaceMapping(Base):
    __tablename__ = "cluster_face_rep_face_mapping"

    mapping_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    face_id = Column(String, ForeignKey("faces.face_id", ondelete="CASCADE"), nullable=False)
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("cluster_rep_face.cluster_id", ondelete="CASCADE"), nullable=False)
    label = Column(String, nullable=False)  # Cluster's label
    run_id = Column(UUID(as_uuid=True), ForeignKey("clustering_runs.run_id", ondelete="CASCADE"), nullable=False)
    partner = Column(String, nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    aws_face_id = Column(String, nullable=True)
    data = Column(JSONB, nullable=True, default=dict)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ClusterClip(Base):
    __tablename__ = "cluster_clip"

    id = Column(Integer, primary_key=True, autoincrement=True)
    photo_id = Column(String, ForeignKey("photos.photo_id", ondelete="CASCADE"), nullable=False)
    label = Column(String, nullable=False)  # Cluster's label
    run_id = Column(UUID(as_uuid=True), ForeignKey("clustering_runs.run_id", ondelete="CASCADE"), nullable=False)
    partner = Column(String, nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    category= Column(String, nullable=True)
    chapter= Column(String, nullable=True)
    data = Column(JSONB, nullable=True, default=dict)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
