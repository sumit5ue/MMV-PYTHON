from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from db.base import Base
import uuid
from sqlalchemy.dialects.postgresql import ARRAY  # ✅ correct import

# class ClusterPhotoRepPhotoMapping(Base):
#     __tablename__ = "cluster_photo_rep_photo_mapping"

#     mapping_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     photo_id = Column(UUID(as_uuid=True), ForeignKey("faces.face_id", ondelete="CASCADE"), nullable=False)
    
#     # Replacing `cluster_id` with `label` and `run_id`
#     label = Column(String, nullable=False)  # Cluster's label
#     run_id = Column(UUID(as_uuid=True), nullable=False)  # Clustering run identifier
    
#     partner = Column(String, nullable=True, index=True)
#     confidence = Column(Float, nullable=True)
#     chapter_id = Column(String, nullable=True)
#     data = Column(JSONB, nullable=True, default=dict)

#     created_at = Column(DateTime, server_default=func.now())
#     updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

#     # Optional index to ensure uniqueness of face_id and label combination
#     # __table_args__ = (
#     #     sqlalchemy.Index("ix_cluster_face_mapping_face_label", "face_id", "label", unique=True),
#     # )
