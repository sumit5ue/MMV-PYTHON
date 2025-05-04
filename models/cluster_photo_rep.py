from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from db.base import Base  # adjust as needed
import uuid
from sqlalchemy.dialects.postgresql import ARRAY  # ✅ correct import


# class ClusterPhotoRep(Base):
#     __tablename__ = "cluster_rep_photo"

#     cluster_id = Column(UUID(as_uuid=True), primary_key=True)
#     run_id = Column(UUID(as_uuid=True), ForeignKey("clustering_runs.run_id", ondelete="CASCADE"), nullable=False)
#     cluster_label = Column(String, nullable=True, index=True)
#     partner = Column(String, nullable=True, index=True)
#     rep_photo_id = Column(UUID(as_uuid=True), ForeignKey("photos.photo_id", ondelete="SET NULL"), nullable=True)
#     centroid = Column(ARRAY(Float), nullable=True)
#     data = Column(JSONB, nullable=True, default=dict)
#     chapter_id=Column(String, nullable=True)
#     chapter=Column(String, nullable=True)
#     created_at = Column(DateTime, server_default=func.now())
#     updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
