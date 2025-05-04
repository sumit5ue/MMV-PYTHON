from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from db.base import Base  # adjust this import to your project
import uuid
from sqlalchemy.dialects.postgresql import ARRAY  # ✅ correct import


class PhotoCluster(Base):
    __tablename__ = "photo_clusters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    partner = Column(String, nullable=False, index=True)
    cluster_ver = Column(String, nullable=False)
    cluster_id = Column(String, nullable=False, index=True)

    rep_id = Column(UUID(as_uuid=True), ForeignKey("photos.id"), nullable=False)
    cluster_item_id = Column(UUID(as_uuid=True), ForeignKey("photos.id"), nullable=False)

    confidence = Column(Float, nullable=False)
    chapter = Column(String, nullable=False)
    data = Column(JSONB, nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


# # Optional composite index for faster multi-column queries
# Index("ix_photo_clusters_partner_cluster", PhotoCluster.partner, PhotoCluster.cluster_id)
