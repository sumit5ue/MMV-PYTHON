from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from db.base import Base  # adjust as needed
import uuid


class FaceCluster(Base):
    __tablename__ = "face_clusters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    partner = Column(String, nullable=False, index=True)
    cluster_ver = Column(String, nullable=False)
    cluster_id = Column(String, nullable=False, index=True)

    rep_id = Column(Integer, ForeignKey("faces.id"), nullable=False)
    cluster_item_id = Column(Integer, ForeignKey("faces.id"), nullable=False)

    confidence = Column(Float, nullable=True)
    member_id = Column(String, nullable=True)
    roster_id = Column(String, nullable=True)
    aws_face_id = Column(String, nullable=True)

    data = Column(JSONB, nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


# Optional composite index (if you query by multiple keys together)
# Index("ix_face_clusters_partner_cluster", FaceCluster.partner, FaceCluster.cluster_id)
