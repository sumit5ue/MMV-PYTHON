from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import ARRAY  # ✅ correct import
from sqlalchemy.dialects.postgresql import UUID, JSONB
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
