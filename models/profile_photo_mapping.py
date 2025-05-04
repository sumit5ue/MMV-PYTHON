from sqlalchemy import Column, String, Float, Boolean, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY  # ✅ correct import

class ProfilePhotoMapping(Base):
    __tablename__ = "profile_photo_mappings"

    mapping_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    photo_id = Column(UUID(as_uuid=True), ForeignKey("photos.photo_id", ondelete="CASCADE"), nullable=False)
    partner = Column(String, nullable=True, index=True)
    aws_face_id = Column(String, nullable=False)
    external_image_id = Column(String, nullable=True)
    member_id = Column(String, nullable=True)
    roster_id = Column(String, nullable=True)
    metadata = Column(JSONB, nullable=True, default=dict)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        sqlalchemy.Index("ix_profile_photo_mappings_aws_face_id", "aws_face_id"),
    )