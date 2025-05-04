# from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
# from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
# from sqlalchemy.sql import func
# from sqlalchemy.orm import declarative_base
# import uuid

# Base = declarative_base()

# class Photo(Base):
#     __tablename__ = "photos"

#     id = Column(Integer, primary_key=True, autoincrement=True)
#     photo_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
#     partner = Column(String, nullable=True, index=True)
#     file_path = Column(String, nullable=False)
#     metadata = Column(JSONB, nullable=True, default=dict)

#     created_at = Column(DateTime, server_default=func.now())
#     updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

# class Face(Base):
#     __tablename__ = "faces"

#     id = Column(Integer, primary_key=True, autoincrement=True)
#     face_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
#     photo_id = Column(UUID(as_uuid=True), ForeignKey("photos.photo_id", ondelete="CASCADE"), nullable=False)
#     partner = Column(String, nullable=True, index=True)
#     assigned_id = Column(UUID(as_uuid=True), nullable=True, index=True)
#     bb_x = Column(Float, nullable=True)
#     bb_y = Column(Float, nullable=True)
#     bb_width = Column(Float, nullable=True)
#     bb_height = Column(Float, nullable=True)
#     embedding = Column(ARRAY(Float), nullable=True)
#     det_score = Column(Float, nullable=True)
#     pitch = Column(Float, nullable=True)
#     yaw = Column(Float, nullable=True)
#     roll = Column(Float, nullable=True)
#     landmarks = Column(ARRAY(Float), nullable=True)
#     member_id = Column(String, nullable=True)
#     roster_id = Column(String, nullable=True)
#     aws_face_id = Column(String, nullable=True)
#     data = Column(JSONB, nullable=True, default=dict)

#     created_at = Column(DateTime, server_default=func.now())
#     updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

# class ClusteringRun(Base):
#     __tablename__ = "clustering_runs"

#     run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     partner = Column(String, nullable=True, index=True)
#     algorithm = Column(String, nullable=False)
#     parameters = Column(JSONB, nullable=True, default=dict)

#     created_at = Column(DateTime, server_default=func.now())
#     updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

# class Cluster(Base):
#     __tablename__ = "clusters"

#     cluster_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     run_id = Column(UUID(as_uuid=True), ForeignKey("clustering_runs.run_id", ondelete="CASCADE"), nullable=False)
#     partner = Column(String, nullable=True, index=True)
#     rep_face_id = Column(UUID(as_uuid=True), ForeignKey("faces.face_id", ondelete="SET NULL"), nullable=True)
#     centroid = Column(ARRAY(Float), nullable=True)
#     data = Column(JSONB, nullable=True, default=dict)

#     created_at = Column(DateTime, server_default=func.now())
#     updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

# class ClusterFaceMapping(Base):
#     __tablename__ = "cluster_face_mapping"

#     mapping_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     face_id = Column(UUID(as_uuid=True), ForeignKey("faces.face_id", ondelete="CASCADE"), nullable=False)
#     cluster_id = Column(UUID(as_uuid=True), ForeignKey("clusters.cluster_id", ondelete="CASCADE"), nullable=False)
#     partner = Column(String, nullable=True, index=True)
#     confidence = Column(Float, nullable=True)
#     member_id = Column(String, nullable=True)
#     roster_id = Column(String, nullable=True)
#     aws_face_id = Column(String, nullable=True)
#     data = Column(JSONB, nullable=True, default=dict)

#     created_at = Column(DateTime, server_default=func.now())
#     updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

#     __table_args__ = (
#         sqlalchemy.Index("ix_cluster_face_mapping_face_cluster", "face_id", "cluster_id", unique=True),
#     )