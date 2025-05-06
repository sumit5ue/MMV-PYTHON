# db/init_db.py

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.session import engine
from db.base import Base

from models.photo import Photo,Face,ClusterPhotoRep,ClusterPhotoRepPhotoMapping,ClusteringRun,ClusterFaceRep,ClusterFaceRepFaceMapping  # ⬅️ This is crucial
# from models.face import Face
# from models.clustering_run import ClusteringRun
# from models.cluster_face_rep import ClusterFaceRep
# from models.cluster_face_rep_face_mapping import ClusterFaceRepFaceMapping
# from models.cluster_photo_rep_photo_mapping import ClusterPhotoRepPhotoMapping
from models.clip import Clip
# from models.cluster_face_rep import FaceCluster
# from models.cluster_photo import PhotoCluster

def create_tables():
    print("📦 Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created.")

if __name__ == "__main__":
    create_tables()
