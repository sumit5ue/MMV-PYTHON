import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from pydantic import ValidationError
import sys
from pymongo import MongoClient

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# ---- Import your models ----
from schemas.photo import PhotoCreate
from models.photo import Photo, Face  # SQLAlchemy model
from db.session import SessionLocal  # DB session factory

# MongoDB URI and collection details
MONGO_URI = "mongodb+srv://sumit5ue:SumitStar2024@cluster0.bpw9q.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "test"
PHOTO_CHAPTER_COLLECTION = "photo_face"  # Correct the name to match your collection

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
mongoDb = client[DATABASE_NAME]
photo_chapter_collection = mongoDb[PHOTO_CHAPTER_COLLECTION]

def transfer_face_to_mongodb(partner: str):
    # Create a session to query PostgreSQL
    session = SessionLocal()

    # Query to fetch records from PostgreSQL for the given partner
    faces = session.query(Face).filter(Face.partner == partner, Face.is_aws_error == False, Face.aws_face_id != None).all()

    print(f"Found {len(faces)} faces for partner: {partner}")

    # Prepare list of documents for MongoDB
    for face in faces:
        document = {
            "face_id":face.face_id,
            "aws_face_id": face.aws_face_id,
            "photo_id": face.photo_id,
            "partner": face.partner,
            "bbox_x": face.bbox_x,
            "bbox_y": face.bbox_y,
            "bbox_width": face.bbox_width,
            "bbox_height": face.bbox_height,
            "det_score": face.det_score,
            "pose": face.pose,
            "similarity": face.similarity,
            "external_image_id": face.external_image_id,  # Fixed typo
        }

        # Perform an upsert operation based on the aws_face_id (or any unique field)
        photo_chapter_collection.update_one(
            {"face_id": face.face_id},  # Query to find existing document
            {"$set": document},  # Update fields with new data
            upsert=True  # If no document is found, insert a new one
        )

    print(f"Successfully upserted {len(faces)} faces to MongoDB.")

    # Close the session
    session.close()

# Example usage
partner = "67ec2a60d4d64df971004210"  # Replace with the actual partner id
transfer_face_to_mongodb(partner)
