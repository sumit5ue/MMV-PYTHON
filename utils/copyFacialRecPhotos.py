import sys
import os
from pathlib import Path
import shutil
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Add the root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from models.photo import Face  # Replace with your actual model import

# Set up the database session (assuming you have a SQLAlchemy session set up)
DATABASE_URL = "postgresql://postgres:sumit123@localhost/vector_db"  # Replace with your database URL
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def copy_images_to_downloads(aws_face_id: str, partner: str):
    # Query to find all photo records with the specified aws_face_id
    faces = session.query(Face).filter(Face.aws_face_id == aws_face_id).all()
    
    print(f"Found {len(faces)} photos for aws_face_id: {aws_face_id}")

    if not faces:
        print(f"No faces found with aws_face_id: {aws_face_id}")
        return

    # Define the destination folder for copying the images
    destination_folder = os.path.expanduser('~/Downloads/sample')

    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate over the faces and copy each corresponding image
    for face in faces:
        photo_id = face.photo_id  # Get photo_id for each face record

        # Define source and destination paths
        source_path = f"/Users/sumit/Documents/ai_analysis/{partner}/photos/{photo_id}.jpg"
        destination_path = os.path.join(destination_folder, f"{photo_id}.jpg")

        # Check if the source file exists
        if not os.path.exists(source_path):
            print(f"Source file does not exist: {source_path}")
            continue  # Skip this file if it doesn't exist

        # Copy the file to the destination folder
        try:
            shutil.copy(source_path, destination_path)
            print(f"File {photo_id} copied successfully to {destination_path}")
        except Exception as e:
            print(f"Error while copying the file {photo_id}: {e}")

# Example usage:
aws_face_id = "7817dcec-1ef4-4c38-bed4-220d2957fec4"  # Replace with the actual aws_face_id
partner = "67ec2a60d4d64df971004210"  # Replace with the actual partner name
copy_images_to_downloads(aws_face_id, partner)
