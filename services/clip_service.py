# services/clip_service.py

import torch
import clip
import numpy as np
import traceback
from PIL import Image
from config import get_embeddings_dir, get_metadata_path, get_photos_dir
from utils.jsonl_utils import load_jsonl, save_jsonl
from utils.error_utils import log_error
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
import os
from uuid import UUID
from models.photo import Photo
from models.clip import Clip
from db.session import SessionLocal
import asyncio
import aiohttp


device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def embed_clip_for_photo(photo_path: str, partner: str, photo_id: str):
    try:
        image = Image.open(photo_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = clip_model.encode_image(image_input)
            embedding = embedding.float()
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        embeddings_dir = get_embeddings_dir(partner)
        clip_npy_path = os.path.join(embeddings_dir, f"{partner}_clip.npy")
        os.makedirs(embeddings_dir, exist_ok=True)

        if os.path.exists(clip_npy_path):
            existing = np.load(clip_npy_path)
            combined = np.vstack([existing, embedding.cpu().numpy()])
        else:
            combined = embedding.cpu().numpy()

        np.save(clip_npy_path, combined)

        metadata_path = get_metadata_path(partner)
        update_clip_metadata(metadata_path, photo_id, combined.shape[0] - 1)

    except Exception as e:
        log_error(
            photoId=photo_id,
            faceId=None,
            step="clip",
            errorMessage=str(e),
            traceback=traceback.format_exc(),
            modelType="clip",
            path=photo_path,
            partner=partner
        )


def get_embed_clip_for_photo(photo_path: str, partner: str, photo_id: str):
    try:
        
        photos_dir = get_photos_dir(partner)
        full_photo_path = os.path.join(photos_dir, str(photo_path)+".jpg")
        print("full_photo_path",full_photo_path)
        if not os.path.exists(full_photo_path):
            print(f"Error: {full_photo_path} does not exist.")
        image = Image.open(full_photo_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(image_input)
            embedding = embedding.float()
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        # Convert the embedding to numpy array for further processing or saving
        embedding_np = embedding.cpu().numpy()

        return embedding_np
    
    except Exception as e:
        print(f"Error {e}")
        traceback.print_exc()


def update_clip_metadata(metadata_path, photo_id, vectorId):
    metadata = load_jsonl(metadata_path)
    for entry in metadata:
        if entry["id"] == photo_id:
            if "clip" not in entry:
                entry["clip"] = {}
            entry["clip"]["vectorId"] = vectorId
            entry["clip"]["processed"] = True
            break
    save_jsonl(metadata, metadata_path)

def process_clip_folder(partner: str):
    photos_dir = get_photos_dir(partner)
    metadata_path = get_metadata_path(partner)

    if not os.path.exists(photos_dir):
        raise Exception(f"No photos found at {photos_dir}")

    photo_files = [f for f in os.listdir(photos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Step 1: Initialize metadata.jsonl if missing
    if not os.path.exists(metadata_path):
        metadata_entries = []
        for file in photo_files:
            photo_id = os.path.splitext(file)[0]
            metadata_entries.append({
                "id": photo_id,
                "fileName": file,
                "isUtility": False,
                "objectSaliency": [],
                "aestheticScore": None,
                "faceCount": None,
                "clip": {"processed": False},
                "dino": {"processed": False}
            })
        save_jsonl(metadata_entries, metadata_path)
        print(f"✅ Created metadata.jsonl for {len(metadata_entries)} photos")

    metadata = load_jsonl(metadata_path)
    id_to_metadata = {item['id']: item for item in metadata}

    processed_count = 0
    skipped_count = 0

    for idx, file in enumerate(photo_files):
        full_path = os.path.join(photos_dir, file)
        photo_id = os.path.splitext(file)[0]

        meta = id_to_metadata.get(photo_id)
        if not meta:
            continue  # no metadata? skip
        if meta.get("clip", {}).get("processed") is True:
            skipped_count += 1
            continue

        embed_clip_for_photo(full_path, partner, photo_id)
        processed_count += 1

        if (processed_count + skipped_count) % 10 == 0:
            print(f"Processed {processed_count}, skipped {skipped_count} out of {len(photo_files)} photos...")

    return {
        "processed": processed_count,
        "skipped": skipped_count,
        "total": len(photo_files)
    }

def process_and_save_clip_embeddings(db: Session, partner: str):
    photos_dir = get_photos_dir(partner)
    print("photos_dir", photos_dir)
    
    if not os.path.exists(photos_dir):
        raise Exception(f"No photos found at {photos_dir}")
    embeddings = []
    photo_ids = []
    photo_files = [f for f in os.listdir(photos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    clips_data = []
    processed_count = 0
    skipped_count = 0
    
    # Step 1: Load metadata for photos
   
    for file in photo_files:
        full_path = os.path.join(photos_dir, file)
        photo_id = os.path.splitext(file)[0]
        
        # Step 2: Check metadata and process clip
        meta = id_to_metadata.get(photo_id)
        if not meta:
            continue  # Skip if metadata is missing
        
        if meta.get("clip", {}).get("processed") is True:
            skipped_count += 1
            continue
        
        # Get CLIP embedding for the photo
        embedding = embed_clip_for_photo(full_path)
        
        # Create Clip entry
        clip_entry = {
            "embedding": embedding.tolist(),  # Convert numpy array to list
            "photoId": photo_id,
            "data": {},  # Optional additional data
        }
        
        clips_data.append(clip_entry)
        processed_count += 1

        # Save clips in batches of 50 (adjust batch size as needed)
        if len(clips_data) >= 50:
            save_clip_batch_to_db(db, clips_data)
            clips_data = []  # Reset the batch

    # Save any remaining clips that didn't fill a full batch
    if clips_data:
        save_clip_batch_to_db(db, clips_data)

    return {
        "processed": processed_count,
        "skipped": skipped_count,
        "total": len(photo_files)
    }

def get_clip_embedding(image_path: str):
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = clip_model.encode_image(image_input)
            embedding = embedding.float()
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


def process_and_save_clip_embeddings_for_partner(db: Session, partner: str, batch_size: int=20):
    try:
        # Step 1: Fetch photos from PG that haven't been processed yet
        photos_to_process = db.query(Photo).filter(
            Photo.partner == partner, Photo.is_clip_created == False
        ).limit(10).all()

        if not photos_to_process:
            print("No photos to process for partner", partner)
            return {"message": "No photos to process"}

        photo_ids = [photo.photo_id for photo in photos_to_process]
        photo_files = [photo.file_path for photo in photos_to_process]  # Assuming you have a `file_path` column
        
        # Step 2: Process each photo to get embeddings
        embeddings = []
        for photo_path, photo_id in zip(photo_files, photo_ids):
            embedding = embed_clip_for_photo(photo_path, partner, photo_id)
            embeddings.append(embedding)

        # Step 3: Save the embeddings to the database in batches
        save_embeddings_to_pg(db, embeddings, photo_ids, batch_size, partner)
        
        return {"message": f"Processed and updated--- {len(embeddings)} photos for partner {partner}"}

    except Exception as e:
        print(f"Error fetching and processing photos: {str(e)}")
        return {"message": f"Error: {str(e)}"}
    

async def clip_photos_from_db(partner: str, batch_size: int):
    try:
        db = SessionLocal()
        # Fetch photo_id values from PostgreSQL for the given partner
        photos_to_process = db.query(Photo).filter(
            Photo.partner == partner, Photo.is_clip_created == False,Photo.is_downloaded == True
        ).all()

        if not photos_to_process:
            print(f"No photos to process for {partner}")
            return {"message": "No photos to process"}

        photo_ids = [photo.photo_id for photo in photos_to_process]
        # Get the folder path where photos are stored
        photos_dir = get_photos_dir(partner)  # Assuming you have this function to get the folder path

        # Get the list of image files from the directory
        # photo_files = [f for f in os.listdir(photos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        

        # Process the photos in batches of `batch_size`
        total_photos = len(photo_ids)
        
        for start in range(0, total_photos, batch_size):
            end = min(start + batch_size, total_photos)
            photos_batch = photo_ids[start:end]
            photo_ids_batch = photo_ids[start:end]

            # Process and save embeddings for the current batch of photos
            process_and_save_embeddings_in_batches(db, partner, photos_batch, photo_ids_batch)

        return {"message": f"Processed and updated ---{total_photos} photos for partner {partner}"}

    except Exception as e:
        print(f"Error fetching and processing photos: {str(e)}")
        return {"message": f"Error: {str(e)}"}

def process_and_save_embeddings_in_batches(db: Session, partner: str, photos_batch: list, photo_ids_batch: list):
    embeddings = []
    clip_records = []
    
    try:
        # Process each photo in the batch to get the embedding
        for photo_path, photo_id in zip(photos_batch, photo_ids_batch):
            embedding = get_embed_clip_for_photo(photo_path, partner, photo_id)  # Assuming this returns the embedding
            # embedding =  [float(x) for x in embedding.tolist()]
            embedding = embedding.flatten().tolist() 
            embeddings.append(embedding)
            # Prepare the Clip records to insert into the database
            clip_record = Clip(
                embedding=embedding,  # Flatten the embedding to match the database column type
                photoId=photo_id
            )
            clip_records.append(clip_record)

        # Bulk insert embeddings into the Clip table
        db.bulk_save_objects(clip_records)
        db.commit()  # Commit the batch

        # Update the corresponding photos as processed
        db.query(Photo).filter(Photo.photo_id.in_(photo_ids_batch)).update(
            {Photo.is_clip_created: True}, synchronize_session=False
        )
        db.commit()

        print(f"Processed and updated {len(embeddings)} photos for partner {partner}")
    
    except Exception as e:
        print(f"Error processing and saving embeddings: {str(e)}")
        db.rollback()  # Rollback the transaction in case of error
        return {"message": f"Error: {str(e)}"}


    