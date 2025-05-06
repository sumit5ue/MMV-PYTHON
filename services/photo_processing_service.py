import asyncio
import aiohttp
from io import BytesIO
from PIL import Image, ExifTags
import numpy as np
from pathlib import Path

from services.insightface_service import detect_and_embed_faces_from_array
from db.session import SessionLocal
from models.photo import Photo,Face
# from models.face import Face
from utils.file_utils import get_photo_path

def apply_exif_rotation(image: Image.Image) -> Image.Image:
    try:
        exif = image._getexif()
        if exif is not None:
            orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
            orientation = exif.get(orientation_key)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception:
        pass
    return image

# async def process_photos_async(partner, concurrency=5, limit=100, update_batch_size=100):
#     db = SessionLocal()
#     photos = (
#         db.query(Photo)
#         .filter(Photo.is_downloaded == False, Photo.partner == partner, Photo.is_video == False)
#         .limit(limit)
#         .all()
#     )

#     if not photos:
#         return {"status": "No photos found to process."}

#     face_objects = []
#     downloaded_photo_ids = []
#     lock = asyncio.Lock()

    # def flush(faces_batch, photo_ids_batch, face_count_batch):
    #     """
    #     This function is used to flush the accumulated data to the database in bulk.
    #     It saves face data and updates photo counts.
    #     """
    #     if faces_batch:
    #         db.bulk_save_objects(faces_batch)  # Bulk insert faces into the database

    #     if photo_ids_batch:
    #         # For each photo_id and face_count, update the respective photo in the database
    #         for photo_id, face_count in zip(photo_ids_batch, face_count_batch):
    #             db.query(Photo).filter(Photo.photo_id == photo_id).update(
    #                 {"is_downloaded": True, "face_count": face_count}, synchronize_session=False
    #             )

    #         db.commit()
    #         print(f"📦 Flushed {len(faces_batch)} faces and {len(photo_ids_batch)} photo updates", flush=True)


    # async def fetch_and_process(index, photo_id, url, session, sem):
    #     """
    #     Function to fetch and process a single photo, detect faces, and save face data to the database.
    #     """
    #     async with sem:
    #         try:
    #             # Download the photo and process the image
    #             print(f"[{index + 1}] Downloading {photo_id}", flush=True)
    #             async with session.get(url) as resp:
    #                 if resp.status != 200:
    #                     print(f"❌ Failed HTTP {resp.status} for {photo_id}", flush=True)
    #                     return

    #                 # Read image data into memory
    #                 img_bytes = await resp.read()
    #                 pil_img = Image.open(BytesIO(img_bytes))
    #                 rotated = apply_exif_rotation(pil_img)
    #                 image = np.array(rotated)

    #                 # Save the image to a local path
    #                 file_path = get_photo_path(partner, photo_id)
    #                 rotated.save(file_path)

    #                 # Detect faces and extract embeddings using InsightFace
    #                 faces_metadata = detect_and_embed_faces_from_array(
    #                     image=image,
    #                     partner=partner,
    #                     photo_id=photo_id
    #                 )

    #                 # Create Face objects for each detected face
    #                 face_objs = [
    #                     Face(
    #                         partner=partner,
    #                         photo_id=photo_id,
    #                         bbox_x= face["bbox"][0],
    #                         bbox_y= face["bbox"][1],
    #                         bbox_width= face["bbox"][2] - face["bbox"][0],
    #                         bbox_height= face["bbox"][3] - face["bbox"][1],
    #                         det_score=face.get("det_score"),
    #                         pose={
    #                             "pitch": face.get("pitch"),
    #                             "yaw": face.get("yaw"),
    #                             "roll": face.get("roll"),
    #                             "pose": face.get("pose"),
    #                         },
    #                         embedding=face["embedding"],
    #                         data=face.get("data")
    #                     )
    #                     for face in faces_metadata
    #                 ]

    #                 # Calculate the face count for this photo
    #                 face_count = len(face_objs)

    #                 # Accumulate data for batch processing
    #                 async with lock:
    #                     face_objects.extend(face_objs)  # Add face objects to the batch
    #                     downloaded_photo_ids.append(photo_id)  # Add photo_id to the batch
    #                     face_count_batch.append(face_count)  # Add face count for the photo

    #                     # If batch size is reached (e.g., 100 photos), flush the data
    #                     if len(downloaded_photo_ids) >= update_batch_size:
    #                         flush(face_objects.copy(), downloaded_photo_ids.copy(), face_count_batch.copy())
    #                         face_objects.clear()
    #                         downloaded_photo_ids.clear()
    #                         face_count_batch.clear()

    #             print(f"✅ Processed {photo_id} with {len(face_objs)} faces", flush=True)

    #         except Exception as e:
    #             print(f"❌ Error for {photo_id}: {e}", flush=True)

    # async def orchestrate():
    #     """
    #     Orchestrates the asynchronous fetching and processing of all photos.
    #     Ensures that all photos are processed and updated in the database.
    #     """
    #     try:
    #         print("🚀 Starting photo processing...", flush=True)
    #         sem = asyncio.Semaphore(concurrency)  # Semaphore to control concurrency
    #         async with aiohttp.ClientSession() as session:
    #             tasks = [
    #                 fetch_and_process(index, photo.photo_id, photo.url, session, sem)
    #                 for index, photo in enumerate(photos)
    #             ]
    #             await asyncio.gather(*tasks)

    #         # Flush any residual photos that didn't complete the full batch
    #         if face_objects or downloaded_photo_ids:
    #             flush(face_objects, downloaded_photo_ids, face_count_batch)

    #         print("✅ All photos processed", flush=True)

    #     except Exception as e:
    #         print(f"🔥 Orchestration error: {e}", flush=True)

async def flush(db, faces_batch: list, photo_ids_batch: list, face_count_batch: list):
    """
    Flush accumulated data to the database in bulk.
    Saves face data and updates photo counts.
    
    Args:
        db: SQLAlchemy session
        faces_batch (list): List of Face objects to save
        photo_ids_batch (list): List of photo IDs to update
        face_count_batch (list): List of face counts for each photo
    """
    try:
        if faces_batch:
            db.bulk_save_objects(faces_batch)  # Bulk insert faces

        if photo_ids_batch:
            # Update photo records with face counts and is_downloaded status
            for photo_id, face_count in zip(photo_ids_batch, face_count_batch):
                db.query(Photo).filter(Photo.photo_id == photo_id).update(
                    {"is_downloaded": True, "face_count": face_count},
                    synchronize_session=False
                )

        db.commit()
        print(f"📦 Flushed {len(faces_batch)} faces and {len(photo_ids_batch)} photo updates", flush=True)

    except Exception as e:
        db.rollback()
        print(f"❌ Flush error: {e}", flush=True)
        raise

async def fetch_and_process(index: int, photo_id: str, url: str, session: aiohttp.ClientSession, 
                         sem: asyncio.Semaphore, db, partner: str, 
                         face_objects: list, downloaded_photo_ids: list, face_count_batch: list, 
                         lock: asyncio.Lock, update_batch_size: int):
    """
    Fetch and process a single photo, detect faces, and save face data to the database.
    
    Args:
        index (int): Index of the photo in the processing list
        photo_id (str): Unique identifier for the photo
        url (str): URL to download the photo
        session (aiohttp.ClientSession): HTTP session for downloading
        sem (asyncio.Semaphore): Semaphore to control concurrency
        db: SQLAlchemy session
        partner (str): Partner identifier
        face_objects (list): Shared list to accumulate Face objects
        downloaded_photo_ids (list): Shared list to accumulate photo IDs
        face_count_batch (list): Shared list to accumulate face counts
        lock (asyncio.Lock): Lock for thread-safe batch updates
        update_batch_size (int): Batch size for flushing data
    """
    async with sem:
        try:
            # Download the photo
            print(f"[{index + 1}] Downloading {photo_id}", flush=True)
            async with session.get(url) as resp:
                if resp.status != 200:
                    print(f"❌ Failed HTTP {resp.status} for {photo_id}", flush=True)
                    return

                # Read and process image
                img_bytes = await resp.read()
                pil_img = Image.open(BytesIO(img_bytes))
                rotated = apply_exif_rotation(pil_img)
                image = np.array(rotated)

                # Save image to local path
                file_path = get_photo_path(partner, photo_id)
                rotated.save(file_path)

                # Detect faces and extract embeddings
                faces_metadata = detect_and_embed_faces_from_array(
                    image=image,
                    partner=partner,
                    photo_id=photo_id
                )

                # Create Face objects
                face_objs = [
                    Face(
                        partner=partner,
                        photo_id=photo_id,
                        bbox_x=face["bbox"][0],
                        bbox_y=face["bbox"][1],
                        bbox_width=face["bbox"][2] - face["bbox"][0],
                        bbox_height=face["bbox"][3] - face["bbox"][1],
                        det_score=face.get("det_score"),
                        pose={
                            "pitch": face.get("pitch"),
                            "yaw": face.get("yaw"),
                            "roll": face.get("roll"),
                            "pose": face.get("pose"),
                        },
                        embedding=face["embedding"],
                        data=face.get("data")
                    )
                    for face in faces_metadata
                ]

                # Calculate face count
                face_count = len(face_objs)

                # Accumulate data for batch processing
                async with lock:
                    face_objects.extend(face_objs)
                    downloaded_photo_ids.append(photo_id)
                    face_count_batch.append(face_count)

                    # Flush if batch size is reached
                    if len(downloaded_photo_ids) >= update_batch_size:
                        await flush(db, face_objects.copy(), downloaded_photo_ids.copy(), face_count_batch.copy())
                        face_objects.clear()
                        downloaded_photo_ids.clear()
                        face_count_batch.clear()

                print(f"✅ Processed {photo_id} with {len(face_objs)} faces", flush=True)

        except Exception as e:
            print(f"❌ Error for {photo_id}: {type(e).__name__}: {str(e)}", flush=True)

async def orchestrate(db, partner: str, photos: list, concurrency: int, update_batch_size: int, 
                     face_objects: list, downloaded_photo_ids: list, face_count_batch: list, lock: asyncio.Lock):
    """
    Orchestrate asynchronous fetching and processing of all photos.
    
    Args:
        db: SQLAlchemy session
        partner (str): Partner identifier
        photos (list): List of Photo objects with photo_id and url
        concurrency (int): Number of concurrent downloads
        update_batch_size (int): Batch size for flushing data
        face_objects (list): Shared list to accumulate Face objects
        downloaded_photo_ids (list): Shared list to accumulate photo IDs
        face_count_batch (list): Shared list to accumulate face counts
        lock (asyncio.Lock): Lock for thread-safe batch updates
    """
    try:
        print("🚀 Starting photo processing...", flush=True)

        sem = asyncio.Semaphore(concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = [
                fetch_and_process(
                    index, photo.photo_id, photo.url, session, sem, db, partner,
                    face_objects, downloaded_photo_ids, face_count_batch, lock, update_batch_size
                )
                for index, photo in enumerate(photos)
            ]
            await asyncio.gather(*tasks)

        # Flush any residual data
        if face_objects or downloaded_photo_ids:
            await flush(db, face_objects, downloaded_photo_ids, face_count_batch)

        print("✅ All photos processed", flush=True)

    except Exception as e:
        print(f"🔥 Orchestration error: {e}", flush=True)
        raise

async def process_photos_async(partner: str, concurrency: int = 5, limit: int = 100, update_batch_size: int = 25):
    """
    Process photos asynchronously for a given partner.
    
    Args:
        partner (str): Partner identifier
        concurrency (int): Number of concurrent downloads
        limit (int): Maximum number of photos to process
        update_batch_size (int): Batch size for flushing data
    
    Returns:
        dict: Status message
    """
    db = SessionLocal()  # Assuming SessionLocal is your SQLAlchemy session factory
    try:
        photos = (
            db.query(Photo)
            .filter(Photo.is_downloaded == False, Photo.partner == partner, Photo.is_video == False)
            .limit(limit)
            .all()
        )

        if not photos:
            return {"status": "No photos found to process."}

        # Initialize shared state
        face_objects = []
        downloaded_photo_ids = []
        face_count_batch = []
        lock = asyncio.Lock()

        # Run orchestration
        await orchestrate(db, partner, photos, concurrency, update_batch_size, 
                         face_objects, downloaded_photo_ids, face_count_batch, lock)

        return {"status": f"Processed {len(photos)} photos for partner {partner}"}

    except Exception as e:
        db.rollback()
        print(f"❌ Process photos error: {e}", flush=True)
        raise
    finally:
        db.close()