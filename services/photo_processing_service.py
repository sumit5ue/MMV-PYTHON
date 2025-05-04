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

async def process_photos_async(partner, concurrency=5, limit=100, update_batch_size=100):
    db = SessionLocal()
    photos = (
        db.query(Photo)
        .filter(Photo.is_downloaded == False, Photo.partner == partner, Photo.is_video == False)
        .limit(limit)
        .all()
    )

    if not photos:
        return {"status": "No photos found to process."}

    face_objects = []
    downloaded_photo_ids = []
    lock = asyncio.Lock()

    def flush(faces_batch, photo_ids_batch):
        if faces_batch:
            db.bulk_save_objects(faces_batch)

        if photo_ids_batch:
            db.query(Photo).filter(Photo.photo_id.in_(photo_ids_batch)).update(
                {"is_downloaded": True}, synchronize_session=False
            )

        db.commit()
        print(f"📦 Flushed {len(faces_batch)} faces and {len(photo_ids_batch)} photo updates", flush=True)

    async def fetch_and_process(index, photo_id, url, session, sem):
        print("photo_id is",photo_id)
        async with sem:
            print(f"[{index + 1}] Downloading {photo_id}", flush=True)
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        print(f"❌ Failed HTTP {resp.status} for {photo_id}", flush=True)
                        return

                    img_bytes = await resp.read()
                    pil_img = Image.open(BytesIO(img_bytes))
                    rotated = apply_exif_rotation(pil_img)
                    image = np.array(rotated)

                    file_path = get_photo_path(partner, photo_id)
                    rotated.save(file_path)

                    faces_metadata = detect_and_embed_faces_from_array(
                        image=image,
                        partner=partner,
                        photo_id=photo_id
                    )

                    face_objs = [
                        Face(
                            partner=partner,
                            photo_id=photo_id,
                            bbox={
                                "x": face["bbox"][0],
                                "y": face["bbox"][1],
                                "width": face["bbox"][2]-face["bbox"][0],
                                "height": face["bbox"][3]-face["bbox"][1]
                            },
                            det_score=face.get("det_score"),                            
                            
                            pose={
                            "pitch": face.get("pitch"),
                            "yaw": face.get("yaw"),
                            "roll": face.get("roll"),
                            "pose":face.get("pose"),
                            },
                            
                            embedding=face["embedding"],
                            data=face.get("data")
                        )
                        for face in faces_metadata
                    ]

                    async with lock:
                        face_objects.extend(face_objs)
                        downloaded_photo_ids.append(photo_id)

                        if len(downloaded_photo_ids) >= update_batch_size:
                            flush(face_objects.copy(), downloaded_photo_ids.copy())
                            face_objects.clear()
                            downloaded_photo_ids.clear()

                    print(f"✅ Processed {photo_id} with {len(face_objs)} faces", flush=True)

            except Exception as e:
                print(f"❌ Error for {photo_id}: {e}", flush=True)

    async def orchestrate():
        try:
            print("🚀 Starting photo processing...", flush=True)
            sem = asyncio.Semaphore(concurrency)
            async with aiohttp.ClientSession() as session:
                tasks = [
                    fetch_and_process(index, photo.photo_id, photo.url, session, sem)
                    for index, photo in enumerate(photos)
                ]
                await asyncio.gather(*tasks)

            if face_objects or downloaded_photo_ids:
                flush(face_objects, downloaded_photo_ids)

            print("✅ All photos processed", flush=True)

        except Exception as e:
            print(f"🔥 Orchestration error: {e}", flush=True)

    await orchestrate()

    return {"status": f"Processed {len(photos)} photos for partner '{partner}'."}
