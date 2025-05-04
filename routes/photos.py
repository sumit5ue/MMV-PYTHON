from fastapi import BackgroundTasks, APIRouter, HTTPException
from pydantic import BaseModel,field_validator
from typing import List, Optional
import sqlite3
import aiohttp
import asyncio
import uuid
import mimetypes
import os
import json
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
from services.insightface_service import detect_and_embed_faces_from_array
import numpy as np
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime

DB_PATH = Path("/Users/sumit/Documents/ai_analysis/data.db")
DOWNLOAD_DIR = Path("/Users/sumit/Documents/ai_analysis")
ERROR_LOG_PATH = Path(__file__).parent / "photo_errors.json"

router = APIRouter()

# ---------- SCHEMAS ----------

class SaliencyBox(BaseModel):
    x: Optional[float]
    y: Optional[float]
    w: Optional[float]
    h: Optional[float]

class Photo(BaseModel):
    id: str
    partner: str
    url: str
    sharedWith: List[str] = []
    rosterId: Optional[str] = None
    caption: Optional[str] = None
    is_video: Optional[bool] = False
    is_clip_created: Optional[bool] = False
    saliency: Optional[SaliencyBox] = None
    aestheticScore: Optional[float] = None
    weightedScore: Optional[float] = None
    path: Optional[str] = None
    created_at: Optional[str] = None  # <-- updated to str

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, v):
        if v is not None:
            try:
                datetime.fromisoformat(v)
            except ValueError:
                raise ValueError("created_at must be in ISO 8601 format (e.g., 2025-05-01T14:33:00)")
        return v


class PhotoDownloadRequest(BaseModel):
    partner: str
    limit: int = 100
    concurrency: int = 20
    update_batch_size: int = 20

# ---------- DB HELPERS ----------

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            id TEXT PRIMARY KEY,
            partner TEXT,
            url TEXT UNIQUE,
            rosterId TEXT,
            caption TEXT,
            isVideo INTEGER,
            sal_x REAL,
            sal_y REAL,
            sal_w REAL,
            sal_h REAL,
            aestheticScore REAL,
            weightedScore REAL,
            path TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS shared_with (
            photoId TEXT,
            sharedWith TEXT,
            PRIMARY KEY (photoId, sharedWith),
            FOREIGN KEY (photoId) REFERENCES photos(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    return conn

# ---------- ENDPOINTS ----------

@router.post("/db/photo")
async def add_or_update_photo(photo: Photo):
    try:
        conn = get_connection()
        cur = conn.cursor()

        sal = photo.saliency or SaliencyBox(x=None, y=None, w=None, h=None)

        cur.execute('''
            INSERT INTO photos (id, partner, url, rosterId, caption, isVideo, sal_x, sal_y, sal_w, sal_h, aestheticScore, weightedScore, path,created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                partner=excluded.partner,
                url=excluded.url,
                rosterId=excluded.rosterId,
                caption=excluded.caption,
                isVideo=excluded.isVideo,
                sal_x=excluded.sal_x,
                sal_y=excluded.sal_y,
                sal_w=excluded.sal_w,
                sal_h=excluded.sal_h,
                aestheticScore=excluded.aestheticScore,
                weightedScore=excluded.weightedScore,
                path=excluded.path,
                created_at=COALESCE(excluded.created_at, photos.created_at)
        ''', (
            photo.id,
            photo.partner,
            photo.url,
            photo.rosterId,
            photo.caption,
            int(photo.isVideo),
            sal.x, sal.y, sal.w, sal.h,
            photo.aestheticScore,
            photo.weightedScore,
            photo.path,
            photo.created_at or datetime.now().isoformat()

        ))

        cur.execute("DELETE FROM shared_with WHERE photoId = ?", (photo.id,))
        for shared in set(photo.sharedWith):
            cur.execute("INSERT INTO shared_with (photoId, sharedWith) VALUES (?, ?)", (photo.id, shared))

        conn.commit()
        return {"status": "ok", "id": photo.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.post("/db/photos/bulk")
async def bulk_insert_update_photos(photos: List[Photo]):
    conn = None
    errors = {
        "shared_with_insertion_errors": [],
        "shared_with_skipped": [],
        "other_errors": []
    }
    debug_log = []  # Track all shared_with outcomes
    
    try:
        conn = get_connection()
        conn.execute("PRAGMA foreign_keys = ON")  # Ensure foreign key constraints
        try:
            conn.execute("PRAGMA strict = ON")  # Enforce strict mode (SQLite >= 3.31.0)
        except sqlite3.OperationalError:
            print("PRAGMA strict not supported in this SQLite version")
        cur = conn.cursor()
        total_shared_with_received = 0
        total_shared_with_inserted = 0
        total_shared_with_skipped = 0

        for photo in photos:
            sal = photo.saliency or SaliencyBox(x=None, y=None, w=None, h=None)

            # Insert or update photo
            try:
                cur.execute('''
                    INSERT INTO photos (id, partner, url, rosterId, caption, isVideo, sal_x, sal_y, sal_w, sal_h, aestheticScore, weightedScore, path,created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        partner=excluded.partner,
                        url=excluded.url,
                        rosterId=excluded.rosterId,
                        caption=excluded.caption,
                        isVideo=excluded.isVideo,
                        sal_x=excluded.sal_x,
                        sal_y=excluded.sal_y,
                        sal_w=excluded.sal_w,
                        sal_h=excluded.sal_h,
                        aestheticScore=excluded.aestheticScore,
                        weightedScore=excluded.weightedScore,
                        path=excluded.path,
                        created_at=COALESCE(excluded.created_at, photos.created_at)

                ''', (
                    photo.id,
                    photo.partner,
                    photo.url,
                    photo.rosterId,
                    photo.caption,
                    int(photo.isVideo),
                    sal.x, sal.y, sal.w, sal.h,
                    photo.aestheticScore,
                    photo.weightedScore,
                    photo.path,
                    photo.created_at or datetime.now().isoformat()



                ))
            except Exception as e:
                print(f"Failed to insert/update photoId {photo.id}: {str(e)}")
                errors["other_errors"].append({"photoId": photo.id, "error": f"Failed to insert/update photo: {str(e)}"})
                debug_log.append({"photoId": photo.id, "action": "photo_insert", "status": "failed", "error": str(e)})
                continue

            # Handle shared_with entries
            cur.execute("DELETE FROM shared_with WHERE photoId = ?", (photo.id,))
            shared_with_unique = set(photo.sharedWith)
            shared_with_count = len(shared_with_unique)
            total_shared_with_received += shared_with_count
            debug_log.append({"photoId": photo.id, "sharedWith_count": shared_with_count, "sharedWith_values": list(shared_with_unique)})
            
            inserted_count = 0
            for shared in shared_with_unique:
                outcome = {"photoId": photo.id, "sharedWith": str(shared), "status": None}
                try:
                    if shared and isinstance(shared, str) and shared.strip():
                        print(f"Attempting to insert shared_with for photoId {photo.id}, sharedWith '{shared}'")
                        cur.execute("INSERT OR FAIL INTO shared_with (photoId, sharedWith) VALUES (?, ?)", (photo.id, shared))
                        inserted_count += 1
                        print(f"Successfully inserted shared_with for photoId {photo.id}, sharedWith '{shared}'")
                        outcome["status"] = "inserted"
                    else:
                        print(f"Skipped shared_with for photoId {photo.id}, sharedWith '{shared}' (invalid: null, empty, or non-string)")
                        errors["shared_with_skipped"].append({
                            "photoId": photo.id,
                            "sharedWith": str(shared),
                            "reason": f"Invalid sharedWith ID (value: '{shared}')"
                        })
                        total_shared_with_skipped += 1
                        outcome["status"] = "skipped"
                except sqlite3.IntegrityError as e:
                    print(f"IntegrityError for photoId {photo.id}, sharedWith '{shared}': {str(e)}")
                    errors["shared_with_insertion_errors"].append({
                        "photoId": photo.id,
                        "sharedWith": str(shared),
                        "error": f"IntegrityError: {str(e)}"
                    })
                    outcome["status"] = "failed_integrity"
                except Exception as e:
                    print(f"Unexpected error for photoId {photo.id}, sharedWith '{shared}': {str(e)}")
                    errors["shared_with_insertion_errors"].append({
                        "photoId": photo.id,
                        "sharedWith": str(shared),
                        "error": f"Unexpected error: {str(e)}"
                    })
                    outcome["status"] = "failed_unexpected"
                debug_log.append(outcome)
                print(f"Outcome for photoId {photo.id}, sharedWith '{shared}': {outcome['status']}")

            total_shared_with_inserted += inserted_count
            if inserted_count != shared_with_count:
                print(f"Warning: Inserted {inserted_count} of {shared_with_count} sharedWith entries for photoId {photo.id}")
                debug_log.append({"photoId": photo.id, "warning": f"Inserted {inserted_count} of {shared_with_count} sharedWith entries"})

        conn.commit()
        print(f"Total shared_with entries received: {total_shared_with_received}")
        print(f"Total shared_with entries inserted: {total_shared_with_inserted}")
        print(f"Total shared_with entries skipped: {total_shared_with_skipped}")
        print(f"Total shared_with entries failed: {len(errors['shared_with_insertion_errors'])}")
        debug_log.append({
            "summary": {
                "received": total_shared_with_received,
                "inserted": total_shared_with_inserted,
                "skipped": total_shared_with_skipped,
                "failed": len(errors["shared_with_insertion_errors"])
            }
        })
        
        # Save errors to JSON
        error_file = "/Users/sumit/MMV-PYTHON/endpoint_errors.json"
        fallback_error_file = "/tmp/endpoint_errors.json"
        try:
            print(f"Attempting to save errors to {error_file}")
            with open(error_file, "w") as f:
                json.dump(errors, f, indent=2)
            print(f"Successfully saved endpoint errors to {error_file}")
        except Exception as e:
            print(f"Failed to save endpoint errors to {error_file}: {str(e)}")
            try:
                print(f"Attempting to save errors to fallback {fallback_error_file}")
                with open(fallback_error_file, "w") as f:
                    json.dump(errors, f, indent=2)
                print(f"Successfully saved endpoint errors to {fallback_error_file}")
            except Exception as e2:
                print(f"Failed to save endpoint errors to {fallback_error_file}: {str(e2)}")
                print(f"Error log contents: {json.dumps(errors, indent=2)}")
        
        # Save debug log
        debug_file = "/Users/sumit/MMV-PYTHON/endpoint_debug.json"
        try:
            print(f"Attempting to save debug log to {debug_file}")
            with open(debug_file, "w") as f:
                json.dump(debug_log, f, indent=2)
            print(f"Successfully saved debug log to {debug_file}")
        except Exception as e:
            print(f"Failed to save debug log to {debug_file}: {str(e)}")
            print(f"Debug log contents: {json.dumps(debug_log, indent=2)}")
        
        return {
            "status": "ok",
            "count": len(photos),
            "shared_with_received": total_shared_with_received,
            "shared_with_inserted": total_shared_with_inserted,
            "shared_with_skipped": total_shared_with_skipped,
            "shared_with_failed": len(errors["shared_with_insertion_errors"]),
            "errors": errors["shared_with_insertion_errors"] + errors["shared_with_skipped"] + errors["other_errors"]
        }
    
    except Exception as e:
        if conn:
            conn.rollback()
        errors["other_errors"].append({"error": f"Unexpected error: {str(e)}"})
        debug_log.append({"error": f"Unexpected error: {str(e)}"})
        
        # Save errors to JSON
        error_file = "/Users/sumit/MMV-PYTHON/endpoint_errors.json"
        fallback_error_file = "/tmp/endpoint_errors.json"
        try:
            print(f"Attempting to save errors to {error_file}")
            with open(error_file, "w") as f:
                json.dump(errors, f, indent=2)
            print(f"Successfully saved endpoint errors to {error_file}")
        except Exception as e:
            print(f"Failed to save endpoint errors to {error_file}: {str(e)}")
            try:
                print(f"Attempting to save errors to fallback {fallback_error_file}")
                with open(fallback_error_file, "w") as f:
                    json.dump(errors, f, indent=2)
                print(f"Successfully saved endpoint errors to {fallback_error_file}")
            except Exception as e2:
                print(f"Failed to save endpoint errors to {fallback_error_file}: {str(e2)}")
                print(f"Error log contents: {json.dumps(errors, indent=2)}")
        
        # Save debug log
        debug_file = "/Users/sumit/MMV-PYTHON/endpoint_debug.json"
        try:
            print(f"Attempting to save debug log to {debug_file}")
            with open(debug_file, "w") as f:
                json.dump(debug_log, f, indent=2)
            print(f"Successfully saved debug log to {debug_file}")
        except Exception as e:
            print(f"Failed to save debug log to {debug_file}: {str(e)}")
            print(f"Debug log contents: {json.dumps(debug_log, indent=2)}")
        
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if conn:
            conn.close()
            
@router.get("/db/photo/{photo_id}")
async def get_photo(photo_id: str):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Photo not found")

        cur.execute("SELECT sharedWith FROM shared_with WHERE photoId = ?", (photo_id,))
        shared_with = [r[0] for r in cur.fetchall()]

        return {
            "id": row[0],
            "partner": row[1],
            "url": row[2],
            "sharedWith": shared_with,
            "rosterId": row[3],
            "caption": row[4],
            "isVideo": bool(row[5]),
            "saliency": {
                "x": row[6], "y": row[7], "w": row[8], "h": row[9]
            } if row[6] is not None else None,
            "aestheticScore": row[10],
            "weightedScore": row[11],
            "path": row[12],
            "created_at": row[13]

        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.delete("/db/photo/{photo_id}")
async def delete_photo(photo_id: str):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
        cur.execute("DELETE FROM shared_with WHERE photoId = ?", (photo_id,))
        conn.commit()
        return {"status": "deleted", "id": photo_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.get("/db/photos")
async def list_photos(partner: Optional[str] = None, sharedWith: Optional[str] = None):
    try:
        conn = get_connection()
        cur = conn.cursor()

        if sharedWith:
            cur.execute('''
                SELECT p.* FROM photos p
                JOIN shared_with s ON p.id = s.photoId
                WHERE s.sharedWith = ?
            ''', (sharedWith,))
        elif partner:
            cur.execute("SELECT * FROM photos WHERE partner = ?", (partner,))
        else:
            cur.execute("SELECT * FROM photos")

        rows = cur.fetchall()
        results = []
        for row in rows:
            cur.execute("SELECT sharedWith FROM shared_with WHERE photoId = ?", (row[0],))
            shared_with = [r[0] for r in cur.fetchall()]
            results.append({
                "id": row[0],
                "partner": row[1],
                "url": row[2],
                "sharedWith": shared_with,
                "rosterId": row[3],
                "caption": row[4],
                "isVideo": bool(row[5]),
                "saliency": {
                    "x": row[6], "y": row[7], "w": row[8], "h": row[9]
                } if row[6] is not None else None,
                "aestheticScore": row[10],
                "weightedScore": row[11],
                "path": row[12],
                "created_at": row[13]

            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@router.post("/db/photos/download-missing")
async def download_missing_photos(
    body: PhotoDownloadRequest,
    background_tasks: BackgroundTasks
):
    partner = body.partner
    limit = body.limit
    concurrency = body.concurrency
    update_batch_size = body.update_batch_size

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, url FROM photos
        WHERE path IS NULL AND partner = ? AND isVideo = 0
    """, (partner,))
    photos = cur.fetchall()
    conn.close()

    if not photos:
        return {"status": "no photos to download"}

    output_dir = DOWNLOAD_DIR / partner / "photos"
    output_dir.mkdir(parents=True, exist_ok=True)

    async def download_and_save(session, photo_id, url, index):
        try:
            print(f"[{index+1}] Downloading photo {photo_id}")
            async with session.get(url) as resp:
                if resp.status == 200:
                    content_type = resp.headers.get("Content-Type", "")
                    ext = mimetypes.guess_extension(content_type.split(";")[0]) or ".jpg"
                    file_name = f"{uuid.uuid4()}{ext}"
                    file_path = output_dir / file_name
                    data = await resp.read()
                    with open(file_path, "wb") as f:
                        f.write(data)
                    return (photo_id, str(file_path))
                else:
                    log_error(photo_id, f"HTTP {resp.status} for {url}")
        except Exception as e:
            log_error(photo_id, e)
        return None

    def update_db_paths(batch):
        try:
            conn = get_connection()
            cur = conn.cursor()
            for photo_id, path in batch:
                cur.execute("UPDATE photos SET path = ? WHERE id = ?", (path, photo_id))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error while updating DB paths: {e}")

    async def worker():
        results = []
        sem = asyncio.Semaphore(concurrency)
        async with aiohttp.ClientSession() as session:
            tasks = []

            async def sem_task(index, photo_id, url):
                async with sem:
                    return await download_and_save(session, photo_id, url, index)

            for index, (photo_id, url) in enumerate(photos):
                tasks.append(sem_task(index, photo_id, url))

            for i, task in enumerate(asyncio.as_completed(tasks), start=1):
                result = await task
                if result:
                    results.append(result)
                if len(results) >= update_batch_size or (i == len(photos) and results):
                    update_db_paths(results)
                    results.clear()

    background_tasks.add_task(worker)
    return {
        "status": f"Started downloading {len(photos)} photos for partner '{partner}'",
        "saving_to": str(output_dir)
    }

def log_error(photo_id: str, error: Exception):
    try:
        if ERROR_LOG_PATH.exists():
            with open(ERROR_LOG_PATH, "r") as f:
                existing = json.load(f)
        else:
            existing = {}

        existing[photo_id] = str(error)

        with open(ERROR_LOG_PATH, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print(f"Error while writing to log file: {e}")

@router.post("/db/photos/download-and-process-faces")
async def download_and_process_faces(
    body: PhotoDownloadRequest,
    background_tasks: BackgroundTasks
):
    partner = body.partner
    concurrency = body.concurrency
    limit = body.limit
    update_batch_size = body.update_batch_size

    SAVE_DIR = Path("/Users/sumit/Documents/ai_analysis") / partner / "photos"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Get photos to process
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, url FROM photos
        WHERE path IS NULL AND partner = ? AND isVideo = 0
        LIMIT ?
    """, (partner, limit))
    photos = cur.fetchall()
    conn.close()

    if not photos:
        return {"status": "No photos found to process."}

    photo_results = []
    all_faces = []
    lock = asyncio.Lock()
    global_vector_id = 0

    # Step 2: Flush functions
    def flush_photos_and_faces(photo_batch, face_batch):
        conn = get_connection()
        cur = conn.cursor()

        for photo in photo_batch:
            cur.execute("""
                UPDATE photos
                SET path = ?
                WHERE id = ?
            """, (photo["path"],  photo["photoId"]))

        for face in face_batch:
            cur.execute("""
                INSERT INTO face_embeddings (
                    faceId, photoId, partner, embedding,
                    bb_x1, bb_y1, bb_x2, bb_y2,
                    gender, age, detScore,
                    yaw, pitch, roll, pose
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                face["faceId"],
                face["photoId"],
                partner,
                face["embedding"].tobytes(),
                face["bbox"][0], face["bbox"][1],
                face["bbox"][2], face["bbox"][3],
                face["gender"],
                face["age"],
                face["detScore"],
                face["yaw"],
                face["pitch"],
                face["roll"],
                face["pose"]
            ))

        conn.commit()
        conn.close()
        print(f"📦 Flushed {len(photo_batch)} photos and {len(face_batch)} faces", flush=True)

    # Step 3: Process one photo
    async def fetch_and_process(index, photo_id, url, session, sem):
        nonlocal global_vector_id
        async with sem:
            print(f"[{index+1}] Downloading {photo_id}", flush=True)
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        img_bytes = await resp.read()
                        pil_img = Image.open(BytesIO(img_bytes))
                        rotated = apply_exif_rotation(pil_img)
                        image = np.array(rotated)

                        # Save image
                        file_path = SAVE_DIR / f"{photo_id}.jpg"
                        rotated.save(file_path)

                        # Detect faces
                        global_vector_id, faces_metadata = detect_and_embed_faces_from_array(
                            image=image,
                            partner=partner,
                            photo_id=photo_id,
                            global_vector_id=global_vector_id,
                        )

                        photo_result = {
                            "photoId": photo_id,
                            "path": str(file_path),
                            "faceCount": len(faces_metadata),
                        }

                        async with lock:
                            photo_results.append(photo_result)
                            all_faces.extend(faces_metadata)

                            if len(photo_results) >= update_batch_size:
                                flush_photos_and_faces(photo_results.copy(), all_faces.copy())
                                photo_results.clear()
                                all_faces.clear()

                        print(f"✅ Completed {photo_id} with {len(faces_metadata)} faces", flush=True)
                    else:
                        log_error(photo_id, f"HTTP {resp.status} for {url}")
            except Exception as e:
                print(f"❌ Error for {photo_id}: {e}", flush=True)
                log_error(photo_id, str(e))

    # Step 4: Run all tasks
    async def orchestrate():
        try:
            print("🚀 Starting processing...", flush=True)
            sem = asyncio.Semaphore(concurrency)
            async with aiohttp.ClientSession() as session:
                tasks = [
                    fetch_and_process(index, photo_id, url, session, sem)
                    for index, (photo_id, url) in enumerate(photos)
                ]
                await asyncio.gather(*tasks)

            # Final flush
            if photo_results or all_faces:
                flush_photos_and_faces(photo_results, all_faces)

            print("✅ All tasks complete", flush=True)
        except Exception as e:
            print(f"🔥 Orchestration error: {e}", flush=True)

    await orchestrate()

    return {
        "status": f"Processed {len(photos)} photos for {partner}."
    }

def apply_exif_rotation(image: Image.Image) -> Image.Image:
    try:
        exif = image._getexif()
        if exif is not None:
            orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
    except Exception:
        pass
    return image
