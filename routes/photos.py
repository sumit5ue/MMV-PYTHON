from fastapi import BackgroundTasks, APIRouter, HTTPException
from pydantic import BaseModel
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
    isVideo: Optional[bool] = False
    saliency: Optional[SaliencyBox] = None
    aestheticScore: Optional[float] = None
    weightedScore: Optional[float] = None
    path: Optional[str] = None


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
            path TEXT
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
            INSERT INTO photos (id, partner, url, rosterId, caption, isVideo, sal_x, sal_y, sal_w, sal_h, aestheticScore, weightedScore, path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                path=excluded.path
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
            photo.path
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
    try:
        conn = get_connection()
        cur = conn.cursor()

        for photo in photos:
            sal = photo.saliency or SaliencyBox(x=None, y=None, w=None, h=None)

            cur.execute('''
                INSERT INTO photos (id, partner, url, rosterId, caption, isVideo, sal_x, sal_y, sal_w, sal_h, aestheticScore, weightedScore, path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    path=excluded.path
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
                photo.path
            ))

            cur.execute("DELETE FROM shared_with WHERE photoId = ?", (photo.id,))
            for shared in set(photo.sharedWith):
                cur.execute("INSERT INTO shared_with (photoId, sharedWith) VALUES (?, ?)", (photo.id, shared))

        conn.commit()
        return {"status": "ok", "count": len(photos)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
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
            "path": row[12]
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
                "path": row[12]
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

    async def fetch_and_process(index, photo_id, url, session, sem, global_counter):
        async with sem:
            print(f"[{index+1}] Downloading {photo_id}")
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        img_bytes = await resp.read()
                        image = np.array(Image.open(BytesIO(img_bytes)))
                        return detect_and_embed_faces_from_array(image, partner, photo_id, global_counter)
                    else:
                        log_error(photo_id, f"HTTP {resp.status} for {url}")
            except Exception as e:
                log_error(photo_id, e)
            return global_counter

    async def orchestrate():
        sem = asyncio.Semaphore(concurrency)
        async with aiohttp.ClientSession() as session:
            global_vector_id = 0
            for index, (photo_id, url) in enumerate(photos):
                global_vector_id = await fetch_and_process(index, photo_id, url, session, sem, global_vector_id)

    background_tasks.add_task(orchestrate)
    return {
        "status": f"Started processing {len(photos)} photos for {partner}."
    }
