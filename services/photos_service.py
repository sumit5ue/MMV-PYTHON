# services/photos_service.py

from config import get_photos_dir, get_metadata_path
from utils.jsonl_utils import append_jsonl, load_jsonl, save_jsonl
import os
import uuid
from PIL import Image

def process_photos_for_partner(partner: str):
    photos_dir = get_photos_dir(partner)
    metadata_path = get_metadata_path(partner)

    if not os.path.exists(photos_dir):
        raise Exception(f"Photos folder does not exist: {photos_dir}")

    photo_files = [f for f in os.listdir(photos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    existing_metadata = load_jsonl(metadata_path)
    existing_ids = set(item['id'] for item in existing_metadata)

    new_metadata = []
    for idx, filename in enumerate(photo_files):
        photo_id = os.path.splitext(filename)[0]

        if photo_id in existing_ids:
            continue  # Skip already processed

        item = {
            "id": photo_id,
            "fileName": filename,
            "isUtility": False,
            "objectSaliency": [],
            "aestheticScore": None,
            "faceCount": None,
            "clip": {"processed": False},
            "dino": {"processed": False}
        }
        new_metadata.append(item)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(photo_files)} photos...")

    append_jsonl_batch(new_metadata, metadata_path)
    print(f"✅ Photo metadata updated for partner {partner}")

def append_jsonl_batch(items, path):
    with open(path, "a") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
