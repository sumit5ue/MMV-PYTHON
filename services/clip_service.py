# services/clip_service.py

import torch
import clip
import numpy as np
import os
import traceback
from PIL import Image
from config import get_embeddings_dir, get_metadata_path, get_photos_dir
from utils.jsonl_utils import load_jsonl, save_jsonl
from utils.error_utils import log_error

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
