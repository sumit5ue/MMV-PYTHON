# services/dino_service.py

import torch
import torchvision.transforms as T
import numpy as np
import os
import traceback
from PIL import Image
from config import get_embeddings_dir, get_metadata_path, get_photos_dir
from utils.jsonl_utils import load_jsonl, save_jsonl
from utils.error_utils import log_error

# Load DINO model (vit_b_16 pretrained on Imagenet)
from torchvision.models import vit_b_16, ViT_B_16_Weights

device = "mps" if torch.backends.mps.is_available() else "cpu"
dino_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
dino_model.eval()
dino_model = dino_model.to(device)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def embed_dino_for_photo(photo_path: str, partner: str, photo_id: str):
    try:
        image = Image.open(photo_path).convert("RGB")
        image_input = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = dino_model(image_input)
            embedding = embedding.float()
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        embeddings_dir = get_embeddings_dir(partner)
        dino_npy_path = os.path.join(embeddings_dir, f"{partner}_dino.npy")
        os.makedirs(embeddings_dir, exist_ok=True)

        if os.path.exists(dino_npy_path):
            existing = np.load(dino_npy_path)
            combined = np.vstack([existing, embedding.cpu().numpy()])
        else:
            combined = embedding.cpu().numpy()

        np.save(dino_npy_path, combined)

        metadata_path = get_metadata_path(partner)
        update_dino_metadata(metadata_path, photo_id, combined.shape[0] - 1)

    except Exception as e:
        log_error(
            photoId=photo_id,
            faceId=None,
            step="dino",
            errorMessage=str(e),
            traceback=traceback.format_exc(),
            modelType="dino",
            path=photo_path,
            partner=partner
        )

def update_dino_metadata(metadata_path, photo_id, vectorId):
    metadata = load_jsonl(metadata_path)
    for entry in metadata:
        if entry["id"] == photo_id:
            if "dino" not in entry:
                entry["dino"] = {}
            entry["dino"]["vectorId"] = vectorId
            entry["dino"]["processed"] = True
            break
    save_jsonl(metadata, metadata_path)

def process_dino_folder(partner: str):
    photos_dir = get_photos_dir(partner)
    metadata_path = get_metadata_path(partner)

    if not os.path.exists(photos_dir):
        raise Exception(f"No photos found at {photos_dir}")

    photo_files = [f for f in os.listdir(photos_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not os.path.exists(metadata_path):
        raise Exception(f"Metadata file {metadata_path} not found. Run clip embedding first to generate metadata.")

    metadata = load_jsonl(metadata_path)
    id_to_metadata = {item['id']: item for item in metadata}

    processed_count = 0
    skipped_count = 0

    for idx, file in enumerate(photo_files):
        full_path = os.path.join(photos_dir, file)
        photo_id = os.path.splitext(file)[0]

        meta = id_to_metadata.get(photo_id)
        if not meta:
            continue
        if meta.get("dino", {}).get("processed") is True:
            skipped_count += 1
            continue

        embed_dino_for_photo(full_path, partner, photo_id)
        processed_count += 1

        if (processed_count + skipped_count) % 10 == 0:
            print(f"Processed {processed_count}, skipped {skipped_count} out of {len(photo_files)} photos...")

    return {
        "processed": processed_count,
        "skipped": skipped_count,
        "total": len(photo_files)
    }
