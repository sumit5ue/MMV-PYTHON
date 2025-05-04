# utils/file_utils.py

import os
from PIL import Image

BASE_DIR = "/Users/sumit/Documents/ai_analysis"  # or use Path if preferred


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_face_crop(image, bbox, save_path):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image.crop((x1, y1, x2, y2))
    cropped.save(save_path)

def get_photo_path(partner: str, photo_id: str) -> str:
    return os.path.join(BASE_DIR , partner, "photos", f"{photo_id}.jpg")

