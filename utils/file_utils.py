# utils/file_utils.py

import os
from PIL import Image
import pillow_heif
BASE_DIR = "/Users/sumit/Documents/ai_analysis"  # or use Path if preferred
pillow_heif.register_heif_opener()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_face_crop(image, bbox, save_path):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image.crop((x1, y1, x2, y2))
    cropped.save(save_path)

def get_photo_path(partner: str, photo_id: str, pil_img: Image.Image) -> str:
    fmt = pil_img.format.upper() if pil_img.format else "JPEG"

    if fmt in ["HEIC", "HEIF", None]:
        ext = "jpg"
    else:
        ext = fmt.lower()

    return os.path.join(BASE_DIR, partner, "photos", f"{photo_id}.{ext}")

# def get_photo_path(partner: str, photo_id: str) -> str:
#     return os.path.join(BASE_DIR , partner, "photos", f"{photo_id}.jpg")
