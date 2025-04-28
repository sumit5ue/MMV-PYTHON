# utils/file_utils.py

import os
from PIL import Image

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_face_crop(image, bbox, save_path):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image.crop((x1, y1, x2, y2))
    cropped.save(save_path)
